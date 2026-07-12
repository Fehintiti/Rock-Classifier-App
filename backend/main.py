from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from PIL import Image
import httpx
import json
import io
import os
import uuid
from datetime import datetime, timezone

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://fehintiti.github.io",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase is used to persist user feedback (image + correction) for future
# retraining. SUPABASE_SERVICE_KEY must be the service_role key (server-side
# only) so writes bypass row-level security; never expose it to the frontend.
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "feedback-images")
SUPABASE_TABLE = os.environ.get("SUPABASE_TABLE", "feedback")


class HierarchicalModel(nn.Module):
    def __init__(self, backbone_name='convnext_tiny', num_l1=3, num_l2=41, dropout=0.6):
        super().__init__()
        if backbone_name == 'convnext_tiny':
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            self.backbone = convnext_tiny(weights=weights)
            num_features = 768

        self.backbone.classifier = nn.Identity()

        self.l1_head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Dropout(dropout * 0.7),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(512, num_l1)
        )

        self.l2_head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Dropout(dropout),
            nn.Linear(num_features, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_l2)
        )

    def forward(self, x):
        features = self.backbone(x)
        if features.dim() == 4:
            features = features.flatten(1)
        l1_logits = self.l1_head(features)
        l2_logits = self.l2_head(features)
        return l1_logits, l2_logits


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model_cleaned_best.pth")
info_path = os.path.join(BASE_DIR, "deployment_info.json")

# The model file is too large for Git (116MB, over GitHub's 100MB limit) so
# it isn't committed to the repo. Locally you can just drop it next to this
# file; in Cloud Run (built from a fresh CI checkout that never has it) it's
# fetched from Cloud Storage at startup instead.
MODEL_GCS_BUCKET = os.environ.get("MODEL_GCS_BUCKET", "")
MODEL_GCS_BLOB = os.environ.get("MODEL_GCS_BLOB", "model_cleaned_best.pth")

if not os.path.exists(model_path):
    if not MODEL_GCS_BUCKET:
        raise RuntimeError(
            "model_cleaned_best.pth not found next to main.py, and "
            "MODEL_GCS_BUCKET is not set to download it from Cloud Storage. "
            "See DEPLOYMENT.md."
        )
    from google.cloud import storage
    print(f"Downloading model from gs://{MODEL_GCS_BUCKET}/{MODEL_GCS_BLOB} ...")
    storage.Client().bucket(MODEL_GCS_BUCKET).blob(MODEL_GCS_BLOB).download_to_filename(model_path)
    print("Model downloaded.")

with open(info_path, 'r') as f:
    info = json.load(f)

device = torch.device('cpu')
model = HierarchicalModel(num_l1=3, num_l2=len(info['l2_classes']))
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


@app.get("/")
def root():
    return {"message": "Rock Classifier API running!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            l1_logits, l2_logits = model(img_tensor)
            l1_probs = torch.softmax(l1_logits, dim=1)
            l2_probs = torch.softmax(l2_logits, dim=1)

        l1_conf, l1_idx = l1_probs.max(dim=1)
        l2_top3_probs, l2_top3_idx = torch.topk(l2_probs, k=3, dim=1)

        return {
            "l1_class": info['l1_classes'][l1_idx.item()],
            "l1_confidence": float(l1_conf.item()),
            "l2_predictions": [
                {"rock_type": info['l2_classes'][idx.item()], "confidence": float(prob.item())}
                for prob, idx in zip(l2_top3_probs[0], l2_top3_idx[0])
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def submit_feedback(
    file: UploadFile = File(...),
    model_prediction_type: str = Form(...),
    model_prediction_name: str = Form(...),
    user_correction_type: str = Form(...),
    user_correction_name: str = Form(...),
    certainty: str = Form(...)
):
    """Save user feedback (image + correction) to Supabase for model improvement"""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise HTTPException(
            status_code=500,
            detail="Feedback storage is not configured: SUPABASE_URL / SUPABASE_SERVICE_KEY are missing.",
        )

    try:
        feedback_id = f"fb_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now(timezone.utc).isoformat()
        image_data = await file.read()
        image_path = f"feedback/{timestamp.split('T')[0]}/{feedback_id}.jpg"
        content_type = file.content_type or "image/jpeg"

        auth_headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        }

        async with httpx.AsyncClient(timeout=30) as client:
            upload_resp = await client.post(
                f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{image_path}",
                headers={**auth_headers, "Content-Type": content_type},
                content=image_data,
            )
            if upload_resp.status_code not in (200, 201):
                raise HTTPException(
                    status_code=502,
                    detail=f"Failed to upload feedback image: {upload_resp.text}",
                )

            insert_resp = await client.post(
                f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}",
                headers={
                    **auth_headers,
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                },
                json={
                    "feedback_id": feedback_id,
                    "image_path": image_path,
                    "model_predicted_type": model_prediction_type,
                    "model_predicted_name": model_prediction_name,
                    "user_corrected_type": user_correction_type,
                    "user_corrected_name": user_correction_name,
                    "certainty": certainty,
                    "created_at": timestamp,
                },
            )
            if insert_resp.status_code not in (200, 201, 204):
                raise HTTPException(
                    status_code=502,
                    detail=f"Failed to save feedback record: {insert_resp.text}",
                )

        return {
            "success": True,
            "message": "Thank you! Your feedback helps improve the model.",
            "feedback_id": feedback_id
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
