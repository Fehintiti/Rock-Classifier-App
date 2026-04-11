from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from PIL import Image
import json
import io
import os
from datetime import datetime
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model_cleaned_best.pth")
info_path = os.path.join(BASE_DIR, "deployment_info.json")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)