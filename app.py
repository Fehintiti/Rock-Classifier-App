import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from PIL import Image
import json
import numpy as np

# Page configuration
st.set_page_config(
    page_title="AI Rock Classifier",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful design
st.markdown("""
<style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .header-title {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
    }
    
    .header-subtitle {
        color: #e0e7ff;
        font-size: 1.2rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    
    /* Top links bar */
    .links-bar {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .links-bar a {
        color: #667eea;
        text-decoration: none;
        font-weight: 600;
        margin: 0 1rem;
        font-size: 1rem;
    }
    
    .links-bar a:hover {
        color: #764ba2;
        text-decoration: underline;
    }
    
    /* Important notice banner */
    .notice-banner {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 5px solid #f59e0b;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .notice-banner strong {
        color: #92400e;
        font-size: 1.1rem;
    }
    
    .notice-banner p {
        color: #78350f;
        margin: 0.5rem 0 0 0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Result cards */
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .confidence-high {
        color: #059669;
        font-weight: 700;
        font-size: 2.5rem;
    }
    
    .confidence-medium {
        color: #d97706;
        font-weight: 700;
        font-size: 2.5rem;
    }
    
    .confidence-low {
        color: #dc2626;
        font-weight: 700;
        font-size: 2.5rem;
    }
    
    .rock-type {
        font-size: 2.5rem;
        font-weight: 800;
        color: #111827;
        margin: 0.5rem 0;
    }
    
    /* Prediction cards - DARKER colors for better visibility */
    .prediction-card {
        background: #f3f4f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #6b4423;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .prediction-card strong {
        color: #1f2937;
        font-size: 1.2rem;
    }
    
    .prediction-card p {
        color: #374151;
        font-weight: 600;
    }
    
    /* Progress bars - darker */
    .progress-bar-bg {
        background: #d1d5db;
        border-radius: 10px;
        height: 20px;
        margin-top: 0.5rem;
        overflow: hidden;
    }
    
    .progress-bar-fill-primary {
        background: linear-gradient(90deg, #059669 0%, #10b981 100%);
        height: 100%;
        transition: width 0.5s;
    }
    
    .progress-bar-fill-secondary {
        background: linear-gradient(90deg, #6b7280 0%, #9ca3af 100%);
        height: 100%;
        transition: width 0.3s;
    }
    
    /* Info boxes */
    .info-box {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fee2e2;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🪨 AI Rock Classifier</h1>
    <p class="header-subtitle">Deep Learning for Field Rock Identification</p>
</div>
""", unsafe_allow_html=True)

# Top links bar
st.markdown("""
<div class="links-bar">
    <a href="YOUR_GITHUB_LINK" target="_blank">📂 Check out the code</a>
    <span style="color: #d1d5db;">|</span>
    <a href="YOUR_MEDIUM_LINK" target="_blank">📖 Story behind the project</a>
    <span style="color: #d1d5db;">|</span>
    <a href="YOUR_LINKEDIN_LINK" target="_blank">👤 Connect on LinkedIn</a>
</div>
""", unsafe_allow_html=True)

# Important notice banner
st.markdown("""
<div class="notice-banner">
    <strong>⚠️ Personal Research Project</strong>
    <p>
        This is a personal project exploring AI applications in geological classification. 
        Built to demonstrate how deep learning can assist field geologists with preliminary rock identification.
        <br><br>
        <strong>Not for professional use:</strong> This is a decision support tool, not a replacement for expert identification. 
        Always confirm predictions with field tests (hardness, acid reaction, hand lens examination).
    </p>
</div>
""", unsafe_allow_html=True)

# Model Architecture - EXACT MATCH from training
class HierarchicalModel(nn.Module):
    """Base hierarchical model - EXACT match from training."""
    def __init__(self, backbone_name='convnext_tiny', num_l1=3, num_l2=41, dropout=0.6):
        super().__init__()

        # Load backbone
        if backbone_name == 'convnext_tiny':
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            self.backbone = convnext_tiny(weights=weights)
            num_features = 768

        # Remove original classifier
        self.backbone.classifier = nn.Identity()

        # L1 head - EXACT architecture
        self.l1_head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Dropout(dropout * 0.7),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(512, num_l1)
        )

        # L2 head - EXACT architecture
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

# Load model and info
@st.cache_resource
def load_model():
    with open('deployment_info.json', 'r') as f:
        info = json.load(f)
    
    device = torch.device('cpu')
    model = HierarchicalModel(
        backbone_name='convnext_tiny',
        num_l1=3, 
        num_l2=len(info['l2_classes']),
        dropout=0.6
    )
    model.load_state_dict(torch.load('model_cleaned_best.pth', map_location=device))
    model.eval()
    
    return model, info, device

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Predict function
def predict_rock(image, model, info, device):
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        l1_logits, l2_logits = model(img_tensor)
        l1_probs = torch.softmax(l1_logits, dim=1)
        l2_probs = torch.softmax(l2_logits, dim=1)
    
    # Get top predictions
    l1_conf, l1_idx = l1_probs.max(dim=1)
    l2_top3_probs, l2_top3_idx = torch.topk(l2_probs, k=3, dim=1)
    
    l1_class = info['l1_classes'][l1_idx.item()]
    l1_confidence = l1_conf.item()
    
    l2_predictions = []
    for prob, idx in zip(l2_top3_probs[0], l2_top3_idx[0]):
        l2_predictions.append({
            'rock_type': info['l2_classes'][idx.item()],
            'confidence': prob.item()
        })
    
    return l1_class, l1_confidence, l2_predictions

# Sidebar
with st.sidebar:
    st.header("📊 Model Performance")
    st.write("""
    **Architecture:** ConvNeXt-Tiny CNN
    
    **Test Set Results:**
    - Rock Group: 77% accuracy
    - Igneous: 89% accuracy
    - Metamorphic: 49% accuracy
    - Sedimentary: 44% accuracy
    
    **Training Data:**
    - 2,734 field rock images
    - 41 rock types
    - 180 GPU-hours
    """)
    
    st.header("🎯 How to Use")
    st.write("""
    1. Upload a clear rock photo
    2. Get instant AI prediction
    3. Check confidence score
    4. Review top 3 possibilities
    """)
    
    st.header("🔬 Best Results For")
    st.success("**Igneous rocks** - 89% accuracy")
    st.warning("**Metamorphic/Sedimentary** - Use with caution (45-49% accuracy)")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Upload Your Rock Photo")
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)

with col2:
    if uploaded_file is not None:
        st.subheader("🤖 AI Analysis")
        
        with st.spinner('Analyzing rock sample...'):
            model, info, device = load_model()
            l1_class, l1_conf, l2_preds = predict_rock(image, model, info, device)
        
        # Determine confidence level
        if l1_conf >= 0.7:
            conf_class = "confidence-high"
            conf_emoji = "✅"
            reliability = "High Confidence"
        elif l1_conf >= 0.5:
            conf_class = "confidence-medium"
            conf_emoji = "⚠️"
            reliability = "Medium Confidence"
        else:
            conf_class = "confidence-low"
            conf_emoji = "❌"
            reliability = "Low Confidence"
        
        # Display main result
        st.markdown(f"""
        <div class="result-card">
            <p style="color: #6b7280; font-size: 1rem; margin: 0; font-weight: 600;">Rock Group</p>
            <p class="rock-type">{l1_class.upper()}</p>
            <p style="color: #6b7280; font-size: 1rem; font-weight: 600;">Confidence</p>
            <p class="{conf_class}">{conf_emoji} {l1_conf*100:.1f}%</p>
            <p style="color: #6b7280; font-size: 0.95rem; margin-top: 1rem; font-weight: 600;">{reliability}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Specific rock type predictions with DARKER colors
        st.markdown("### 🔍 Specific Rock Type (Top 3)")
        
        for i, pred in enumerate(l2_preds):
            if i == 0:
                st.markdown(f"""
                <div class="prediction-card" style="border-left: 4px solid #059669;">
                    <strong style="font-size: 1.3rem;">1️⃣ {pred['rock_type'].title()}</strong>
                    <div class="progress-bar-bg">
                        <div class="progress-bar-fill-primary" style="width: {pred['confidence']*100}%;"></div>
                    </div>
                    <p style="margin-top: 0.5rem; color: #111827; font-size: 1rem;">{pred['confidence']*100:.1f}% confidence</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-card">
                    <strong style="font-size: 1.1rem;">{i+1}️⃣ {pred['rock_type'].title()}</strong>
                    <div class="progress-bar-bg" style="height: 15px;">
                        <div class="progress-bar-fill-secondary" style="width: {pred['confidence']*100}%;"></div>
                    </div>
                    <p style="margin-top: 0.3rem; color: #374151; font-size: 0.9rem;">{pred['confidence']*100:.1f}% confidence</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Reliability warning for non-igneous
        if l1_class != 'igneous':
            st.markdown(f"""
            <div class="warning-box">
                <strong style="color: #991b1b;">⚠️ Reliability Note</strong><br>
                <p style="color: #7f1d1d; margin-top: 0.5rem;">
                The model is less reliable for {l1_class} rocks (49% accuracy) compared to igneous rocks (89% accuracy).
                Consider these predictions as preliminary and confirm with field tests.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Educational content
        if l1_conf < 0.7:
            st.markdown("""
            <div class="info-box">
                <strong style="color: #92400e;">💡 Why is the model uncertain?</strong><br>
                <p style="color: #78350f; margin-top: 0.5rem; line-height: 1.6;">
                • Weathering may have obscured diagnostic features<br>
                • The rock may be transitional between types<br>
                • Photo angle/lighting may not show key characteristics<br>
                • Field tests (hardness, acid reaction) needed for confirmation
                </p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 1.5rem;">
    <p style="font-size: 0.9rem; color: #9ca3af;">
        Model trained on 2,734 field rock images from Kaggle • ConvNeXt-Tiny architecture • 77% rock group accuracy
    </p>
    <p style="font-size: 0.85rem; margin-top: 0.5rem; color: #9ca3af;">
        Built by a Data Geoscientist as a personal learning project
    </p>
</div>
""", unsafe_allow_html=True)