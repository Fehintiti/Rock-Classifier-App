# 🪨 AI Rock Classifier

A deep learning system that identifies 41 rock types from a single field photograph — and a fully productionized app around it, not just a model. Upload a photo, get a rock group and specific rock type prediction with confidence scores, and submit a correction if it's wrong. Every correction is persisted for future retraining.

## 🎯 Try it Live
**[fehintiti.github.io/Rock-Classifier-App](https://fehintiti.github.io/Rock-Classifier-App/)**

## 📊 Project Overview
- **Model:** ConvNeXt-Tiny CNN with a custom dual-head architecture, predicting rock group and specific rock type simultaneously
- **Performance:** 77% rock group accuracy, 89% for igneous rocks
- **Dataset:** 2,734 field rock images (41 rock types)
- **Training:** 180 GPU-hours on Google Colab

## 🚀 Features
- Upload rock photos for instant AI identification
- Get rock group classification (igneous/metamorphic/sedimentary)
- View top 3 specific rock type predictions with confidence scores
- Submit a correction when a prediction is wrong — stored for future model retraining
- Educational insights about classification uncertainty

## 📈 Model Performance

| Rock Group | Test Accuracy |
|------------|---------------|
| Igneous | 89.4% |
| Metamorphic | 49.0% |
| Sedimentary | 44.2% |
| **Overall** | **77%** |

## 🛠️ Tech Stack
- **Deep Learning:** PyTorch, ConvNeXt-Tiny
- **Backend:** FastAPI, deployed on Google Cloud Run (scale-to-zero)
- **Frontend:** React, deployed on GitHub Pages
- **Feedback Storage:** Supabase (Postgres + Storage), for future retraining
- **CI/CD:** GitHub Actions (auto-deploy to Cloud Run on backend changes)
- **Training:** Google Colab (NVIDIA L4 GPU)
- **Dataset Source:** Kaggle Rock Classification Dataset

See [DEPLOYMENT.md](DEPLOYMENT.md) for how the backend and feedback storage are set up and deployed.

## 🔄 Built for Production, Not Just a Demo
This started as a Streamlit prototype, then was rebuilt into a full production system: a FastAPI backend serving real-time inference, a React frontend, and a closed feedback loop where user corrections are persisted to Supabase for future retraining. When the original AWS-hosted backend was unexpectedly taken offline (a free-tier account closure), the entire service — inference backend and feedback pipeline alike — was migrated to Google Cloud Run with zero data loss, and redesigned to scale to zero instead of running (and billing) around the clock.

## 📖 Read More
- **Technical Article:** [Coming Soon]
- **Training Notebook:** [View on Google Colab]

## ⚠️ Important Note
This is a personal research project demonstrating AI applications in geology. Not intended for professional geological identification. Always confirm predictions with field tests.

## 🙋‍♂️ About
Built by Tomisin Okunlola

## 📄 License
MIT License - Feel free to use for educational purposes