# 🪨 AI Rock Classifier

An AI-powered tool for field rock identification using deep learning.

## 🎯 Try it Live
**[fehintiti.github.io/Rock-Classifier-App](https://fehintiti.github.io/Rock-Classifier-App/)**

## 📊 Project Overview
- **Model:** ConvNeXt-Tiny CNN with hierarchical classification
- **Performance:** 77% rock group accuracy, 89% for igneous rocks
- **Dataset:** 2,734 field rock images (41 rock types)
- **Training:** 180 GPU-hours on Google Colab

## 🚀 Features
- Upload rock photos for instant AI identification
- Get rock group classification (igneous/metamorphic/sedimentary)
- View top 3 specific rock type predictions with confidence scores
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

## 📖 Read More
- **Technical Article:** [Coming Soon]
- **Training Notebook:** [View on Google Colab]

## ⚠️ Important Note
This is a personal research project demonstrating AI applications in geology. Not intended for professional geological identification. Always confirm predictions with field tests.

## 🙋‍♂️ About
Built by Tomisin Okunlola

## 📄 License
MIT License - Feel free to use for educational purposes