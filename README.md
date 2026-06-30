# 🧠 Brain Tumor Detection (VGG16 + Gradio)

An AI-assisted brain MRI classifier that detects **Glioma**, **Meningioma**,
**Pituitary** tumors, or **No Tumor**, using transfer learning on VGG16.
Includes an interactive Gradio dashboard with probability breakdown and an
auto-generated PDF diagnostic report.

🔗 **Live demo:** [Add your Hugging Face Space link here after deployment]

## Features
- 4-class MRI classification (glioma / meningioma / pituitary / no tumor)
- Transfer learning on VGG16 (ImageNet weights, fine-tuned top layers)
- Interactive Gradio dashboard: upload an MRI, get instant prediction
- Confidence score + probability bar chart for all classes
- Auto-generated PDF diagnostic report with findings and suggested specialists
- Grad-CAM visualization support (see `train.py`)

## Project Structure
```
.
├── app.py              # Gradio app — run this to launch the live demo
├── train.py            # Model training script (run separately, needs GPU + dataset)
├── requirements.txt    # Dependencies
├── BrainTumor_VGG16_Final.h5   # Trained model weights (add this file — see below)
└── README.md
```

## Model File
The trained model (`BrainTumor_VGG16_Final.h5`) is **not included in this repo
by default** if it's large. You have two options:

1. **Git LFS** (recommended) — track the file with Git LFS so it lives in the repo.
2. **External hosting** — upload the `.h5` to Hugging Face Hub / Google Drive and
   download it inside `app.py` at startup.

## Run Locally
```bash
git clone https://github.com/<your-username>/brain-tumor-detection.git
cd brain-tumor-detection
pip install -r requirements.txt
python app.py
```
This launches a local Gradio server (usually at `http://127.0.0.1:7860`).

## Dataset
Trained on the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
(Kaggle), containing 4 classes: glioma, meningioma, pituitary, no tumor.

## Tech Stack
- TensorFlow / Keras (VGG16 transfer learning)
- Gradio (web UI)
- ReportLab (PDF report generation)
- OpenCV / Matplotlib (Grad-CAM, visualizations)

## Disclaimer
This project is for educational/portfolio purposes only. It is **not** a
certified medical diagnostic tool and should not be used for real clinical
decisions.

## Author
Navadeep Reddy — [GitHub](https://github.com/NavadeepReddy21)
