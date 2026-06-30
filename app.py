"""
app.py
Brain Tumor Detection - Gradio Dashboard

Loads a pretrained VGG16-based classifier and serves an interactive
dashboard: upload an MRI -> get tumor type, risk level, findings,
probability chart, and a downloadable PDF diagnostic report.

This file is meant to be deployed as-is (e.g. on Hugging Face Spaces).
The model file BrainTumor_VGG16_Final.h5 must sit alongside this script.
"""

import io
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering, required for server deployment
import matplotlib.pyplot as plt
import gradio as gr
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import HexColor, black

# ===================== CONFIG =====================
MODEL_PATH = "BrainTumor_VGG16_Final.h5"
IMG_SIZE = 128

CLASS_LABELS = ["glioma", "meningioma", "notumor", "pituitary"]

DISPLAY_LABELS = {
    "glioma": "Glioma Tumor",
    "meningioma": "Meningioma Tumor",
    "notumor": "No Tumor Detected",
    "pituitary": "Pituitary Tumor",
}

RISK_DICT = {
    "glioma": "High",
    "meningioma": "Medium",
    "pituitary": "Medium",
    "notumor": "Low",
}

FINDINGS_DICT = {
    "glioma": (
        "An intra-axial lesion with heterogeneous signal intensity is noted. "
        "Features suggest an infiltrative gliomatous pathology with possible "
        "surrounding edema."
    ),
    "meningioma": (
        "A well-circumscribed extra-axial lesion is identified, likely "
        "arising from the meninges. Imaging features are suggestive of "
        "meningioma."
    ),
    "pituitary": (
        "A sellar/suprasellar lesion is observed involving the pituitary "
        "gland. Findings are suggestive of a pituitary adenoma."
    ),
    "notumor": (
        "No focal intracranial mass lesion is identified. Brain parenchyma "
        "appears within normal limits."
    ),
}

DOCTOR_DICT = {
    "glioma": [
        ("Dr. Kanwaljeet Garg", "MBBS, MS, MCh", "AIIMS", "New Delhi"),
        ("Dr. Aliasgar Moiyadi", "MBBS, MS, DNB", "Tata Memorial Hospital", "Mumbai"),
    ],
    "meningioma": [
        ("Dr. Guru Dutta Satyarthee", "MBBS, MS, MCh", "AIIMS", "New Delhi"),
        ("Dr. Hemanth B. S.", "MBBS, MS, MCh", "NIMHANS", "Bengaluru"),
    ],
    "pituitary": [
        ("Dr. Muffazal Lakdawala", "MBBS, MS", "Saifee Hospital", "Mumbai"),
        ("Dr. K. Sridhar", "MBBS, MS, MCh", "NIMHANS", "Bengaluru"),
    ],
    "notumor": [
        ("General Neurologist", "MBBS, MD", "Multispecialty Hospital", "Nearby City"),
    ],
}

# ===================== LOAD MODEL (once, at startup) =====================
print("Loading model...")
model = load_model(MODEL_PATH, compile=False)
print("Model loaded.")


# ===================== PDF GENERATOR =====================
def generate_pdf(name, age, gender, tumor, confidence, risk,
                  findings, doctors, mri_img, prob_img):
    path = "Brain_Tumor_Report.pdf"
    c = canvas.Canvas(path, pagesize=A4)
    PAGE_W, PAGE_H = A4
    MARGIN = 40

    header_color = HexColor("#0F766E")
    section_color = HexColor("#ECFEFF")

    y = PAGE_H - MARGIN

    # Header
    c.setFillColor(header_color)
    c.rect(0, y - 45, PAGE_W, 45, fill=1)
    c.setFillColor(HexColor("#FFFFFF"))
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(PAGE_W / 2, y - 30, "BRAIN MRI DIAGNOSTIC REPORT")
    y -= 65

    # Patient details
    c.setFillColor(section_color)
    c.rect(MARGIN, y - 60, PAGE_W - 2 * MARGIN, 60, fill=1)
    c.setFillColor(black)
    c.setFont("Helvetica", 11)
    c.drawString(MARGIN + 10, y - 25, f"Patient Name : {name}")
    c.drawString(MARGIN + 10, y - 45, f"Age : {age}    Gender : {gender}")
    c.drawRightString(PAGE_W - MARGIN - 10, y - 45,
                       f"Date : {datetime.now().strftime('%d-%m-%Y')}")
    y -= 80

    # Images
    box_h = 200
    box_w = (PAGE_W - 2 * MARGIN - 20) / 2
    c.rect(MARGIN, y - box_h, box_w, box_h)
    c.rect(MARGIN + box_w + 20, y - box_h, box_w, box_h)

    c.drawImage(ImageReader(mri_img), MARGIN + 10, y - box_h + 10,
                box_w - 20, box_h - 20, preserveAspectRatio=True)
    c.drawImage(ImageReader(prob_img), MARGIN + box_w + 30, y - box_h + 40,
                box_w - 40, box_h - 80, preserveAspectRatio=True)
    y -= box_h + 30

    # Diagnosis
    diag_h = 80
    c.setFillColor(section_color)
    c.rect(MARGIN, y - diag_h, PAGE_W - 2 * MARGIN, diag_h, fill=1)
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(MARGIN + 10, y - 25, "Diagnosis")
    c.setFont("Helvetica", 11)
    c.drawString(MARGIN + 10, y - 45, f"Tumour Type : {tumor}")
    c.drawString(MARGIN + 10, y - 65, f"Risk Level   : {risk}")
    c.drawString(PAGE_W / 2, y - 65, f"Confidence : {confidence:.2f}%")
    y -= diag_h + 20

    # Findings
    find_h = 110
    c.setFillColor(section_color)
    c.rect(MARGIN, y - find_h, PAGE_W - 2 * MARGIN, find_h, fill=1)
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(MARGIN + 10, y - 25, "Findings")
    text = c.beginText(MARGIN + 10, y - 45)
    text.setFont("Helvetica", 10)
    for line in findings.split(". "):
        text.textLine(line.strip())
    c.drawText(text)
    y -= find_h + 20

    # Doctors
    doc_h = 110
    c.setFillColor(section_color)
    c.rect(MARGIN, y - doc_h, PAGE_W - 2 * MARGIN, doc_h, fill=1)
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(MARGIN + 10, y - 25, "Suggested Specialist Doctors")
    text = c.beginText(MARGIN + 10, y - 45)
    text.setFont("Helvetica", 10)
    for d in doctors:
        text.textLine(f"{d[0]} ({d[1]})")
        text.textLine(f"{d[2]}, {d[3]}")
        text.textLine("")
    c.drawText(text)

    # Footer
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(MARGIN, 40,
                 "Disclaimer: AI-assisted preliminary report. Not a medical diagnosis.")

    c.showPage()
    c.save()
    return path


# ===================== PREDICTION =====================
def predict_dashboard(img, name, age, gender):
    if img is None:
        return "Please upload an MRI image.", None, None, None

    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))

    preds = model.predict(arr)[0]
    idx = np.argmax(preds)
    key = CLASS_LABELS[idx]

    tumor = DISPLAY_LABELS[key]
    confidence = preds[idx] * 100
    risk = RISK_DICT[key]
    findings = FINDINGS_DICT[key]
    doctors = DOCTOR_DICT[key]

    fig = plt.figure(figsize=(5, 3))
    plt.barh([DISPLAY_LABELS[c] for c in CLASS_LABELS], preds * 100)
    plt.xlim(0, 100)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    prob_img = Image.open(buf)

    pdf_path = generate_pdf(name or "N/A", age or "N/A", gender or "N/A",
                             tumor, confidence, risk, findings, doctors,
                             img, prob_img)

    result = (
        f"Tumour Type: {tumor}\n"
        f"Risk Level: {risk}\n"
        f"Confidence: {confidence:.2f}%\n\n"
        f"Findings:\n{findings}"
    )

    return result, img, prob_img, pdf_path


# ===================== UI =====================
with gr.Blocks(title="Brain Tumor Detection") as demo:
    gr.Markdown("## 🧠 Brain Tumor Detection – Interactive Dashboard")
    gr.Markdown(
        "Upload a brain MRI scan to get an AI-assisted classification "
        "(Glioma / Meningioma / Pituitary / No Tumor), along with a "
        "downloadable diagnostic report.\n\n"
        "**Disclaimer:** This is an educational AI demo, not a medical "
        "diagnostic tool. Always consult a qualified radiologist/neurologist."
    )

    with gr.Row():
        with gr.Column():
            name = gr.Textbox(label="Patient Name")
            age = gr.Number(label="Age", value=25)
            gender = gr.Dropdown(["Male", "Female", "Other"], label="Gender")
            img_in = gr.Image(type="pil", label="Upload MRI Image")
            btn = gr.Button("Analyze MRI", variant="primary")

        with gr.Column():
            result = gr.Textbox(label="Diagnosis & Findings", lines=12)
            mri_out = gr.Image(label="Uploaded MRI")
            prob_out = gr.Image(label="Probability Analysis")
            pdf_out = gr.File(label="Download Diagnostic PDF")

    btn.click(
        predict_dashboard,
        inputs=[img_in, name, age, gender],
        outputs=[result, mri_out, prob_out, pdf_out],
    )

if __name__ == "__main__":
    demo.launch()
