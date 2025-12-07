# 🩺 Cancer Detection Deep Learning System

A web-based Deep Learning application that predicts whether a breast cancer tumor is **Benign** or **Malignant** using digitized medical imaging features.  
Users enter tumor characteristics through a **7-step guided web form**, and the model returns a prediction along with confidence scores.

---

## 🚀 Features

- **Deep Learning Classification:**  
  Powered by a TensorFlow/Keras neural network (`model.h5`) for reliable cancer detection.

- **Step-by-Step Wizard:**  
  A clean UI where users input tumor metrics in 7 stages (Radius, Perimeter, Area, Texture, Symmetry, etc.).

- **Instant Prediction:**  
  Outputs **Benign** or **Malignant** with probability scores.

- **Color-Coded Alerts:**  
  Green / Yellow / Red based on prediction confidence.

- **Docker Ready:**  
  Fully containerized for deployment on any platform.

---

## 🛠️ Tech Stack

**Frontend:**  
- HTML5  
- CSS  
- Jinja2 Templates  

**Backend:**  
- Python 3.10  
- Flask  

**ML / Data Science:**  
- TensorFlow (Keras)  
- Scikit-Learn  
- NumPy  
- Pandas  
- Joblib  

**DevOps:**  
- Docker  

---

## 📂 Project Structure

```text
├── app.py                  # Main Flask backend
├── Dockerfile              # Docker setup for deployment
├── requirements.txt        # Required Python packages
├── model.h5                # Trained Deep Learning model
├── scaler.pkl              # Scikit-learn StandardScaler
├── test_api.py             # Script for local model testing
├── static/                 # Static files (CSS, JS, reports, etc.)
└── templates/              # HTML template pages
    ├── home.html
    ├── step1_radius.html
    ├── step2_perimeter.html
    ├── step3_area.html
    ├── step4_texture.html
    ├── step5_symmetry.html
    ├── step6_smoothness.html
    ├── step7_concavity.html
    └── step8_summary.html
