# ❤️ Heart Stroke Prediction App

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![ML](https://img.shields.io/badge/MachineLearning-KNN-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 📸 App Preview
![App Screenshot](https://via.placeholder.com/800x400.png?text=Heart+Stroke+Prediction+App)

---

## 📌 About the Project
This is a **Machine Learning-powered web application** that predicts the risk of heart disease based on user health parameters.

It uses a **K-Nearest Neighbors (KNN)** model and provides real-time predictions through an interactive UI built with Streamlit.

---

## ✨ Features
✔️ Clean and interactive UI  
✔️ Real-time prediction  
✔️ Probability score display  
✔️ One-hot encoded inputs  
✔️ Scalable ML pipeline  

---

## 🧠 Tech Stack
| Category | Tools |
|--------|------|
| Language | Python |
| Frontend | Streamlit |
| ML Model | KNN (Scikit-learn) |
| Libraries | Pandas, NumPy, Joblib |

---

## 📂 Project Structure\
heart-stroke-predictor/
│── heart_stroke.py
│── knn_heart_model.pkl
│── heart_scaler.pkl
│── heart_columns.pkl
│── requirements.txt
│── README.md


---

## ⚙️ Installation & Setup

```bash
# Clone repository
git clone https://github.com/Kartiksh123/heart-stroke-predictor.git

# Go to project folder
cd heart-stroke-predictor

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run heart_stroke.py