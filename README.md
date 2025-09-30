# 🚀 Employee Burnout Risk Analysis Dashboard

This project is a **Streamlit-based interactive dashboard** designed to analyze and predict **employee burnout risk** using HR data and machine learning models.

---

## 📌 Features

- **Data Loading & Preprocessing**
  - Loads HR data from Excel (or generates a sample dataset if unavailable)
  - Feature engineering to compute **Burnout Risk Score** and **Burnout Risk (binary target)**

- **Exploratory Data Analysis (EDA)**
  - Summary statistics and feature distributions
  - Correlation heatmap
  - Burnout risk analysis across categorical features

- **Machine Learning Models**
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - XGBoost  

- **Model Evaluation**
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix
  - ROC Curves
  - Cross-validation scores

- **Model Explainability**
  - Feature importance visualization

- **Burnout Prediction**
  - 📊 **HR Data-based Prediction** (input employee details)
  - 🧠 **Self-Assessment Questionnaire** (predict burnout from personal responses)

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **Backend:** Python (Pandas, NumPy, Scikit-learn, XGBoost)  
- **Visualization:** Matplotlib, Seaborn  
- **Deployment-ready:** Works locally with Streamlit, can be hosted on Streamlit Cloud  

---

## 📂 Project Structure
    │── app.py # Main Streamlit app
    │── HR Data.xlsx # Dataset
    │── requirements.txt # Dependencies
    │── README.md # Project documentation

## ⚡ Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Employee-Burnout-Prediction.git
   cd Employee-Burnout-Prediction
   ```
2. **Install dependencies**
 ```
   pip install -r requirements.txt
```
3. **Run the Streamlit app**
```
streamlit run app.py
```
## 📊 Example Outputs

- Burnout Risk Distribution
- Feature Correlation Heatmap
- Model Performance Comparison
- ROC Curves & Confusion Matrix
- Interactive Burnout Risk Prediction

## 📈 Future Enhancements

- Add deep learning models (TensorFlow / PyTorch)
- Add more real-world HR datasets

## Live Demo
- 🎯 HR-based prediction & questionnaire assessment app: [employee-burnout-prediction-using-ml.streamlit.app](https://employee-burnout-prediction-using-ml.streamlit.app/)  

