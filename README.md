# Live App : https://rainprediction-logisticregressionclassification.streamlit.app/
# 🌧️ Rain Prediction using Logistic Regression

## 📌 Project Overview

This project predicts whether it will rain or not based on weather conditions using a Machine Learning Logistic Regression model.

The model is trained on weather-related features such as Temperature, Humidity, Wind Speed, Cloud Cover, and Pressure.

A Streamlit web application is also developed to allow users to make real-time rain predictions.

---

## 🎯 Problem Statement

Weather forecasting is important for agriculture, transportation, event planning, and disaster management.

The goal of this project is to predict rainfall using historical weather parameters.

---

## 📊 Dataset Features

| Feature     | Description                      |
| ----------- | -------------------------------- |
| Temperature | Atmospheric temperature          |
| Humidity    | Moisture level in air            |
| Wind_Speed  | Wind velocity                    |
| Cloud_Cover | Percentage of cloud coverage     |
| Pressure    | Atmospheric pressure             |
| Rain        | Target Variable (Rain / No Rain) |

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Pickle
* Streamlit
* Git & GitHub

---

## 🤖 Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Selection
5. Train-Test Split
6. Feature Scaling
7. Logistic Regression Model Training
8. Model Evaluation
9. Model Serialization using Pickle
10. Streamlit Deployment

---

## 📈 Model Evaluation Metrics

* Accuracy Score
* Precision
* Recall
* F1 Score
* Confusion Matrix
* ROC Curve
* ROC-AUC Score

---

## 💾 Saved Files

* rain_prediction_model.pkl
* scaler.pkl

These files are used for deployment without retraining the model.

---

## 🚀 Streamlit Application

Run the application:

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```text
Rain-Prediction-Logistic-Regression/
│
├── app.py
├── rain_prediction_model.pkl
├── scaler.pkl
├── weather.csv
├── requirements.txt
├── README.md
└── notebook.ipynb
```

---

## 📊 Sample Prediction

Input:

* Temperature = 28
* Humidity = 85
* Wind Speed = 12
* Cloud Cover = 90
* Pressure = 1005

Output:

```text
Rain Expected 🌧️
```

---

## 🔮 Future Improvements

* Random Forest Implementation
* XGBoost Model
* Real-Time Weather API Integration
* Cloud Deployment
* Model Monitoring

---

## 👨‍💻 Author

**Nikhil Dongare**

Aspiring Data Scientist | Machine Learning Enthusiast


LinkedIn:www.linkedin.com/in/nikhil-dongare-5958092ba

---

## ⭐ If you like this project

Give this repository a star on GitHub.
