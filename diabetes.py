import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from streamlit_lottie import st_lottie
import requests

# 🎞️ Load Lottie animation from URL
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# 🧠 Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("c:\\Users\\medra\\Downloads\\diabetes_prediction_dataset.csv")
    data = data.drop_duplicates()
    selected_columns = ['HbA1c_level', 'blood_glucose_level', 'age', 'diabetes']
    return data[selected_columns]

# 🧠 Train model
@st.cache_resource
def train_model(df):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = SVC(C=10, kernel='rbf')
    clf.fit(X_train, y_train)
    return clf

# 💥 Main app
def main():
    st.set_page_config(page_title="Diabetes Predictor", layout="centered")

    # 💫 Animated title
    st.markdown("""
        <div style="text-align: center;">
            <h1 style="color: #4CAF50; animation: pulse 2s infinite;">🩺 Diabetes Prediction App 🧠</h1>
        </div>
        <style>
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.05); }
                100% { transform: scale(1); }
            }
        </style>
    """, unsafe_allow_html=True)

    # ✅ WORKING Lottie animation URL
    lottie_animation = load_lottieurl("https://lottie.host/4ec59fe6-2e68-4ab6-bb8b-269ee68291ae/Zr8ABiZK5g.json")
    if lottie_animation:
        st_lottie(lottie_animation, height=250, key="health")

    # 🔍 Load + train
    df = load_data()
    clf = train_model(df)

    st.subheader("Enter patient details:")
    HbA1c = st.slider("HbA1c Level", min_value=3.0, max_value=15.0, step=0.1)
    glucose = st.slider("Blood Glucose Level", min_value=50, max_value=300, step=1)
    age = st.slider("Age", min_value=1, max_value=120)

    if st.button("Predict"):
        input_data = np.array([[HbA1c, glucose, age]])
        prediction = clf.predict(input_data)[0]

        if prediction == 0:
            st.success("✅ This person is **not diabetic**.")
            st.balloons()
        else:
            st.error("⚠️ This person is **diabetic**.")
            st.snow()

# 🚀 Run
if __name__ == "__main__":
    main()
