import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def heart_disease_data():
    df = pd.read_csv('cardio_train.csv')
    # st.dataframe(df, use_container_width=True)
    # age = df['age'].values
    # # convert age string to int
    # age = [int(x.split()[0]) for x in age]
    # st.write(age.min())
    # st.write(age.max())
    # age = df['age'].values/365
    # st.write(age.round())
    # df.loc[:, 'age'] = age.round()
    # df.to_csv('cardio_train.csv', index=False)
    st.title("Heart Disease")
    # heart_data = pd.read_csv('/content/heart_disease_data.csv')
    age = st.number_input("Age", min_value=0, max_value=100, value=0)
    gender= st.selectbox("Sex",  options=['male','female'])
    if gender=='male':
        sex=0
    if gender=='female':
        sex=1
    cp= st.number_input("Chest Pain", min_value=0, max_value=4, value=0)
    trestbps = st.number_input("Resting Blood Pressure", min_value=0, max_value=200, value=0)
    chol = st.number_input("Serum Cholestoral", min_value=0, max_value=600, value=0)
    fbs = st.number_input("Fasting Blood Sugar", min_value=0, max_value=1, value=0)
    restecg = st.number_input("Resting Electrocardiographic Results", min_value=0, max_value=2, value=0)
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=200, value=0)
    exang = st.number_input("Exercise Induced Angina", min_value=0, max_value=1, value=0)
    oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest", min_value=0.0, max_value=6.2, value=0.0)
    slope = st.number_input("Slope of the Peak Exercise ST Segment", min_value=0, max_value=2, value=0)
    ca = st.number_input("Number of Major Vessels (0-3) Colored by Flourosopy", min_value=0, max_value=4, value=0)
    thal = st.number_input("Thalassemia", min_value=0, max_value=3, value=0)

    heart_data = pd.read_csv("heart_disease_data.csv")
    # st.dataframe(heart_data)
    X = heart_data.drop(columns = 'target', axis=1)
    Y = heart_data['target']
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
    model = LogisticRegression()
    model.fit(X_train, Y_train)
   
    X_test_prediction = model.predict(X_test)

    if st.button("Predict"):
        input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
        input_data_as_numpy_array= np.asarray(input_data)

        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        prediction = model.predict(input_data_reshaped)
        # st.write(prediction)
        if prediction[0]== 0:
            st.success('The Person does not have a Heart Disease')
            st.balloons()
        else:
            st.error('The Person has Heart Disease')