import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


def diabetes_predict():
    st.title("Diabetes")

    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=17, value=0)
    glucose = st.number_input("Glucose", min_value=0, max_value=199, value=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=122, value=0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=99, value=0)
    insulin = st.number_input("Insulin", min_value=0, max_value=846, value=0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=67.1, value=0.0)
    diabetespedigreefunction	 = st.number_input("DiabetesPedigreeFunction",  value=0.0)
    age= st.number_input("Age",  value=0)

    

    diabetes_dataset = pd.read_csv("diabetes.csv")

    # st.dataframe(diabetes_predict)

    # diabetes_predict = diabetes_predict["Outcome"].value_counts()

    # st.write(diabetes_predict)

    # separating the data and labels
    X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
    Y = diabetes_dataset['Outcome']
    print(X,Y)
    scaler = StandardScaler()
    scaler.fit(X)
    standardized_data = scaler.transform(X)
    X = standardized_data
    # Y = diabetes_dataset['Outcome']
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
    classifier = svm.SVC(kernel='linear')
    #training the support vector Machine Classifier
    classifier.fit(X_train, Y_train)
    if st.button("Predict"):
        input_data = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetespedigreefunction, age)
        # changing the input data to numpy array
        input_data_as_numpy_array = np.asarray(input_data)
        # reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        # standardize the input data
        std_data = scaler.transform(input_data_reshaped)
        print(std_data)
        prediction = classifier.predict(std_data)
        print(prediction)
        if prediction[0]==0:
            st.success("You are not diabetic")
            st.balloons()
        else:
            st.error("You are diabetic")

