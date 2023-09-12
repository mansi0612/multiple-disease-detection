import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


def diabetes_predict():
    diabetes_dataset = pd.read_csv("diabetes-dataset-cleaned.csv")
    diabetes_dataset = diabetes_dataset.drop(columns = 'Pdiabetes', axis=1)
    diabetes_dataset=diabetes_dataset.dropna(axis=0)
    print(diabetes_dataset.shape)
    # diabetes_dataset.groupby('Diabetes').mean()
    st.title("Diabetes")
    Age_select= st.selectbox("Age",  ['Select','Less than 40', '40-49','50-59', 'Greater than 60'])
    if Age_select == 'Less than 40':
        Age = 0 
    if Age_select == '40-49':
        Age = 1
    if Age_select == '50-59':
        Age = 2
    if Age_select == 'Greater than 60':
        Age = 3
    Gender_select=st.selectbox("Select Gender", ['Select','Male', 'Female'])
    if Gender_select == 'Male':
        Gender = 0
    if Gender_select == 'Female':
        Gender = 1
    Family_Diabetes_select=st.selectbox("Family Diabetes", ['Select','No', 'Yes'])
    if Family_Diabetes_select == 'No':
        Family_Diabetes = 0
    if Family_Diabetes_select == 'Yes':
        Family_Diabetes = 1
    highBP_select=st.selectbox("highBP", ['Select','No', 'Yes'])
    if highBP_select == 'No':
        highBP = 0
    if highBP_select == 'Yes':
        highBP = 1
    PhysicallyActive_select=st.selectbox("PhysicallyActive",  ['Select','Not at all', 'Less than Half hr.','More than Half hr.', '1 hr. or more'])
    if PhysicallyActive_select == 'Not at all':
        PhysicallyActive = 0
    if PhysicallyActive_select == 'Less than Half hr.':
        PhysicallyActive = 1
    if PhysicallyActive_select == 'More than Half hr.': 
        PhysicallyActive = 2
    if PhysicallyActive_select == '1 hr. or more':
        PhysicallyActive = 3
    BMI=st.number_input("BMI", min_value=0.0, max_value=100.0, value=0.0)
    Smoking_select=st.selectbox("Smoking",  ['Select','No', 'Yes'])
    if Smoking_select == 'No':
        Smoking = 0
    if Smoking_select == 'Yes':
        Smoking = 1
    Alcohol_select=st.selectbox("Alcohol", ['Select','No' ,'Yes'] )
    if Alcohol_select == 'No':
        Alcohol = 0
    if Alcohol_select == 'Yes':
        Alcohol = 1
    Sleep=st.number_input("Sleep", min_value=0, max_value=12, value=0)
    SoundSleep=st.number_input("SoundSleep", min_value=0, max_value=12, value=0)
    RegularMedicine_select=st.selectbox("RegularMedicine",['Select','No', 'Yes'] )
    if RegularMedicine_select == 'No':
        RegularMedicine = 0
    if RegularMedicine_select == 'Yes':
        RegularMedicine = 1
    JunkFood_select=st.selectbox("JunkFood", ['Select','Occassional', 'Often','Very Often'])
    if JunkFood_select == 'Occassional':
        JunkFood = 0
    if JunkFood_select == 'Often':
        JunkFood = 1
    if JunkFood_select == 'Very Often':
        JunkFood = 2
    Stress_select=st.selectbox("Stress", ['Select','Not at all', 'Sometimes', 'Very Often', 'Always'])
    if Stress_select == 'Not at all':   
        Stress = 0
    if Stress_select == 'Sometimes':
        Stress = 1
    if Stress_select == 'Very Often':
        Stress = 2
    if Stress_select == 'Always':
        Stress = 3
    BPLevel_select=st.selectbox("BPLevel", ['Select','Normal', 'Low','High'])
    if BPLevel_select == 'Normal':
        BPLevel = 0
    if BPLevel_select == 'Low':
        BPLevel = 1
    if BPLevel_select == 'High':
        BPLevel = 2
    Pregancies=st.number_input("Pregancies", min_value=0, max_value=3, value=0)
    UriationFreq_select=st.selectbox("Uriation nFrequency", ['Select','Not much', 'Very often'])
    if UriationFreq_select == 'Not much':
        UriationFreq = 0
    if UriationFreq_select == 'Very often':
        UriationFreq = 1
    # Diabetic=  st.number_input("Diabetic", min_value=0, max_value=1, value=0)
       
    # pregnancies = st.number_input("Pregnancies", min_value=0, max_value=17, value=0)
    # glucose = st.number_input("Glucose", min_value=0, max_value=199, value=0)
    # blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=122, value=0)
    # skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=99, value=0)
    # insulin = st.number_input("Insulin", min_value=0, max_value=846, value=0)
    # bmi = st.number_input("BMI", min_value=0.0, max_value=67.1, value=0.0)
    # diabetespedigreefunction	 = st.number_input("DiabetesPedigreeFunction",  value=0.0)
    # age= st.number_input("Age",  value=0)

    

    # diabetes_dataset = pd.read_csv("./diabetes-dataset-cleaned.csv")
    # diabetes_dataset = diabetes_dataset.dropna(axis=0)
    

    # st.dataframe(diabetes_dataset)

    diabetes_dataset_count = diabetes_dataset["Diabetic"].value_counts()
    
    # st.write(diabetes_dataset_count)

    # # separating the data and labels
   # All columns Dataset
    X = diabetes_dataset.drop(columns = 'Diabetic', axis=1)
    Y = diabetes_dataset['Diabetic']
    print(X,Y)
    scaler = StandardScaler()
    scaler.fit(X)
    standardized_data = scaler.transform(X)
    X = standardized_data


    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.1, stratify=Y, random_state=1)
    print(X.shape, X_train.shape, X_test.shape)

    # # X = standardized_data
    # # Y = diabetes_dataset['Diabetic']
    
  
    classifier = svm.SVC(kernel='linear')
    # # training the support vector Machine Classifier
    classifier.fit(X_train, Y_train)
    
    # acc = sendaccuracy(diabetes_dataset)
    # st.write("Accuracy of the model is: ", acc)
    # # X_train_prediction = classifier.predict(X_train)
    
    # training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    #training the support vector Machine Classifier
    
    if st.button("Predict"):
        try:
            input_data = (Age,Gender,Family_Diabetes,highBP,PhysicallyActive,BMI,Smoking,Alcohol,Sleep, SoundSleep, RegularMedicine, JunkFood, Stress, BPLevel, Pregancies, UriationFreq)
        #     # changing the input data to numpy array
            input_data_as_numpy_array = np.asarray(input_data)
        # #     # reshape the array as we are predicting for one instance
            input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        # #     # standardize the input data
            std_data = scaler.transform(input_data_reshaped)
            print(std_data)
            prediction = classifier.predict(std_data)
            print(prediction)
            if prediction[0]==0:
                st.success("You are not diabetic")
                st.balloons()
            else:
                st.error("You are diabetic")
        except:
            st.error("Please enter all the values")


# def sendaccuracy(diabetes_dataset):
#     # # separating the data and labels
#     # All columns Dataset
#     X = diabetes_dataset.drop(columns = 'Diabetic', axis=1)
#     Y = diabetes_dataset['Diabetic']
#     print(X,Y)
#     scaler = StandardScaler()
#     scaler.fit(X)
#     standardized_data = scaler.transform(X)
#     X = standardized_data


    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=1)

    # # X = standardized_data
    # # Y = diabetes_dataset['Diabetic']
    
  
    classifier = svm.SVC(kernel='linear')
    # # training the support vector Machine Classifier
    classifier.fit(X_train, Y_train)
    
    # X_train_prediction = classifier.predict(X_train)
    
    # training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    #training the support vector Machine Classifier
    X_test_prediction = classifier.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
    return test_data_accuracy