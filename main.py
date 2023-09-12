import streamlit as st
from streamlit_option_menu import option_menu
from diabetes_predict import diabetes_predict
import pandas as pd
from heart_disease_data import heart_disease_data

# page config
st.set_page_config(
    page_title="My App",
    page_icon=":smiley:",
    layout="wide", # wide or center
    initial_sidebar_state="expanded",
)

def main():
    

# Print the value of my_variable
     
    # streamit option menu
    with st.sidebar:
        selected = option_menu("Main Menu", ["Home", "Diabetes",'Heart Disease'], 
            icons=['house', "file-earmark-medical",'activity'], menu_icon="cast", default_index=0)
    
    if selected == "Home":
        st.title("Diabetes and Heart detection website")
        st.write("""This is a diabetes and heart prediction website! This both disease are nowadays very common to people as number of people are having these disease and many people are dying due to this disease
                    So to prevent people from dying they should know whether they are having this disease or not?
                    When they feel like having some symtomps realted to this disease they can visit our website and predict the disease 
                    without even going to doctor and this model is very accurate.
                    We wanted to take care of the people so we made a good prediction model for this disease to save the life of the people""")
        st.write("""This website is made by Mansi Tanna of the Charotar University of Science and Technology .""")
        # diabetes_dataset = pd.read_csv("./diabetes-dataset-cleaned.csv")
        # diabetes_dataset = diabetes_dataset.drop(columns = 'Pdiabetes', axis=1)
        # diabetes_dataset=diabetes_dataset.dropna(axis=0)
        # accuracy = sendaccuracy(diabetes_dataset)
        # st.write("Accuracy of the model is: ", round((accuracy*100),2))
    elif selected == "Diabetes":
        diabetes_predict()
    elif selected == "Heart Disease":
        heart_disease_data()
    # elif selected == "Settings":
        # st.title("Settin

if __name__ == "__main__":
    main()