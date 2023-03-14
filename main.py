import streamlit as st
from streamlit_option_menu import option_menu
from diabetes_predict import diabetes_predict
from heart_disease_data import heart_disease_data

# page config
st.set_page_config(
    page_title="My App",
    page_icon=":smiley:",
    layout="wide", # wide or center
    initial_sidebar_state="expanded",
)

def main():
    # streamit option menu
    with st.sidebar:
        selected = option_menu("Main Menu", ["Home", "Diabetes",'Heart Disease','Settings'], 
            icons=['house', "file-earmark-medical",'activity','0-cirle-fill'], menu_icon="cast", default_index=2)
    
    if selected == "Home":
        st.title("Home")
    elif selected == "Diabetes":
        diabetes_predict()
    elif selected == "Heart Disease":
        heart_disease_data()
    elif selected == "Settings":
        st.title("Settings")

if __name__ == "__main__":
    main()