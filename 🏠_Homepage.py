import streamlit as st
import base64

st.set_page_config(
    page_title = "Welcome to my app!",
    page_icon = "👋"
)

st.write("# Welcome to my app! 👋")

st.sidebar.success("Select a page above")

st.markdown(
    """
    Welcome to my app! This app aims to classify what drug type someone takes based on certain inputs (e.g. Age, Gender, Blood Pressure etc).
    An XGBoost classifier is used to train the model in the backend. The dataset used is called 'Drug Classification' and it contains information 
    about certain drug types. It can be found on github using the following link:
    """
)

st.write('https://www.kaggle.com/datasets/prathamtripathi/drug-classification?resource=download')


st.markdown("![Alt Text](https://media.giphy.com/media/xT5LMuHEQeETeJPwu4/giphy.gif)")
st.write("Shutdown Media we're back")


