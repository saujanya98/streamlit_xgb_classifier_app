import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import pickle

st.set_page_config(page_title = "XGBoost Classifier", page_icon = "🧮")

st.markdown("# XGBoost Classifier 🧮")
st.sidebar.header("XGBoost Classifier")
st.markdown(
    """
    Here is where you can input predictions and see what the model returns! The code used to train the model will be on my GitHub:
    """
)

st.write('https://github.com/saujanya98/streamlit_xgb_classifier_app')

predictions = st.container()

with predictions:
    st.header('Predictions')
    # xgboost model
    df = pd.read_csv('pages/data/drug200.csv')
    encoder = LabelEncoder()
    df['Sex'] = encoder.fit_transform(df['Sex'])
    df['BP'] = encoder.fit_transform(df['BP'])
    df['Cholesterol'] = encoder.fit_transform(df['Cholesterol'])
    
    
    st.markdown(
    """
    Here's a correlation plot to see how the features are related to one another:
    """ )
    fig1 = plt.figure(figsize=(10, 4))
    ax1 = fig1.add_subplot(111)
    plt.xticks(rotation=90)
    plt.title('Correlation Plot')
    corr = df.drop(columns='Drug').corr()
    sns.heatmap(corr,cmap='coolwarm',annot=True, fmt='.2f')
    st.pyplot(fig1)
    
    # Define mapping function
    def map_drug_names(drug_name):
        drug_map = {'drugA': 0, 'drugB': 1, 'drugC': 2, 'drugX': 3, 'DrugY': 4}
        return drug_map[drug_name]

    # Apply mapping function to 'Drug' column in a pandas DataFrame
    df['Drug'] = df['Drug'].apply(map_drug_names)
    
    X = df.drop('Drug',axis=1).values
    y = df['Drug'].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    with open('pages/xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    
    st.markdown(
    """
    Use the interactive buttons/sliders to input values into the model to output a prediction!
    """ )
    Age = [st.slider('What Age?', min_value=1, max_value=100, value=20, step=1)]
    Sex = [st.selectbox('What Gender?',('M','F'))]
    BloodPressure = [st.selectbox('What Blood Pressure Level?', ('HIGH','LOW','NORMAL'))]
    Cholesterol = [st.selectbox('What Cholesterol Level?', ('HIGH','NORMAL'))]
    SodiumToPotassium = [st.slider('What Sodium to Potassium Level?', min_value=0.1, max_value=40.0, step=0.1)]
    
    Sex = encoder.fit_transform(Sex)
    BloodPressure = encoder.fit_transform(BloodPressure)
    Cholesterol = encoder.fit_transform(Cholesterol)
    inputs = [Age,Sex,BloodPressure,Cholesterol,SodiumToPotassium]
    inputs = scaler.fit_transform(inputs)
    
    # Define mapping function
    def reverse_map_drug_num(drug_num):
        map_drug = {0: 'drugA', 1: 'drugB', 2: 'drugC', 3: 'drugX', 4: 'drugY'}
        return map_drug[drug_num]
        
    prediction = xgb_model.predict(inputs.T)[0]
    prediction = reverse_map_drug_num(prediction)
    
    if st.button('Predict'):
        st.write(prediction)
    
