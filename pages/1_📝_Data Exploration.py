import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title = "Data Exploration", page_icon = "📝")

st.markdown("# Data Exploration 📝")
st.sidebar.header("Data Exploration")
st.write(
    """
    There are 5 different drug types that are present in the dataset. They are: DrugA, DrugB, DrugC, DrugX and DrugY.
    Note: The count of each drug type is not equal (whereby there may be more cases of a particular drug type present in the dataset compared to others).
    This ultimately will affect results, but this app is purely for experimentation so in this case, it is fine. Also, the dataset only has 200 rows.
    """
)

# creating the containers i.e. sections for the app

dataset = st.container()
eda = st.container()

with dataset:
    st.header('Dataset Used')
    st.markdown(
    """
    Lets look at the first five rows of the dataset. It is a useful method to quickly inspect the dataset and get a feel for its structure
    and content.
    """ )
    
    df = pd.read_csv('pages/data/drug200.csv')
    st.write(df.head())
    
    st.markdown(
    """
    The features used are Age, Sex, Blood Pressure, Cholesterol level and Sodium to Potassium Ratio in Blood. Based on the values of these
    features, we return a Drug type that the patient should take.
    """ )
    
with eda:
    st.header('EDA')
    st.markdown(
    """
    Lets look at some visualisations to explore the data:
    """
    )
    
    st.write('Figure 1')
    fig1 = plt.figure(figsize=(10, 4))
    ax1 = fig1.add_subplot(111)
    plt.xticks(rotation=90)
    plt.title('Count of Drug Type')
    drug_counts = df['Drug'].value_counts()
    sns.countplot(x='Drug', data=df, order=drug_counts.index)
    st.pyplot(fig1)

    st.write('Figure 2')
    fig2 = plt.figure(figsize=(10, 4))
    ax2 = fig1.add_subplot(111)
    plt.xticks(rotation=90)
    plt.title('Ratio of Male/Female consuming a particular Drug type')
    sns.histplot(x = 'Drug', hue = df['Sex'], multiple = 'stack', palette = 'cool', data = df)
    st.pyplot(fig2)
    
    st.write('Figure 3')
    fig3 = plt.figure(figsize=(10, 4))
    ax3 = fig1.add_subplot(111)
    plt.xticks(rotation=90)
    plt.title('Histogram of Na_to_K feature')
    sns.histplot(x='Na_to_K',data=df)
    st.pyplot(fig3)
    
    st.write('Figure 4')
    fig4 = plt.figure(figsize=(10, 4))
    plt.title('Histogram of Na_to_K feature')
    plt.scatter(x=df.Age[df.Sex=='F'], y=df.Na_to_K[(df.Sex=='F')])
    plt.scatter(x=df.Age[df.Sex=='M'], y=df.Na_to_K[(df.Sex=='M')])
    plt.legend(["Female", "Male"])
    plt.xlabel("Age")
    plt.ylabel("Na_to_K")
    plt.title('Sodium to Potassium Distribution based on Gender and Age')
    st.pyplot(fig4)        
