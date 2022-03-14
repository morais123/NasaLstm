# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 09:31:37 2022

@author: Mohamed-Aziz.Rais
"""


import streamlit as st
import pandas as pd
import shap
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
st.set_page_config(layout="wide", page_title='Explaining  Predictive Maintenance ML Model')

st.header('NASA turbofan engines')
st.markdown("""The turbofan dataset features four datasets of increasing complexity 
            The engines operate normally in the beginning but develop a fault over time. For the training sets, the engines are run to failure, while in the test sets the time series end ‘sometime’ before failure. 
            The goal is to predict the Remaining Useful Life (RUL) of each turbofan engine to prevent failure and alert the company about the 20 cycles before it happens""")
st.subheader('Description of the database')
Description=pd.DataFrame([['FD001',1,1,100,100],['FD002',6,1,260,259],['FD003',1,2,100,100],['FD004',6,2,248,249]])
Description=Description.rename(columns={0:'Dataset',1:'Operating conditions',2:'Fault modes',3:'Train size',4:'Test size'})
Description=Description.set_index('Dataset')
st.dataframe(Description)
st.sidebar.markdown("""
    **Author**: Rais Mohamed-Aziz
    
    **Last Published**: 04-Feb-2022
    
    **Feature Detailed Description**: [Turbo engine degradation NASA IOT](https://data.nasa.gov/widgets/vrks-gjie)
   
    **References**:
    - [Kaggle Explainable AI Course](https://www.kaggle.com/learn/machine-learning-explainability)
    - [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/)
    - [Kaggle Notebooks](https://www.kaggle.com/chingchunyeh/heart-disease-report)
    
""")
add_selectbox = st.sidebar.selectbox(
    "Select the component",
    ("Component 1: FD001", "Component 2: FD002", "Component 3: FD003","Component 4: FD004")
)

if add_selectbox=="Component 1: FD001":
# Create a page dropdown
    
    page = st.selectbox("Choose your page", ["Exploratory data analysis", "Explainable AI: Modelisation XGBOOST", "Explainable AI: Modelisation IForest"]) 
    if page == "Exploratory data analysis":
        
    
        st.header('Exploratory data analysis')
        
    
        shap.initjs()
    
        header = st.container()
        dataset = st.container()
        model = st.container()
        explainable = st.container()
    
    
        
        with header:
            st.title("Explaining Predictive Maintenance ML Model")
            st.markdown("""
                Many people say machine learning models are **black boxes**, in the sense that they can make good predictions but you can't understand the logic behind those predictions. This statement is true in the sense that most data scientists don't know how to extract insights from models yet.
                
                This interactive application explains the turbo engine degradation based on [Turbo engine degradation NASA IOT](https://data.nasa.gov/widgets/vrks-gjie) dataset using **Explainable AI** technique.
            """)
    
    
    
    
    
    
        #df_all_variables_train2,df_all_variables_train3,df_all_variables_train4,df_all_variables_test1,df_all_variables_test2,df_all_variables_test3,df_all_variables_test4
        
        
            
            
    
        with dataset:
            st.header("**Dataset**")
            st.markdown("""
                This database contains 26 features of simulated data from turbo engine sensors, we will try first to extract the RUL from a dataset for each ID of component.
                
            """)
            st.write("Upload the dataset to Register them in DataBase")
           
            

