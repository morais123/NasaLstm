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

@st.cache(suppress_st_warning=True)
def data_retrieving():
    """
    Function : data_retrieving
    Param : data_train : retrieves data and perform data preparation tasks
    Output: dataframe with recoded RUL 1 if RUL<20 or 0 of RUL>20 
    
    
    
    """
    dfs=[]
    files=['data/train_FD001.txt','data/train_FD002.txt','data/train_FD003.txt','data/train_FD004.txt','data/test_FD001.txt','data/test_FD002.txt','data/test_FD003.txt','data/train_FD004.txt']
    for file in files:
        data=pd.read_csv(file,header=None,sep=' ')
        data=data[[x for x in range(0,26)]]
        data.columns=['ID', 'Cycle', 'OpSet1', 'OpSet2', 'OpSet3', 'SensorMeasure1', 'SensorMeasure2', 'SensorMeasure3', 'SensorMeasure4', 'SensorMeasure5', 'SensorMeasure6', 'SensorMeasure7', 'SensorMeasure8', 'SensorMeasure9', 'SensorMeasure10', 'SensorMeasure11', 'SensorMeasure12', 'SensorMeasure13', 'SensorMeasure14', 'SensorMeasure15', 'SensorMeasure16', 'SensorMeasure17', 'SensorMeasure18', 'SensorMeasure19', 'SensorMeasure20', 'SensorMeasure21']
        max_cycles_df=data.groupby(['ID'],sort=False)['Cycle'].max().reset_index().rename(columns={'Cycle':'MaxCycle'})
        data=pd.merge(data,max_cycles_df,how='inner',on='ID')
        
        data['RUL']=data['MaxCycle']-data['Cycle']
        #df_all_variables_train1=df_all_variables_train1.set_index('ID')
        #df_all_variables_train1['RUL']=[1 if out<20 else 0 for out in df_all_variables_train1['RUL']]
        dfs.append(data)
    return dfs
@st.cache(suppress_st_warning=True)
def get_data():
    dataframe_model=[]
    dataframes=data_retrieving()
    

    for data_model in dataframes:
        data_model=data_model.set_index('ID')
        data_model['RUL']=[1 if out<20 else 0 for out in data_model['RUL']]
        dataframe_model.append(data_model)

    return dataframes,dataframe_model
    
dataframes,dataframe_model=get_data()


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
           
            #,df_all_variables_train2,df_all_variables_train3,df_all_variables_train4,df_all_variables_test1,df_all_variables_test2,df_all_variables_test3,df_all_variables_test4
            st.dataframe(dataframes[4])
            st.subheader('Frequency of RUL accross the engines')
            df_max_rul = dataframes[0][['ID', 'RUL']].groupby('ID').max().reset_index()
            fig=plt.figure(figsize=(15,7))
            df_max_rul['RUL'].hist(bins=15)
            plt.xlabel('RUL')
            plt.ylabel('frequency')
            
            st.pyplot(fig)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.subheader('Visualise evolution of each sensor value by cycle')
            sensors=dataframes[0].iloc[:,5:26]
            plt.figure(figsize=(13,5))
            sensor_name = st.selectbox('Select the Sensor',(sensors.columns))
            for i in dataframes[0]['ID'].unique():
                if (i % 10 == 0):  # only plot every 10th unit_nr
                    plt.plot('RUL', sensor_name, 
                             data=dataframes[0][dataframes[0]['ID']==i])
            plt.xlim(250, 0)  # reverse the x-axis so RUL counts down to zero
            plt.xticks(np.arange(0, 275, 25))
            plt.ylabel(sensor_name)
            plt.xlabel('Remaining Use fulLife')
            plt.show()
            
            #plot_sensor(select_person)
            st.pyplot()
            st.subheader('Detect relationships between variables')
                
            fig = plt.figure(figsize=(10, 4))
            f_cor = dataframes[0].corr()
            sns.heatmap(f_cor)
            st.pyplot()

