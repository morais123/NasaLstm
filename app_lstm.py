# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 09:31:37 2022

@author: Mohamed-Aziz.Rais
"""


import  wx
import streamlit as st
import pandas as pd

app = wx.App()
@st.cache(suppress_st_warning=True)
def data_retrieving(dirname):
    """
    Function : data_retrieving
    Param : data_train : retrieves data and perform data preparation tasks
    Output: dataframe with recoded RUL 1 if RUL<20 or 0 of RUL>20 
    
    
    
    """
    dfs=[]
    files=['/train_FD001.txt','/train_FD002.txt','/train_FD003.txt','/train_FD004.txt','/test_FD001.txt','/test_FD002.txt','/test_FD003.txt','/train_FD004.txt']
    for file in files:
        df_all_variables_train1=pd.read_csv(dirname+file,header=None,sep=' ')
        df_all_variables_train1=df_all_variables_train1[[x for x in range(0,26)]]
        df_all_variables_train1.columns=['ID', 'Cycle', 'OpSet1', 'OpSet2', 'OpSet3', 'SensorMeasure1', 'SensorMeasure2', 'SensorMeasure3', 'SensorMeasure4', 'SensorMeasure5', 'SensorMeasure6', 'SensorMeasure7', 'SensorMeasure8', 'SensorMeasure9', 'SensorMeasure10', 'SensorMeasure11', 'SensorMeasure12', 'SensorMeasure13', 'SensorMeasure14', 'SensorMeasure15', 'SensorMeasure16', 'SensorMeasure17', 'SensorMeasure18', 'SensorMeasure19', 'SensorMeasure20', 'SensorMeasure21']
        max_cycles_df=df_all_variables_train1.groupby(['ID'],sort=False)['Cycle'].max().reset_index().rename(columns={'Cycle':'MaxCycle'})
        df_all_variables_train1=pd.merge(df_all_variables_train1,max_cycles_df,how='inner',on='ID')
        
        df_all_variables_train1['RUL']=df_all_variables_train1['MaxCycle']-df_all_variables_train1['Cycle']
        #df_all_variables_train1=df_all_variables_train1.set_index('ID')
        #df_all_variables_train1['RUL']=[1 if out<20 else 0 for out in df_all_variables_train1['RUL']]
        dfs.append(df_all_variables_train1)
    
    
    
    return dfs
@st.cache(suppress_st_warning=True)
def data_retrieving_models(dirname):
    """
    Function : data_retrieving
    Param : data_train : retrieves data and perform data preparation tasks
    Output: dataframe with recoded RUL 1 if RUL<20 or 0 of RUL>20 
    
    
    
    """
    dfs=[]
    files=['/train_FD001.txt','/train_FD002.txt','/train_FD003.txt','/train_FD004.txt','/test_FD001.txt','/test_FD002.txt','/test_FD003.txt','/train_FD004.txt']
    for file in files:
        df_all_variables_train1=pd.read_csv(dirname+file,header=None,sep=' ')
        df_all_variables_train1=df_all_variables_train1[[x for x in range(0,26)]]
        df_all_variables_train1.columns=['ID', 'Cycle', 'OpSet1', 'OpSet2', 'OpSet3', 'SensorMeasure1', 'SensorMeasure2', 'SensorMeasure3', 'SensorMeasure4', 'SensorMeasure5', 'SensorMeasure6', 'SensorMeasure7', 'SensorMeasure8', 'SensorMeasure9', 'SensorMeasure10', 'SensorMeasure11', 'SensorMeasure12', 'SensorMeasure13', 'SensorMeasure14', 'SensorMeasure15', 'SensorMeasure16', 'SensorMeasure17', 'SensorMeasure18', 'SensorMeasure19', 'SensorMeasure20', 'SensorMeasure21']
        max_cycles_df=df_all_variables_train1.groupby(['ID'],sort=False)['Cycle'].max().reset_index().rename(columns={'Cycle':'MaxCycle'})
        df_all_variables_train1=pd.merge(df_all_variables_train1,max_cycles_df,how='inner',on='ID')
        
        df_all_variables_train1['RUL']=df_all_variables_train1['MaxCycle']-df_all_variables_train1['Cycle']
        df_all_variables_train1=df_all_variables_train1.set_index('ID')
        df_all_variables_train1['RUL']=[1 if out<20 else 0 for out in df_all_variables_train1['RUL']]
        dfs.append(df_all_variables_train1)
    
    
    
    return dfs

@st.cache(suppress_st_warning=True)
def get_data():
    
    st.title('Folder Picker')
    st.write('Please select a folder:')
    clicked = st.button('Click to select the folder', key = "FolderSelectionButton")
    if clicked:
        dlg_obj = wx.DirDialog (None, "Choose input directory", "",
                            wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
                    
        if dlg_obj.ShowModal() == wx.ID_OK:
            folder_path = dlg_obj.GetPath()

        st.header('Selected Folder')
        
        dataframes=data_retrieving(folder_path)
        dataframe_model=data_retrieving_models(folder_path)
    return dataframes,dataframe_model

dataframes,dataframe_model=get_data()


st.dataframe(dataframes[0])
