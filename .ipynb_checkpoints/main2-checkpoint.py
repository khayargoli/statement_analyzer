import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tabula as tb
import math

pd.pandas.set_option("display.max_rows", None)
pd.pandas.set_option("display.max_columns", None)

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    dfs = tb.read_pdf(uploaded_file, pages='all', multiple_tables = True)
    df = pd.concat(dfs)
    df = df.dropna()

    st.title('Statement Analysis')
    st.markdown("Record size:")
    st.write(df.shape)

    st.markdown("Anaylyzing statement from: " + df.head(1)['Transaction Date'].values[0].split(' ')[0] + " to " + df.tail(1)['Transaction Date'].values[0].split(' ')[0])
    
    df['Debit'] = df["Debit"].replace('-','0',inplace=False)
    df['Debit'] = df["Debit"].replace(',','',inplace=False, regex=True)
    df['Credit'] = df["Credit"].replace('-','0',inplace=False)
    df['Credit'] = df["Credit"].replace(',','',inplace=False, regex=True)
    df['Debit'] = pd.to_numeric(df['Debit'])
    df['Credit'] = pd.to_numeric(df['Credit'])
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], format='%d-%m-%Y') 
    #pd.to_date(df['Transaction Date'], format="%Y-%m-%d")
    #df['Transaction Date'] = df['Transaction Date'].dt.date
    #df['Transaction Date'] = pd.to_date(df['Transaction Date'], format="%Y-%m-%d")
    df.head()

    
    df = df[(df['Debit'] < 1000000)] 
    df = df[(df['Credit'] < 1000000)]
    st.write(df.head(1))
    st.write(df.tail(1))
    # df = df.drop(['S.N'], axis=1)
    df['Day'] = df['Transaction Date'].dt.day_name()
    
    
    df_trans = df.groupby(["Transaction Date"] , as_index=False).sum()
    
    
    st.markdown("Statistical Overview")
    date_sums = df.groupby(["Transaction Date"] , as_index=False).sum()
    st.dataframe(date_sums.describe())
    
    st.markdown("TOP 5 DAYS WHERE YOU SPENT THE MOST")
    st.write(df_trans.nlargest(5, 'Debit'))
    # st.bar_chart(data=df_trans.nlargest(5, 'Debit'))
    
    st.markdown("TOP 5 DAYS WHERE YOU EARNED THE MOST")
    st.write(df_trans.nlargest(5, 'Credit'))
   # st.bar_chart(data=df_trans.nlargest(5, 'Credit'))
    
    st.markdown("MONTHLY SPENDING")
    df2=date_sums.groupby(pd.Grouper(key='Transaction Date', freq='1M')).sum()
    st.bar_chart(data=df2)
    
    st.markdown("MONTHLY AVG SPENDING")
    df4=date_sums.groupby(pd.Grouper(key='Transaction Date', freq='1M')).mean()
    st.bar_chart(data=df4)

    st.markdown("EARN VS SPENDING WITH RESPECT TO DAYS")
    df3=df[['Day', 'Debit', 'Credit']].groupby('Day').sum()
    #st.write(df3)
    st.bar_chart(data=df3)
    
