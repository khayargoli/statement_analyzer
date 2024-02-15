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
    dfs = tb.read_pdf(uploaded_file, pages='all', multiple_tables = True, columns=[27.0,68.0,272.0,357.5,397.0,474.5,553.0,631.0])
    df = pd.concat(dfs)
    df = df.dropna()

    st.title('Statement Analysis')
    st.markdown("Record size:")
    st.write(df.shape)

    st.markdown("Anaylyzing statement from: " + df.head(1)['Transaction Date'].values[0].split(' ')[0] + " to " + df.tail(1)['Transaction Date'].values[0].split(' ')[0])
    
    df['Withdraw'] = df["Withdraw"].replace('-','0',inplace=False)
    df['Withdraw'] = df["Withdraw"].replace(',','',inplace=False, regex=True)
    df['Deposit'] = df["Deposit"].replace('-','0',inplace=False)
    df['Deposit'] = df["Deposit"].replace(',','',inplace=False, regex=True)
    df['Withdraw'] = pd.to_numeric(df['Withdraw'])
    df['Deposit'] = pd.to_numeric(df['Deposit'])
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], format="%Y-%m-%d %H:%M:%S")
    df['Transaction Date'] = df['Transaction Date'].dt.date
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], format="%Y-%m-%d")
    df.head()

    
    df = df[(df['Withdraw'] < 1000000)] 
    df = df[(df['Deposit'] < 1000000)]
    st.write(df.head(1))
    st.write(df.tail(1))
    df = df.drop(['S.N'], axis=1)
    df['Day'] = df['Transaction Date'].dt.day_name()
    df_trans = df.groupby(["Transaction Date"] , as_index=False).sum()
    
    
    st.markdown("Statistical Overview")
    date_sums = df.groupby(["Transaction Date"] , as_index=False).sum()
    st.dataframe(date_sums.describe())
    
    st.markdown("TOP 10 DAYS WHERE YOU SPENT THE MOST")
    st.write(df_trans.nlargest(10, 'Withdraw'))
    df_spent = df_trans.nlargest(10, 'Withdraw').set_index('Transaction Date')['Withdraw']

    st.bar_chart(data=df_spent)
    
    st.markdown("TOP 10 DAYS WHERE YOU EARNED THE MOST")
    st.write(df_trans.nlargest(10, 'Deposit'))
    df_earned = df_trans.nlargest(10, 'Withdraw').set_index('Transaction Date')['Withdraw']
    st.bar_chart(data=df_earned)
    
    st.markdown("MONTHLY SPENDING")
    df2=df_trans.groupby(pd.Grouper(key='Transaction Date', freq='1ME')).sum()
    st.write(df2)
    st.bar_chart(data=df2)

    st.markdown("MONTHLY AVG SPENDING")
    df4=date_sums.groupby(pd.Grouper(key='Transaction Date', freq='1ME')).mean()
    st.write(df4)
    st.bar_chart(data=df4)

    st.markdown("EARN VS SPENDING WITH RESPECT TO DAYS")
    df3=df[['Day', 'Withdraw', 'Deposit']].groupby('Day').sum()
    st.write(df3)
    st.bar_chart(data=df3)
    
