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
    st.write(df)

    st.markdown("Anaylyzing statement from: " + df.head(1)['Transaction Date'].values[0].split(' ')[0] + " to " + df.tail(1)['Transaction Date'].values[0].split(' ')[0])
    
    df['Withdraw'] = df["Withdraw"].replace('-','0',inplace=False)
    df['Withdraw'] = df["Withdraw"].replace(',','',inplace=False, regex=True)
    df['Deposit'] = df["Deposit"].replace('-','0',inplace=False)
    df['Deposit'] = df["Deposit"].replace(',','',inplace=False, regex=True)
    df['Withdraw'] = pd.to_numeric(df['Withdraw'])
    df['Deposit'] = pd.to_numeric(df['Deposit'])
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])

    st.write(df['Transaction Date'].dtype)
    
    #df['Transaction Date'] = df['Transaction Date'].dt.date
    #df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], format="%Y-%m-%d")

    
    df = df[(df['Withdraw'] < 100000)] 
    df = df[(df['Deposit'] < 1000000)]
    df = df.drop(['S.N'], axis=1)
    df['Day'] = df['Transaction Date'].dt.day_name()
    
    
    st.markdown("TOP 10 DAYS WHERE YOU SPENT THE MOST")
    df_spent = df.sort_values(by='Withdraw', ascending=False).head(10)
    st.write(df_spent)
    df_spent_index = df_spent.set_index('Transaction Date')['Withdraw']

    st.bar_chart(data=df_spent_index)
    
    st.markdown("TOP 10 DAYS WHERE YOU EARNED THE MOST")
    df_earned =  df.sort_values(by='Deposit', ascending=False).head(10) 
    st.write(df_earned)
    df_earned_index = df_earned.set_index('Transaction Date')['Deposit']
    st.bar_chart(data=df_earned_index)
    
    st.markdown("MONTHLY EARNING VS SPENDING")
    
    df['Month'] = df['Transaction Date'].dt.month_name()
    df['Year'] = df['Transaction Date'].dt.year.astype(str)
    
    df['Year-Month'] = df['Year'].astype(str) + '-' + df['Month'].astype(str)

    # Set this new column as the index
    df.set_index('Year-Month', inplace=True)

    # Drop the now redundant 'Year' and 'Month' columns if you wish (optional)
    df.drop(['Year', 'Month'], axis=1, inplace=True)

    
    monthly_expenses = df.groupby('Year-Month')['Withdraw'].sum()
    monthly_income = df.groupby('Year-Month')['Deposit'].sum()

    st.write(monthly_expenses)
    st.write(monthly_income)
    # Combine monthly_expenses and monthly_income into a single DataFrame
    monthly_finances = pd.DataFrame({
        'Withdrawls': monthly_expenses,
        'Deposits': monthly_income
    }).reset_index()

    # Now, plot with Streamlit
    st.bar_chart(monthly_finances.set_index('Year-Month'))
   # st.markdown("MONTHLY AVG SPENDING")
   # df_monthly_total = df.groupby(['Year-Month'])[['Withdraw']].sum()
   # df_monthly_spend = df_monthly_total['Withdraw'].mean()
    #st.write(df_monthly_spend)
    #st.bar_chart(data=df_monthly_spend)

    # st.markdown("EARN VS SPENDING WITH RESPECT TO DAYS")
    # df3=df[['Day', 'Withdraw', 'Deposit']].groupby('Day').sum()
    # st.write(df3)
    # st.bar_chart(data=df3)
    
    st.markdown("Saving Rate")
    saving_rate = ((monthly_income - monthly_expenses) / monthly_income) * 100
    st.write(saving_rate)