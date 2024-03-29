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
    dfs = tb.read_pdf(uploaded_file, pages='all',  multiple_tables = True,  area=[185, 420, 1130, 800], columns=[420, 500,  590, 660, 730])
    df = pd.concat(dfs)
    st.write(df.head())
    df = df.drop(['Unnamed: 0', 'Cheque Number'], axis=1)

    #df = df.dropna()
    df = df[df['Value Date'] != 'null']
    
    st.title('Statement Analysis')
    st.markdown("Record size:")
    st.write(df.shape)
    st.markdown("Anaylyzing statement from: " + df.head(1)['Value Date'].values[0].split(' ')[0] + " to " + df.tail(1)['Value Date'].values[0].split(' ')[0])
    
    df['Withdraw'] = df["Withdraw"].replace('-','0',inplace=False)
    df['Withdraw'] = df["Withdraw"].replace(',','',inplace=False, regex=True)
    df['Deposit'] = df["Deposit"].replace('-','0',inplace=False)
    df['Deposit'] = df["Deposit"].replace(',','',inplace=False, regex=True)
    df['Withdraw'] = pd.to_numeric(df['Withdraw'])
    df['Deposit'] = pd.to_numeric(df['Deposit'])
    df['Value Date'] = pd.to_datetime(df['Value Date'])

    st.write(df['Value Date'].dtype)
    
    #df['Value Date'] = df['Value Date'].dt.date
    #df['Value Date'] = pd.to_datetime(df['Value Date'], format="%Y-%m-%d")

    
    df = df[(df['Withdraw'] < 20000)] 
    df = df[(df['Deposit'] < 100000)]
    st.markdown("Record size after removing transactions more than 1 lakh: " + str(df.shape))
    #df = df.drop(['S.N'], axis=1)
    df['Day'] = df['Value Date'].dt.day_name()
    
    
    st.markdown("TOP 30 TRANS WHERE YOU SPENT THE MOST")
    df_spent = df.sort_values(by='Withdraw', ascending=False).head(30)
    st.write(df_spent)
    df_spent_index = df_spent.set_index('Value Date')['Withdraw']

    st.bar_chart(data=df_spent_index)
    
    st.markdown("TOP 30 TRANS WHERE YOU EARNED THE MOST")
    df_earned =  df.sort_values(by='Deposit', ascending=False).head(30) 
    st.write(df_earned)
    df_earned_index = df_earned.set_index('Value Date')['Deposit']
    st.bar_chart(data=df_earned_index)
    
    st.markdown("MONTHLY SPENDING / EARNING / SAVINGS")
    
    df['Month'] = df['Value Date'].dt.month_name()
    df['Year'] = df['Value Date'].dt.year.astype(str)
    
    df['Year-Month'] = df['Year'].astype(str) + '-' + df['Month'].astype(str)

    # Set this new column as the index
    df.set_index('Year-Month', inplace=True)

    # Drop the now redundant 'Year' and 'Month' columns if you wish (optional)
    df.drop(['Year', 'Month'], axis=1, inplace=True)

    
    monthly_expenses = df.groupby('Year-Month')['Withdraw'].sum()
    monthly_income = df.groupby('Year-Month')['Deposit'].sum()
    monthly_savings = monthly_income - monthly_expenses

  
    
    # Combine monthly_expenses and monthly_income into a single DataFrame
    monthly_finances = pd.DataFrame({
        'Withdrawls': monthly_expenses,
        'Deposits': monthly_income,
        'Savings': monthly_savings
    }).reset_index()
    st.write(monthly_finances)
    # Now, plot with Streamlit
    st.line_chart(monthly_finances.set_index('Year-Month'))
   # st.markdown("MONTHLY AVG SPENDING")
   # df_monthly_total = df.groupby(['Year-Month'])[['Withdraw']].sum()
   # df_monthly_spend = df_monthly_total['Withdraw'].mean()
    #st.write(df_monthly_spend)
    #st.bar_chart(data=df_monthly_spend)

    # st.markdown("EARN VS SPENDING WITH RESPECT TO DAYS")
    # df3=df[['Day', 'Withdraw', 'Deposit']].groupby('Day').sum()
    # st.write(df3)
    # st.bar_chart(data=df3)
    
    
    st.markdown(" Saving Rate")
   
    

    saving_rate = (monthly_savings / monthly_income) * 100
    st.write(saving_rate)

    savings_target = st.number_input('Enter earning target')
   
    if st.button('Calculate Time'):
        savings_target = float(savings_target)
        average_monthly_savings = monthly_savings.mean()
        months_to_target = savings_target / average_monthly_savings
        st.markdown("### Time to Reach Earning Target")
        #st.write(f"Months to reach savings target: {months_to_target:.2f} months")
        years = months_to_target // 12
        remaining_months = months_to_target % 12  # Using modulo operator to find the remainder

        st.write(f"{years:.0f} years and {remaining_months:.0f} months.")

    