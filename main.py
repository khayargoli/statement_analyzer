import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tabula as tb

pd.pandas.set_option("display.max_rows", None)
pd.pandas.set_option("display.max_columns", None)

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    dfs = tb.read_pdf(uploaded_file, pages="all", multiple_tables=True)
    df = pd.concat(dfs)
    df = df.reset_index(drop=True)
    st.write("Raw data:")
    st.write(df.head(20))

    # Show column names to debug
    st.write("Column names:")
    st.write(df.columns.tolist())

    # Drop rows where Transaction Date is null or empty
    df = df.dropna(subset=["Transaction Date"])
    df = df[df["Transaction Date"] != ""]
    df = df[df["Transaction Date"].notna()]

    st.write("After filtering:")
    st.write(df.head(20))

    st.title("Statement Analysis")
    st.markdown("Record size:")
    st.write(df.shape)
    st.markdown(
        "Anaylyzing statement from: "
        + df.head(1)["Transaction Date"].values[0].split(" ")[0]
        + " to "
        + df.tail(1)["Transaction Date"].values[0].split(" ")[0]
    )

    df["Debit"] = df["Debit"].replace("-", "0", inplace=False)
    df["Debit"] = df["Debit"].replace(",", "", inplace=False, regex=True)
    df["Credit"] = df["Credit"].replace("-", "0", inplace=False)
    df["Credit"] = df["Credit"].replace(",", "", inplace=False, regex=True)
    df["Debit"] = pd.to_numeric(df["Debit"])
    df["Credit"] = pd.to_numeric(df["Credit"])
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], format="%d-%m-%Y")

    # Create a separate display column with only date (no time)
    df["Date"] = df["Transaction Date"].dt.date

    # Create a clean display DataFrame with Date first and no Transaction Date
    display_columns = ["Date"] + [
        col for col in df.columns if col not in ["Date", "Transaction Date"]
    ]
    df_display = df[display_columns].copy()

    df_high = df[(df["Debit"] >= 20000)]
    st.markdown(
        "TOP 30 TRANS WHERE YOU SPENT MORE THAN 20 thousand rupees (Most probably investments)"
    )
    df_spent_high = df_high.sort_values(
        by=["Debit", "Date"], ascending=[False, False]
    ).head(30)
    df_spent_high = df_spent_high.reset_index(drop=True)
    st.write(df_spent_high[display_columns])

    df = df[(df["Debit"] < 20000)]
    df = df[(df["Credit"] < 180000)]

    st.markdown("TOP 30 TRANS WHERE YOU SPENT THE MOST (LESS THAN 20 thousand rupees)")
    df_spent = df.sort_values(by=["Debit", "Date"], ascending=[False, False]).head(30)
    df_spent = df_spent.reset_index(drop=True)
    st.write(df_spent[display_columns])
    df_spent_index = df_spent.set_index("Date")["Debit"]

    st.bar_chart(data=df_spent_index)

    st.markdown("TOP 30 TRANS WHERE YOU EARNED THE MOST")
    df_earned = df.sort_values(by="Credit", ascending=False).head(30)
    st.write(df_earned[display_columns])
    df_earned_index = df_earned.set_index("Date")["Credit"]
    st.bar_chart(data=df_earned_index)

    st.markdown("MONTHLY SPENDING / EARNING / SAVINGS")

    df["Month"] = df["Transaction Date"].dt.month_name()
    df["Year"] = df["Transaction Date"].dt.year.astype(str)

    df["Year-Month"] = df["Year"].astype(str) + "-" + df["Month"].astype(str)

    # Set this new column as the index
    df.set_index("Year-Month", inplace=True)

    # Drop the now redundant 'Year' and 'Month' columns if you wish (optional)
    df.drop(["Year", "Month"], axis=1, inplace=True)

    monthly_expenses = df.groupby("Year-Month")["Debit"].sum().round(0).astype(int)
    monthly_income = df.groupby("Year-Month")["Credit"].sum().round(0).astype(int)
    monthly_savings = (monthly_income - monthly_expenses).round(0).astype(int)

    # Combine monthly_expenses and monthly_income into a single DataFrame
    monthly_finances = pd.DataFrame(
        {
            "Debits": monthly_expenses,
            "Credits": monthly_income,
            "Savings": monthly_savings,
        }
    ).reset_index()

    # Convert Year-Month to datetime for proper sorting, then sort in descending order
    monthly_finances["Year-Month-Date"] = pd.to_datetime(
        monthly_finances["Year-Month"], format="%Y-%B"
    )
    monthly_finances = monthly_finances.sort_values("Year-Month-Date", ascending=False)
    monthly_finances = monthly_finances.drop(
        "Year-Month-Date", axis=1
    )  # Remove the temporary column
    monthly_finances = monthly_finances.reset_index(drop=True)
    st.write(monthly_finances)
    # Now, plot with Streamlit
    st.line_chart(monthly_finances.set_index("Year-Month"))

    st.markdown(" Saving Rate")

    saving_rate = (monthly_savings / monthly_income) * 100
    st.write(saving_rate)

    savings_target = st.number_input("Enter earning target")

    if st.button("Calculate Time"):
        savings_target = float(savings_target)
        average_monthly_savings = monthly_savings.mean()
        months_to_target = savings_target / average_monthly_savings
        st.markdown("### Time to Reach Earning Target")
        # st.write(f"Months to reach savings target: {months_to_target:.2f} months")
        years = months_to_target // 12
        remaining_months = (
            months_to_target % 12
        )  # Using modulo operator to find the remainder

        st.write(f"{years:.0f} years and {remaining_months:.0f} months.")
