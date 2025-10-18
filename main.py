import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tabula as tb
from sklearn.linear_model import LinearRegression

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
        "Analyzing Statement From: "
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

    df_original = df.copy()

    # Add Year-Month processing to df_original for forecasting
    df_original["Month"] = df_original["Transaction Date"].dt.month_name()
    df_original["Year"] = df_original["Transaction Date"].dt.year.astype(str)
    df_original["Year-Month"] = (
        df_original["Transaction Date"].dt.to_period("M").astype(str)
    )

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

    # Create Year-Month in YYYY-MM format for proper sorting
    df["Year-Month"] = df["Transaction Date"].dt.to_period("M").astype(str)

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

    # Sort in descending order (most recent first)
    monthly_finances = monthly_finances.sort_values("Year-Month", ascending=False)
    monthly_finances = monthly_finances.reset_index(drop=True)
    st.write(monthly_finances)

    # Create a properly ordered DataFrame for the chart
    monthly_finances_chart = monthly_finances.copy()
    monthly_finances_chart = monthly_finances_chart.set_index("Year-Month")

    # Now, plot with Streamlit - the data is already sorted in descending order
    st.line_chart(monthly_finances_chart)

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

    # ===== NEW FEATURES ======

    st.markdown("##  SPENDING FORECASTS")

    # Prepare data for forecasting
    df_forecast = df_original.copy()

    # Apply the same filters as main analysis for realistic spending forecasts
    df_forecast = df_forecast[(df_forecast["Debit"] < 20000)]

    # Set Year-Month as the index (already created in df_original)
    df_forecast.set_index("Year-Month", inplace=True)

    # Drop the now redundant 'Year' and 'Month' columns if you wish (optional)
    df_forecast.drop(["Year", "Month"], axis=1, inplace=True)

    df_forecast["Date"] = df_forecast["Transaction Date"].dt.date
    daily_spending = df_forecast.groupby("Date")["Debit"].sum().reset_index()
    daily_spending["Date"] = pd.to_datetime(daily_spending["Date"])
    daily_spending = daily_spending.set_index("Date").sort_index()

    # Fill missing dates with 0 spending
    date_range = pd.date_range(
        start=daily_spending.index.min(), end=daily_spending.index.max(), freq="D"
    )
    daily_spending = daily_spending.reindex(date_range, fill_value=0)

    # Simple moving average forecast
    st.markdown("### Simple Moving Average Forecast")

    # Calculate different moving averages
    daily_spending["MA_7"] = daily_spending["Debit"].rolling(window=7).mean()
    daily_spending["MA_14"] = daily_spending["Debit"].rolling(window=14).mean()
    daily_spending["MA_30"] = daily_spending["Debit"].rolling(window=30).mean()

    # Forecast next 30 days using 30-day moving average
    last_ma30 = daily_spending["MA_30"].dropna().iloc[-1]
    forecast_dates = pd.date_range(
        start=daily_spending.index.max() + pd.Timedelta(days=1), periods=30, freq="D"
    )
    forecast_data = pd.DataFrame(
        {"Date": forecast_dates, "Forecasted_Spending": [last_ma30] * 30}
    )

    # Create forecast visualization
    fig3, ax3 = plt.subplots(figsize=(15, 8))

    # Plot historical data
    ax3.plot(
        daily_spending.index,
        daily_spending["Debit"],
        label="Actual Daily Spending",
        alpha=0.7,
        linewidth=1,
    )
    ax3.plot(
        daily_spending.index,
        daily_spending["MA_7"],
        label="7-Day Moving Average",
        linewidth=2,
    )
    ax3.plot(
        daily_spending.index,
        daily_spending["MA_14"],
        label="14-Day Moving Average",
        linewidth=2,
    )
    ax3.plot(
        daily_spending.index,
        daily_spending["MA_30"],
        label="30-Day Moving Average",
        linewidth=2,
    )

    # Plot forecast
    ax3.plot(
        forecast_data["Date"],
        forecast_data["Forecasted_Spending"],
        label="30-Day Forecast",
        linestyle="--",
        linewidth=3,
        color="red",
    )

    ax3.set_title("Spending Forecast (Next 30 Days)")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Daily Spending (Rs. )")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Format y-axis to show values in thousands
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1000:.0f}k"))

    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig3)

    # Monthly spending forecast
    st.markdown("### Monthly Spending Forecast")

    # Calculate monthly spending trends
    monthly_spending = df_original_forecast.groupby(
        df_original_forecast["Transaction Date"].dt.to_period("M")
    )["Debit"].sum()
    monthly_spending.index = monthly_spending.index.to_timestamp()

    # Simple linear trend for monthly forecast
    monthly_df_original = pd.DataFrame(
        {"Date": monthly_spending.index, "Spending": monthly_spending.values}
    )
    monthly_df_original["Month_Number"] = range(len(monthly_df_original))

    # Use multiple forecasting methods and take the best approach
    from sklearn.linear_model import LinearRegression

    X = monthly_df_original[["Month_Number"]]
    y = monthly_df_original["Spending"]

    # Method 1: Linear regression
    model = LinearRegression()
    model.fit(X, y)

    # Method 2: Simple moving average (last 3 months)
    recent_avg = monthly_df_original["Spending"].tail(3).mean()

    # Method 3: Exponential smoothing (simple)
    if len(monthly_df_original) >= 2:
        alpha = 0.3  # Smoothing factor
        exp_smooth = monthly_df_original["Spending"].iloc[0]
        for i in range(1, len(monthly_df_original)):
            exp_smooth = (
                alpha * monthly_df_original["Spending"].iloc[i]
                + (1 - alpha) * exp_smooth
            )
    else:
        exp_smooth = monthly_df_original["Spending"].mean()

    # Forecast next 6 months using the best method
    future_months = range(len(monthly_df_original), len(monthly_df_original) + 6)
    future_dates = pd.date_range(
        start=monthly_df_original["Date"].max() + pd.DateOffset(months=1),
        periods=6,
        freq="M",
    )

    # Linear regression predictions
    linear_predictions = model.predict([[m] for m in future_months])

    # Use the most conservative approach: take the maximum of recent average and linear prediction
    # but ensure no negative values
    forecast_values = []
    for i, linear_pred in enumerate(linear_predictions):
        # Use a weighted average: 60% recent average, 40% linear trend
        # but ensure it's not negative and not unreasonably high
        conservative_forecast = max(0, 0.6 * recent_avg + 0.4 * linear_pred)

        # Cap the forecast to be within reasonable bounds (not more than 3x recent average)
        conservative_forecast = min(conservative_forecast, recent_avg * 3)

        forecast_values.append(conservative_forecast)

    forecast_monthly = pd.DataFrame(
        {"Date": future_dates, "Forecasted_Spending": forecast_values}
    )

    # Create monthly forecast visualization
    fig4, ax4 = plt.subplots(figsize=(12, 6))

    ax4.plot(
        monthly_df_original["Date"],
        monthly_df_original["Spending"],
        marker="o",
        label="Actual Monthly Spending",
        linewidth=2,
    )
    ax4.plot(
        forecast_monthly["Date"],
        forecast_monthly["Forecasted_Spending"],
        marker="s",
        label="6-Month Forecast",
        linestyle="--",
        linewidth=2,
        color="red",
    )

    # Add a horizontal line for recent average
    ax4.axhline(
        y=recent_avg,
        color="green",
        linestyle=":",
        alpha=0.7,
        label=f"Recent 3-Month Avg (Rs. {recent_avg:.0f})",
    )

    ax4.set_title("Monthly Spending Forecast (Next 6 Months)")
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Monthly Spending (Rs. )")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Format y-axis to show values in thousands
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1000:.0f}k"))

    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig4)

    # Add forecast method explanation
    st.markdown("#### Forecast Method")
    st.write(f"**Recent 3-Month Average**: Rs. {recent_avg:.0f}")
    st.write(
        f"**Linear Trend Prediction**: Rs. {linear_predictions[0]:.0f} (next month)"
    )
    st.write(
        f"**Final Forecast**: Weighted average ensuring realistic values (no negative predictions)"
    )

    # Forecast summary
    st.markdown("### Forecast Summary")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Next 30 Days (Daily Avg)", f"Rs. {last_ma30:.0f}")
        st.metric("Next 30 Days (Total)", f"Rs. {last_ma30 * 30:.0f}")

    with col2:
        next_month_forecast = forecast_monthly.iloc[0]["Forecasted_Spending"]
        st.metric("Next Month Forecast", f"Rs. {next_month_forecast:.0f}")
        st.metric(
            "6-Month Average",
            f"Rs. {forecast_monthly['Forecasted_Spending'].mean():.0f}",
        )

    with col3:
        # Calculate spending trend
        recent_avg = monthly_df_original["Spending"].tail(3).mean()
        forecast_avg = forecast_monthly["Forecasted_Spending"].mean()
        trend_pct = ((forecast_avg - recent_avg) / recent_avg) * 100
        st.metric("Spending Trend", f"{trend_pct:+.1f}%")

        # Spending volatility
        volatility = daily_spending["Debit"].std()
        st.metric("Daily Volatility", f"Rs. {volatility:.0f}")

    # ===== NEW SECTION: DATE RANGE FILTERED SPENDING ANALYSIS =====

    st.markdown("## 📅 CUSTOM DATE RANGE SPENDING ANALYSIS")

    # Get the date range from the original data
    min_date = df_original["Transaction Date"].min().date()
    max_date = df_original["Transaction Date"].max().date()

    # Create date range selector
    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Select Start Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            key="start_date_filter",
        )

    with col2:
        end_date = st.date_input(
            "Select End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="end_date_filter",
        )

    # Add checkbox to exclude investments
    exclude_investments = st.checkbox("Exclude Investments", value=False)

    # Add threshold input that appears when checkbox is checked
    investment_threshold = 20000  # default value
    if exclude_investments:
        investment_threshold = st.number_input(
            "Exclude Transactions Above (Rs.)",
            min_value=1000,
            max_value=100000,
            value=20000,
            step=1000,
            help="Transactions above this amount will be considered investments and excluded",
        )

    # Filter data based on selected date range
    if start_date <= end_date:
        filtered_df_original = df_original[
            (df_original["Transaction Date"].dt.date >= start_date)
            & (df_original["Transaction Date"].dt.date <= end_date)
        ].copy()

        # Filter out credit transactions (debit = 0) - only show actual spending
        filtered_df_original = filtered_df_original[filtered_df_original["Debit"] > 0]

        # Apply investment exclusion filter if checkbox is checked
        if exclude_investments:
            filtered_df_original = filtered_df_original[
                filtered_df_original["Debit"] < investment_threshold
            ]

        if not filtered_df_original.empty:

            # All spending transactions in the selected period
            st.markdown("### All Spending Transactions")
            all_spending_filtered = filtered_df_original[display_columns].sort_values(
                "Debit", ascending=False
            )
            st.write(all_spending_filtered)

            st.markdown(f"### Spending Analysis:")

            # Calculate total spending in the selected period
            total_spending = filtered_df_original["Debit"].sum()
            avg_daily_spending = total_spending / len(
                filtered_df_original["Transaction Date"].dt.date.unique()
            )

            # Display key metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Spending", f"Rs. {total_spending:,.0f}")

            with col2:
                st.metric("Average Daily Spending", f"Rs. {avg_daily_spending:,.0f}")

            with col3:
                days_in_period = (end_date - start_date).days + 1
                st.metric("Days in Period", f"{days_in_period}")

            # Daily spending trend for the selected period
            daily_spending_filtered = (
                filtered_df_original.groupby(
                    filtered_df_original["Transaction Date"].dt.date
                )["Debit"]
                .sum()
                .reset_index()
            )
            daily_spending_filtered.columns = ["Date", "Daily_Spending"]

            if len(daily_spending_filtered) > 1:
                # Spending pattern analysis
                st.markdown("### Spending Pattern Analysis")

                # Day of week analysis
                filtered_df_original["Day_of_Week"] = filtered_df_original[
                    "Transaction Date"
                ].dt.day_name()
                weekly_spending = (
                    filtered_df_original.groupby("Day_of_Week")["Debit"]
                    .sum()
                    .sort_values(ascending=False)
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Spending by Day of Week**")
                    st.bar_chart(weekly_spending)

                with col2:
                    # Spending distribution
                    spending_ranges = [
                        (0, 1000, "Rs. 0 - Rs. 1,000"),
                        (1000, 5000, "Rs. 1,000 - Rs. 5,000"),
                        (5000, 10000, "Rs. 5,000 - Rs. 10,000"),
                        (10000, 20000, "Rs. 10,000 - Rs. 20,000"),
                        (20000, float("inf"), "Above Rs. 20,000"),
                    ]

                    range_counts = []
                    range_labels = []

                    for min_val, max_val, label in spending_ranges:
                        count = len(
                            filtered_df_original[
                                (filtered_df_original["Debit"] >= min_val)
                                & (filtered_df_original["Debit"] < max_val)
                            ]
                        )
                        range_counts.append(count)
                        range_labels.append(label)

                    spending_dist = pd.DataFrame(
                        {"Range": range_labels, "Count": range_counts}
                    ).set_index("Range")

                    st.markdown("**Spending Distribution**")
                    st.bar_chart(spending_dist)

            # Summary insights
            st.markdown("### 📊 Summary Insights")

            # Calculate insights
            highest_spending_day = daily_spending_filtered.loc[
                daily_spending_filtered["Daily_Spending"].idxmax()
            ]
            lowest_spending_day = daily_spending_filtered.loc[
                daily_spending_filtered["Daily_Spending"].idxmin()
            ]

            col1, col2 = st.columns(2)

            with col1:
                st.info(
                    f"**Highest Spending Day**: {highest_spending_day['Date']} (Rs. {highest_spending_day['Daily_Spending']:,.0f})"
                )
                st.info(
                    f"**Lowest Spending Day**: {lowest_spending_day['Date']} (Rs. {lowest_spending_day['Daily_Spending']:,.0f})"
                )

            with col2:
                spending_variance = daily_spending_filtered["Daily_Spending"].std()
                st.info(f"**Spending Variance**: Rs. {spending_variance:,.0f}")

                if len(daily_spending_filtered) > 0:
                    zero_spending_days = len(
                        daily_spending_filtered[
                            daily_spending_filtered["Daily_Spending"] == 0
                        ]
                    )
                    st.info(f"**Zero Spending Days**: {zero_spending_days}")

        else:
            st.warning("No spending data found for the selected date range.")

    else:
        st.error("Start date must be before or equal to end date.")
