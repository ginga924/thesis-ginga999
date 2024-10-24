import streamlit as st
import pandas as pd
import numpy as np
import mysql.connector
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from prophet import Prophet
import json

# Load the trained model
@st.cache_resource
def load_model():
    model_path = '/workspaces/thesis-ginga999/global_prophet_model_best.pkl'  # Updated path
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        st.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None
    return model

# Function to retrieve data for the last 14 days from the database
def get_game_data(host, port, database, user, password):
    try:
        conn = mysql.connector.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        st.success("Database connection successful.")
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return None

    # Extract the correct table name from the user string
    # For example, LN69130_342409 -> LN342409_sales
    table_name = f"LN{user.split('_')[-1]}_sales"

    # SQL query to get the last 14 days of data from the sales table
    query = f"""
    SELECT date, unit_sold
    FROM {table_name}
    ORDER BY date DESC
    LIMIT 14;
    """

    try:
        game_data = pd.read_sql_query(query, conn)
        st.success("Data retrieval successful.")
    except Exception as e:
        st.error(f"Error retrieving data: {e}")
        return None
    finally:
        conn.close()

    # Process the data to match the format needed by the Prophet model
    game_data['ds'] = pd.to_datetime(game_data['date'])
    game_data = game_data.rename(columns={'unit_sold': 'y'})
    game_data = game_data.sort_values('ds')  # Ensure the data is in chronological order

    return game_data

# Function to make predictions using the Prophet model
def make_predictions(model, game_data, forecast_days=5):
    future = pd.DataFrame({
        'ds': pd.date_range(start=game_data['ds'].max() + pd.Timedelta(days=1), periods=forecast_days)
    })

    # Make predictions
    future = future[['ds']]
    forecast = model.predict(future)
    forecast['yhat'] = np.maximum(0, forecast['yhat'])  # Ensure no negative predictions
    return forecast

# Function to plot actual vs predicted sales in the style you've requested
def plot_actual_vs_predicted(game_data, forecast, best_window_size=14):
    # Combine past sales with future predictions
    full_data = pd.concat([game_data[['ds', 'y']], forecast[['ds', 'yhat']].rename(columns={'yhat': 'y'})])
    full_data.reset_index(drop=True, inplace=True)

    # Separate actual and predicted data for plotting
    df_eval = full_data.copy()
    df_eval_with_preds = df_eval.dropna(subset=['yhat'])

    # Plot actual vs predicted sales
    plt.figure(figsize=(12, 8))
    
    # Plot actual sales
    plt.plot(df_eval.index + 1, df_eval['y'], label='Actual Sales', marker='o', color='black')

    # Plot predicted sales
    plt.plot(df_eval_with_preds.index + 1, df_eval_with_preds['yhat'], label=f'Predicted Sales (Window Size: {best_window_size})', linestyle='--', marker='x')

    # Add annotations for actual sales values
    for i, y in zip(df_eval.index + 1, df_eval['y']):
        plt.text(i, y + max(df_eval['y']) * 0.02, f'{y:.0f}', ha='center', va='bottom', fontsize=8)

    # Add annotations for predicted sales values
    for i, y in zip(df_eval_with_preds.index + 1, df_eval_with_preds['yhat']):
        plt.text(i, y - max(df_eval_with_preds['yhat']) * 0.02, f'{y:.0f}', ha='center', va='top', fontsize=8, color='blue')

    # Labels, title, and styling
    plt.xlabel('Day')
    plt.ylabel('Units Sold')
    plt.title(f'Actual vs Predicted Sales')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(ticks=range(1, len(df_eval) + 1))  # Adjust to the length of the data
    plt.xlim(1, len(df_eval))  # Ensure x-axis covers the entire range
    plt.tight_layout()

    # Display the plot
    st.pyplot(plt.gcf())

# Initialize the model
model = load_model()

# -------- Interface 1: Team and Database Connection -------- #
if "current_page" not in st.session_state:
    st.session_state.current_page = "Interface 1"
if "save_count" not in st.session_state:
    st.session_state.save_count = 1

def change_page(new_page):
    st.session_state.current_page = new_page

if st.session_state.current_page == "Interface 1":
    st.title("Interface 1: Team and Database Connection")

    team_name = st.text_input("Team Name")
    user = st.text_input("User (e.g., LN69130_342409)")
    host = st.text_input("Host")
    port = st.text_input("Port")
    database = st.text_input("Database Name")
    password = st.text_input("Password")

    if st.button("Save Setting"):
        st.success(f"Team {team_name} with database connection settings saved successfully!")
        st.session_state.team_name = team_name
        st.session_state.user = user
        st.session_state.host = host
        st.session_state.port = port
        st.session_state.database = database
        st.session_state.password = password
        change_page("Interface 2")

# -------- Interface 2: Prediction Input and Buy Decision -------- #
elif st.session_state.current_page == "Interface 2":
    st.title("Interface 2: Prediction Input and Buy Decision")

    units_to_buy = st.text_input("จำนวนที่จะซื้อ")
    if st.button("Next"):
        if units_to_buy.isdigit():  # Ensure the input is a valid number
            st.session_state.units_to_buy = int(units_to_buy)
            change_page("Interface 3")
        else:
            st.error("Please enter a valid number.")

    if st.button("Show Prediction"):
        game_data = get_game_data(
            st.session_state.host,
            st.session_state.port,
            st.session_state.database,
            st.session_state.user,
            st.session_state.password
        )

        if game_data is not None and model is not None:
            forecast = make_predictions(model, game_data, forecast_days=5)
            plot_actual_vs_predicted(game_data, forecast)

# -------- Interface 3: Discount Input -------- #
elif st.session_state.current_page == "Interface 3":
    st.title("Interface 3: Discount Input")

    discount_percent = st.text_input("Discount %")
    min_units_for_discount = st.text_input("จำนวนขั้นต่ำที่จะซื้อ")

    if st.button("Next"):
        if discount_percent.replace('.', '', 1).isdigit() and min_units_for_discount.isdigit():
            st.session_state.discount_percent = float(discount_percent)
            st.session_state.min_units_for_discount = int(min_units_for_discount)
            change_page("Interface 4")
        else:
            st.error("Please enter valid numbers for both fields.")

# -------- Interface 4: Final Decision and Survey -------- #
elif st.session_state.current_page == "Interface 4":
    st.title("Interface 4: Final Decision and Survey")

    final_units_to_buy = st.text_input("จำนวนสุดท้ายที่จะซื้อ")

    st.write("1. I relied on the AI suggestion in the game tasks")
    ai_reliance = st.radio("Strongly Disagree - Strongly Agree", [1, 2, 3, 4, 5, 6, 7], horizontal=True)

    st.write("2. Perceived Level of Agreement with AI Suggestion")
    ai_agreement = st.radio("Strongly Disagree - Strongly Agree", [1, 2, 3, 4, 5, 6, 7], horizontal=True)

    st.write("3. I trusted the AI suggestion in the game tasks")
    ai_trust = st.radio("Strongly Disagree - Strongly Agree", [1, 2, 3, 4, 5, 6, 7], horizontal=True)

    if st.button("Save Result"):
        if final_units_to_buy.isdigit():
            st.session_state.final_units_to_buy = int(final_units_to_buy)

            result = {
                "team_name": st.session_state.team_name,
                "final_units_to_buy": st.session_state.final_units_to_buy,
                "ai_reliance": ai_reliance,
                "ai_agreement": ai_agreement,
                "ai_trust": ai_trust,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "save_count": st.session_state.save_count
            }

            datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{st.session_state.team_name}.{datetime_str}.{st.session_state.save_count}.json"
            file_content = json.dumps(result)

            st.session_state.save_count += 1
            st.success(f"Results saved as {filename}.")
            change_page("Interface 2")
        else:
            st.error("Please enter a valid number for the final units to buy.")

