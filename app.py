import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- PAGE CONFIG & PREMIUM CSS ---
st.set_page_config(page_title="Logistics ETA Predictor", layout="wide", page_icon="⏱️")

st.markdown("""
    <style>
    .main { background-color: #f8fafc; font-family: 'Inter', sans-serif; }
    .header-box {
        background: linear-gradient(135deg, #0f172a 0%, #334155 100%);
        padding: 2.5rem; border-radius: 10px; color: white; text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 2rem;
    }
    .header-box h1 { margin: 0; font-weight: 700; color: #ffffff; }
    .stButton>button {
        background-color: #2563eb; color: white; font-weight: 600;
        padding: 0.75rem; border-radius: 8px; width: 100%; border: none; transition: 0.3s;
    }
    .stButton>button:hover { background-color: #1d4ed8; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2); }
    .result-card {
        background: white; padding: 2rem; border-radius: 8px; border-left: 6px solid #2563eb;
        text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-top: 2rem;
    }
    .result-value { font-size: 3.5rem; font-weight: 800; color: #0f172a; }
    </style>
""", unsafe_allow_html=True)

# --- LOAD ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    with open('models/xgb_best.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/standard_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

# --- UI LAYOUT ---
st.markdown('<div class="header-box"><h1>⏱️ Intelligent Delivery ETA Engine</h1><p>Predicting delivery transit times using advanced machine learning.</p></div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Courier Metrics")
    age = st.number_input("Courier Age", min_value=18, max_value=65, value=30)
    rating = st.slider("Courier Rating", 1.0, 5.0, 4.5, 0.1)
    multiple_deliveries = st.selectbox("Active Deliveries", [0, 1, 2, 3])
    order_hour = st.slider("Hour of Order (24h)", 0, 23, 19)

with col2:
    st.subheader("Transit Conditions")
    distance = st.number_input("Delivery Distance (km)", 0.1, 50.0, 5.0)
    traffic = st.selectbox("Traffic Density", ['Low', 'Medium', 'High', 'Jam'])
    weather = st.selectbox("Weather", ['Cloudy', 'Fog', 'Sandstorms', 'Stormy', 'Sunny', 'Windy'])
    city = st.selectbox("City Zone", ['Metropolitian', 'Urban', 'Semi-Urban'])

with col3:
    st.subheader("Order Specifications")
    vehicle = st.selectbox("Vehicle Type", ['bicycle', 'electric_scooter', 'motorcycle', 'scooter'])
    vehicle_cond = st.selectbox("Vehicle Condition Index", [0, 1, 2, 3])
    order_type = st.selectbox("Order Type", ['Buffet', 'Drinks', 'Meal', 'Snack'])
    festival = st.selectbox("Festival Day?", ['No', 'Yes'])

# --- PREDICTION LOGIC ---
if st.button("Generate Forecast"):
    # 1. Initialize a dataframe with zeros for all 21 exact columns the scaler expects
    feature_cols = ['Delivery_person_Age', 'Delivery_person_Ratings', 'distance_km', 'order_hour', 
                    'Road_traffic_density', 'Vehicle_condition', 'multiple_deliveries', 
                    'Type_of_order_Drinks', 'Type_of_order_Meal', 'Type_of_order_Snack', 
                    'Type_of_vehicle_electric_scooter', 'Type_of_vehicle_motorcycle', 'Type_of_vehicle_scooter', 
                    'Festival_Yes', 'City_Semi-Urban', 'City_Urban', 'Weatherconditions_Fog', 
                    'Weatherconditions_Sandstorms', 'Weatherconditions_Stormy', 'Weatherconditions_Sunny', 
                    'Weatherconditions_Windy']
    
    input_df = pd.DataFrame(0, index=[0], columns=feature_cols)
    
    # 2. Map direct numeric inputs
    input_df['Delivery_person_Age'] = age
    input_df['Delivery_person_Ratings'] = rating
    input_df['distance_km'] = distance
    input_df['order_hour'] = order_hour
    input_df['Vehicle_condition'] = vehicle_cond
    input_df['multiple_deliveries'] = multiple_deliveries
    
    # 3. Map Ordinal Encoded traffic
    traffic_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Jam': 3}
    input_df['Road_traffic_density'] = traffic_map[traffic]
    
    # 4. Map One-Hot Encoded categorical variables
    if order_type != 'Buffet': input_df[f'Type_of_order_{order_type}'] = 1
    if vehicle != 'bicycle': input_df[f'Type_of_vehicle_{vehicle}'] = 1
    if festival == 'Yes': input_df['Festival_Yes'] = 1
    if city != 'Metropolitian': input_df[f'City_{city}'] = 1
    if weather != 'Cloudy': input_df[f'Weatherconditions_{weather}'] = 1
    
    # 5. Scale and Predict
    scaled_features = scaler.transform(input_df)
    prediction = model.predict(scaled_features)[0]
    
    # 6. Output UI
    st.markdown(f"""
        <div class="result-card">
            <p style="color: #64748b; font-weight: 600; text-transform: uppercase;">Estimated Time of Arrival</p>
            <div class="result-value">{int(round(prediction))} <span style="font-size: 1.5rem; color: #64748b;">minutes</span></div>
        </div>
    """, unsafe_allow_html=True)
