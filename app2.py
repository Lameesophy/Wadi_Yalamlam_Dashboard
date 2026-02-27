import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Wadi Yalamlam Groundwater Dashboard",
    page_icon="💧",
    layout="wide"
)

# --- 2. TITLE & PROJECT OVERVIEW ---
st.title("🇸🇦 Wadi Yalamlam Groundwater Potential Dashboard")
st.markdown("""
This interactive dashboard predicts groundwater potential in **Wadi Yalamlam, Saudi Arabia** 
using Machine Learning (Random Forest) and Geospatial Analysis.
""")

# --- 3. DATA LOADING (WITH FILE CHECKER) ---
@st.cache_data
def load_data():
    file_name = "wadi_yalamlam_data.csv"
    
    # Check if the file exists on the server
    if os.path.exists(file_name):
        return pd.read_csv(file_name)
    else:
        st.error(f"❌ Error: '{file_name}' not found on GitHub!")
        st.write("### 🔍 Debugging Info for the Server:")
        st.write(f"Current Directory: `{os.getcwd()}`")
        st.write("Files visible to the server:", os.listdir("."))
        st.info("💡 Tip: Make sure your CSV file is named exactly 'wadi_yalamlam_data.csv' and is in the same folder as this script.")
        return None

df = load_data()

if df is not None:
    # --- 4. MACHINE LEARNING MODEL SETUP ---
    # Preprocessing: Convert 'lithology' text into numbers (One-Hot Encoding)
    df_ml = pd.get_dummies(df, columns=['lithology'], drop_first=True)
    
    # Define Features (X) and Target (y)
    X = df_ml.drop(['point_id', 'latitude', 'longitude', 'potential_category'], axis=1)
    y = df_ml['potential_category']
    
    # Train the Random Forest Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # --- 5. SIDEBAR: INTERACTIVE PREDICTION TOOL ---
    st.sidebar.header("🔍 Predict Groundwater Potential")
    dist_input = st.sidebar.slider("Distance to Fault (meters)", 0, 5000, 500)
    lith_input = st.sidebar.selectbox("Select Lithology (Rock Type)", df['lithology'].unique())
    
    # Prepare input for prediction
    input_data = pd.DataFrame(columns=X.columns)
    input_data.loc[0] = 0 
    input_data['distance_to_fault'] = dist_input
    if f'lithology_{lith_input}' in input_data.columns:
        input_data[f'lithology_{lith_input}'] = 1
        
    # Make Prediction
    prediction = model.predict(input_data)[0]
    
    # Display Prediction with Color Coding
    pred_colors = {'High': 'green', 'Moderate': 'orange', 'Low': 'red'}
    st.sidebar.subheader("Model Prediction:")
    st.sidebar.markdown(f"### :{pred_colors.get(prediction, 'blue')}[{prediction} Potential]")

    # --- 6. MAIN CONTENT: GEOSPATIAL MAP & DATA ANALYSIS ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🌍 Interactive Geospatial Map")
        st.info("The **black dashed line** represents the major fault line. Click on points to see well details.")
        
        # Create the base map centered on the data
        m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=11)
        
        # A. DRAW THE FAULT LINE
        fault_coords = [[21.15, 39.85], [21.05, 39.80], [20.95, 39.75]]
        folium.PolyLine(fault_coords, color="black", weight=4, opacity=0.8, dash_array='10').add_to(m)

        # B. DRAW THE WELL POINTS
        colors = {'High': 'green', 'Moderate': 'orange', 'Low': 'red'}
        for _, row in df.iterrows():
            popup_content = f"""
            <b>Well ID:</b> {row['point_id']}  

            <b>Potential:</b> {row['potential_category']}  

            <b>Lithology:</b> {row['lithology']}  

            <b>Dist. to Fault:</b> {row['distance_to_fault']:.2f}m
            """
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=7,
                popup=folium.Popup(popup_content, max_width=300),
                color=colors.get(row['potential_category'], 'blue'),
                fill=True,
                fill_opacity=0.7
            ).add_to(m)
        
        # Render the map in Streamlit
        st_folium(m, width=700, height=500)

    with col2:
        st.subheader("📊 Data Insights")
        st.write("Summary of well characteristics:")
        st.write(df[['potential_category', 'lithology', 'distance_to_fault']].describe())
        
        st.markdown("**Distribution of Potential Categories:**")
        st.bar_chart(df['potential_category'].value_counts())

# --- 7. FOOTER ---
st.markdown("---")
st.caption("Built by [Your Name] | BSc Data Science Portfolio Project | Data Source: SGS / Madani & Niyazi (2023)")
