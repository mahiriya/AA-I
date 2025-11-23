import streamlit as st
import pandas as pd
import joblib
import json
import os
from datetime import datetime

st.set_page_config(page_title='House Price Prediction', layout='centered')
st.title('🏠 House Price Prediction')

MODEL_FILE = 'model.pkl'
FEATURE_FILE = 'feature_columns.json'
KMEANS_MODEL_FILE = 'kmeans.pkl'
KMEANS_SCALER_FILE = 'kmeans_scaler.pkl'

# Load model
model = joblib.load(MODEL_FILE) if os.path.exists(MODEL_FILE) else None
if model:
    st.success('✅ Prediction model loaded successfully')
else:
    st.error('❌ Prediction model file not found. Please train and save the model as model.pkl')

# Load feature list
feature_cols = json.load(open(FEATURE_FILE)) if os.path.exists(FEATURE_FILE) else []
if not feature_cols:
    st.error('❌ Feature list file not found. Please create feature_columns.json')

# Load KMeans model and scaler
kmeans_model = joblib.load(KMEANS_MODEL_FILE) if os.path.exists(KMEANS_MODEL_FILE) else None
kmeans_scaler = joblib.load(KMEANS_SCALER_FILE) if os.path.exists(KMEANS_SCALER_FILE) else None

if kmeans_model and kmeans_scaler:
    st.success('✅ KMeans model and scaler loaded successfully for clustering')
else:
    st.warning('⚠️ KMeans model or scaler not found. Clustering feature might be inaccurate or unavailable.')
    st.info('To ensure accurate clustering, please ensure kmeans.pkl and kmeans_scaler.pkl are saved.')


st.header('Enter Property Details')

current_year = datetime.now().year

col1, col2 = st.columns(2)

with col1:
    size_sqft = st.number_input('Size in SqFt', min_value=100, max_value=10000, value=1500, help="Area of the property in square feet.")
    price_per_sqft = st.number_input('Price per SqFt', min_value=0.01, max_value=10.0, value=0.1, format="%.2f", help="Price per square foot. (Note: This is used as a feature in the model, and should ideally be derived from total price.)")
    year_built = st.number_input('Year Built', min_value=1900, max_value=current_year, value=2000, help="Year the property was constructed.")
    floor_no = st.number_input('Floor Number', min_value=0, max_value=50, value=5, help="The floor on which the property is located.")
    total_floors = st.number_input('Total Floors in Building', min_value=1, max_value=50, value=10, help="Total number of floors in the building.")
    nearby_schools = st.number_input('Number of Nearby Schools', min_value=0, max_value=15, value=5, help="Count of schools in close proximity.")
    nearby_hospitals = st.number_input('Number of Nearby Hospitals', min_value=0, max_value=15, value=3, help="Count of hospitals in close proximity.")
    parking_space_input = st.selectbox('Parking Space', ['Yes', 'No'], help="Is parking space available?")
    security_input = st.selectbox('Security Available', ['Yes', 'No'], help="Is security available in the property?")
    availability_status_input = st.selectbox('Availability Status', ['Ready_to_Move', 'Under_Construction'], help="Current availability status of the property.")

with col2:
    property_type_input = st.selectbox('Property Type', ['Apartment', 'Independent House', 'Villa'], help="Type of property.")
    public_transport_input = st.selectbox('Public Transport Accessibility', ['High', 'Medium', 'Low'], help="Level of access to public transport.")
    furnished_status_input = st.selectbox('Furnished Status', ['Furnished', 'Semi-furnished', 'Unfurnished'], help="Current furnishing status of the property.")
    facing_input = st.selectbox('Facing', ['North', 'South', 'East', 'West'], help="Direction the property faces.")
    owner_type_input = st.selectbox('Owner Type', ['Owner', 'Builder', 'Broker'], help="Type of owner selling the property.")
    bhk_input = st.selectbox('BHK', [1, 2, 3, 4, 5], help="Number of bedrooms, hall, and kitchen.")
    
    amenities_options = ['Clubhouse', 'Garden', 'Gym', 'Playground', 'Pool']
    selected_amenities = st.multiselect('Amenities', amenities_options, help="Select available amenities.")

    state_options = ['Odisha', 'Tamil Nadu', 'West Bengal', 'Gujarat', 'Delhi', 'Telangana', 'Maharashtra', 'Punjab', 'Uttar Pradesh', 'Uttarakhand', 'Assam', 'Kerala', 'Jharkhand', 'Andhra Pradesh', 'Chhattisgarh', 'Madhya Pradesh', 'Karnataka', 'Rajasthan', 'Bihar', 'Haryana']
    state_input = st.selectbox('State', state_options, help="State where the property is located.")


if st.button('🔍 Predict House Price'):
    if model is None or not feature_cols:
        st.error('Model or feature list not loaded. Cannot make prediction.')
    else:
        # Create a dictionary to hold all feature values, initialized to 0 for one-hot encoded features
        input_data_dict = {col_name: 0 for col_name in feature_cols}

        # Fill in numerical inputs
        input_data_dict['Size_in_SqFt'] = size_sqft
        input_data_dict['Price_per_SqFt'] = price_per_sqft
        input_data_dict['Year_Built'] = year_built
        input_data_dict['Floor_No'] = floor_no
        input_data_dict['Total_Floors'] = total_floors
        input_data_dict['Nearby_Schools'] = nearby_schools
        input_data_dict['Nearby_Hospitals'] = nearby_hospitals

        # Derived features
        age_of_property = current_year - year_built
        input_data_dict['Age_of_Property'] = age_of_property

        # Binary inputs
        input_data_dict['Parking_Space'] = 1 if parking_space_input == 'Yes' else 0
        input_data_dict['Security'] = 1 if security_input == 'Yes' else 0
        input_data_dict['Availability_Status'] = 1 if availability_status_input == 'Ready_to_Move' else 0

        # One-hot encoded categorical inputs
        if property_type_input == 'Independent House':
            input_data_dict['Property_Type_Independent House'] = 1
        elif property_type_input == 'Villa':
            input_data_dict['Property_Type_Villa'] = 1

        if furnished_status_input == 'Semi-furnished':
            input_data_dict['Furnished_Status_Semi-furnished'] = 1
        elif furnished_status_input == 'Unfurnished':
            input_data_dict['Furnished_Status_Unfurnished'] = 1

        if public_transport_input == 'Low':
            input_data_dict['Public_Transport_Accessibility_Low'] = 1
        elif public_transport_input == 'Medium':
            input_data_dict['Public_Transport_Accessibility_Medium'] = 1

        if facing_input == 'North':
            input_data_dict['Facing_North'] = 1
        elif facing_input == 'South':
            input_data_dict['Facing_South'] = 1
        elif facing_input == 'West':
            input_data_dict['Facing_West'] = 1

        if owner_type_input == 'Builder':
            input_data_dict['Owner_Type_Builder'] = 1
        elif owner_type_input == 'Owner':
            input_data_dict['Owner_Type_Owner'] = 1

        if bhk_input == 2:
            input_data_dict['BHK_2'] = 1
        elif bhk_input == 3:
            input_data_dict['BHK_3'] = 1
        elif bhk_input == 4:
            input_data_dict['BHK_4'] = 1
        elif bhk_input == 5:
            input_data_dict['BHK_5'] = 1

        for amenity in selected_amenities:
            if amenity in input_data_dict: 
                input_data_dict[amenity] = 1

        state_col_name = f'State_{state_input}'
        if state_col_name in input_data_dict:
            input_data_dict[state_col_name] = 1

        # --- Calculate Cluster Feature ---
        cluster_val = 0 # Default if KMeans not loaded
        if kmeans_model and kmeans_scaler:
            # Features used for KMeans clustering: ['BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 'Age_of_Property']
            # Price_in_Lakhs is the target, so we use a placeholder (0.0) here for clustering new inputs.
            # This is a known limitation due to how the clustering model was originally trained.
            kmeans_input_features = pd.DataFrame([[bhk_input, size_sqft, 0.0, age_of_property]],
                                                 columns=['BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 'Age_of_Property'])
            scaled_kmeans_input = kmeans_scaler.transform(kmeans_input_features)
            cluster_val = kmeans_model.predict(scaled_kmeans_input)[0]
        input_data_dict['Cluster'] = cluster_val

        # Create DataFrame for prediction, ensuring column order matches training data
        input_df = pd.DataFrame([input_data_dict])
        input_df = input_df[feature_cols] # Ensure column order for the model


        st.subheader('Input Preview for Prediction')
        st.dataframe(input_df)

        try:
            prediction_lakhs = model.predict(input_df)[0]
            st.success(f'### Predicted House Price: ₹ {prediction_lakhs:,.2f} Lakhs')
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.write("Please ensure all necessary inputs are provided and match the model's expectations, and that all models are loaded correctly.")
