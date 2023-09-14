import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Set background color to look swee
st.set_page_config(
    page_title="Vivalamuffins' HDB Resale Price Prediction App",
    page_icon="🏠",
    layout="wide",  # Optional
    initial_sidebar_state="expanded"
)

custom_bg_color = """
    <style>
        body {
            background-color: #40E0D0;
        }
        .streamlit-app .streamlit-layout .element-container {
            background-color: #40E0D0;
        }
        .block-container {
            background-color: #40E0D0;
        }
        .stApp {
            background-color: #40E0D0;
        }
    </style>
"""

st.markdown(custom_bg_color, unsafe_allow_html=True)


# Load the trained model
model = joblib.load('best_model.pkl')

# Initialize the LabelEncoder for 'storey_range' and 'town'
label_encoder_storey = LabelEncoder()
known_storey_range_classes = [i for i in range(51)]  # Replace with the actual labels
label_encoder_storey.fit(known_storey_range_classes)

label_encoder_town = LabelEncoder()
known_town_classes = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL', 'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN', 'NORTHSHORE']  
label_encoder_town.fit(known_town_classes)

mature_estates = ["ANG MO KIO", "BEDOK", "BISHAN", "BUKIT MERAH", "BUKIT TIMAH", "CENTRAL AREA", "CLEMENTI", "GEYLANG", "KALLANG/WHAMPOA", "MARINE PARADE"]

# Streamlit app
st.title('HDB Resale Price Prediction App')

# Collect user input
Tranc_Month = st.slider('Transaction Month:', 1, 12, 5)
Tranc_Year = st.slider('Transaction Year:', 1990, 2023, 2021)
storey_range = st.slider('Storey Range:', 1, 51, 25)
town = st.selectbox('Town:', known_town_classes)

if town in mature_estates:
    st.write(f"You've selected {town}, which is a mature estate.")
else:
    st.write(f"You've selected {town}, which is not a mature estate.")

# Label encode the 'storey_range' and 'town'
encoded_storey_range = label_encoder_storey.transform([storey_range])[0]	
encoded_town = label_encoder_town.transform([town])[0]

# User input dictionary
input_data = {
    'Tranc_Month': Tranc_Month,
    'Tranc_Year': Tranc_Year,
    'storey_range': encoded_storey_range,
    'town_encoded': encoded_town,
    'floor_area_sqm': st.slider('Floor Area (in sqm):', 0, 200, 100),
    'mrt_nearest_distance': st.slider('Nearest MRT Distance:', 0, 2000, 500),
    'Hawker_Within_1km': st.slider('Hawker Centers Within 1km:', 0, 10, 2),
    'Mall_Nearest_Distance': st.slider('Mall Nearest Distance:', 0, 2000, 300),
    'Mall_Within_1km': st.slider('Mall Within 1km:', 0, 10, 2),
    'max_floor_lvl': st.slider('Floor Level:', 1, 50, 12),
    'unit_per_floor': st.slider('Units Per Floor:', 1, 20, 8),
    'multistorey_carpark': st.checkbox('Multi-storey Carpark'),
    'precinct_pavilion': st.checkbox('Precinct Pavilion'),
    'commercial': st.checkbox('Commercial'),
    'total_dwelling_units': st.slider('Total Dwelling Units:', 0, 200, 100),
    'hdb_age': st.slider('HDB Age:', 0, 100, 20),
    'mid_storey': st.slider('Mid Storey:', 0, 50, 3),
    'vacancy': st.slider('Vacancy:', 0, 100, 0),
    'avg_distance_to_key_locations': st.slider('Avg. Distance to Key Locations:', 0, 2000, 400),
    'flat_model': st.selectbox('Enter HDB Flat Model:', ['Model A', 'Model A2', 'Improved', 'Apartment', 'Simplified', 'New Generation', 'Premium Apartment', 'Standard', 'DBSS', 'Terrace', 'Premium Apartment Loft', 'Adjoined flat', 'Multi Generation', 'Type S1', 'Type S2', 'Maisonette', 'Model A-Maisonette', 'Premium Maisonette', 'Improved-Maisonette', '2-room', 'Others']),
    'total_schools_within_1km': st.slider('Total Schools Within 1km:', 0, 10, 2),
    'hawker_market_stalls': st.slider('Hawker Market Stalls:', 0, 100, 20),
    'unique_model': 0,  # Default value
    'central': 0,  # Default value
    'east': 0,  # Default value
    'north': 0,  # Default value
    'north_east': 0,  # Default value
    'west': 0  # Default value
    
}

# Prepare the input data
input_df = pd.DataFrame([input_data])

# Make a prediction
predicted_price = model.predict(input_df)[0]

# Display the prediction
st.write(f'Predicted HDB Resale Price: SGD {predicted_price:,.2f}')
