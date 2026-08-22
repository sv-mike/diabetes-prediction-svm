import pickle
import numpy as np
import pandas as pd
import streamlit as st

model = pickle.load(open('trained_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

feature_cols = ['bedrooms', 'bathrooms', 'floor_area', 'amenities_count', 'category_Duplex', 'category_Flats', 'category_Mansion', 'category_Semi-Detached', 'category_Townhouse', 'condition_Renovated', 'condition_Used', 'is_furnished_Semi-Furnished', 'is_furnished_Unfurnished', 'locality_grouped_Ablekuma', 'locality_grouped_Abokobi', 'locality_grouped_Accra Metropolitan', 'locality_grouped_Achimota', 'locality_grouped_Adenta', 'locality_grouped_Adjiriganor', 'locality_grouped_Agbogba', 'locality_grouped_Airport Residential Area', 'locality_grouped_Ashaiman Municipal', 'locality_grouped_Ashaley Botwe', 'locality_grouped_Ashomang Estate', 'locality_grouped_Atwima Nwabiagya', 'locality_grouped_Awoshie', 'locality_grouped_Awutu Senya East Municipal', 'locality_grouped_Cantonments', 'locality_grouped_Dansoman', 'locality_grouped_Dome', 'locality_grouped_Dzorwulu', 'locality_grouped_East Legon', 'locality_grouped_Ga East Municipal', 'locality_grouped_Ga South Municipal', 'locality_grouped_Ga West Municipal', 'locality_grouped_Gbawe', 'locality_grouped_Haatso', 'locality_grouped_Kaneshie', 'locality_grouped_Kasoa', 'locality_grouped_Kumasi Metropolitan', 'locality_grouped_Labadi', 'locality_grouped_Labone', 'locality_grouped_Lapaz', 'locality_grouped_Ledzokuku-Krowor', 'locality_grouped_Madina', 'locality_grouped_New-Juaben Municipal', 'locality_grouped_North Legon', 'locality_grouped_Nungua', 'locality_grouped_Ofankor', 'locality_grouped_Old Ashomang', 'locality_grouped_Osu', 'locality_grouped_Other', 'locality_grouped_Oyarifa', 'locality_grouped_Oyibi', 'locality_grouped_Pokuase', 'locality_grouped_Sowutwuom', 'locality_grouped_Spintex', 'locality_grouped_Taifa-Burkina', 'locality_grouped_Takoradi', 'locality_grouped_Tamale Municipal', 'locality_grouped_Tantra Hills', 'locality_grouped_Tema Metropolitan', 'locality_grouped_Tesano', 'locality_grouped_Teshie', 'locality_grouped_Weija', 'locality_grouped_West Legon']

st.title('Ghana House Rental Price Predictor')

bedrooms = st.number_input('Bedrooms', min_value=1, max_value=10, value=2)
bathrooms = st.number_input('Bathrooms', min_value=1, max_value=10, value=2)
floor_area = st.number_input('Floor area (sqm)', min_value=10, max_value=5000, value=100)
amenities_count = st.number_input('Number of amenities', min_value=0, max_value=20, value=5)

category = st.selectbox('Property type', ['Detached', 'Duplex', 'Flats', 'Mansion', 'Semi-Detached', 'Townhouse'])
condition = st.selectbox('Condition', ['New', 'Renovated', 'Used'])
is_furnished = st.selectbox('Furnishing', ['Furnished', 'Semi-Furnished', 'Unfurnished'])
area = st.selectbox('Area', ['Abelemkpe', 'Ablekuma', 'Abokobi', 'Accra Metropolitan', 'Achimota', 'Adenta', 'Adjiriganor', 'Agbogba', 'Airport Residential Area', 'Ashaiman Municipal', 'Ashaley Botwe', 'Ashomang Estate', 'Atwima Nwabiagya', 'Awoshie', 'Awutu Senya East Municipal', 'Cantonments', 'Dansoman', 'Dome', 'Dzorwulu', 'East Legon', 'Ga East Municipal', 'Ga South Municipal', 'Ga West Municipal', 'Gbawe', 'Haatso', 'Kaneshie', 'Kasoa', 'Kumasi Metropolitan', 'Labadi', 'Labone', 'Lapaz', 'Ledzokuku-Krowor', 'Madina', 'New-Juaben Municipal', 'North Legon', 'Nungua', 'Ofankor', 'Old Ashomang', 'Osu', 'Other', 'Oyarifa', 'Oyibi', 'Pokuase', 'Sowutwuom', 'Spintex', 'Taifa-Burkina', 'Takoradi', 'Tamale Municipal', 'Tantra Hills', 'Tema Metropolitan', 'Tesano', 'Teshie', 'Weija', 'West Legon'])  

if st.button('Predict Rent'):
    input_df = pd.DataFrame([[0]*len(feature_cols)], columns=feature_cols)

    input_df['bedrooms'] = bedrooms
    input_df['bathrooms'] = bathrooms
    input_df['floor_area'] = floor_area
    input_df['amenities_count'] = amenities_count

    for col, val in [('category_'+category, 1), ('condition_'+condition, 1),
                  ('is_furnished_'+is_furnished, 1), ('locality_grouped_'+area, 1)]:
        if col in input_df.columns:
            input_df[col] = val

    input_df[['bedrooms','bathrooms','floor_area','amenities_count']] = scaler.transform(
        input_df[['bedrooms','bathrooms','floor_area','amenities_count']]
    )

    prediction_log = model.predict(input_df)
    prediction_price = np.exp(prediction_log)

    st.write(f'Estimated rent: GHS {round(prediction_price[0]):,}')
