import streamlit as st
import pandas as pd
import joblib

# Load your dataset
df = pd.read_csv('fantasy_scores_clean.csv')

# Streamlit UI components
st.title("Fantasy Score Predictor")

# File uploader widget for the model
uploaded_model = st.file_uploader("Choose a .pkl model file", type="pkl")

if uploaded_model is not None:
    # Load the uploaded model
    model = joblib.load(uploaded_model)
    st.success("Model loaded successfully!")

    # Dropdown for inputs
    venue = st.selectbox('Select Venue', df['venue'].unique())
    player_name = st.selectbox('Select Player Name', df['player_name'].unique())
    opposition_team = st.selectbox('Select Opposition Team', df['opposition_team'].unique())
    player_team_first_role = st.selectbox('Select Player Team First Role', df['player_team_first_role'].unique())
    
    # Button to predict score
    if st.button('Predict Fantasy Score'):
        # Create input data for the model
        input_data = pd.DataFrame({
            'venue': [venue],
            'player_name': [player_name],
            'opposition_team': [opposition_team],
            'player_team_first_role': [player_team_first_role]
        })

        # Encoding categorical features
        input_data_encoded = pd.get_dummies(input_data)
        input_data_encoded = input_data_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

        # Make prediction
        predicted_score = model.predict(input_data_encoded)

        # Display the predicted score
        st.write(f"Predicted Fantasy Score for {player_name}: {predicted_score[0]}")

    # Options Route
    st.sidebar.title("Options")
    st.sidebar.write("Unique Options from the Dataset:")

    st.sidebar.write("**Venues:**")
    st.sidebar.write(df['venue'].unique().tolist())

    st.sidebar.write("**Player Names:**")
    st.sidebar.write(df['player_name'].unique().tolist())

    st.sidebar.write("**Opposition Team:**")
    st.sidebar.write(df['opposition_team'].unique().tolist())

    st.sidebar.write("**Player Team First Role:**")
    st.sidebar.write(df['player_team_first_role'].unique().tolist())

else:
    st.info("Please upload a .pkl model file to proceed.")
