import streamlit as st
import pandas as pd
import joblib

# Load your dataset
df = pd.read_csv('fantasy_scores_clean.csv')

# Streamlit UI components
st.title("Fantasy Score Predictor")

# File uploader widget for the model
#uploaded_model = st.file_uploader("Choose a .pkl model file", type="pkl")
import streamlit as st
import gdown
import pandas as pd
from docx import Document
import pickle



# Function to download and display file
file_url = 'https://drive.google.com/uc?id=1xPzABbOoQTFFo7mz41zs2GdqOIoWawKr'


# Download the file using gdown
output = 'model_clean.pkl'  # Change the file name and extension as needed
gdown.download(file_url, output, quiet=False)

with open(output, 'rb') as file:
        data = pickle.load(file)
st.write(data)


# Input for Google Drive file URL



#https://drive.google.com/file/d/1xPzABbOoQTFFo7mz41zs2GdqOIoWawKr/view?usp=drive_link


# Load the uploaded model
model = joblib.load(data)
#st.success("Model loaded successfully!")

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

