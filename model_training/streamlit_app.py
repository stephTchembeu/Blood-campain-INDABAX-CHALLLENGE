import streamlit as st
import requests
import json

# Title of the Streamlit app
st.title("Blood Donor Eligibility Prediction")

# Custom CSS to increase the width and height of the input form
st.markdown(
    """
    <style>
    .stTextArea textarea {
        height: 500px;
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Form to input data
with st.form(key='input_form'):
    input_data = st.text_area("Input Data (in JSON format)", "")
    submit_button = st.form_submit_button(label='Submit')

# When the form is submitted
if submit_button:
    try:
        # We convert the input data to a dictionary
        input_data_dict = json.loads(input_data)
        
        # We send a POST request to our Flask API lauched previously
        response = requests.post("http://127.0.0.1:5001/predict", json=input_data_dict)
        
        # Display the response
        if response.status_code == 200:
            st.success("Request successful!")
            st.json(response.json())
        else:
            st.error(f"Request failed with status code {response.status_code}")
    except json.JSONDecodeError:
        st.error("Invalid JSON format")