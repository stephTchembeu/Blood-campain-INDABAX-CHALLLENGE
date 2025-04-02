import streamlit as st

# Custom CSS to style the container
st.markdown("""
    <style>
        .custom-container {
            background-color: #f0f2f6; /* Light grey background */
            padding: 20px;
            border-radius: 15px; /* Rounded corners */
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2); /* Shadow effect */
            text-align: center; /* Centered text */
            width: 80%; /* Width of the container */
            margin: auto; /* Center it horizontally */
        }
    </style>
""", unsafe_allow_html=True)

# Using the styled container
st.markdown('<div class="custom-container">Hello, this is a styled container in Streamlit!</div>', unsafe_allow_html=True)
