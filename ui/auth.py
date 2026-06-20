"""
Streamlit Authenticator integration for the Streamlit dashboard.
Provides secure authentication utilizing bcrypt.
"""
import os
import yaml
from yaml.loader import SafeLoader
import streamlit as st
import streamlit_authenticator as stauth

_DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD")

def check_auth() -> bool:
    """
    Returns True if the user is authenticated.
    Call at the top of app.py to protect the page.
    """
    if _DASHBOARD_PASSWORD is None:
        return True
    try:
        with open('config.yaml', 'r', encoding='utf-8') as file:
            config = yaml.load(file, Loader=SafeLoader)
    except Exception as e:
        st.error(f"Failed to load Auth config from config.yaml: {e}")
        return False

    if 'credentials' not in config:
        st.error("Auth configuration missing in config.yaml.")
        return False

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )

    name, authentication_status, username = authenticator.login('Login to LLM Detector', 'main')

    if authentication_status:
        st.sidebar.markdown(f"**Account**: {name}")
        authenticator.logout('Logout', 'sidebar')
        return True
    elif authentication_status == False:
        st.error('Username or password is incorrect. (Use admin / admin_password)')
        return False
    elif authentication_status == None:
        st.info('Please enter your username and password. Hint: admin / admin_password')
        return False

    return False
