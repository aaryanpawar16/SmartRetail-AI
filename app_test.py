# app_test.py
import streamlit as st
st.title("Dependency test")

try:
    import pandas as pd
    st.success(f"pandas import OK — version {pd.__version__}")
except Exception as e:
    st.error("pandas import FAILED")
    st.text(str(e))
