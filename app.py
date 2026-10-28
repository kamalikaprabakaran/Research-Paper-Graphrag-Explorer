import streamlit as st
from config.settings import Settings

st.set_page_config(page_title="Research Paper Connection Explorer", layout="centered")

st.title("Research Paper Connection Explorer")
st.markdown(
    """
    **Phase 0 – Project Foundation**

    This prototype demonstrates the intended pipeline:
    1. PDF ingestion (future)
    2. Vector retrieval (future)
    3. Knowledge graph construction (future)
    4. Hybrid GraphRAG (future)
    5. Evidence‑grounded answer generation (future)

    The application currently shows only this static overview and confirms that the environment loads without requiring any external credentials.
    """
)

# Load settings – this will succeed even if optional credentials are missing.
settings = Settings()
st.success("Configuration loaded successfully.")
