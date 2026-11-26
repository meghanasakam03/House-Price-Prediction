
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="HomeValue | Property Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "house_price_pipeline.pkl"
    return joblib.load(model_path)

model = load_model()

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #F1F5F9;
    color: #173B50;
}

.block-container {
    max-width: 1200px;
    padding-top: 2.5rem;
    padding-bottom: 2rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #102438, #153D50);
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

.brand {
    font-size: 32px;
    font-weight: 800;
    margin-top: 20px;
}

.brand span {
    color: #5EEAD4 !important;
}

.sidebar-caption {
    color: #A8C2D2 !important;
    font-size: 12px;
    line-height: 1.8;
    letter-spacing: 1px;
}

/* DASHBOARD HEADER */

.main-header {
    position: relative;
    overflow: hidden;
    background: linear-gradient(120deg, #102C43, #155E75);
    padding: 38px;
    border-radius: 22px;
    margin-bottom: 28px;
    box-shadow: 0 12px 30px rgba(15, 45, 65, 0.15);
}

.main-header::after {
    content: "";
    position: absolute;
    width: 220px;
    height: 220px;
    border-radius: 50%;
    background: rgba(94, 234, 212, 0.09);
    right: -40px;
    top: -100px;
}

.header-tag {
    color: #83DED5;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 3px;
    margin-bottom: 20px;
}

.main-header h1 {
    color: #FFFFFF;
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 15px;
}

.main-header p {
    color: #C2DCE7;
    font-size: 15px;
    margin-bottom: 0;
}

/* INFORMATION CARDS */

.info-card {
    background: #FFFFFF;
    border: 1px solid #DFE8EF;
    border-radius: 16px;
    padding: 22px;
    min-height: 110px;
    margin-bottom: 20px;
    box-shadow: 0 4px 12px rgba(15, 45, 65, 0.04);
}

.info-label {
    color: #64748B;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.info-value {
    color: #16465D;
    font-size: 22px;
    font-weight: 800;
    margin-top: 8px;
}

/* FORM */

div[data-testid="stForm"] {
    background: #FFFFFF;
    border: 1px solid #DCE5EB;
    border-radius: 20px;
    padding: 26px;
    box-shadow: 0 5px 18px rgba(15, 45, 65, 0.05);
}

[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] label {
    color: #173B50;
}

[data-testid="stMain"] [data-testid="stCaptionContainer"] {
    color: #64748B;
}

div[data-testid="stNumberInput"] input {
    background-color: #F8FAFC;
    color: #173B50;
}

div[data-baseweb="select"] > div {
    background-color: #F8FAFC;
    color: #173B50;
}

/* BUTTON */

div.stFormSubmitButton > button {
    background: linear-gradient(90deg, #0F766E, #155E75);
    color: white;
    border: none;
    border-radius: 12px;
    min-height: 52px;
    font-size: 16px;
    font-weight: 700;
    width: 100%;
}

div.stFormSubmitButton > button:hover {
    background: #115E59;
    color: white;
    border: none;
}

/* PREDICTION RESULT */

.result-card {
    background: linear-gradient(120deg, #ECFDF5, #E0F2F1);
    border: 1px solid #A7D8C9;
    border-radius: 20px;
    padding: 32px;
    margin-top: 25px;
    text-align: center;
}

.result-label {
    color: #16634A;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
}

.result-price {
    color: #065F46;
    font-size: 42px;
    font-weight: 800;
    margin: 12px 0;
}

.result-caption {
    color: #52796F;
    font-size: 14px;
}

/* FOOTER */

.footer {
    text-align: center;
    color: #64748B;
    font-size: 12px;
    letter-spacing: 1px;
    padding: 35px 0 10px 0;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown(
        '<div class="brand">Home<span>Value.</span></div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="sidebar-caption">
        HOUSE PRICE INTELLIGENCE<br>
        PLATFORM
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### Navigation")

    page = st.radio(
        "Navigation",
        [
            "🏠 House Price Prediction",
            "📊 Model Overview"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### Model Information")

    st.markdown("""
    <div class="sidebar-caption">
        Algorithm: Linear Regression<br><br>
        Task: Regression<br><br>
        Input Features: 8<br><br>
        Framework: Scikit-learn<br><br>
        Version: 1.0
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div class="sidebar-caption">
        Powered by Python, Scikit-learn
        and Streamlit.
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# HOUSE PRICE PREDICTION PAGE
# --------------------------------------------------

if page == "🏠 House Price Prediction":

    st.markdown("""
    <div class="main-header">
        <div class="header-tag">
            HOMEVALUE / PROPERTY INSIGHTS
        </div>
        <h1>House Price Prediction Dashboard</h1>
        <p>
            Estimate a property's selling price using
            its features, condition and location.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # INFORMATION CARDS

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="info-card">
            <div class="info-label">ML Algorithm</div>
            <div class="info-value">Linear Regression</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <div class="info-label">Prediction Type</div>
            <div class="info-value">Regression</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="info-card">
            <div class="info-label">Input Features</div>
            <div class="info-value">8 Features</div>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Enter Property Information")

    st.caption(
        "Enter the property's details below and "
        "click Predict House Price."
    )

    # --------------------------------------------------
    # PROPERTY INPUT FORM
    # --------------------------------------------------

    with st.form("property_form"):

        st.markdown("### 🏡 Property Details")

        c1, c2, c3 = st.columns(3)

        with c1:

            area = st.number_input(
                "Property Area",
                min_value=501,
                max_value=4999,
                value=2500
            )

            bedrooms = st.slider(
                "Bedrooms", 1, 5, 3
            )

        with c2:

            bathrooms = st.slider(
                "Bathrooms", 1, 4, 2
            )

            floors = st.slider(
                "Floors", 1, 3, 1
            )

        with c3:

            year_built = st.number_input(
                "Year Built",
                min_value=1900,
                max_value=2023,
                value=2000
            )

            garage = st.radio(
                "Garage Availability",
                ["No", "Yes"],
                horizontal=True
            )

        st.divider()

        st.markdown("### 📍 Location & Condition")

        c1, c2 = st.columns(2)

        with c1:

            location = st.selectbox(
                "Property Location",
                ["Downtown", "Suburban", "Urban", "Rural"]
            )

        with c2:

            condition = st.selectbox(
                "Property Condition",
                ["Excellent", "Good", "Fair", "Poor"]
            )

        st.write("")

        submitted = st.form_submit_button(
            "🔍 Predict House Price",
            use_container_width=True
        )

    # --------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------

    if submitted:

        input_data = pd.DataFrame({
            "Area": [area],
            "Bedrooms": [bedrooms],
            "Bathrooms": [bathrooms],
            "Floors": [floors],
            "YearBuilt": [year_built],
            "Location": [location],
            "Condition": [condition],
            "Garage": [garage]
        })

        prediction = model.predict(input_data)[0]

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">
                ESTIMATED PROPERTY PRICE
            </div>
            <div class="result-price">
                {prediction:,.2f}
            </div>
            <div class="result-caption">
                Predicted using the trained
                Machine Learning model.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        st.subheader("Property Summary")

        p1, p2, p3 = st.columns(3)

        p1.metric("Property Area", f"{area:,}")
        p2.metric("Bedrooms", bedrooms)
        p3.metric("Bathrooms", bathrooms)

        with st.expander("View Complete Property Details"):
            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )

        st.info(
            "This prediction is a model-generated estimate "
            "based on historical data and is not a guaranteed "
            "market valuation."
        )

# --------------------------------------------------
# MODEL OVERVIEW PAGE
# --------------------------------------------------

else:

    st.markdown("""
    <div class="main-header">
        <div class="header-tag">
            HOMEVALUE / MACHINE LEARNING
        </div>
        <h1>Model Overview</h1>
        <p>
            Explore the Machine Learning methodology
            behind property price prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🎯 Problem Statement")

    st.write(
        "Predict house prices using numerical and "
        "categorical property features through an "
        "end-to-end Machine Learning pipeline."
    )

    st.divider()

    st.subheader("📊 Dataset Features")

    feature_data = pd.DataFrame({
        "Feature": [
            "Area",
            "Bedrooms",
            "Bathrooms",
            "Floors",
            "YearBuilt",
            "Location",
            "Condition",
            "Garage"
        ],
        "Data Type": [
            "Numerical",
            "Numerical",
            "Numerical",
            "Numerical",
            "Numerical",
            "Categorical",
            "Categorical",
            "Categorical"
        ]
    })

    st.dataframe(
        feature_data,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("⚙️ Data Preprocessing")

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("#### Numerical Pipeline")

        st.markdown("""
        - SimpleImputer
        - StandardScaler
        """)

    with c2:

        st.markdown("#### Categorical Pipeline")

        st.markdown("""
        - SimpleImputer
        - OneHotEncoder
        """)

    st.write(
        "ColumnTransformer combines numerical and "
        "categorical preprocessing into a unified pipeline."
    )

    st.divider()

    st.subheader("🤖 Machine Learning Algorithm")

    st.markdown("**Linear Regression**")

    st.write(
        "Linear Regression estimates the relationship "
        "between property features and house prices "
        "to predict a continuous numerical value."
    )

    st.divider()

    st.subheader("🔄 Project Workflow")

    st.code("""
Dataset
   ↓
Data Preprocessing
   ↓
ColumnTransformer
   ↓
Linear Regression
   ↓
Trained Pipeline (.pkl)
   ↓
Streamlit Application
   ↓
GitHub & Cloud Deployment
    """, language="text")

    st.subheader("🛠️ Technologies Used")

    st.markdown("""
    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Joblib
    - Streamlit
    - GitHub
    """)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
    HOMEVALUE | PROPERTY INTELLIGENCE PLATFORM<br><br>
    Machine Learning Demonstration Project
</div>
""", unsafe_allow_html=True)
