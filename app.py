import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="California Housing | MLOps Deployment",
    page_icon="🏠",
    layout="wide"
)

# -------------------------------------------------
# LOAD TRAINED MODEL
# -------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("california_housing_knn_model.pkl")


model = load_model()

# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.title("🏠 California Housing Price Predictor")

st.subheader("Assignment - Week 8: Machine Learning Operations (MLOps)")

st.markdown(
    """
    **From Notebook to Production**

    This project demonstrates an end-to-end machine learning workflow where a
    trained model is **preprocessed, tuned, packaged, saved, and deployed**
    as an interactive web application.

    The application uses a trained **K-Nearest Neighbours (KNN) regression model**
    to estimate median house values using the California Housing dataset.
    """
)

st.divider()

# -------------------------------------------------
# PROJECT INFORMATION
# -------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("ML Algorithm", "KNN")

with col2:
    st.metric("Input Features", "8")

with col3:
    st.metric("Test R² Score", "0.7221")

with col4:
    st.metric("Deployment", "Streamlit")

st.divider()

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

with st.sidebar:

    st.header("📘 Project Information")

    st.write(
        """
        **Dataset:** California Housing

        **Problem:** Regression

        **Model:** K-Nearest Neighbours

        **Preprocessing:** StandardScaler + ColumnTransformer

        **Tuning:** GridSearchCV

        **Cross-validation:** 5-Fold

        **Evaluation:** R² Score

        **Model storage:** Joblib

        **Deployment:** Streamlit
        """
    )

    st.divider()

    st.caption(
        "Machine Learning Operations (MLOps) practical project"
    )

# -------------------------------------------------
# PREDICTION SECTION
# -------------------------------------------------

st.header("🔮 House Value Prediction")

st.write(
    "Enter the housing characteristics below and click "
    "**Predict House Value** to generate an estimate."
)

# Two-column layout
left_col, right_col = st.columns(2)

with left_col:

    st.subheader("🏘️ Housing Information")

    med_inc = st.number_input(
        "Median Income",
        min_value=0.0,
        value=3.5,
        step=0.1,
        help="Median income of households in the block group."
    )

    house_age = st.number_input(
        "Median House Age",
        min_value=0.0,
        value=30.0,
        step=1.0,
        help="Median age of houses in the area."
    )

    ave_rooms = st.number_input(
        "Average Rooms per Household",
        min_value=0.0,
        value=5.0,
        step=0.1
    )

    ave_bedrms = st.number_input(
        "Average Bedrooms per Household",
        min_value=0.0,
        value=1.0,
        step=0.1
    )


with right_col:

    st.subheader("📍 Area Information")

    population = st.number_input(
        "Population",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )

    ave_occup = st.number_input(
        "Average Household Members",
        min_value=0.0,
        value=3.0,
        step=0.1
    )

    latitude = st.number_input(
        "Latitude",
        value=34.0,
        step=0.01
    )

    longitude = st.number_input(
        "Longitude",
        value=-118.0,
        step=0.01
    )

st.divider()

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

if st.button(
    "🏠 Predict House Value",
    type="primary",
    use_container_width=True
):

    input_data = pd.DataFrame({
        "MedInc": [med_inc],
        "HouseAge": [house_age],
        "AveRooms": [ave_rooms],
        "AveBedrms": [ave_bedrms],
        "Population": [population],
        "AveOccup": [ave_occup],
        "Latitude": [latitude],
        "Longitude": [longitude]
    })

    try:

        prediction = model.predict(input_data)[0]

        # Dataset target is measured in $100,000 units
        estimated_price = prediction * 100000

        st.success("Prediction completed successfully.")

        st.subheader("📊 Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                label="Estimated Median House Value",
                value=f"${estimated_price:,.2f}"
            )

        with result_col2:

            st.metric(
                label="Model Prediction",
                value=f"{prediction:.4f}"
            )

        with st.expander("View Input Data"):
            st.dataframe(
                input_data,
                use_container_width=True
            )

        st.info(
            "This prediction is an estimate produced by the trained "
            "machine learning model and should be used for demonstration "
            "purposes only."
        )

    except Exception as error:

        st.error(
            "The model could not generate a prediction. "
            "Please check the input values."
        )

        st.exception(error)

# -------------------------------------------------
# MLOPS WORKFLOW
# -------------------------------------------------

st.divider()

st.header("⚙️ MLOps Workflow")

st.markdown(
    """
    **1. Data Preparation**  
    California Housing data was loaded and prepared for modelling.

    **2. Preprocessing**  
    Numerical features were standardized using `StandardScaler`
    through a `ColumnTransformer`.

    **3. Model Training**  
    A `KNeighborsRegressor` was used for house-value prediction.

    **4. Hyperparameter Tuning**  
    `GridSearchCV` and 5-fold cross-validation were used to identify
    the best model configuration.

    **5. Model Evaluation**  
    The final model was evaluated using the R² score.

    **6. Model Packaging**  
    The complete preprocessing and prediction pipeline was saved
    using Joblib.

    **7. Deployment**  
    The saved model was deployed as this interactive Streamlit
    web application.
    """
)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.divider()

st.caption(
    "Assignment - Week 8 | Machine Learning Operations (MLOps) | "
    "California Housing Price Prediction"
)
