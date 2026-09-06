import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Churn Retention Assistant",
    page_icon="🤖",
    layout="wide"
)

# ============================================================
# LOAD DATA AND MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("models/churn_model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("data/telco.csv")


model = load_model()
df = load_data()

# ============================================================
# TITLE
# ============================================================

st.title("🤖 AI Churn Retention Assistant")
st.write(
    "Machine Learning platform for customer churn prediction, "
    "analytics and retention recommendations."
)

st.divider()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "📊 Dashboard",
        "🔮 Churn Prediction",
        "💡 Retention Recommendations",
        "📈 Model Performance"
    ]
)

# ============================================================
# DASHBOARD
# ============================================================

if page == "📊 Dashboard":

    st.header("📊 Customer Analytics Dashboard")

    total_customers = len(df)
    churned_customers = (df["Churn"] == "Yes").sum()
    retained_customers = (df["Churn"] == "No").sum()
    churn_rate = churned_customers / total_customers

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Churned Customers",
        f"{churned_customers:,}"
    )

    col3.metric(
        "Retained Customers",
        f"{retained_customers:,}"
    )

    col4.metric(
        "Churn Rate",
        f"{churn_rate:.1%}"
    )

    st.divider()

    # Churn distribution
    st.subheader("Customer Churn Distribution")

    churn_data = df["Churn"].value_counts()

    st.bar_chart(churn_data)

    # Contract analysis
    st.subheader("Churn by Contract Type")

    contract_data = pd.crosstab(
        df["Contract"],
        df["Churn"]
    )

    st.bar_chart(contract_data)

    # Internet service
    st.subheader("Churn by Internet Service")

    internet_data = pd.crosstab(
        df["InternetService"],
        df["Churn"]
    )

    st.bar_chart(internet_data)

    # Tenure
    st.subheader("Average Monthly Charges by Tenure")

    tenure_data = (
        df.groupby("tenure")["MonthlyCharges"]
        .mean()
        .reset_index()
    )

    tenure_data = tenure_data.set_index("tenure")

    st.line_chart(tenure_data)

# ============================================================
# CHURN PREDICTION
# ============================================================

elif page == "🔮 Churn Prediction":

    st.header("🔮 Predict Customer Churn")

    st.write(
        "Enter customer information and the AI model "
        "will estimate the probability of churn."
    )

    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12
        )

        phone = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )

        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

    with col2:

        device = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            value=70.0
        )

        total = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            value=800.0
        )

    st.divider()

    if st.button(
        "🔮 Predict Churn",
        type="primary"
    ):

        customer = pd.DataFrame([{

            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": device,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total

        }])

        prediction = model.predict(customer)[0]

        probability = model.predict_proba(
            customer
        )[0][1]

        st.divider()

        if prediction == 1:

            st.error("⚠️ HIGH CHURN RISK")

        else:

            st.success("✅ LOW CHURN RISK")

        col1, col2 = st.columns(2)

        col1.metric(
            "Churn Probability",
            f"{probability:.1%}"
        )

        col2.metric(
            "Retention Probability",
            f"{1 - probability:.1%}"
        )

        st.progress(float(probability))

# ============================================================
# RETENTION RECOMMENDATIONS
# ============================================================

elif page == "💡 Retention Recommendations":

    st.header("💡 Retention Recommendation Engine")

    st.write(
        "This section provides business actions that can "
        "reduce customer churn."
    )

    st.subheader("Recommended Strategies")

    recommendations = [

        (
            "🎁 Loyalty Discounts",
            "Offer a personalized discount to high-risk customers."
        ),

        (
            "📄 Contract Upgrade",
            "Encourage month-to-month customers to move "
            "to a one-year or two-year contract."
        ),

        (
            "💰 Lower-Cost Plans",
            "Offer a more affordable plan to customers "
            "with high monthly charges."
        ),

        (
            "🛡️ Security Packages",
            "Offer online security and device protection "
            "packages."
        ),

        (
            "🔧 Technical Support",
            "Provide free or discounted technical support "
            "to customers who do not currently use it."
        ),

        (
            "❤️ Customer Engagement",
            "Contact high-risk customers before they decide "
            "to leave."
        )

    ]

    for title, description in recommendations:

        with st.container():

            st.subheader(title)

            st.write(description)

            st.divider()

    st.info(
        "💡 Use the Churn Prediction page to identify "
        "high-risk customers first."
    )

# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📈 Model Performance":

    st.header("📈 Machine Learning Model Performance")

    # Prepare evaluation data

    evaluation_df = df.copy()

    evaluation_df = evaluation_df.drop(
        "customerID",
        axis=1
    )

    evaluation_df["TotalCharges"] = pd.to_numeric(
        evaluation_df["TotalCharges"],
        errors="coerce"
    )

    evaluation_df = evaluation_df.dropna()

    X = evaluation_df.drop(
        "Churn",
        axis=1
    )

    y = evaluation_df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    predictions = model.predict(X)

    accuracy = accuracy_score(
        y,
        predictions
    )

    st.metric(
        "Model Accuracy",
        f"{accuracy:.2%}"
    )

    st.divider()

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y,
        predictions
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Actual No", "Actual Yes"],
        columns=["Predicted No", "Predicted Yes"]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

    st.subheader("Classification Report")

    report = classification_report(
        y,
        predictions,
        target_names=[
            "No Churn",
            "Churn"
        ],
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df,
        use_container_width=True
    )

    st.success(
        "Random Forest model successfully loaded "
        "from models/churn_model.pkl"
    )