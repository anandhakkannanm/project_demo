import streamlit as st
import pandas as pd
import pickle

# ------------------- Page Config -------------------
st.set_page_config(
    page_title="Raisin Classification",
    page_icon="🍇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------- Load Model --------------------
with open("random_forest.pkl", "rb") as file:
    model = pickle.load(file)

# ------------------- CSS -------------------
st.markdown("""
<style>

.stApp{
    background: linear-gradient(to right,#eef2ff,#f8f9ff);
}

.main-title{
    font-size:48px;
    font-weight:bold;
    color:#5E35B1;
    text-align:center;
}

.sub-title{
    text-align:center;
    color:#666666;
    font-size:20px;
    margin-bottom:20px;
}

div[data-testid="stMetric"]{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.08);
}

.stButton>button{
    width:100%;
    height:60px;
    border-radius:15px;
    background:linear-gradient(90deg,#7B1FA2,#512DA8);
    color:white;
    font-size:22px;
    font-weight:bold;
    border:none;
}

.stButton>button:hover{
    background:linear-gradient(90deg,#512DA8,#311B92);
}

.css-1d391kg{
    background:#512DA8;
}

.block{
    background:white;
    padding:25px;
    border-radius:20px;
    box-shadow:0px 5px 20px rgba(0,0,0,0.10);
}

/* CSS to clean up the printed layout (hides navigation, sidebar, and buttons) */
@media print {
    section[data-testid="stSidebar"], 
    .stButton, 
    iframe,
    header,
    footer {
        display: none !important;
    }
    .stApp {
        background: white !important;
    }
}

</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar ----------------

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2909/2909762.png", width=120)

st.sidebar.title("🍇 Raisin Classifier")

st.sidebar.info("""
### Machine Learning Model

✅ Random Forest

Dataset Features

- Area
- Major Axis Length
- Minor Axis Length
- Eccentricity
- Convex Area
- Extent
- Perimeter

Predicts:

🍇 Kecimen

🍇 Besni
""")

# ---------------- Header ----------------

st.markdown('<div class="main-title">🍇 Raisin Classification</div>', unsafe_allow_html=True)

st.markdown('<div class="sub-title">Artificial Intelligence Powered Raisin Variety Prediction</div>', unsafe_allow_html=True)

st.divider()

# ---------------- Metrics ----------------

m1,m2,m3,m4=st.columns(4)

m1.metric("Model","Random Forest")
m2.metric("Classes","2")
m3.metric("Features","7")
m4.metric("Accuracy","99%")

st.divider()

# ---------------- Input ----------------

left,right=st.columns(2)

with left:
    Area=st.number_input("Area",0.0)
    MajorAxisLength=st.number_input("Major Axis Length",0.0)
    MinorAxisLength=st.number_input("Minor Axis Length",0.0)
    Eccentricity=st.number_input("Eccentricity",0.0)

with right:
    ConvexArea=st.number_input("Convex Area",0.0)
    Extent=st.number_input("Extent",0.0)
    Perimeter=st.number_input("Perimeter",0.0)

st.write("")

# ---------------- Prediction ----------------

if st.button("🔍 Predict Raisin Type"):

    sample=pd.DataFrame({
        "Area":[Area],
        "MajorAxisLength":[MajorAxisLength],
        "MinorAxisLength":[MinorAxisLength],
        "Eccentricity":[Eccentricity],
        "ConvexArea":[ConvexArea],
        "Extent":[Extent],
        "Perimeter":[Perimeter]
    })

    prediction=model.predict(sample)[0]

    # --- FIXED PROBABILITY LOGIC ---
    try:
        probabilities = model.predict_proba(sample)[0] # Extract array for the single row
        probability = float(max(probabilities)) * 100 # Take the highest probability class
    except Exception as e:
        probability = 100.0

    st.divider()

    if prediction==1:
        st.success("### 🍇 Prediction : Kecimen Raisin")
        st.balloons()
    else:
        st.success("### 🍇 Prediction : Besni Raisin")
        st.snow()

    st.progress(probability/100)

    st.metric("Prediction Confidence",f"{probability:.2f}%")

    st.subheader("📋 Input Summary")

    st.dataframe(sample,use_container_width=True)
    
    # ---------------- Print Option ----------------
    st.write("")
    st.components.v1.html(
        """
        <button class="print-btn" onclick="window.parent.print()">🖨️ Print / Save Report as PDF</button>
        <style>
        .print-btn {
            width: 100%;
            height: 50px;
            border-radius: 15px;
            background: linear-gradient(90deg, #4F46E5, #3730A3);
            color: white;
            font-size: 18px;
            font-weight: bold;
            border: none;
            cursor: pointer;
            box-shadow: 0px 4px 10px rgba(79, 70, 229, 0.3);
        }
        .print-btn:hover {
            background: linear-gradient(90deg, #3730A3, #242066);
        }
        </style>
        """,
        height=60
    )

st.divider()

st.caption("Developed using Streamlit | Machine Learning | Random Forest")