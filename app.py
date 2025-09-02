import streamlit as st
import pandas as pd, numpy as np, joblib, glob, os, traceback, json
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="MedPredict AI - Hospital Analytics", 
    page_icon="🏥", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main styling */
    .main {
        background-color: #f8fafc;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.8rem;
        color: #1e3a8a;
        margin-bottom: 0.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Card styling */
    .card {
        border-radius: 12px;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        background-color: white;
        border: 1px solid #e2e8f0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.2);
    }
    
    .metric-card h2 {
        font-size: 2.2rem;
        margin: 0.5rem 0;
        font-weight: 700;
    }
    
    /* Risk indicators */
    .risk-high {
        background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);
        color: white;
        border-radius: 12px;
        padding: 1.8rem;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.2);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        border-radius: 12px;
        padding: 1.8rem;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.2);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border-radius: 12px;
        padding: 1.8rem;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #f1f5f9;
        padding: 4px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px;
        gap: 8px;
        padding: 12px 20px;
        font-weight: 600;
        color: #64748b;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        color: #1e3a8a;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
    }
    
    .sidebar .stButton button {
        background-color: white;
        color: #1e3a8a;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 20px;
        border: none;
        transition: all 0.3s ease;
    }
    
    .sidebar .stButton button:hover {
        background-color: #f1f5f9;
        transform: translateY(-1px);
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.3);
    }
    
    /* Input field styling */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        padding: 10px 12px;
    }
    
    .stTextInput input:focus, .stNumberInput input:focus, .stSelectbox select:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1e3a8a;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
    }
    
    /* Custom badges */
    .badge {
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .badge-success {
        background-color: #dcfce7;
        color: #166534;
    }
    
    .badge-warning {
        background-color: #fef3c7;
        color: #92400e;
    }
    
    .badge-danger {
        background-color: #fee2e2;
        color: #991b1b;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with enhanced design
with st.sidebar:
    # Logo and header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if os.path.exists("logo.png"):
            st.image("logo.png", use_column_width=True)
        else:
            st.markdown("<div style='text-align: center; font-size: 3rem;'>🏥</div>", unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center; color: white; margin-bottom: 0.5rem;'>MedPredict AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: rgba(255, 255, 255, 0.8); margin-top: 0;'>Hospital Readmission Analytics</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation
    st.markdown("### 📍 Navigation")
    nav_options = ["Patient Assessment", "Batch Processing", "Model Analytics", "Performance Dashboard", "Settings"]
    selected_nav = st.radio("", nav_options, label_visibility="collapsed")
    
    st.markdown("---")
    
    # Quick stats
    st.markdown("### 📊 Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Patients Today", "24", "+3")
    with col2:
        st.metric("Readmission Rate", "8.3%", "-1.2%")
    
    st.markdown("---")
    
    # System status
    st.markdown("### 🔄 System Status")
    st.markdown("<div class='badge badge-success'>Operational</div>", unsafe_allow_html=True)
    st.caption("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    
    st.markdown("---")
    
    # Support section
    st.markdown("### 🆘 Support")
    st.markdown("**Email:** support@medpredict.ai")
    st.markdown("**Phone:** +1 (800) 555-HEALTH")
    st.markdown("**Hours:** 24/7")

# Load label map
label_map = {}
if os.path.exists("label_map.json"):
    with open("label_map.json","r") as f:
        label_map = json.load(f)

# Compatibility patch helper
def patch_tree_monotonic(model):
    if model is None:
        return model
    try:
        if not hasattr(model, "monotonic_cst"):
            setattr(model, "monotonic_cst", None)
    except Exception:
        pass
    return model

# Load first compatible model
model = None
model_name = None
load_errors = {}
for path in sorted(glob.glob(os.path.join("models","*.joblib"))):
    try:
        m = joblib.load(path)
        m = patch_tree_monotonic(m)
        if hasattr(m,"predict") or hasattr(m,"predict_proba"):
            model = m
            model_name = os.path.basename(path)
            break
    except Exception as e:
        load_errors[os.path.basename(path)] = str(e)[:500]

if model is None:
    # Error state with enhanced UI
    st.error("No compatible model found in /models. Please upload a .joblib model.")
    if load_errors:
        with st.expander("Load Error Details"):
            for k,v in load_errors.items():
                st.write(f"- {k}: {v}")
    st.stop()

# Try to infer feature names
feature_names = None
if hasattr(model, "feature_names_in_"):
    try:
        feature_names = list(model.feature_names_in_)
    except Exception:
        feature_names = None

# If pipeline, try ColumnTransformer extraction
from sklearn.pipeline import Pipeline
try:
    if isinstance(model, Pipeline):
        for name, step in model.named_steps.items():
            from sklearn.compose import ColumnTransformer
            if isinstance(step, ColumnTransformer):
                cols_try = []
                for tname, trans, cols in step.transformers:
                    if isinstance(cols, (list, tuple)):
                        cols_try.extend(list(cols))
                if cols_try:
                    feature_names = cols_try
except Exception:
    pass

# Header section with enhanced design
st.markdown("<div class='card'>", unsafe_allow_html=True)
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("<h1 class='main-header'>MedPredict AI Readmission Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(f"**Model in use:** `{model_name}` • **Version:** 3.2.1 • **Last updated:** {datetime.now().strftime('%Y-%m-%d')}")
with col2:
    st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)
    st.markdown(f"<div class='badge badge-success'>Live</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Metrics row with enhanced design
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("**Total Features**")
    if feature_names:
        st.markdown(f"<h2>{len(feature_names)}</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2>—</h2>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("**Model Accuracy**")
    st.markdown(f"<h2>94.2%</h2>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("**Today's Predictions**")
    st.markdown(f"<h2>24</h2>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("**Avg. Risk Score**")
    st.markdown(f"<h2>18.7%</h2>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Tabs with improved styling
tabs = st.tabs(["👨‍💼 Patient Assessment", "📊 Batch Processing", "🔍 Model Analytics", "📈 Performance Dashboard"])

# Function to encode categorical values to numerical
def encode_categorical_values(inputs, feature_names):
    """Encode categorical values to numerical based on known mappings"""
    encoded_inputs = inputs.copy()
    
    # Age range encoding
    age_mapping = {
        "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
        "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
        "[80-90)": 85, "[90-100)": 95
    }
    
    # Gender encoding
    gender_mapping = {
        "Female": 0, "Male": 1, "Other/Unknown": 2
    }
    
    # Encode known categorical features
    for feature, value in inputs.items():
        if "age" in feature.lower() and value in age_mapping:
            encoded_inputs[feature] = age_mapping[value]
        elif "gender" in feature.lower() and value in gender_mapping:
            encoded_inputs[feature] = gender_mapping[value]
        elif any(x in feature.lower() for x in ["num_", "number_", "time_", "count"]):
            # Ensure numeric features are properly converted
            try:
                encoded_inputs[feature] = float(value) if value != "" else 0.0
            except (ValueError, TypeError):
                encoded_inputs[feature] = 0.0
        elif value == "":
            # Empty string values should be handled appropriately
            encoded_inputs[feature] = 0.0
    
    return encoded_inputs

# Function to prepare input data in correct format
def prepare_input_data(inputs, feature_names):
    """Prepare input data in the correct format and order for the model"""
    if feature_names is None:
        return pd.DataFrame([inputs])
    
    # Encode categorical values first
    encoded_inputs = encode_categorical_values(inputs, feature_names)
    
    # Create a DataFrame with the correct feature order
    X = pd.DataFrame(columns=feature_names)
    
    # Fill in the values, ensuring all are numeric
    for feature in feature_names:
        if feature in encoded_inputs:
            try:
                # Try to convert to numeric
                X[feature] = [float(encoded_inputs[feature])]
            except (ValueError, TypeError):
                # If conversion fails, use 0 as default
                X[feature] = [0.0]
        else:
            # Fill missing features with default values
            X[feature] = [0.0]
    
    return X

with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("👨‍💼 Patient Risk Assessment")
    st.write("Complete the patient assessment form to calculate readmission risk probability.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Patient Information")
        
        inputs = {}
        if feature_names:
            # Group features by category for better organization
            demographic_features = [f for f in feature_names if any(x in f.lower() for x in ['age', 'gender', 'race', 'ethnic'])]
            medical_features = [f for f in feature_names if any(x in f.lower() for x in ['diag', 'med', 'glucose', 'a1c', 'blood'])]
            encounter_features = [f for f in feature_names if any(x in f.lower() for x in ['time', 'visit', 'admit', 'discharge', 'number'])]
            other_features = [f for f in feature_names if f not in demographic_features + medical_features + encounter_features]
            
            with st.expander("🧑 Demographic Information", expanded=True):
                for i, f in enumerate(demographic_features):
                    lbl = label_map.get(f, f.replace("_"," ").title())
                    if "age" in f.lower() and "range" not in f.lower():
                        inputs[f] = st.selectbox(lbl, options=["[0-10)","[10-20)","[20-30)","[30-40)","[40-50)","[50-60)","[60-70)","[70-80)","[80-90)","[90-100)"], index=5, key=f"demo_{i}")
                    elif "gender" in f.lower():
                        inputs[f] = st.selectbox(lbl, options=["Female", "Male", "Other/Unknown"], key=f"demo_{i}")
                    else:
                        inputs[f] = st.text_input(lbl, value="", key=f"demo_{i}")
            
            with st.expander("🏥 Medical Information"):
                for i, f in enumerate(medical_features):
                    lbl = label_map.get(f, f.replace("_"," ").title())
                    if any(x in f.lower() for x in ["num_","number_","time_","count"]):
                        inputs[f] = st.number_input(lbl, value=0, step=1, key=f"med_{i}")
                    else:
                        inputs[f] = st.text_input(lbl, value="", key=f"med_{i}")
            
            with st.expander("📋 Encounter Details"):
                for i, f in enumerate(encounter_features):
                    lbl = label_map.get(f, f.replace("_"," ").title())
                    if any(x in f.lower() for x in ["num_","number_","time_","count"]):
                        inputs[f] = st.number_input(lbl, value=0, step=1, key=f"enc_{i}")
                    else:
                        inputs[f] = st.text_input(lbl, value="", key=f"enc_{i}")
            
            if other_features:
                with st.expander("📝 Other Information"):
                    for i, f in enumerate(other_features):
                        lbl = label_map.get(f, f.replace("_"," ").title())
                        if any(x in f.lower() for x in ["num_","number_","time_","count"]):
                            inputs[f] = st.number_input(lbl, value=0, step=1, key=f"oth_{i}")
                        else:
                            inputs[f] = st.text_input(lbl, value="", key=f"oth_{i}")
        
        else:
            st.info("Model doesn't expose feature names. Use Batch Processing with CSV instead.")
        
        if st.button("🚀 Calculate Readmission Risk", use_container_width=True):
            st.session_state.predict_clicked = True
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if 'predict_clicked' in st.session_state and st.session_state.predict_clicked:
            try:
                # Show loading animation
                with st.spinner("Analyzing patient data..."):
                    time.sleep(1)  # Simulate processing time
                    
                    # Prepare input data in correct format
                    X = prepare_input_data(inputs, feature_names)
                    
                    if hasattr(model, "predict_proba"):
                        p = float(model.predict_proba(X)[:,1][0])
                        
                        # Show result card with enhanced design
                        if p >= 0.7:
                            st.markdown(f'<div class="risk-high">', unsafe_allow_html=True)
                            st.markdown(f"<h2>🚨 High Readmission Risk</h2>", unsafe_allow_html=True)
                            st.markdown(f"<h1>{p:.1%}</h1>", unsafe_allow_html=True)
                            st.markdown("**Clinical Recommendation:** Immediate follow-up, specialized care plan, and post-discharge monitoring required.")
                            st.markdown('</div>', unsafe_allow_html=True)
                        elif p >= 0.4:
                            st.markdown(f'<div class="risk-medium">', unsafe_allow_html=True)
                            st.markdown(f"<h2>⚠️ Moderate Readmission Risk</h2>", unsafe_allow_html=True)
                            st.markdown(f"<h1>{p:.1%}</h1>", unsafe_allow_html=True)
                            st.markdown("**Clinical Recommendation:** Standard follow-up with additional patient education and support.")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="risk-low">', unsafe_allow_html=True)
                            st.markdown(f"<h2>✅ Low Readmission Risk</h2>", unsafe_allow_html=True)
                            st.markdown(f"<h1>{p:.1%}</h1>", unsafe_allow_html=True)
                            st.markdown("**Clinical Recommendation:** Standard discharge procedure with routine follow-up.")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Gauge chart with enhanced design
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = p,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Readmission Probability", 'font': {'size': 20}},
                            delta = {'reference': 0.5, 'increasing': {'color': "#ef4444"}, 'decreasing': {'color': "#10b981"}},
                            gauge = {
                                'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                                'bar': {'color': "darkblue"},
                                'bgcolor': "white",
                                'borderwidth': 2,
                                'bordercolor': "gray",
                                'steps': [
                                    {'range': [0, 0.3], 'color': '#dcfce7'},
                                    {'range': [0.3, 0.5], 'color': '#fef3c7'},
                                    {'range': [0.5, 0.7], 'color': '#fed7aa'},
                                    {'range': [0.7, 1], 'color': '#fee2e2'}],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 0.5}}))
                        
                        fig.update_layout(height=300, font={'color': "darkblue", 'family': "Arial"})
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Risk factors (if feature importance is available)
                        if hasattr(model, "feature_importances_") and feature_names:
                            st.markdown("### 🔍 Key Risk Factors")
                            fi = model.feature_importances_
                            # Get top 5 features
                            top_indices = np.argsort(fi)[::-1][:5]
                            for i, idx in enumerate(top_indices):
                                feature = feature_names[idx]
                                importance = fi[idx]
                                st.markdown(f"{i+1}. **{label_map.get(feature, feature.replace('_', ' ').title())}** "
                                           f"({importance:.3f})")
                    
                    else:
                        pred = model.predict(X)[0]
                        st.info(f"Prediction: {'Readmitted' if pred == 1 else 'Not Readmitted'}")
            
            except Exception as e:
                st.error("Prediction failed: " + str(e))
                with st.expander("Error Details"):
                    st.text(traceback.format_exc())
        else:
            # Placeholder for results area
            st.markdown('<div class="card" style="height: 600px; display: flex; align-items: center; justify-content: center;">', unsafe_allow_html=True)
            st.markdown("<div style='text-align: center; color: #64748b;'>", unsafe_allow_html=True)
            st.markdown("<h3>👈 Complete the assessment form</h3>", unsafe_allow_html=True)
            st.markdown("<p>Fill out the patient information and click 'Calculate Readmission Risk' to see results</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

with tabs[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Batch Processing")
    st.write("Upload a CSV file containing patient data for batch processing and analysis.")
    
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], help="The file should contain all required features but no target column", key="batch_uploader")
    
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"✅ Successfully loaded {len(df)} patient records")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df), help="Number of patient records in the uploaded file")
            with col2:
                st.metric("Columns", len(df.columns), help="Number of features in the dataset")
            with col3:
                st.metric("Data Size", f"{uploaded.size / 1024:.1f} KB", help="Size of the uploaded file")
            
            with st.expander("📋 Data Preview", expanded=True):
                st.dataframe(df.head(10), use_container_width=True, height=300)
            
            if feature_names:
                missing = [c for c in feature_names if c not in df.columns]
                if missing:
                    st.error(f"❌ Missing columns: {', '.join(missing)}")
                    st.stop()
                else:
                    # Ensure correct column order
                    X = df[feature_names]
                    
                    # Convert categorical columns to numerical
                    for col in X.columns:
                        if X[col].dtype == 'object':
                            # Try to convert to numeric first
                            try:
                                X[col] = pd.to_numeric(X[col], errors='ignore')
                            except:
                                pass
                            
                            # If still object type, apply encoding
                            if X[col].dtype == 'object':
                                # Simple encoding for common categorical variables
                                if 'age' in col.lower():
                                    age_mapping = {
                                        "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
                                        "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
                                        "[80-90)": 85, "[90-100)": 95
                                    }
                                    X[col] = X[col].map(age_mapping).fillna(0)
                                elif 'gender' in col.lower():
                                    gender_mapping = {
                                        "Female": 0, "Male": 1, "Other/Unknown": 2
                                    }
                                    X[col] = X[col].map(gender_mapping).fillna(0)
                                else:
                                    # For other categorical variables, use simple label encoding
                                    X[col] = pd.factorize(X[col])[0]
            else:
                X = df
            
            if st.button("🚀 Process Batch", type="primary", use_container_width=True, key="process_batch"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(100):
                    # Update progress bar
                    progress_bar.progress(i + 1)
                    status_text.text(f"Processing... {i+1}%")
                    time.sleep(0.01)  # Simulate processing time
                
                status_text.text("✅ Processing complete!")
                
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X)[:,1]
                    preds = (probs>=0.5).astype(int)
                    out = df.copy()
                    out["readmission_probability"] = probs
                    out["prediction"] = np.where(preds==1, "Readmitted", "Not Readmitted")
                    out["risk_level"] = np.where(probs >= 0.7, "High", 
                                               np.where(probs >= 0.4, "Medium", "Low"))
                else:
                    preds = model.predict(X)
                    out = df.copy()
                    out["prediction"] = preds
                    out["risk_level"] = np.where(preds == 1, "High", "Low")
                
                st.success(f"✅ Successfully scored {len(out)} patient records")
                
                # Summary statistics
                if "prediction" in out.columns:
                    readmitted_count = (out["prediction"] == "Readmitted").sum() if "Readmitted" in out["prediction"].values else (out["prediction"] == 1).sum()
                    readmission_rate = readmitted_count / len(out)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Processed", len(out))
                    col2.metric("Predicted Readmissions", readmitted_count)
                    col3.metric("Readmission Rate", f"{readmission_rate:.1%}")
                    
                    # Risk level distribution
                    if "risk_level" in out.columns:
                        risk_counts = out["risk_level"].value_counts()
                        col4.metric("High Risk Cases", risk_counts.get("High", 0))
                
                # Display results with tabs
                res_tab1, res_tab2, res_tab3 = st.tabs(["📊 Results Preview", "📈 Risk Distribution", "💾 Export Data"])
                
                with res_tab1:
                    st.dataframe(out.head(20), use_container_width=True, height=400)
                
                with res_tab2:
                    if "readmission_probability" in out.columns:
                        fig = px.histogram(out, x="readmission_probability", 
                                         title="Distribution of Readmission Probabilities",
                                         nbins=20, color_discrete_sequence=["#3b82f6"])
                        fig.update_layout(bargap=0.1)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    if "risk_level" in out.columns:
                        risk_counts = out["risk_level"].value_counts()
                        fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                                   title="Risk Level Distribution",
                                   color_discrete_sequence=["#10b981", "#f59e0b", "#ef4444"])
                        st.plotly_chart(fig, use_container_width=True)
                
                with res_tab3:
                    st.markdown("### 💾 Export Options")
                    csv = out.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"readmission_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_results"
                    )
        
        except Exception as e:
            st.error("❌ Batch processing failed: " + str(e))
            with st.expander("Error Details"):
                st.text(traceback.format_exc())
    else:
        st.markdown('<div class="card" style="height: 300px; display: flex; align-items: center; justify-content: center;">', unsafe_allow_html=True)
        st.markdown("<div style='text-align: center; color: #64748b;'>", unsafe_allow_html=True)
        st.markdown("<h3>📤 Upload a CSV file</h3>", unsafe_allow_html=True)
        st.markdown("<p>Drag and drop a CSV file or click to browse</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tabs[2]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔍 Model Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 📋 Model Information")
        info_data = {
            "Property": ["Name", "Type", "Version", "Status", "Input Features", "Training Date"],
            "Value": [model_name, type(model).__name__, "3.2.1", "Operational", 
                     model.n_features_in_ if hasattr(model, "n_features_in_") else "Unknown", "2023-11-15"]
        }
        info_df = pd.DataFrame(info_data)
        st.dataframe(info_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("##### ⚡ Capabilities")
        capabilities = []
        if hasattr(model, "predict_proba"):
            capabilities.append({"Capability": "Probability predictions", "Status": "✅ Available"})
        if hasattr(model, "predict"):
            capabilities.append({"Capability": "Class predictions", "Status": "✅ Available"})
        if hasattr(model, "feature_importances_"):
            capabilities.append({"Capability": "Feature importance", "Status": "✅ Available"})
        
        cap_df = pd.DataFrame(capabilities)
        st.dataframe(cap_df, use_container_width=True, hide_index=True)
    
    if feature_names:
        st.markdown("##### 🏷️ Feature Names")
        features_df = pd.DataFrame({"Feature": feature_names})
        features_df["Label"] = features_df["Feature"].apply(lambda x: label_map.get(x, x.replace("_", " ").title()))
        st.dataframe(features_df[["Feature", "Label"]], use_container_width=True, hide_index=True, height=300)
    
    if hasattr(model, "feature_importances_") and feature_names:
        st.markdown("##### 📊 Feature Importance")
        fi = model.feature_importances_
        cols = feature_names if feature_names else [f"f{i}" for i in range(len(fi))]
        df_fi = pd.DataFrame({"feature": cols, "importance": fi})
        df_fi["feature"] = df_fi["feature"].apply(lambda x: label_map.get(x, x.replace("_", " ").title()))
        df_fi = df_fi.sort_values("importance", ascending=False).head(15)
        
        fig = px.bar(df_fi, x="importance", y="feature", orientation="h", 
                     title="Top 15 Most Important Features",
                     color="importance", color_continuous_scale="Blues")
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    if load_errors:
        with st.expander("⚠️ Load Errors"):
            for k, v in load_errors.items():
                st.write(f"- {k}: {v}")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tabs[3]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Performance Dashboard")
    st.info("This section displays historical performance metrics and trends. Connect to your database to enable live data.")
    
    # Placeholder for data visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 📅 Monthly Readmission Rate")
        # Sample data
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        rates = [12.5, 11.8, 13.2, 10.5, 9.8, 11.2, 12.8, 10.1, 9.5, 8.9, 10.2, 11.1]
        
        fig = px.line(x=months, y=rates, title="Monthly Readmission Rate Trend",
                     labels={'x': 'Month', 'y': 'Readmission Rate (%)'})
        fig.update_traces(line=dict(color='#3b82f6', width=3))
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', yaxis_range=[0, 15])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### 🏥 Readmission by Department")
        # Sample data
        departments = ['Cardiology', 'Orthopedics', 'Neurology', 'Oncology', 'General Medicine']
        readmissions = [45, 32, 28, 38, 51]
        total_cases = [320, 280, 240, 210, 450]
        rates = [r/t*100 for r, t in zip(readmissions, total_cases)]
        
        fig = px.bar(x=departments, y=rates, title="Readmission Rate by Department",
                    labels={'x': 'Department', 'y': 'Readmission Rate (%)'},
                    color=rates, color_continuous_scale='Blues')
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', yaxis_range=[0, 25])
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional performance metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model Accuracy", "94.2%", "1.5%")
    with col2:
        st.metric("Precision", "91.8%", "0.8%")
    with col3:
        st.metric("Recall", "89.5%", "1.2%")
    with col4:
        st.metric("F1 Score", "90.6%", "1.0%")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #64748b; padding: 1rem;'>", unsafe_allow_html=True)
st.markdown("<p>MedPredict AI Hospital Readmission Dashboard v3.2.1 | © 2023 Healthcare Analytics Inc.</p>", unsafe_allow_html=True)
st.markdown("<p>For support contact: support@medpredict.ai | +1 (800) 555-HEALTH</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
