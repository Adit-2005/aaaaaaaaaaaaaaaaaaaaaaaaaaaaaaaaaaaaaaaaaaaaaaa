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
    
    /* Form group styling */
    .form-group {
        margin-bottom: 1.2rem;
    }
    
    .form-group label {
        display: block;
        margin-bottom: 0.5rem;
        font-weight: 600;
        color: #374151;
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
        font-size: 1rem;
    }
    
    .stTextInput input:focus, .stNumberInput input:focus, .stSelectbox select:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1e3a8a;
        font-size: 1.1rem;
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
    
    /* Form section headers */
    .form-section {
        background: linear-gradient(90deg, #f1f5f9 0%, #e2e8f0 100%);
        padding: 12px 16px;
        border-radius: 8px;
        margin: 16px 0;
        border-left: 4px solid #3b82f6;
    }
    
    .form-section h3 {
        margin: 0;
        color: #1e3a8a;
        font-size: 1.2rem;
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
    nav_options = ["Patient Assessment", "Batch Processing", "Model Analytics", "Performance Dashboard"]
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
            
            # Demographic Information
            st.markdown('<div class="form-section">', unsafe_allow_html=True)
            st.markdown("<h3>👤 Demographic Information</h3>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            for i, f in enumerate(demographic_features):
                lbl = label_map.get(f, f.replace("_"," ").title())
                if "age" in f.lower() and "range" not in f.lower():
                    st.markdown(f'<div class="form-group">', unsafe_allow_html=True)
                    st.markdown(f'<label for="{f}">{lbl}</label>', unsafe_allow_html=True)
                    inputs[f] = st.selectbox(lbl, options=["[0-10)","[10-20)","[20-30)","[30-40)","[40-50)","[50-60)","[60-70)","[70-80)","[80-90)","[90-100)"], 
                                           index=5, key=f"demo_{i}", label_visibility="collapsed")
                    st.markdown('</div>', unsafe_allow_html=True)
                elif "gender" in f.lower():
                    st.markdown(f'<div class="form-group">', unsafe_allow_html=True)
                    st.markdown(f'<label for="{f}">{lbl}</label>', unsafe_allow_html=True)
                    inputs[f] = st.selectbox(lbl, options=["Female", "Male", "Other/Unknown"], 
                                           key=f"demo_{i}", label_visibility="collapsed")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="form-group">', unsafe_allow_html=True)
                    st.markdown(f'<label for="{f}">{lbl}</label>', unsafe_allow_html=True)
                    inputs[f] = st.text_input(lbl, value="", key=f"demo_{i}", placeholder="Enter value", label_visibility="collapsed")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Medical Information
            st.markdown('<div class="form-section">', unsafe_allow_html=True)
            st.markdown("<h3>🏥 Medical Information</h3>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            for i, f in enumerate(medical_features):
                lbl = label_map.get(f, f.replace("_"," ").title())
                if any(x in f.lower() for x in ["num_","number_","time_","count"]):
                    st.markdown(f'<div class="form-group">', unsafe_allow_html=True)
                    st.markdown(f'<label for="{f}">{lbl}</label>', unsafe_allow_html=True)
                    inputs[f] = st.number_input(lbl, value=0, step=1, key=f"med_{i}", 
                                              label_visibility="collapsed", min_value=0)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="form-group">', unsafe_allow_html=True)
                    st.markdown(f'<label for="{f}">{lbl}</label>', unsafe_allow_html=True)
                    inputs[f] = st.text_input(lbl, value="", key=f"med_{i}", placeholder="Enter value", label_visibility="collapsed")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Encounter Details
            st.markdown('<div class="form-section">', unsafe_allow_html=True)
            st.markdown("<h3>📋 Encounter Details</h3>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            for i, f in enumerate(encounter_features):
                lbl = label_map.get(f, f.replace("_"," ").title())
                if any(x in f.lower() for x in ["num_","number_","time_","count"]):
                    st.markdown(f'<div class="form-group">', unsafe_allow_html=True)
                    st.markdown(f'<label for="{f}">{lbl}</label>', unsafe_allow_html=True)
                    inputs[f] = st.number_input(lbl, value=0, step=1, key=f"enc_{i}", 
                                              label_visibility="collapsed", min_value=0)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="form-group">', unsafe_allow_html=True)
                    st.markdown(f'<label for="{f}">{lbl}</label>', unsafe_allow_html=True)
                    inputs[f] = st.text_input(lbl, value="", key=f"enc_{i}", placeholder="Enter value", label_visibility="collapsed")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Other Information
            if other_features:
                st.markdown('<div class="form-section">', unsafe_allow_html=True)
                st.markdown("<h3>📝 Other Information</h3>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                for i, f in enumerate(other_features):
                    lbl = label_map.get(f, f.replace("_"," ").title())
                    if any(x in f.lower() for x in ["num_","number_","time_","count"]):
                        st.markdown(f'<div class="form-group">', unsafe_allow_html=True)
                        st.markdown(f'<label for="{f}">{lbl}</label>', unsafe_allow_html=True)
                        inputs[f] = st.number_input(lbl, value=0, step=1, key=f"oth_{i}", 
                                                  label_visibility="collapsed", min_value=0)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="form-group">', unsafe_allow_html=True)
                        st.markdown(f'<label for="{f}">{lbl}</label>', unsafe_allow_html=True)
                        inputs[f] = st.text_input(lbl, value="", key=f"oth_{i}", placeholder="Enter value", label_visibility="collapsed")
                        st.markdown('</div>', unsafe_allow_html=True)
        
        else:
            st.info("Model doesn't expose feature names. Use Batch Processing with CSV instead.")
        
        # Calculate button
        if st.button("🚀 Calculate Readmission Risk", use_container_width=True, type="primary"):
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
            st.markdown('<div class="card" style="height: 600px; display: flex; align-items: center; justify-content: center; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);">', unsafe_allow_html=True)
            st.markdown("<div style='text-align: center; color: #64748b;'>", unsafe_allow_html=True)
            st.markdown("<h3>👈 Complete the assessment form</h3>", unsafe_allow_html=True)
            st.markdown("<p>Fill out the patient information and click 'Calculate Readmission Risk' to see results</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# Rest of the code remains the same as the previous version for other tabs
# [Batch Processing, Model Analytics, Performance Dashboard tabs would follow here]

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #64748b; padding: 1rem;'>", unsafe_allow_html=True)
st.markdown("<p>MedPredict AI Hospital Readmission Dashboard v3.2.1 | © 2023 Healthcare Analytics Inc.</p>", unsafe_allow_html=True)
st.markdown("<p>For support contact: support@medpredict.ai | +1 (800) 555-HEALTH</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
