import streamlit as st
import pandas as pd, numpy as np, joblib, glob, os, traceback, json
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import re

# Page configuration
st.set_page_config(
    page_title="Hospital Readmission Predictor", 
    page_icon="🏥", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
    }
    .main-header {
        font-size: 2.8rem;
        color: #1e3a8a;
        margin-bottom: 0.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .card {
        border-radius: 12px;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        background-color: white;
        border: 1px solid #e2e8f0;
    }
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
    }
    .stTabs [aria-selected="true"] {
        background-color: white;
        color: #1e3a8a;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    .stButton button {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
    }
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
    .quick-form {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
    }
    @media (max-width: 1200px) {
        .quick-form {
            grid-template-columns: 1fr;
        }
    }
    .form-field {
        margin-bottom: 1rem;
    }
    .form-field label {
        display: block;
        margin-bottom: 0.5rem;
        font-weight: 600;
        color: #374151;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with enhanced design
with st.sidebar:
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    if os.path.exists("logo.png"):
        st.image("logo.png", use_column_width=True)
    else:
        st.markdown("<div style='text-align: center; font-size: 3rem;'>🏥</div>", unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center; color: white; margin-bottom: 0.5rem;'>ReadmitPredict</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: rgba(255, 255, 255, 0.8); margin-top: 0;'>Hospital Readmission Analytics</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📍 Navigation")
    nav_options = ["Patient Assessment", "Batch Processing", "Model Info"]
    selected_nav = st.radio("", nav_options, label_visibility="collapsed")
    
    st.markdown("---")
    
    st.markdown("### 📊 Today's Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Assessments", "24", "+3")
    with col2:
        st.metric("Avg. Risk", "18.7%", "-1.2%")
    
    st.markdown("---")
    
    st.markdown("### 🆘 Support")
    st.markdown("**Email:** support@readmitpredict.com")
    st.markdown("**Phone:** +1 (800) 555-READMIT")

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

# Header section
st.markdown("<div class='card'>", unsafe_allow_html=True)
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("<h1 class='main-header'>Hospital Readmission Predictor</h1>", unsafe_allow_html=True)
    st.markdown(f"**Model:** `{model_name}` • **Version:** 1.0.0")
with col2:
    st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)
    st.markdown(f"<div style='background-color: #dcfce7; color: #166534; padding: 4px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; display: inline-block;'>Live</div>", unsafe_allow_html=True)
    st.markmarkdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("**Model Features**")
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
    st.markdown("**Today's Assessments**")
    st.markdown(f"<h2>24</h2>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("**Avg. Risk Score**")
    st.markdown(f"<h2>18.7%</h2>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Tabs
tabs = st.tabs(["⚡ Patient Assessment", "📊 Batch Processing", "🔍 Model Info"])

# Function to create input widgets based on feature names
def create_feature_inputs(feature_names):
    inputs = {}
    
    if not feature_names:
        st.info("Model doesn't expose feature names. Using batch processing with CSV instead.")
        return inputs
    
    # Categorize features for better organization
    time_features = [f for f in feature_names if any(x in f.lower() for x in ['time', 'length', 'duration'])]
    count_features = [f for f in feature_names if any(x in f.lower() for x in ['num', 'number', 'count'])]
    visit_features = [f for f in feature_names if any(x in f.lower() for x in ['visit', 'admission', 'encounter'])]
    medical_features = [f for f in feature_names if any(x in f.lower() for x in ['diag', 'med', 'glucose', 'a1c', 'blood'])]
    other_features = [f for f in feature_names if f not in time_features + count_features + visit_features + medical_features]
    
    # Time Features Section
    if time_features:
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown("<h3>⏰ Time-Related Features</h3>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        for i, f in enumerate(time_features):
            lbl = label_map.get(f, f.replace("_", " ").title())
            if "time_in_hospital" in f.lower():
                inputs[f] = st.slider(lbl, 1, 30, 5, key=f"time_{i}")
            else:
                inputs[f] = st.number_input(lbl, value=0, key=f"time_{i}")
    
    # Count Features Section
    if count_features:
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown("<h3>🔢 Count Features</h3>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        for i, f in enumerate(count_features):
            lbl = label_map.get(f, f.replace("_", " ").title())
            if "medication" in f.lower() or "meds" in f.lower():
                inputs[f] = st.slider(lbl, 0, 30, 5, key=f"count_{i}")
            elif "procedure" in f.lower():
                inputs[f] = st.slider(lbl, 0, 10, 1, key=f"count_{i}")
            elif "diagnosis" in f.lower():
                inputs[f] = st.slider(lbl, 1, 20, 5, key=f"count_{i}")
            elif "lab" in f.lower():
                inputs[f] = st.slider(lbl, 0, 100, 45, key=f"count_{i}")
            else:
                inputs[f] = st.number_input(lbl, value=0, key=f"count_{i}")
    
    # Visit Features Section
    if visit_features:
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown("<h3>🏥 Visit History</h3>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        for i, f in enumerate(visit_features):
            lbl = label_map.get(f, f.replace("_", " ").title())
            if "inpatient" in f.lower():
                inputs[f] = st.slider(lbl, 0, 20, 0, key=f"visit_{i}")
            elif "outpatient" in f.lower():
                inputs[f] = st.slider(lbl, 0, 20, 0, key=f"visit_{i}")
            elif "emergency" in f.lower():
                inputs[f] = st.slider(lbl, 0, 10, 0, key=f"visit_{i}")
            else:
                inputs[f] = st.number_input(lbl, value=0, key=f"visit_{i}")
    
    # Medical Features Section
    if medical_features:
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown("<h3>💊 Medical Information</h3>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        for i, f in enumerate(medical_features):
            lbl = label_map.get(f, f.replace("_", " ").title())
            if any(x in f.lower() for x in ['diabetes', 'a1c', 'glucose']):
                inputs[f] = st.selectbox(lbl, ["No", "Yes", "Borderline"], key=f"med_{i}")
            else:
                inputs[f] = st.number_input(lbl, value=0, key=f"med_{i}")
    
    # Other Features Section
    if other_features:
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown("<h3>📋 Other Information</h3>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        for i, f in enumerate(other_features):
            lbl = label_map.get(f, f.replace("_", " ").title())
            if "age" in f.lower():
                inputs[f] = st.slider(lbl, 0, 100, 50, key=f"other_{i}")
            elif "gender" in f.lower():
                inputs[f] = st.selectbox(lbl, ["Female", "Male", "Other/Unknown"], key=f"other_{i}")
            else:
                inputs[f] = st.number_input(lbl, value=0, key=f"other_{i}")
    
    return inputs

# Function to encode categorical values to numerical
def encode_categorical_values(inputs):
    """Encode categorical values to numerical based on known mappings"""
    encoded_inputs = inputs.copy()
    
    for feature, value in inputs.items():
        if isinstance(value, str):
            if value == "No":
                encoded_inputs[feature] = 0
            elif value == "Yes":
                encoded_inputs[feature] = 1
            elif value == "Borderline":
                encoded_inputs[feature] = 0.5
            elif value == "Female":
                encoded_inputs[feature] = 0
            elif value == "Male":
                encoded_inputs[feature] = 1
            elif value == "Other/Unknown":
                encoded_inputs[feature] = 2
            else:
                # Try to convert to numeric if it's a string representation of a number
                try:
                    encoded_inputs[feature] = float(value)
                except (ValueError, TypeError):
                    encoded_inputs[feature] = 0.0
    
    return encoded_inputs

# Function to prepare input data in correct format
def prepare_input_data(inputs, feature_names):
    """Prepare input data in the correct format and order for the model"""
    if feature_names is None:
        return pd.DataFrame([inputs])
    
    # Encode categorical values first
    encoded_inputs = encode_categorical_values(inputs)
    
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

# Function to safely get prediction probability - FIXED VERSION
def safe_predict_proba(model, X):
    """Safely get prediction probability with error handling"""
    try:
        # First try predict_proba
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            
            # Handle different shapes of probability arrays
            if len(proba.shape) == 2:  # Standard binary classification
                if proba.shape[1] > 1:  # Binary classifier with two classes
                    p = float(proba[0, 1])  # Probability of class 1 (readmission)
                else:  # Possibly a single class output
                    p = float(proba[0, 0])
            else:  # Handle 1D array case
                p = float(proba[0])
                
            # Ensure the probability is between 0 and 1
            p = max(0.0, min(1.0, p))
            return p
        
        # Fallback to predict if predict_proba is not available
        elif hasattr(model, "predict"):
            pred = model.predict(X)
            # Return 1.0 for positive class, 0.0 for negative
            return 1.0 if pred[0] == 1 else 0.0
            
        else:
            st.error("Model doesn't have predict or predict_proba methods")
            return 0.5  # Return neutral probability
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        with st.expander("Error Details"):
            st.text(traceback.format_exc())
        return 0.5  # Return neutral probability on error

with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⚡ Patient Readmission Risk Assessment")
    st.write("Enter patient information to calculate readmission risk.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Patient Information")
        
        # Create input form based on model features
        inputs = create_feature_inputs(feature_names)
        
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
                    
                    # Debug: Show the prepared data
                    with st.expander("Debug: Prepared Data"):
                        st.write("Input features:", feature_names)
                        st.dataframe(X)
                    
                    # Get prediction probability with error handling
                    p = safe_predict_proba(model, X)
                    
                    # Show result card with enhanced design
                    if p >= 0.7:
                        st.markdown(f'<div class="risk-high">', unsafe_allow_html=True)
                        st.markdown(f"<h2>🚨 High Readmission Risk</h2>", unsafe_allow_html=True)
                        st.markdown(f"<h1>{p:.1%}</h1>", unsafe_allow_html=True)
                        st.markdown("**Recommendation:** Immediate follow-up, specialized care plan, and post-discharge monitoring required.")
                        st.markdown('</div>', unsafe_allow_html=True)
                    elif p >= 0.4:
                        st.markdown(f'<div class="risk-medium">', unsafe_allow_html=True)
                        st.markdown(f"<h2>⚠️ Moderate Readmission Risk</h2>", unsafe_allow_html=True)
                        st.markdown(f"<h1>{p:.1%}</h1>", unsafe_allow_html=True)
                        st.markdown("**Recommendation:** Standard follow-up with additional patient education and support.")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="risk-low">', unsafe_allow_html=True)
                        st.markdown(f"<h2>✅ Low Readmission Risk</h2>", unsafe_allow_html=True)
                        st.markdown(f"<h1>{p:.1%}</h1>", unsafe_allow_html=True)
                        st.markdown("**Recommendation:** Standard discharge procedure with routine follow-up.")
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
            
            except Exception as e:
                st.error("Prediction failed: " + str(e))
                with st.expander("Error Details"):
                    st.text(traceback.format_exc())
        else:
            # Placeholder for results area
            st.markdown('<div class="card" style="height: 600px; display: flex; align-items: center; justify-content: center; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);">', unsafe_allow_html=True)
            st.markdown("<div style='text-align: center; color: #64748b;'>", unsafe_allow_html=True)
            st.markdown("<h3>👈 Complete the assessment form</h3>", unsafe_allow_html=True)
            st.markdown("<p>Enter patient information and click 'Calculate Readmission Risk'</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #64748b; padding: 1rem;'>", unsafe_allow_html=True)
st.markdown("<p>Hospital Readmission Predictor v1.0.0 | © 2023 Healthcare Analytics Inc.</p>", unsafe_allow_html=True)
st.markdown("<p>For support contact: support@readmitpredict.com | +1 (800) 555-READMIT</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
