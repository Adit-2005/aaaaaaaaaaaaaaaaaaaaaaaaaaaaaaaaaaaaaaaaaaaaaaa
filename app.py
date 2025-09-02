import streamlit as st
import pandas as pd, numpy as np, joblib, glob, os, traceback, json
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Hospital Readmission — Modern Dashboard", 
    page_icon="🏥", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1a237e;
        margin-bottom: 0.5rem;
    }
    .card {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .risk-high {
        background: linear-gradient(135deg, #ff5252 0%, #b71c1c 100%);
        color: white;
        border-radius: 10px;
        padding: 1.5rem;
    }
    .risk-low {
        background: linear-gradient(135deg, #66bb6a 0%, #2e7d32 100%);
        color: white;
        border-radius: 10px;
        padding: 1.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0px 0px;
        gap: 8px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1a237e;
        color: white;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a237e 0%, #283593 100%);
        color: white;
    }
    .sidebar .sidebar-content .stButton button {
        background-color: white;
        color: #1a237e;
        border-radius: 8px;
        font-weight: 600;
    }
    .sidebar-logo {
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with enhanced design
with st.sidebar:
    st.markdown('<div class="sidebar-logo">', unsafe_allow_html=True)
    if os.path.exists("logo.png"):
        st.image("logo.png", use_column_width=True)
    else:
        st.markdown("<h1 style='text-align: center; color: white;'>🏥</h1>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center; color: white;'>Hospital AI Toolkit</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: white;'>Modern Dashboard — Single & Batch scoring</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation
    st.markdown("### Navigation")
    page_options = ["Single Patient", "Batch Scoring", "Model Insights", "Data Overview"]
    selected_page = st.radio("Go to", page_options, label_visibility="collapsed")
    
    st.markdown("---")
    
    # Information section
    st.markdown("### System Info")
    st.markdown(f"**Version:** 2.1.0")
    st.markdown(f"**Last updated:** March 2023")
    
    st.markdown("---")
    
    # Help section
    st.markdown("### Need Help?")
    st.markdown("Contact support: support@hospitalai.com")

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

# try to infer feature names
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

# Header
st.markdown("<h1 class='main-header'>🏥 Hospital Readmission Predictor</h1>", unsafe_allow_html=True)
st.markdown(f"**Model:** `{model_name}` — Modern Dashboard")

# Metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("**Total Features**")
    if feature_names:
        st.markdown(f"<h2>{len(feature_names)}</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2>Unknown</h2>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("**Model Type**")
    st.markdown(f"<h2>{type(model).__name__}</h2>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("**Status**")
    st.markdown("<h2>Operational</h2>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("**Version**")
    st.markdown("<h2>2.1.0</h2>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Tabs with improved styling
tabs = st.tabs(["📋 Single Patient", "📊 Batch Scoring", "🔍 Model Insights", "📈 Data Overview"])

with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Single Patient Scoring")
    st.write("Fill patient details in the form below to calculate readmission risk.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    left, right = st.columns([1, 1])
    
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Patient Details")
        
        inputs = {}
        if feature_names:
            # Group features by category for better organization
            demographic_features = [f for f in feature_names if any(x in f.lower() for x in ['age', 'gender', 'race', 'ethnic'])]
            medical_features = [f for f in feature_names if any(x in f.lower() for x in ['diag', 'med', 'glucose', 'a1c', 'blood'])]
            encounter_features = [f for f in feature_names if any(x in f.lower() for x in ['time', 'visit', 'admit', 'discharge', 'number'])]
            other_features = [f for f in feature_names if f not in demographic_features + medical_features + encounter_features]
            
            with st.expander("Demographic Information", expanded=True):
                for i, f in enumerate(demographic_features):
                    lbl = label_map.get(f, f.replace("_"," ").title())
                    if "age" in f.lower() and "range" not in f.lower():
                        inputs[f] = st.selectbox(lbl, options=["[0-10)","[10-20)","[20-30)","[30-40)","[40-50)","[50-60)","[60-70)","[70-80)","[80-90)","[90-100)"], index=5, key=f"demo_{i}")
                    elif "gender" in f.lower():
                        inputs[f] = st.selectbox(lbl, options=["Female", "Male", "Other/Unknown"], key=f"demo_{i}")
                    else:
                        inputs[f] = st.text_input(lbl, key=f"demo_{i}")
            
            with st.expander("Medical Information"):
                for i, f in enumerate(medical_features):
                    lbl = label_map.get(f, f.replace("_"," ").title())
                    if any(x in f.lower() for x in ["num_","number_","time_","count"]):
                        inputs[f] = st.number_input(lbl, value=0, step=1, key=f"med_{i}")
                    else:
                        inputs[f] = st.text_input(lbl, key=f"med_{i}")
            
            with st.expander("Encounter Details"):
                for i, f in enumerate(encounter_features):
                    lbl = label_map.get(f, f.replace("_"," ").title())
                    if any(x in f.lower() for x in ["num_","number_","time_","count"]):
                        inputs[f] = st.number_input(lbl, value=0, step=1, key=f"enc_{i}")
                    else:
                        inputs[f] = st.text_input(lbl, key=f"enc_{i}")
            
            if other_features:
                with st.expander("Other Information"):
                    for i, f in enumerate(other_features):
                        lbl = label_map.get(f, f.replace("_"," ").title())
                        if any(x in f.lower() for x in ["num_","number_","time_","count"]):
                            inputs[f] = st.number_input(lbl, value=0, step=1, key=f"oth_{i}")
                        else:
                            inputs[f] = st.text_input(lbl, key=f"oth_{i}")
        
        else:
            st.info("Model doesn't expose feature names. Use Batch Scoring with CSV instead.")
        
        if st.button("Predict Readmission Risk", use_container_width=True):
            st.session_state.predict_clicked = True
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with right:
        if 'predict_clicked' in st.session_state and st.session_state.predict_clicked:
            try:
                X = pd.DataFrame([inputs])
                # convert numeric-like
                for c in X.columns:
                    try:
                        X[c] = pd.to_numeric(X[c])
                    except Exception:
                        pass
                
                if hasattr(model, "predict_proba"):
                    p = float(model.predict_proba(X)[:,1][0])
                    
                    # Show result card
                    if p >= 0.5:
                        st.markdown(f'<div class="risk-high">', unsafe_allow_html=True)
                        st.markdown(f"<h2>High Readmission Risk</h2>", unsafe_allow_html=True)
                        st.markdown(f"<h1>{p:.1%}</h1>", unsafe_allow_html=True)
                        st.markdown("**Recommendation:** Close follow-up and specialized care plan")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="risk-low">', unsafe_allow_html=True)
                        st.markdown(f"<h2>Low Readmission Risk</h2>", unsafe_allow_html=True)
                        st.markdown(f"<h1>{p:.1%}</h1>", unsafe_allow_html=True)
                        st.markdown("**Recommendation:** Standard discharge procedure")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Gauge chart
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = p,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Readmission Probability"},
                        delta = {'reference': 0.5, 'increasing': {'color': "#ff5252"}, 'decreasing': {'color': "#66bb6a"}},
                        gauge = {
                            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "darkblue"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 0.3], 'color': '#c8e6c9'},
                                {'range': [0.3, 0.5], 'color': '#fff9c4'},
                                {'range': [0.5, 1], 'color': '#ffcdd2'}],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.5}}))
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Risk factors (if feature importance is available)
                    if hasattr(model, "feature_importances_") and feature_names:
                        st.markdown("### Key Risk Factors")
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

with tabs[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Batch Scoring")
    st.write("Upload a CSV file containing patient data for batch processing.")
    
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], help="The file should contain all required features but no target column", key="batch_uploader")
    
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"Successfully loaded {len(df)} records")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            
            with st.expander("Data Preview"):
                st.dataframe(df.head(10), use_container_width=True)
            
            if feature_names:
                missing = [c for c in feature_names if c not in df.columns]
                if missing:
                    st.error(f"Missing columns: {', '.join(missing)}")
                    st.stop()
                else:
                    X = df[feature_names]
            else:
                X = df
            
            if st.button("Process Batch", type="primary", use_container_width=True, key="process_batch"):
                with st.spinner("Processing data..."):
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X)[:,1]
                        preds = (probs>=0.5).astype(int)
                        out = df.copy()
                        out["readmission_probability"] = probs
                        out["prediction"] = np.where(preds==1, "Readmitted", "Not Readmitted")
                    else:
                        preds = model.predict(X)
                        out = df.copy()
                        out["prediction"] = preds
                
                st.success(f"Successfully scored {len(out)} records")
                
                # Summary statistics
                if "prediction" in out.columns:
                    readmitted_count = (out["prediction"] == "Readmitted").sum() if "Readmitted" in out["prediction"].values else (out["prediction"] == 1).sum()
                    readmission_rate = readmitted_count / len(out)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Processed", len(out))
                    col2.metric("Predicted Readmissions", readmitted_count)
                    col3.metric("Readmission Rate", f"{readmission_rate:.1%}")
                
                # Display results
                with st.expander("Results Preview"):
                    st.dataframe(out.head(10), use_container_width=True)
                
                # Download button
                csv = out.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="readmission_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_results"
                )
        
        except Exception as e:
            st.error("Batch processing failed: " + str(e))
            with st.expander("Error Details"):
                st.text(traceback.format_exc())
    
    st.markdown('</div>', unsafe_allow_html=True)

with tabs[2]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Model Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Model Information")
        st.write(f"**Name:** {model_name}")
        st.write(f"**Type:** {type(model).__name__}")
        if hasattr(model, "n_features_in_"):
            st.write(f"**Input Features:** {model.n_features_in_}")
        st.write(f"**Loaded Successfully:** Yes")
    
    with col2:
        st.markdown("##### Capabilities")
        capabilities = []
        if hasattr(model, "predict_proba"):
            capabilities.append("Probability predictions")
        if hasattr(model, "predict"):
            capabilities.append("Class predictions")
        if hasattr(model, "feature_importances_"):
            capabilities.append("Feature importance")
        
        for cap in capabilities:
            st.write(f"✅ {cap}")
    
    if feature_names:
        st.markdown("##### Feature Names")
        features_df = pd.DataFrame({"Feature": feature_names})
        features_df["Label"] = features_df["Feature"].apply(lambda x: label_map.get(x, x.replace("_", " ").title()))
        st.dataframe(features_df[["Feature", "Label"]], use_container_width=True, hide_index=True)
    
    if hasattr(model, "feature_importances_") and feature_names:
        st.markdown("##### Feature Importance")
        fi = model.feature_importances_
        cols = feature_names if feature_names else [f"f{i}" for i in range(len(fi))]
        df_fi = pd.DataFrame({"feature": cols, "importance": fi})
        df_fi["feature"] = df_fi["feature"].apply(lambda x: label_map.get(x, x.replace("_", " ").title()))
        df_fi = df_fi.sort_values("importance", ascending=False).head(20)
        
        fig = px.bar(df_fi, x="importance", y="feature", orientation="h", 
                     title="Top 20 Most Important Features")
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    if load_errors:
        with st.expander("Load Errors"):
            for k, v in load_errors.items():
                st.write(f"- {k}: {v}")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tabs[3]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Data Overview")
    st.info("This section would display historical data trends and statistics. Connect to your database to enable this feature.")
    
    # Placeholder for data visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Monthly Readmission Rate")
        # Sample data
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        rates = [12.5, 11.8, 13.2, 10.5, 9.8, 11.2, 12.8, 10.1, 9.5, 8.9, 10.2, 11.1]
        
        fig = px.line(x=months, y=rates, title="Monthly Readmission Rate Trend")
        fig.update_layout(xaxis_title="Month", yaxis_title="Readmission Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### Readmission by Department")
        # Sample data
        departments = ['Cardiology', 'Orthopedics', 'Neurology', 'Oncology', 'General Medicine']
        readmissions = [45, 32, 28, 38, 51]
        total_cases = [320, 280, 240, 210, 450]
        rates = [r/t*100 for r, t in zip(readmissions, total_cases)]
        
        fig = px.bar(x=departments, y=rates, title="Readmission Rate by Department")
        fig.update_layout(xaxis_title="Department", yaxis_title="Readmission Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>Hospital AI Toolkit v2.1.0 | © 2023 Healthcare Analytics Inc.</div>", unsafe_allow_html=True)
