
import streamlit as st
import pandas as pd, numpy as np, joblib, glob, os, traceback, json
from PIL import Image
import plotly.express as px

st.set_page_config(page_title="Hospital Readmission — Modern Dashboard", page_icon="🏥", layout="wide")

# Sidebar
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", use_column_width=False, width=120)
st.sidebar.title("Hospital AI Toolkit")
st.sidebar.caption("Modern Dashboard — Single & Batch scoring")

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
        st.write("Load attempts:")
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
st.markdown("<h1 style='margin:0'>🏥 Hospital Readmission Predictor</h1>", unsafe_allow_html=True)
st.markdown(f"**Model:** `{model_name}` — Modern Dashboard")

# Layout: metrics row
c1,c2,c3,c4 = st.columns([2,1,1,1])
c1.subheader("Overview")
c1.write("Use the form to score a single patient or upload a CSV for batch scoring.")
if feature_names:
    c2.metric("Features", len(feature_names))
else:
    c2.metric("Features", "Unknown")
# show sample size if model has attribute (not mandatory)
if hasattr(model, "n_features_in_"):
    c3.metric("n_features_in_", int(getattr(model, "n_features_in_")))
else:
    c3.metric("Model type", type(model).__name__)
c4.metric("Mode", "Exact-model" if model_name else "Fallback")

tabs = st.tabs(["Single Patient", "Batch Scoring", "Model Insights"])
tab1, tab2, tab3 = tabs

with tab1:
    st.subheader("Single Patient Scoring")
    st.write("Fill patient details in the left panel. Results appear on the right.")
    left, right = st.columns([1,2])
    with left:
        st.markdown("### Inputs")
        inputs = {}
        if feature_names:
            for f in feature_names:
                lbl = label_map.get(f, f.replace("_"," ").title())
                # choose widget type heuristically
                if "age" in f.lower() and "range" not in f.lower():
                    inputs[f] = st.selectbox(lbl, options=["[0-10)","[10-20)","[20-30)","[30-40)","[40-50)","[50-60)","[60-70)","[70-80)","[80-90)","[90-100)"], index=5, key=f)
                elif any(x in f.lower() for x in ["num_","number_","time_","count"]):
                    inputs[f] = st.number_input(lbl, value=0, step=1, key=f)
                else:
                    inputs[f] = st.text_input(lbl, key=f)
        else:
            st.info("Model doesn't expose feature names. Use Batch Scoring with CSV instead.")
        if st.button("Predict Patient"):
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
                    left_metric = right
                    # show result card
                    with right:
                        st.markdown(f\"\"\"<div style='border-radius:12px;padding:18px;background:linear-gradient(90deg,#fff7e6,#fff2de);'>
<h2 style='color:#b35b00;margin:0'>Risk: { 'High' if p>=0.5 else 'Low' } — {p:.2%}</h2>
<p style='margin:0'>Recommendation: { 'Close follow-up' if p>=0.5 else 'Standard discharge' }</p>
</div>\"\"\", unsafe_allow_html=True)
                        # gauge chart
                        fig = px.pie(values=[p,1-p], names=['Readmission','No Readmission'], hole=0.6)
                        fig.update_traces(textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    pred = model.predict(X)[0]
                    st.write("Prediction:", pred)
            except Exception as e:
                st.error("Prediction failed: " + str(e))
                st.text(traceback.format_exc())

with tab2:
    st.subheader("Batch Scoring")
    st.write("Upload a CSV (no target column). Use the sample template if included.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.write("Preview")
            st.dataframe(df.head(50))
            if feature_names:
                missing = [c for c in feature_names if c not in df.columns]
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    X = df[feature_names]
            else:
                X = df
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
            st.success(f"Scored {len(out)} rows.")
            st.dataframe(out.head(200))
            st.download_button("Download results", out.to_csv(index=False).encode('utf-8'), "predictions.csv", "text/csv")
        except Exception as e:
            st.error("Batch scoring failed: " + str(e))
            st.text(traceback.format_exc())

with tab3:
    st.subheader("Model Insights")
    st.write("Feature names (detected):")
    st.write(feature_names if feature_names else "Not available")
    st.write("Model type: " + type(model).__name__)
    if hasattr(model, "feature_importances_"):
        try:
            import pandas as pd, plotly.express as px
            fi = model.feature_importances_
            cols = feature_names if feature_names else [f"f{i}" for i in range(len(fi))]
            df_fi = pd.DataFrame({"feature": cols, "importance": fi}).sort_values("importance", ascending=False).head(20)
            fig = px.bar(df_fi, x="importance", y="feature", orientation="h")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.write("Failed to show feature importances:", e)

    st.write("Load attempts/errors:")
    for k,v in load_errors.items():
        st.write(f"- {k}: {v}")

