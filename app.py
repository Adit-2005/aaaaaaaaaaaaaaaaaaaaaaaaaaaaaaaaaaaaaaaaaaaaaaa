
import streamlit as st
import pandas as pd, numpy as np, joblib, os, glob, traceback, base64
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
st.set_page_config(page_title="Hospital Readmission Predictor", page_icon="🏥", layout="wide")

# --- Utils ---
def patch_tree_monotonic(model):
    """If a loaded sklearn DecisionTree/Forest lacks monotonic_cst (older save), set to None to avoid attribute error."""
    try:
        # For single estimator
        if hasattr(model, "monotonic_cst"):
            return model
        # Some ensemble wraps estimators
        setattr(model, "monotonic_cst", None)
    except Exception:
        pass
    return model

def load_first_compatible_model():
    MODEL_DIR = "models"
    load_errors = {}
    for path in sorted(glob.glob(os.path.join(MODEL_DIR, "*.joblib"))):
        name = os.path.basename(path)
        try:
            m = joblib.load(path)
            # Quick patch for known tree attribute missing in newer sklearn
            try:
                m = patch_tree_monotonic(m)
            except Exception:
                pass
            if hasattr(m, "predict") or hasattr(m, "predict_proba"):
                return m, name, load_errors
        except Exception as e:
            load_errors[name] = str(e)
    return None, None, load_errors

model, model_name, load_errors = load_first_compatible_model()
if model is None:
    st.error("No compatible model loaded. See errors below. The app includes a 'models' folder — ensure the correct .joblib is present.")
    for k,v in load_errors.items():
        st.write(f"- **{k}**: {v[:500]}")
    st.stop()

# Header UI
st.markdown("<div style='display:flex;align-items:center;gap:16px'><div style='font-size:34px'>🏥</div><div><h1 style='margin:0'>Hospital Readmission Predictor</h1><div style='color:#586e95'>AI-assisted readmission risk scoring — polished UI</div></div></div>", unsafe_allow_html=True)
st.write("---")

# Try to infer feature names
feature_names = None
if hasattr(model, "feature_names_in_"):
    try:
        feature_names = list(model.feature_names_in_)
    except Exception:
        feature_names = None

# Attempt to get column list from ColumnTransformer inside a pipeline
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

# Layout: two columns for single input form
tabs = st.tabs(["🧍 Single Prediction", "🗂️ Batch Scoring", "ℹ️ Model Info"])
tab1, tab2, tab3 = tabs

with tab1:
    st.subheader("Single patient prediction")
    st.write("Fill the form below. Fields are auto-detected from the model when possible. For categorical fields, use the suggested values.")
    if feature_names:
        cols = st.columns(2)
        inputs = {}
        left = feature_names[0::2]
        right = feature_names[1::2]
        for i, fname in enumerate(left):
            with cols[0]:
                lbl = fname.replace("_"," ").title()
                inputs[fname] = st.text_input(lbl, value="", key=f"l_{fname}")
        for i, fname in enumerate(right):
            with cols[1]:
                lbl = fname.replace("_"," ").title()
                inputs[fname] = st.text_input(lbl, value="", key=f"r_{fname}")
        submit = st.button("Predict", type="primary")
        if submit:
            try:
                X = pd.DataFrame([inputs])
                # attempt numeric conversion where possible
                for c in X.columns:
                    try:
                        X[c] = pd.to_numeric(X[c])
                    except Exception:
                        pass
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X)[:,1]
                    p = float(probs[0])
                    score = int(p*100)
                    # show polished card
                    if p >= 0.5:
                        st.markdown(f\"\"\"<div style='border-radius:12px;padding:18px;background:linear-gradient(90deg,#ffefef,#ffecec);'>
<h3 style='color:#b00020;margin:0'>⚠️ High risk — {score}%</h3>
<p style='margin:0'>Recommendation: Consider close follow-up and review discharge plan.</p></div>\"\"\", unsafe_allow_html=True)
                    else:
                        st.markdown(f\"\"\"<div style='border-radius:12px;padding:18px;background:linear-gradient(90deg,#f0fbf6,#e8f7ee);'>
<h3 style='color:#116530;margin:0'>✅ Low risk — {score}%</h3>
<p style='margin:0'>Recommendation: Standard follow-up.</p></div>\"\"\", unsafe_allow_html=True)
                    st.progress(p)
                else:
                    pred = model.predict(X)[0]
                    st.write("Prediction:", pred)
            except Exception as e:
                st.error("Prediction failed: " + str(e))
                st.text(traceback.format_exc())
    else:
        st.info("Model does not expose feature names. Use Batch Scoring or upload a CSV with proper columns.")

with tab2:
    st.subheader("Batch scoring")
    st.write("Upload a CSV with the same feature columns used during training (no target column).")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], help="CSV should contain the same feature columns used during training.")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.write("Preview:")
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
                preds = (probs >= 0.5).astype(int)
                out = df.copy()
                out["readmission_probability"] = probs
                out["prediction"] = np.where(preds==1, "Readmitted", "Not Readmitted")
            else:
                preds = model.predict(X)
                out = df.copy()
                out["prediction"] = preds
            st.success(f"Scored {len(out)} rows.")
            st.dataframe(out.head(200))
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download results", csv_bytes, "predictions.csv", "text/csv")
        except Exception as e:
            st.error("Batch scoring failed: " + str(e))
            st.text(traceback.format_exc())

with tab3:
    st.subheader("Model info & compatibility")
    st.write(f"Loaded model: **{model_name}**")
    st.write("Available feature names (detected):")
    st.write(feature_names if feature_names else "Not available")
    st.write("Model load errors (if any):")
    for k,v in list(load_errors.items())[:20]:
        st.write(f"- **{k}**: {v[:400]}")
    st.write(\"\"\"---
**Notes**: The app includes a small compatibility patch to set missing `monotonic_cst` on tree models. For exact reproduction, pin `scikit-learn` to the version used during training in `requirements.txt`.
\"\"\")
