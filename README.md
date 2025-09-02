# Hospital Readmission Predictor — Final Polished App

This Streamlit app loads your exported .joblib models from the `models/` directory. It includes:
- A polished single-patient prediction form (auto-detected fields when possible).
- Batch CSV scoring with downloadable results.
- A small compatibility patch for older DecisionTree models (monotonic_cst).

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Notes on compatibility
If your exported model was trained with an older scikit-learn, pin the version in `requirements.txt` to match (e.g., `scikit-learn==1.2.2`). The app also tries a fallback patch at runtime.