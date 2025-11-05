# Easy Data Science Project: Iris Species Classifier

This is a **one-day, exam-ready** project showing the full DS pipeline:
- Data collection (Iris dataset)
- EDA (basic)
- Model training & evaluation (Random Forest)
- Saved model artifact
- Simple Streamlit UI for demo

## 📦 Folder Structure
```
iris_ds_project/
├─ app/                # Streamlit UI
│  └─ app.py
├─ data/
│  └─ iris.csv         # dataset
├─ models/
│  └─ best_model.joblib
├─ reports/            # add your screenshots here
├─ scripts/
│  └─ train.py         # training script
├─ requirements.txt
└─ README.md
```

## ✅ Quick Start (5 steps)

1. **Create & activate a virtual env (recommended)**
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   # Mac/Linux:
   source .venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Re-train the model**
   ```bash
   python scripts/train.py
   ```

4. **Run the app**
   ```bash
   streamlit run app/app.py
   ```

5. **Demo**
   - Enter sepal/petal measurements and click **Predict**.
   - Show predicted species + class probabilities.
   - Open the expander to show sample training rows.

## 📊 What to show in your report/slides
- Problem: Predict Iris species from flower measurements.
- Dataset: 150 rows, 4 numeric features, 3 classes.
- Model: RandomForest (200 trees) inside a scaling pipeline.
- Metrics: Accuracy on held-out test (printout from training).
- Explainability: Feature importance intuition (petal length & width are most important).
- Demo: Streamlit form + prediction.
- Limitations: Small dataset, basic model, no hyperparameter tuning.
- Future work: Hyperparameter search, SHAP plots, additional models.

## 🧪 Reproducible Training
Run `python scripts/train.py` to:
- Load data from `data/iris.csv`
- Split train/test (80/20, stratified)
- Train a RandomForest
- Print accuracy + classification report
- Save model to `models/best_model.joblib`

## 📝 Viva Points
- Why RandomForest? Robust, handles non-linearities, low tuning.
- Why scaling? Keeps features on similar ranges (although RF is scale-invariant, pipeline is future-proof).
- Avoided leakage? Yes, only used 4 measurements as features.
- Class balance? Balanced (50/50/50).
- Metric choice? Accuracy + per-class precision/recall.

Good luck!
