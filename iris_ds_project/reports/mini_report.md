# Iris Species Classifier – Mini Report

**Objective:** Build a quick, end-to-end classifier to predict Iris species from flower measurements.

## Dataset
- Source: scikit-learn Iris
- Samples: 150 | Classes: 3 (setosa, versicolor, virginica)
- Features: sepal_length, sepal_width, petal_length, petal_width

## Method
- Split: 80/20 stratified train/test
- Model: RandomForestClassifier (200 trees) inside a pipeline with StandardScaler
- Metric: Accuracy + classification report

## Results
- Test Accuracy: 0.9000

(Add the `classification_report` output from `python scripts/train.py` here as a screenshot or paste.)

## Demo
- Run `streamlit run app/app.py`
- Input the four measurements and view predictions + probabilities.

## Discussion
- Petal length/width are most informative (domain knowledge).
- Small, clean dataset → high accuracy.

## Limitations & Future Work
- Limited dataset size and diversity.
- Add hyperparameter tuning, SHAP explanations, and model comparison.
