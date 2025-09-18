# DM2 Prediction ML Pipeline Documentation

## Overview
This document describes the machine learning pipeline used to create the DM2 (Type 2 Diabetes) prediction model deployed in the web application.

## Notebook File
- **File**: `notebook_dm2_pipeline.ipynb`
- **Created**: September 17, 2025
- **Purpose**: Complete ML pipeline for Type 2 Diabetes prediction

## Generated Model
- **Model File**: `outputs/execution_20250917_221933/models/final_best_model_logistic_regression.joblib`
- **Algorithm**: Logistic Regression (selected as best performing model)
- **Alternative**: Gradient Boosting Classifier (also trained and evaluated)
- **Format**: Joblib serialized scikit-learn model

## Pipeline Structure
The notebook contains 12 cells covering:

1. **Data Loading and Setup**
   - Library imports (pandas, numpy, scikit-learn)
   - Configuration and output directory setup

2. **Data Preprocessing**
   - Feature engineering with 21 specific medical features
   - Data validation and cleaning

3. **Model Training**
   - Logistic Regression model
   - Gradient Boosting Classifier model
   - Cross-validation evaluation

4. **Model Selection and Serialization**
   - Performance comparison
   - Best model selection
   - Model serialization using joblib

## Features Used (21 total)
The model uses comprehensive medical and lifestyle features including:
- HOMA-IR (Homeostatic Model Assessment of Insulin Resistance)
- BMI and waist circumference
- Blood pressure measurements
- Glucose and insulin levels
- Lipid profiles
- Family history and lifestyle factors

## Model Performance
The notebook includes cross-validation results and model comparison to ensure optimal performance for medical predictions.

## Integration with Web Application
The generated model file is loaded by the Flask application (`app.py`) using:
```python
model = joblib.load('outputs/execution_20250917_221933/models/final_best_model_logistic_regression.joblib')
```

## Usage
To reproduce the model:
1. Run all cells in `notebook_dm2_pipeline.ipynb`
2. The trained model will be saved to the outputs directory
3. Update the model path in `app.py` if using a new execution timestamp

## Dependencies
- pandas
- numpy
- scikit-learn
- joblib
- matplotlib (for visualizations)

## Medical Context
This model is designed for diabetes prediction in clinical settings and incorporates established medical indicators for metabolic syndrome and insulin resistance assessment.