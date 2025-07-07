# Customer Churn Prediction API Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Data Specification](#data-specification)
4. [Model Information](#model-information)
5. [Web Application API](#web-application-api)
6. [Machine Learning Pipeline](#machine-learning-pipeline)
7. [Usage Instructions](#usage-instructions)
8. [Examples](#examples)
9. [Technical Requirements](#technical-requirements)
10. [Troubleshooting](#troubleshooting)

## Project Overview

The Customer Churn Prediction project is a machine learning application that predicts whether a customer will churn (leave the service) based on their demographic and usage characteristics. The project consists of:

- **Streamlit Web Application**: Interactive user interface for real-time predictions
- **Machine Learning Pipeline**: Complete ML workflow from data preprocessing to model training
- **Trained Models**: Serialized Random Forest classifier and StandardScaler for feature normalization

## Architecture

```
├── app.py                          # Streamlit web application
├── project.ipynb                   # ML pipeline and model training
├── model.pkl                       # Trained Random Forest model
├── scaler.pkl                      # Fitted StandardScaler
└── dataset/
    └── customer_churn_data.csv     # Training dataset
```

## Data Specification

### Input Features

The model uses the following 4 features for prediction:

| Feature | Type | Range | Description | Encoding |
|---------|------|-------|-------------|----------|
| `Age` | Integer | 18-100 | Customer's age in years | Numeric value |
| `Gender` | String | "Male"/"Female" | Customer's gender | Female=1, Male=0 |
| `Tenure` | Integer | 0-140 | Number of months as customer | Numeric value |
| `MonthlyCharges` | Float | 0.0-200.0 | Monthly subscription fee | Numeric value |

### Output

| Field | Type | Values | Description |
|-------|------|--------|-------------|
| `Churn` | Integer | 0, 1 | Churn prediction (0=No, 1=Yes) |
| `Predicted` | String | "Yes"/"No" | Human-readable prediction |

### Feature Processing

- **Gender Encoding**: Female → 1, Male → 0
- **Feature Scaling**: All features are standardized using `StandardScaler`
- **Feature Order**: [Age, Gender, Tenure, MonthlyCharges]

## Model Information

### Model Details

- **Algorithm**: Random Forest Classifier
- **Training Method**: Grid Search with 5-fold Cross-Validation
- **Performance**: ~89% accuracy on test set
- **Feature Importance**: Age, Tenure, and MonthlyCharges are key predictors

### Hyperparameter Optimization

The model was selected using GridSearchCV with the following parameter grid:

```python
param_grid = {
    "n_estimators": [50, 100, 200, 400],
    "max_features": [2, 3, 4], 
    "bootstrap": [True, False]
}
```

### Alternative Models Evaluated

| Model | Accuracy | Notes |
|-------|----------|-------|
| Logistic Regression | 89% | Baseline model |
| K-Nearest Neighbors | 88.5% | Optimized with GridSearch |
| Support Vector Machine | 89% | Tested multiple kernels |
| Decision Tree | - | Part of ensemble evaluation |
| **Random Forest** | **89%** | **Selected final model** |

## Web Application API

### Streamlit Application (`app.py`)

#### Main Interface Components

##### Input Components

```python
# Age input
age = st.number_input("Age", min_value=18, max_value=100, value=30)

# Tenure input  
tenure = st.number_input("Enter tenure", min_value=0, max_value=140, value=10)

# Monthly charges input
monthly_charges = st.number_input("Enter monthly charge", 
                                  min_value=0.0, max_value=200.0, value=50.0)

# Gender selection
gender = st.selectbox("Enter the Gender", ["Male", "Female"])
```

##### Prediction Interface

```python
# Prediction trigger
predict_button = st.button("Predict")

# Prediction logic
if predict_button:
    # Process inputs and generate prediction
    prediction = model.predict(processed_features)[0]
    result = "Yes" if prediction == 1 else "No"
    st.write(f"Predicted: {result}")
```

#### Application Flow

1. **Input Collection**: User provides customer demographics
2. **Feature Processing**: Convert inputs to model format
3. **Prediction**: Apply trained model to generate churn probability
4. **Output Display**: Show human-readable prediction result

#### Key Functions

##### Gender Encoding Function
```python
gender_selected = 1 if gender == "Female" else 0
```

##### Feature Array Construction
```python
X = [age, gender_selected, tenure, monthly_charges]
X1 = np.array(X)
X_array = scaler.transform([X1])
```

##### Prediction Processing
```python
prediction = model.predict(X_array)[0]
predicted = "Yes" if prediction == 1 else "No"
```

## Machine Learning Pipeline

### Data Processing Functions

#### Data Loading and Cleaning
```python
# Load dataset
df = pd.read_csv("dataset/customer_churn_data.csv")

# Handle missing values
df["InternetService"] = df["InternetService"].fillna("")
```

#### Feature Engineering
```python
# Feature selection
X = df[["Age", "Gender", "Tenure", "MonthlyCharges"]]
y = df[["Churn"]]

# Gender encoding
X["Gender"] = X["Gender"].apply(lambda x: 1 if x == "Female" else 0)

# Target encoding  
y["Churn"] = y["Churn"].apply(lambda x: 1 if x == "Yes" else 0)
```

#### Data Splitting and Scaling
```python
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Model Training Functions

#### Performance Evaluation Function
```python
def modelperformance(predictions):
    """
    Calculate and print model accuracy.
    
    Parameters:
    -----------
    predictions : array-like
        Model predictions for test set
        
    Returns:
    --------
    None
        Prints accuracy score to console
    """
    print("Accuracy score on model is {}".format(accuracy_score(y_test, predictions)))
```

#### Grid Search Configuration
```python
# Random Forest Grid Search
param_grid = {
    "n_estimators": [50, 100, 200, 400],
    "max_features": [2, 3, 4],
    "bootstrap": [True, False]
}

grid_rfc = GridSearchCV(rfc_model, param_grid, cv=5)
grid_rfc.fit(X_train, y_train)
```

#### Model Serialization
```python
# Save best model
best_model = grid_rfc.best_estimator_
joblib.dump(best_model, "model.pkl")

# Save scaler
joblib.dump(scaler, "scaler.pkl")
```

## Usage Instructions

### Running the Web Application

#### Prerequisites
```bash
pip install streamlit pandas numpy scikit-learn joblib
```

#### Start the Application
```bash
streamlit run app.py
```

#### Application Access
- Local URL: `http://localhost:8501`
- The application will open automatically in your default browser

### Using the Prediction Interface

1. **Enter Customer Information**:
   - Age: Integer between 18-100
   - Tenure: Number of months (0-140)
   - Monthly Charges: Amount in dollars (0.00-200.00)
   - Gender: Select "Male" or "Female"

2. **Generate Prediction**:
   - Click the "Predict" button
   - View the churn prediction result

3. **Interpret Results**:
   - "Yes": Customer is likely to churn
   - "No": Customer is likely to stay

### Programmatic Usage

#### Loading the Model
```python
import joblib
import numpy as np

# Load trained components
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
```

#### Making Predictions
```python
# Example customer data
age = 35
gender = "Female"  # Will be encoded as 1
tenure = 24
monthly_charges = 75.50

# Encode gender
gender_encoded = 1 if gender == "Female" else 0

# Create feature array
features = np.array([[age, gender_encoded, tenure, monthly_charges]])

# Scale features
features_scaled = scaler.transform(features)

# Make prediction
prediction = model.predict(features_scaled)[0]
result = "Yes" if prediction == 1 else "No"

print(f"Churn Prediction: {result}")
```

## Examples

### Example 1: High Churn Risk Customer
```python
# Customer profile
customer_data = {
    "age": 65,
    "gender": "Male",      # Encoded as 0
    "tenure": 3,           # Short tenure
    "monthly_charges": 95.0 # High charges
}

# Expected prediction: High churn probability ("Yes")
```

### Example 2: Low Churn Risk Customer
```python
# Customer profile  
customer_data = {
    "age": 30,
    "gender": "Female",    # Encoded as 1
    "tenure": 36,          # Long tenure
    "monthly_charges": 45.0 # Moderate charges
}

# Expected prediction: Low churn probability ("No")
```

### Example 3: Batch Predictions
```python
import pandas as pd

# Multiple customers
customers = pd.DataFrame({
    'Age': [25, 45, 60],
    'Gender': ['Female', 'Male', 'Female'], 
    'Tenure': [12, 24, 6],
    'MonthlyCharges': [50.0, 80.0, 120.0]
})

# Process batch
customers['Gender_Encoded'] = customers['Gender'].apply(
    lambda x: 1 if x == "Female" else 0
)

# Select and order features
features = customers[['Age', 'Gender_Encoded', 'Tenure', 'MonthlyCharges']].values

# Scale and predict
features_scaled = scaler.transform(features)
predictions = model.predict(features_scaled)

# Add results
customers['Churn_Prediction'] = ['Yes' if p == 1 else 'No' for p in predictions]
print(customers[['Age', 'Gender', 'Tenure', 'MonthlyCharges', 'Churn_Prediction']])
```

### Example 4: Feature Importance Analysis
```python
# Get feature importance from the model
feature_names = ['Age', 'Gender', 'Tenure', 'MonthlyCharges']
importances = model.feature_importances_

# Create importance DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("Feature Importance Rankings:")
print(importance_df)
```

## Technical Requirements

### Dependencies
```
streamlit>=1.0.0
pandas>=1.3.0  
numpy>=1.21.0
scikit-learn>=1.0.0
joblib>=1.0.0
matplotlib>=3.4.0 (for notebook visualization)
```

### System Requirements
- Python 3.7+
- 4GB RAM minimum (for model loading)
- Modern web browser for Streamlit interface

### File Dependencies
| File | Purpose | Required |
|------|---------|----------|
| `model.pkl` | Trained Random Forest model | Yes |
| `scaler.pkl` | Fitted StandardScaler | Yes |
| `dataset/customer_churn_data.csv` | Training data | No (for inference) |

## Troubleshooting

### Common Issues

#### Model Loading Errors
```python
# Error: FileNotFoundError: model.pkl not found
# Solution: Ensure model files are in the correct directory
import os
print("Current directory:", os.getcwd())
print("Files:", os.listdir("."))
```

#### Feature Scaling Issues
```python
# Error: Input features not scaled properly
# Solution: Always use the same scaler used during training
features_scaled = scaler.transform(features)  # Correct
# NOT: features_scaled = StandardScaler().fit_transform(features)
```

#### Streamlit Port Issues
```bash
# Error: Port 8501 already in use
# Solution: Use different port
streamlit run app.py --server.port 8502
```

#### Prediction Format Errors
```python
# Error: Input array must be 2D
# Solution: Ensure proper array shape
features = np.array([[age, gender, tenure, charges]])  # Correct (2D)
# NOT: features = np.array([age, gender, tenure, charges])  # Incorrect (1D)
```

### Validation Checks

#### Input Validation
```python
def validate_input(age, gender, tenure, monthly_charges):
    """
    Validate user inputs before prediction.
    
    Parameters:
    -----------
    age : int
        Customer age
    gender : str  
        Customer gender
    tenure : int
        Customer tenure in months
    monthly_charges : float
        Monthly charges amount
        
    Returns:
    --------
    bool
        True if inputs are valid, False otherwise
    """
    if not (18 <= age <= 100):
        return False, "Age must be between 18 and 100"
    
    if gender not in ["Male", "Female"]:
        return False, "Gender must be 'Male' or 'Female'"
        
    if not (0 <= tenure <= 140):
        return False, "Tenure must be between 0 and 140 months"
        
    if not (0.0 <= monthly_charges <= 200.0):
        return False, "Monthly charges must be between $0.00 and $200.00"
        
    return True, "Valid inputs"
```

### Performance Monitoring

#### Model Performance Metrics
```python
# Calculate additional metrics if needed
from sklearn.metrics import classification_report, confusion_matrix

# Generate comprehensive performance report
def evaluate_model_performance(model, X_test, y_test):
    """
    Generate comprehensive model performance metrics.
    
    Parameters:
    -----------
    model : sklearn.ensemble.RandomForestClassifier
        Trained model
    X_test : array-like
        Test features
    y_test : array-like  
        Test labels
        
    Returns:
    --------
    dict
        Performance metrics dictionary
    """
    predictions = model.predict(X_test)
    
    return {
        'accuracy': accuracy_score(y_test, predictions),
        'classification_report': classification_report(y_test, predictions),
        'confusion_matrix': confusion_matrix(y_test, predictions)
    }
```

---

**Last Updated**: Generated from codebase analysis  
**Version**: 1.0  
**Contact**: For issues or questions about this API documentation, please refer to the project repository.