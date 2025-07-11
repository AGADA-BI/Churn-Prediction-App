# Customer Churn Prediction

A machine learning project that predicts customer churn using various algorithms and provides an interactive web interface for real-time predictions.

## 📋 Project Overview

Customer churn prediction is a critical business metric that helps companies identify customers who are likely to stop using their services. This project implements multiple machine learning models to predict customer churn and provides a user-friendly Streamlit web application for making predictions.

## ✨ Features

- **Multiple ML Models**: Implements and compares various algorithms including:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Decision Tree
  - Random Forest
- **Interactive Web App**: Streamlit-based interface for real-time predictions
- **Data Preprocessing**: Automated feature engineering and scaling
- **Model Persistence**: Trained models saved as pickle files for deployment
- **Performance Evaluation**: Comprehensive model comparison with accuracy metrics

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd customer-churn-prediction
```

2. Install required dependencies:
```bash
pip install streamlit pandas numpy scikit-learn matplotlib joblib
```

### Running the Application

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Enter customer information:
   - **Age**: Customer's age (18-100)
   - **Gender**: Male or Female
   - **Tenure**: Length of service in months (0-140)
   - **Monthly Charges**: Monthly service charges ($0-$200)

4. Click "Predict" to get the churn prediction

## 📊 Dataset

The dataset contains 1,000 customer records with the following features:

- **CustomerID**: Unique customer identifier
- **Age**: Customer age
- **Gender**: Customer gender (Male/Female)
- **Tenure**: Number of months as a customer
- **MonthlyCharges**: Monthly subscription fee
- **ContractType**: Type of contract
- **InternetService**: Internet service type
- **TotalCharges**: Total charges over tenure
- **TechSupport**: Technical support usage
- **Churn**: Target variable (Yes/No)

### Data Preprocessing

- **Feature Selection**: Uses Age, Gender, Tenure, and MonthlyCharges as input features
- **Encoding**: Gender encoded as binary (1 for Female, 0 for Male)
- **Target Encoding**: Churn encoded as binary (1 for Yes, 0 for No)
- **Scaling**: StandardScaler applied for feature normalization

## 🤖 Model Performance

The project evaluates multiple machine learning algorithms:

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 89.0% |
| K-Nearest Neighbors | 88.5% |
| Support Vector Machine | 89.0% |
| Decision Tree | (Training in progress) |
| Random Forest | (Training in progress) |

### Model Selection

The final deployed model achieves approximately **89% accuracy** on the test set. The model was selected based on:
- Cross-validation performance
- Generalization capability
- Prediction speed for real-time inference

## 📁 Project Structure

```
├── README.md                    # Project documentation
├── project.ipynb              # Jupyter notebook with data analysis and model training
├── app.py                      # Streamlit web application
├── model.pkl                   # Trained machine learning model
├── scaler.pkl                  # Fitted StandardScaler for preprocessing
└── dataset/
    └── customer_churn_data.csv # Customer dataset
```

## 🔧 Technical Details

### Machine Learning Pipeline

1. **Data Loading**: Load customer data from CSV file
2. **Exploratory Data Analysis**: Analyze data distribution and patterns
3. **Data Cleaning**: Handle missing values and duplicates
4. **Feature Engineering**: Select relevant features and encode categorical variables
5. **Data Splitting**: 80-20 train-test split
6. **Preprocessing**: StandardScaler for feature normalization
7. **Model Training**: Train multiple algorithms with hyperparameter tuning
8. **Model Evaluation**: Compare performance using accuracy metrics
9. **Model Persistence**: Save best model and scaler as pickle files

### Web Application Features

- **Input Validation**: Ensures all inputs are within valid ranges
- **Real-time Predictions**: Instant churn probability calculation
- **User-friendly Interface**: Clean and intuitive Streamlit design
- **Error Handling**: Graceful handling of invalid inputs

## 📈 Usage Examples

### Making Predictions

Example customer profile:
- Age: 35
- Gender: Female
- Tenure: 24 months
- Monthly Charges: $75.50

The model will output either:
- **"No"**: Customer is likely to stay
- **"Yes"**: Customer is at risk of churning

### Interpreting Results

- **High Risk Customers**: Take proactive retention actions
- **Low Risk Customers**: Focus on upselling opportunities
- **Medium Risk Customers**: Monitor closely and engage appropriately

## 🔄 Model Retraining

To retrain the model with new data:

1. Update the dataset file (`dataset/customer_churn_data.csv`)
2. Run the Jupyter notebook (`project.ipynb`)
3. The notebook will automatically save updated `model.pkl` and `scaler.pkl` files
4. Restart the Streamlit application to use the new model

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 📞 Support

For questions or issues, please:
1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed information

---

**Note**: This project is for educational and demonstration purposes. For production use, consider additional validation, monitoring, and security measures.