# Loan Prediction Project  

A **Machine Learning project** to predict loan eligibility based on applicant details. This project walks through **data preprocessing**, **exploratory analysis**, and the application of various **classification algorithms** to automate a loan qualification procedure.  

---

## 🌟 **Project Highlights**  

- 📊 **Exploratory Data Analysis (EDA):** Gained insights into applicant demographics and loan trends.  
- 🛠️ **Data Preprocessing:** Dealt with missing values, imbalanced data, and outliers.  
- 📈 **ML Models:** Implemented multiple classification algorithms like Logistic Regression, K-Nearest Neighbour, SVM, Decision Trees, Random Forests, and Gradient Boosting.  
- ⚖️ **Model Comparison:** Evaluated and compared model performances to identify the best solution.  
- 🎯 **Deployment-Ready Models:** Saved trained models for potential integration with a user interface.  

---

## 🗂️ **Dataset**  

The dataset, `loan_data_set.csv`, was obtained from [Analytics Vidhya](https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/).  

### **Dataset Details**:  

- **Rows:** 614 (loan applications)  
- **Columns:** 13  
  - **Categorical Variables:** Gender, Married, Education, Self_Employed, Property_Area, etc.  
  - **Numerical Variables:** ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term.  
  - **Target Variable:** Loan_Status (Y/N).
- - Structure: 
<img width="562" alt="Screenshot 2024-11-29 at 20 39 09" src="https://github.com/user-attachments/assets/b5a81ce7-adc8-4595-9d94-1aa4cfc71231"> 

---

## 📓 **Notebook Structure**  

The Python notebook `loan_prediction.ipynb` is structured as follows:  

### **1. Importing Libraries**  
Key libraries include pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, and SMOTE.  

### **2. Data Exploration**  
- Visualised the distribution of categorical and numerical variables.  
- Analysed relationships between features using heatmaps and bivariate analysis.  
- Dealt with skewed distributions, missing values, and outliers.  

### **3. Data Preprocessing**  
- Dropped unnecessary columns (e.g., Loan_ID).  
- Imputed missing values using statistical methods (mode and median).  
- One-hot encoded categorical features to prepare data for ML models.  
- Handled class imbalance using SMOTE to ensure equal representation of approved and rejected loans.  
- Normalised features using MinMaxScaler.  

### **4. Machine Learning Models**  
Implemented and evaluated the following models:  
- **Logistic Regression**  
- **K-Nearest Neighbours (KNN)**  
- **Support Vector Machine (SVM)**  
- **Naive Bayes (Categorical and Gaussian)**  
- **Decision Tree**  
- **Random Forest**  
- **Gradient Boosting (with hyperparameter tuning)**  

### **5. Model Comparison**  
Compared the accuracy of models to identify the best-performing one.  

### **6. Saving Trained Models**  
Saved models using both `pickle` and `joblib` for deployment.  

---

## 🔍 **Exploratory Data Analysis (EDA)**  

- **Heatmaps and Correlation:** Identified feature relationships and their impact on loan approval.  
- **Bivariate Analysis:** Explored connections between features (e.g., Loan_Status vs. Income).  
- **Outlier Treatment:** Addressed outliers using the Interquartile Range (IQR) method.  

---

## 📈 **Key Findings**  

1. **Loan Approvals:** Applicants with good credit history and higher incomes are more likely to have their loans approved.  
2. **Skewed Distributions:** Features like ApplicantIncome and LoanAmount were positively skewed, requiring transformations.  
3. **Property Area:** Semi-urban applicants were more likely to receive loan approvals.  

---

## 🚀 **How to Run**  

### **Using the Python Notebook:**  
- Open the `loan_prediction.ipynb` file in Jupyter Notebook or any compatible IDE.  
- Run the notebook sequentially to explore data, preprocess it, and train ML models.  

### **User Interface:**  
- Navigate to the `Interface` folder:  
  ```bash
  cd Interface
  ```
- Run the Streamlit app:  
  ```bash
  streamlit run interface.py
  ```  
- <img width="500" alt="Screenshot 2024-12-16 at 05 36 47" src="https://github.com/user-attachments/assets/93879edb-21ab-4f01-8326-f9404a264466" />

---

## 📊 **Model Comparison Table**  

| **Model**               | **Accuracy (%)** |  
|--------------------------|------------------|  
| K-Nearest Neighbours     | **91.11**        |  
| Random Forest            | 88.89            |  
| SVM                      | 86.67            |  
| Decision Tree            | 86.67            |  
| Logistic Regression      | 82.22            |  
| Gradient Boost           | 73.33            |  
| Gaussian Naive Bayes     | 68.89            |  
| Categorical Naive Bayes  | 66.67            |  

---

## 📂 **Directory Structure**  

```plaintext
Loan_Prediction_Project/  
├── Dataset/  
│   └── loan_data_set.csv  
├── Models/  
│   ├── logistic_model.pkl  
│   ├── knn_model.pkl  
│   └── ...  
├── Interface/  
│   └── interface.py  
├── loan_prediction.ipynb  
├── README.md  
└── requirements.txt  
```  

---

## 🛠️ **Tools and Technologies Used**  

- **Programming Language:** Python  
- **Libraries:** pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, SMOTE, GradientBoosting, XGBoost  
- **Deployment:** Streamlit (Optional)  

---

## 💡 **Future Enhancements**  

1. Add advanced feature engineering techniques to improve model accuracy.  
2. Integrate the saved models into a web or mobile application.  
3. Explore deep learning models for potential accuracy improvement.  

---

## 🏆 **Acknowledgements**  

This project is inspired by: 

[Dataset Link](https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/)  


---

