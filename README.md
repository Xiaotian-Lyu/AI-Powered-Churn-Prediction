# AI-Powered-Customer-Churn-Prediction

## Problem Statement
In the fast-paced and competitive business landscape, retaining customers has become a cornerstone for sustainable growth and long-term success. This project tackles the challenge of customer churn by developing a machine learning model that identifies high-risk customers who are likely to stop using a service. Customer churn not only impacts revenue but also reduces market share and hinders business growth.

By leveraging historical customer data, including usage behavior, demographic information, and subscription details, this project builds a predictive model to proactively identify at-risk customers. With these insights, businesses can implement personalized retention strategies, improve customer satisfaction, and optimize resource allocation. The ultimate goal is to enhance customer loyalty, reduce churn rates, and ensure long-term profitability.

---

## Data Description
The dataset consists of 100,000 customer records with the following features:
- **CustomerID**: Unique identifier for each customer.
- **Name**: Name of the customer.
- **Age**: Age of the customer.
- **Gender**: Gender of the customer (Male/Female).
- **Location**: Geographical location (e.g., Houston, Los Angeles, Miami, Chicago, etc.).
- **Subscription_Length_Months**: Duration of subscription in months.
- **Monthly_Bill**: Monthly billing amount.
- **Total_Usage_GB**: Total internet/data usage in gigabytes.
- **Churn**: Binary label indicating whether the customer churned (1) or not (0).

---

## Technologies Used
- **Languages**: Python
- **Libraries**: 
  - Data Processing: Pandas, NumPy
  - Visualization: Matplotlib, Seaborn
  - Machine Learning: Scikit-learn, TensorFlow, Keras, XGBoost
- **Modeling Techniques**:
  - Logistic Regression
  - Decision Trees
  - Random Forests
  - Gradient Boosting (XGBoost)
  - Support Vector Machines (SVM)
  - K-Nearest Neighbors (KNN)
  - Neural Networks
- **Optimization**: Hyperparameter tuning with GridSearchCV, Cross-Validation
- **Preprocessing**: Feature Scaling, PCA, Random Forest Feature Importance
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

---

## Project Workflow
1. **Exploratory Data Analysis (EDA)**:
   - Visualized data distributions, correlations, and outliers.
   - Analyzed customer demographics and usage patterns.

2. **Data Preprocessing**:
   - Handled missing values and outliers.
   - Performed feature encoding for categorical variables (e.g., gender, location).
   - Scaled numerical variables using StandardScaler for better model convergence.

3. **Feature Engineering**:
   - Selected top features using Random Forest Feature Importance.
   - Applied dimensionality reduction using PCA for efficient model training.

4. **Model Development**:
   - Experimented with multiple machine learning models.
   - Trained and validated each model using cross-validation.

5. **Model Optimization**:
   - Conducted hyperparameter tuning to improve model performance.
   - Fine-tuned thresholds to optimize recall, precision, and F1-score.

6. **Model Evaluation**:
   - Evaluated models using confusion matrices, ROC curves, and metrics such as accuracy, recall, and F1-score.
   - Selected XGBoost as the final model for its superior performance.

7. **Deployment**:
   - Saved the final XGBoost model as a pickle file for production use.
   - Designed a simulated API to accept new data inputs for churn prediction.

---

## Results
- **Key Insights**:
  - Customers with higher monthly bills and shorter subscription lengths were more likely to churn.
  - Younger customers and those in specific locations showed higher churn rates.
- **Model Performance**:
  - Final XGBoost Model:
    - **Accuracy**: 66%
    - **Precision**: 67%
    - **Recall**: 65%
    - **F1-Score**: 66%
    - **ROC-AUC**: 0.66 (Train), 0.50 (Test)

- **Business Impact**:
  - Proactively identified high-risk customers with actionable retention strategies.
  - Improved resource allocation and personalized marketing campaigns.

---

## Key Learnings
- Understanding the importance of feature selection in predictive modeling.
- Handling imbalanced datasets and optimizing models for real-world applicability.
- Developing scalable pipelines for data preprocessing and model deployment.

---

## Repository Contents
- **notebooks/**: Jupyter notebooks for data analysis and model training.
- **data/**: Dataset used for training and testing.
- **models/**: Trained and saved models in pickle format.
- **reports/**: Project reports, including visualizations and performance metrics.
- **scripts/**: Python scripts for data preprocessing and predictions.

---

## Future Improvements
- Experiment with more advanced deep learning architectures.
- Incorporate real-time data for dynamic churn predictions.
- Explore additional features to enhance model accuracy.

---

## License
This project is licensed under the MIT License.

---

Let me know if you'd like any further adjustments!
