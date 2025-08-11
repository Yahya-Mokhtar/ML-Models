Elevvo Internship Machine Learning Projects
Overview
This repository contains the code and documentation for six machine learning projects completed during my internship at Elevvo. The projects demonstrate proficiency in regression, clustering, classification, recommendation systems, and time series forecasting, using Python and popular libraries like Pandas, Scikit-learn, Matplotlib, NumPy, and XGBoost.
Projects
1. Student Score Prediction

Objective: Predict students' exam scores based on study hours using linear regression.
Dataset: Student Performance Factors (Kaggle)
Tasks:
Cleaned and preprocessed the dataset.
Visualized data trends using Matplotlib.
Split data into training and testing sets.
Trained a linear regression model and evaluated performance using metrics like MSE and R².
Bonus: Implemented polynomial regression to compare performance.



2. Customer Segmentation

Objective: Cluster mall customers based on income and spending habits using K-Means.
Dataset: Mall Customer (Kaggle)
Tasks:
Scaled features and performed exploratory data analysis.
Applied K-Means clustering and determined optimal cluster count using the elbow method.
Visualized clusters with 2D scatter plots.
Bonus: Experimented with DBSCAN for alternative clustering.


3. Forest Cover Type Classification

Objective: Predict forest cover types using multi-class classification models.
Dataset: Covertype (UCI)
Tasks:
Preprocessed data, including handling categorical features.
Trained and evaluated models like Random Forest and XGBoost.
Visualized confusion matrix and feature importance.
Bonus: Compared Random Forest vs. XGBoost and performed hyperparameter tuning.


4. Loan Approval Prediction

Objective: Predict loan approval outcomes using classification models.
Dataset: Loan-Approval-Prediction-Dataset (Kaggle)
Tasks:
Handled missing values and encoded categorical features.
Trained models on imbalanced data, focusing on precision, recall, and F1-score.
Bonus: Applied SMOTE to address class imbalance and compared logistic regression vs. decision tree.


5. Movie Recommendation System

Objective: Build a recommendation system to suggest movies based on user similarity.
Dataset: MovieLens 100K Dataset (Kaggle)
Tasks:
Created a user-item matrix to compute similarity scores.
Recommended top-rated unseen movies for users.
Evaluated performance using precision at K.
Bonus: Implemented item-based collaborative filtering.


6. Sales Forecasting

Objective: Predict future sales using regression models on historical data.
Dataset: Walmart Sales Forecast (Kaggle)
Tasks:
Created time-based features (e.g., day, month, lag values).
Trained regression models to forecast sales.
Plotted actual vs. predicted values over time.
Bonus: Applied XGBoost with time-aware validation.


Tools and Libraries

Programming Language: Python
Libraries:
Pandas: Data manipulation and preprocessing
Scikit-learn: Machine learning models and evaluation
Matplotlib: Data visualization
NumPy: Numerical computations
XGBoost: Advanced regression and classification models



Installation

Clone the repository:git clone https://github.com/[your-username]/[your-repo-name].git


Install required Python libraries:pip install pandas scikit-learn matplotlib numpy xgboost


Run individual project scripts (e.g., python student_score_prediction.py).

Usage

Each project folder contains a Python script and associated visualization outputs.
Update dataset paths in scripts if using custom datasets.
Ensure all required libraries are installed before running scripts.

Results

Each project includes evaluation metrics (e.g., MSE, R², precision, recall, F1-score) and visualizations (e.g., scatter plots, confusion matrices, time series plots).
Bonus tasks enhanced model performance and provided deeper insights (e.g., polynomial regression, SMOTE, DBSCAN).

Acknowledgments

Thanks to Elevvo for providing the opportunity to work on these diverse machine learning tasks.
Datasets sourced from Kaggle and UCI Machine Learning Repository.
