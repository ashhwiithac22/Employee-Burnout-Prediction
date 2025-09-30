import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Load and prepare the dataset
@st.cache_data
def load_data():
    try:
        # Load the HR dataset directly from Excel
        df = pd.read_excel("D:\\PROJECTS\\Employee Burnout Prediction\\HR Data.xlsx")
        # Display basic info about the dataset
        st.sidebar.info(f"Dataset Shape: {df.shape}")
        
    except Exception as e:
        st.error(f"❌ Error loading Excel file: {e}")
        # Create a sample dataset for demonstration
        st.info("📊 Creating sample dataset for demonstration...")
        np.random.seed(42)
        n_samples = 1000
        
        df = pd.DataFrame({
            'Age': np.random.randint(22, 60, n_samples),
            'DailyRate': np.random.randint(500, 1500, n_samples),
            'DistanceFromHome': np.random.randint(1, 30, n_samples),
            'Education': np.random.randint(1, 5, n_samples),
            'EnvironmentSatisfaction': np.random.randint(1, 5, n_samples),
            'JobInvolvement': np.random.randint(1, 5, n_samples),
            'JobLevel': np.random.randint(1, 5, n_samples),
            'JobSatisfaction': np.random.randint(1, 5, n_samples),
            'MonthlyIncome': np.random.randint(2000, 20000, n_samples),
            'NumCompaniesWorked': np.random.randint(0, 10, n_samples),
            'PercentSalaryHike': np.random.randint(10, 25, n_samples),
            'PerformanceRating': np.random.randint(1, 5, n_samples),
            'RelationshipSatisfaction': np.random.randint(1, 5, n_samples),
            'StockOptionLevel': np.random.randint(0, 3, n_samples),
            'TotalWorkingYears': np.random.randint(0, 40, n_samples),
            'WorkLifeBalance': np.random.randint(1, 5, n_samples),
            'YearsAtCompany': np.random.randint(0, 20, n_samples),
            'YearsInCurrentRole': np.random.randint(0, 15, n_samples),
            'YearsSinceLastPromotion': np.random.randint(0, 15, n_samples),
            'YearsWithCurrManager': np.random.randint(0, 15, n_samples),
            'OverTime': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
            'Department': np.random.choice(['Sales', 'R&D', 'HR'], n_samples),
            'Gender': np.random.choice(['Male', 'Female'], n_samples)
        })
    
    return df

# Feature engineering for burnout risk
def engineer_burnout_features(df):
    # Clean column names (remove spaces and special characters)
    df.columns = df.columns.str.replace(' ', '').str.replace('-', '')
    
    # Create burnout risk score based on multiple factors
    # Use available columns from the dataset
    available_columns = df.columns.tolist()
    
    # Define weights for different factors
    burnout_score = 0
    weight_total = 0
    
    # Job Satisfaction (if available)
    if 'JobSatisfaction' in available_columns:
        if df['JobSatisfaction'].dtype in [np.int64, np.float64]:
            burnout_score += (df['JobSatisfaction'] / 5) * 0.2
            weight_total += 0.2
    
    # Work Life Balance (if available)
    if 'WorkLifeBalance' in available_columns:
        if df['WorkLifeBalance'].dtype in [np.int64, np.float64]:
            burnout_score += (1 - df['WorkLifeBalance'] / 4) * 0.3
            weight_total += 0.3
    
    # Job Involvement (if available)
    if 'JobInvolvement' in available_columns:
        if df['JobInvolvement'].dtype in [np.int64, np.float64]:
            burnout_score += (df['JobInvolvement'] / 4) * 0.15
            weight_total += 0.15
    
    # Overtime (if available)
    if 'OverTime' in available_columns:
        burnout_score += (df['OverTime'] == 'Yes').astype(int) * 0.2
        weight_total += 0.2
    
    # Environment Satisfaction (if available)
    if 'EnvironmentSatisfaction' in available_columns:
        if df['EnvironmentSatisfaction'].dtype in [np.int64, np.float64]:
            burnout_score += (1 - df['EnvironmentSatisfaction'] / 4) * 0.15
            weight_total += 0.15
    
    # Normalize the score
    if weight_total > 0:
        df['Burnout_Risk_Score'] = burnout_score / weight_total
    else:
        # Fallback: use random scores if no suitable columns found
        df['Burnout_Risk_Score'] = np.random.random(len(df))
    
    # Create binary burnout target (1 = High Risk, 0 = Low Risk)
    df['Burnout_Risk'] = (df['Burnout_Risk_Score'] > df['Burnout_Risk_Score'].median()).astype(int)
    
    return df

# Prepare features for modeling
def prepare_features(df):
    # Select relevant numeric features for burnout prediction
    numeric_features = [
        'Age', 'DailyRate', 'DistanceFromHome', 'Education',
        'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 
        'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked',
        'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
        'StockOptionLevel', 'TotalWorkingYears', 'WorkLifeBalance',
        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
        'YearsWithCurrManager'
    ]
    
    # Use only available numeric columns
    available_numeric = [col for col in numeric_features if col in df.columns and df[col].dtype in [np.int64, np.float64]]
    
    # Handle categorical variables
    categorical_cols = ['OverTime', 'Department', 'Gender', 'BusinessTravel', 'MaritalStatus']
    available_categorical = [col for col in categorical_cols if col in df.columns]
    
    X_numeric = df[available_numeric].copy()
    
    # Handle missing values in numeric columns
    for col in available_numeric:
        if X_numeric[col].isna().any():
            X_numeric[col] = X_numeric[col].fillna(X_numeric[col].median())
    
    # Encode categorical variables
    le = LabelEncoder()
    X_categorical = pd.DataFrame()
    for col in available_categorical:
        try:
            X_categorical[col] = le.fit_transform(df[col].astype(str))
        except Exception as e:
            st.warning(f"Could not encode {col}: {e}")
    
    if not X_categorical.empty:
        X = pd.concat([X_numeric, X_categorical], axis=1)
        feature_names = available_numeric + list(X_categorical.columns)
    else:
        X = X_numeric
        feature_names = available_numeric
    
    y = df['Burnout_Risk']
    
    return X, y, feature_names

# Train and evaluate models
def train_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'model': model,
                'y_pred_proba': y_pred_proba
            }
            trained_models[name] = model
        except Exception as e:
            st.warning(f"⚠️ Error training {name}: {e}")
    
    return results, trained_models

# Streamlit App
def main():
    st.set_page_config(page_title="Employee Burnout Risk Analysis", layout="wide")
    
    st.title("🚀 Employee Burnout Risk Analysis Dashboard")
    st.markdown("---")
    
    # Load data
    with st.spinner("📊 Loading and processing data..."):
        df = load_data()
        df = engineer_burnout_features(df)
    
    # Sidebar for navigation - MERGED THE TWO PREDICTION OPTIONS
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", 
                           ["Home / Overview", "EDA", "Model Training & Prediction", 
                            "Explainability", "Burnout Prediction & Assessment"])  # Merged into one
    
    if page == "Home / Overview":
        st.header("📊 Project Overview")
        st.write("""
        This dashboard analyzes employee burnout risk using HR metrics and machine learning.
        
        **Key Features:**
        - Exploratory Data Analysis of HR metrics
        - Multiple ML models for burnout prediction
        - Interactive burnout risk prediction
        - Employee burnout assessment questionnaire
        """)
        
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10))
        
        st.subheader("Dataset Info")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Shape:**", df.shape)
            st.write("**Columns:**", len(df.columns))
            
        with col2:
            st.write("**Burnout Risk Distribution:**")
            burnout_counts = df['Burnout_Risk'].value_counts()
            st.write(f"Low Risk: {burnout_counts.get(0, 0)}")
            st.write(f"High Risk: {burnout_counts.get(1, 0)}")
            
        with col3:
            st.write("**Burnout Risk Score Statistics:**")
            st.write(f"Mean: {df['Burnout_Risk_Score'].mean():.3f}")
            st.write(f"Std: {df['Burnout_Risk_Score'].std():.3f}")
            
        # Show available columns by type
        st.subheader("Available Features")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Numeric Features:**")
            for col in numeric_cols[:10]:  # Show first 10
                st.write(f"- {col}")
            if len(numeric_cols) > 10:
                st.write(f"... and {len(numeric_cols) - 10} more")
                
        with col2:
            st.write("**Categorical Features:**")
            for col in categorical_cols[:10]:  # Show first 10
                st.write(f"- {col}")
            if len(categorical_cols) > 10:
                st.write(f"... and {len(categorical_cols) - 10} more")
    
    elif page == "EDA":
        st.header("📈 Exploratory Data Analysis")
        
        # Summary statistics for numeric columns only
        st.subheader("Summary Statistics (Numeric Features)")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        st.dataframe(df[numeric_cols].describe())
        
        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        corr_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, center=0)
        ax.set_title("Feature Correlation Heatmap")
        st.pyplot(fig)
        
        # Distribution plots
        st.subheader("Feature Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            feature1 = st.selectbox("Select feature 1", numeric_cols, index=0)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(df[feature1], kde=True, ax=ax, bins=20)
            ax.set_title(f"Distribution of {feature1}")
            st.pyplot(fig)
        
        with col2:
            feature2 = st.selectbox("Select feature 2", numeric_cols, index=1)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(df[feature2], kde=True, ax=ax, bins=20)
            ax.set_title(f"Distribution of {feature2}")
            st.pyplot(fig)
        
        # Burnout risk by categorical features
        st.subheader("Burnout Risk Analysis")
        
        categorical_features = [col for col in df.columns if df[col].dtype == 'object' and col in df.columns]
        
        if categorical_features:
            selected_cat = st.selectbox("Select categorical feature", categorical_features)
            fig, ax = plt.subplots(figsize=(10, 6))
            burnout_by_cat = df.groupby(selected_cat)['Burnout_Risk'].mean().sort_values(ascending=False)
            burnout_by_cat.plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title(f"Average Burnout Risk by {selected_cat}")
            ax.set_ylabel("Burnout Risk Rate")
            ax.set_ylim(0, 1)
            plt.xticks(rotation=45)
            st.pyplot(fig)
    
    elif page == "Model Training & Prediction":
        st.header("🤖 Model Training & Evaluation")
        
        # Prepare features
        with st.spinner("🔄 Preparing features for modeling..."):
            X, y, feature_names = prepare_features(df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        
        st.info(f"📋 Using {len(feature_names)} features for modeling")
        st.write("**Selected Features:**", feature_names)
        
        # Train models
        with st.spinner("🏋️ Training machine learning models..."):
            results, trained_models = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Display results
        st.subheader("Model Performance Comparison")
        
        if results:
            metrics_df = pd.DataFrame({
                model: [results[model]['accuracy'], results[model]['precision'], 
                       results[model]['recall'], results[model]['f1']]
                for model in results
            }, index=['Accuracy', 'Precision', 'Recall', 'F1-Score'])
            
            st.dataframe(metrics_df.T.style.highlight_max(axis=0))
            
            # Find best model
            best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
            best_model = results[best_model_name]['model']
            st.success(f"🎯 Best Model: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.3f})")
            
            # Confusion Matrix
            st.subheader("Confusion Matrix - Best Model")
            y_pred_best = results[best_model_name]['model'].predict(X_test_scaled)
            cm = confusion_matrix(y_test, y_pred_best)
            
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'Confusion Matrix - {best_model_name}')
            st.pyplot(fig)
            
            # ROC Curve
            st.subheader("ROC Curves")
            fig, ax = plt.subplots(figsize=(8, 6))
            
            for name, result in results.items():
                fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
            
            ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curves')
            ax.legend()
            st.pyplot(fig)
            
            # Cross-validation
            st.subheader("Cross-Validation Results")
            cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            st.write(f"Cross-validation scores: {[f'{score:.3f}' for score in cv_scores]}")
            st.write(f"Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        else:
            st.error("❌ No models were successfully trained. Please check your data.")
    
    elif page == "Explainability":
        st.header("🔍 Model Explainability")
        
        with st.spinner("🔄 Preparing data for explainability analysis..."):
            X, y, feature_names = prepare_features(df)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Train Random Forest for explainability
            rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
            rf_model.fit(X_train_scaled, y_train)
        
        # Feature Importance
        st.subheader("Feature Importance")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(data=importance_df.head(15), x='importance', y='feature', ax=ax)
        ax.set_title("Top 15 Most Important Features for Burnout Prediction")
        ax.set_xlabel("Feature Importance")
        st.pyplot(fig)
        
    elif page == "Burnout Prediction & Assessment":  # MERGED PREDICTION INTERFACE
        st.header("🎯 Burnout Prediction & Assessment")
        
        # Create tabs for the two different input methods
        tab1, tab2 = st.tabs(["📊 HR Data Prediction", "🧠 Self-Assessment Questionnaire"])
        
        with tab1:
            st.subheader("Predict Burnout Risk from HR Data")
            
            with st.spinner("🔄 Loading prediction model..."):
                X, y, feature_names = prepare_features(df)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                
                # Train Random Forest model
                rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
                rf_model.fit(X_train_scaled, y_train)
            
            # Input form for HR data prediction
            st.write("Enter employee details to predict burnout risk:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.slider("Age", 18, 65, 35, key="hr_age")
                daily_rate = st.slider("Daily Rate", 100, 2000, 800, key="hr_daily_rate")
                distance_from_home = st.slider("Distance From Home (miles)", 1, 50, 10, key="hr_distance")
                education = st.slider("Education Level (1-5)", 1, 5, 3, key="hr_education")
                environment_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 3, key="hr_env_sat")
                job_involvement = st.slider("Job Involvement (1-4)", 1, 4, 3, key="hr_job_inv")
                job_level = st.slider("Job Level (1-5)", 1, 5, 2, key="hr_job_level")
            
            with col2:
                job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3, key="hr_job_sat")
                monthly_income = st.slider("Monthly Income ($)", 1000, 25000, 6500, key="hr_income")
                num_companies_worked = st.slider("Number of Companies Worked", 0, 15, 3, key="hr_companies")
                work_life_balance = st.slider("Work Life Balance (1-4)", 1, 4, 3, key="hr_wlb")
                years_at_company = st.slider("Years at Company", 0, 40, 5, key="hr_years_company")
                years_in_current_role = st.slider("Years in Current Role", 0, 20, 2, key="hr_years_role")
                overtime = st.selectbox("Works Overtime", ["No", "Yes"], key="hr_overtime")
            
            # Create input array based on available features
            input_data = []
            for feature in feature_names:
                if feature == 'Age':
                    input_data.append(age)
                elif feature == 'DailyRate':
                    input_data.append(daily_rate)
                elif feature == 'DistanceFromHome':
                    input_data.append(distance_from_home)
                elif feature == 'Education':
                    input_data.append(education)
                elif feature == 'EnvironmentSatisfaction':
                    input_data.append(environment_satisfaction)
                elif feature == 'JobInvolvement':
                    input_data.append(job_involvement)
                elif feature == 'JobLevel':
                    input_data.append(job_level)
                elif feature == 'JobSatisfaction':
                    input_data.append(job_satisfaction)
                elif feature == 'MonthlyIncome':
                    input_data.append(monthly_income)
                elif feature == 'NumCompaniesWorked':
                    input_data.append(num_companies_worked)
                elif feature == 'WorkLifeBalance':
                    input_data.append(work_life_balance)
                elif feature == 'YearsAtCompany':
                    input_data.append(years_at_company)
                elif feature == 'YearsInCurrentRole':
                    input_data.append(years_in_current_role)
                elif feature == 'OverTime':
                    input_data.append(1 if overtime == "Yes" else 0)
                else:
                    # For other features, use median values from training data
                    input_data.append(X[feature].median())
            
            input_array = np.array([input_data])
            
            # Scale input
            input_scaled = scaler.transform(input_array)
            
            # Prediction button for HR data
            if st.button("Predict Burnout Risk from HR Data", key="hr_predict"):
                prediction = rf_model.predict(input_scaled)[0]
                probability = rf_model.predict_proba(input_scaled)[0][1]
                
                st.subheader("HR Data Prediction Result")
                
                if prediction == 1:
                    st.error(f"🚨 High Burnout Risk Detected!")
                    st.write(f"Risk Probability: {probability:.2%}")
                    st.write("""
                    **Recommendations:**
                    - Consider workload reduction and delegation
                    - Encourage taking regular breaks and time off
                    - Provide access to mental health resources
                    - Review and improve work-life balance
                    - Consider flexible working arrangements
                    """)
                else:
                    st.success(f"✅ Low Burnout Risk")
                    st.write(f"Risk Probability: {probability:.2%}")
                    st.write("""
                    **Good job!** Employee shows healthy work patterns.
                    
                    **Maintenance Tips:**
                    - Continue monitoring work-life balance
                    - Encourage regular breaks and vacations
                    - Maintain open communication channels
                    - Promote team building activities
                    """)
                
                # Show contributing factors
                st.subheader("Key Contributing Factors")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if work_life_balance <= 2:
                        st.warning("⚠️ Poor Work-Life Balance")
                    else:
                        st.success("✓ Good Work-Life Balance")
                        
                with col2:
                    if job_satisfaction <= 2:
                        st.warning("⚠️ Low Job Satisfaction")
                    else:
                        st.success("✓ Good Job Satisfaction")
                        
                with col3:
                    if overtime == "Yes":
                        st.warning("⚠️ Regular Overtime")
                    else:
                        st.success("✓ Reasonable Working Hours")
        
        with tab2:
            st.subheader("Self-Assessment Burnout Questionnaire")
            st.write("Complete this assessment to evaluate burnout risk based on current symptoms and work conditions.")
            
            with st.form("burnout_assessment"):
                st.subheader("Work-Related Factors")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    work_hours = st.slider("Average weekly work hours", 20, 80, 45, key="assess_hours")
                    work_pressure = st.slider("Perceived work pressure (1-10)", 1, 10, 6, key="assess_pressure")
                    work_autonomy = st.slider("Work autonomy/control (1-10)", 1, 10, 7, key="assess_autonomy")
                    
                with col2:
                    overtime_frequency = st.selectbox("How often do you work overtime?", 
                                                   ["Never", "Rarely", "Sometimes", "Often", "Always"], key="assess_overtime")
                    deadline_frequency = st.selectbox("How often do you face tight deadlines?",
                                                   ["Never", "Rarely", "Sometimes", "Often", "Always"], key="assess_deadlines")
                
                st.subheader("Personal Well-being")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    sleep_hours = st.slider("Average hours of sleep per night", 3, 12, 7, key="assess_sleep")
                    stress_level = st.slider("Current stress level (1-10)", 1, 10, 5, key="assess_stress")
                    
                with col2:
                    energy_level = st.slider("Daily energy level (1-10)", 1, 10, 7, key="assess_energy")
                    motivation_level = st.slider("Work motivation level (1-10)", 1, 10, 7, key="assess_motivation")
                
                st.subheader("Symptoms & Feelings")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    exhaustion = st.selectbox("Emotional exhaustion?",
                                            ["Never", "Rarely", "Sometimes", "Often", "Always"], key="assess_exhaustion")
                    physical_tiredness = st.selectbox("Physical tiredness?",
                                                    ["Never", "Rarely", "Sometimes", "Often", "Always"], key="assess_tiredness")
                    
                with col2:
                    cynicism = st.selectbox("Cynicism/detachment from work?",
                                          ["Never", "Rarely", "Sometimes", "Often", "Always"], key="assess_cynicism")
                    irritability = st.selectbox("Irritability with colleagues?",
                                              ["Never", "Rarely", "Sometimes", "Often", "Always"], key="assess_irritability")
                    
                with col3:
                    reduced_accomplishment = st.selectbox("Reduced professional efficacy?",
                                                       ["Never", "Rarely", "Sometimes", "Often", "Always"], key="assess_accomplishment")
                    concentration = st.selectbox("Difficulty concentrating?",
                                               ["Never", "Rarely", "Sometimes", "Often", "Always"], key="assess_concentration")
                
                submitted = st.form_submit_button("Assess Burnout Risk")
                
                if submitted:
                    # Comprehensive scoring system
                    score = 0
                    
                    # Work factors (max 20 points)
                    if work_hours > 50: score += 2
                    if work_hours > 60: score += 3
                    
                    overtime_scores = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3, "Always": 4}
                    score += overtime_scores[overtime_frequency]
                    
                    if work_pressure >= 7: score += 2
                    if work_pressure >= 9: score += 1
                    
                    deadline_scores = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3, "Always": 4}
                    score += deadline_scores[deadline_frequency]
                    
                    if work_autonomy <= 4: score += 2
                    
                    # Well-being factors (max 15 points)
                    if sleep_hours < 6: score += 3
                    if sleep_hours < 5: score += 2
                    
                    if stress_level >= 7: score += 2
                    if stress_level >= 9: score += 1
                    
                    if energy_level <= 4: score += 2
                    if energy_level <= 3: score += 1
                    
                    if motivation_level <= 4: score += 2
                    if motivation_level <= 2: score += 1
                    
                    # Symptoms (max 25 points)
                    symptom_scores = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3, "Always": 4}
                    score += symptom_scores[exhaustion]
                    score += symptom_scores[cynicism]
                    score += symptom_scores[reduced_accomplishment]
                    score += symptom_scores[physical_tiredness]
                    score += symptom_scores[irritability]
                    score += symptom_scores[concentration]
                    
                    # Risk assessment
                    max_score = 60
                    risk_percentage = (score / max_score) * 100
                    
                    st.subheader("📋 Assessment Results")
                    st.metric("Your Burnout Risk Score", f"{score}/{max_score} ({risk_percentage:.1f}%)")
                    
                    if risk_percentage < 25:
                        st.success("🟢 LOW BURNOUT RISK")
                        st.balloons()
                        st.write("""
                        **You're doing great!** Your work habits and well-being indicators are healthy.
                        
                        **Maintenance Tips:**
                        - Continue your current work-life balance practices
                        - Keep taking regular breaks and vacations
                        - Maintain your support networks
                        - Stay proactive about stress management
                        """)
                        
                    elif risk_percentage < 50:
                        st.warning("🟡 MODERATE BURNOUT RISK")
                        st.write("""
                        **Caution advised.** You're showing some signs of strain that could lead to burnout.
                        
                        **Action Recommendations:**
                        - Monitor your workload and set boundaries
                        - Practice regular stress management techniques
                        - Ensure you're taking proper breaks
                        - Consider discussing workload with your manager
                        - Prioritize sleep and physical activity
                        """)
                        
                    elif risk_percentage < 75:
                        st.error("🟠 HIGH BURNOUT RISK")
                        st.write("""
                        **Immediate attention needed.** You're showing significant burnout risk factors.
                        
                        **Urgent Actions:**
                        - Review and reduce your workload immediately
                        - Take time off if possible
                        - Seek support from HR or mental health professionals
                        - Discuss flexible working arrangements
                        - Prioritize self-care and recovery
                        """)
                        
                    else:
                        st.error("🔴 CRITICAL BURNOUT RISK")
                        st.write("""
                        **Critical situation.** You need immediate intervention and support.
                        
                        **Emergency Recommendations:**
                        - Seek professional mental health support immediately
                        - Take medical leave if necessary
                        - Discuss your situation with HR and management
                        - Delegate responsibilities wherever possible
                        - Focus completely on recovery and well-being
                        """)

if __name__ == "__main__":
    main()