# ==============================================
# PISA 2015 UK ANALYSIS: PRIVATE VS PUBLIC SCHOOLS
# ==============================================
# This comprehensive analysis examines:
# 1. Performance differences between private and public schools
# 2. Impact of home resources and family wealth
# 3. School-level factors affecting performance

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import pingouin as pg
from factor_analyzer import FactorAnalyzer
import warnings

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('seaborn')
pd.set_option('display.max_columns', 50)
sns.set_palette("husl")
%matplotlib inline

# =============================
# SECTION 1: DATA LOADING & CLEANING
# =============================

def load_and_clean_data(filepath):
    """
    Comprehensive data loading and cleaning function
    Args:
        filepath (str): Path to CSV file
    Returns:
        DataFrame: Cleaned pandas DataFrame
    """
    # Load raw data with proper encoding
    print("\n[1.1] Loading raw data...")
    try:
        df = pd.read_csv(filepath)
        print(f"Initial data shape: {df.shape}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
    
    # Basic data quality checks
    print("\n[1.2] Performing initial data quality checks...")
    print("Missing values per column:")
    print(df.isnull().sum())
    
    # Handle missing values - strategy depends on column
    print("\n[1.3] Handling missing values...")
    
    # For test scores, we'll drop rows with missing values (critical for analysis)
    score_cols = ['math_score', 'read_score', 'scie_score']
    df.dropna(subset=score_cols, inplace=True)
    
    # For categorical variables, we'll create 'Unknown' category
    cat_cols = ['gender', 'immig', 'homelang', 'schltype', 'schllocation']
    for col in cat_cols:
        df[col].fillna('Unknown', inplace=True)
    
    # For numerical variables, we'll impute with median (less sensitive to outliers)
    num_cols = ['hedres', 'wealth', 'stratio']
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Data type conversion
    print("\n[1.4] Converting data types...")
    df['booksqty'] = df['booksqty'].astype('category')
    df['clsize'] = df['clsize'].astype('category')
    
    # Create composite score (average of three subjects)
    df['composite_score'] = df[score_cols].mean(axis=1)
    
    # Create binary flags for resources
    resource_cols = ['desk', 'room', 'quietplace', 'homecomputer', 'homeinternet', 'books']
    for col in resource_cols:
        df[f'{col}_bin'] = df[col].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Create total resources count
    df['total_resources'] = df[[f'{col}_bin' for col in resource_cols]].sum(axis=1)
    
    print("\n[1.5] Data cleaning complete. Final shape:", df.shape)
    return df

# =============================
# SECTION 2: EXPLORATORY DATA ANALYSIS
# =============================

def perform_eda(df):
    """
    Comprehensive exploratory data analysis with visualizations
    Args:
        df (DataFrame): Cleaned dataset
    """
    print("\n[2.1] Starting Exploratory Data Analysis...")
    
    # 1. Target Variable Distribution
    plt.figure(figsize=(15, 5))
    plt.suptitle("Distribution of Test Scores", y=1.02)
    
    for i, subject in enumerate(['math_score', 'read_score', 'scie_score'], 1):
        plt.subplot(1, 3, i)
        sns.histplot(df[subject], kde=True, bins=30)
        plt.title(f"{subject.replace('_', ' ').title()}")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    
    # 2. School Type Comparison
    print("\n[2.2] School Type Comparison")
    plt.figure(figsize=(15, 5))
    
    # Boxplot comparison
    plt.subplot(1, 2, 1)
    sns.boxplot(x='schltype', y='composite_score', data=df)
    plt.title("Test Scores by School Type")
    plt.xlabel("School Type")
    plt.ylabel("Composite Score")
    
    # Distribution plot
    plt.subplot(1, 2, 2)
    for school_type in df['schltype'].unique():
        sns.kdeplot(df[df['schltype'] == school_type]['composite_score'], 
                   label=school_type, shade=True)
    plt.title("Score Distribution by School Type")
    plt.xlabel("Composite Score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 3. Background Characteristics Analysis
    print("\n[2.3] Background Characteristics Analysis")
    background_vars = ['hedres', 'wealth', 'total_resources']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, var in enumerate(background_vars):
        sns.scatterplot(x=var, y='composite_score', hue='schltype', data=df, ax=axes[i])
        axes[i].set_title(f"Scores by {var.replace('_', ' ').title()}")
        axes[i].set_xlabel(var.replace('_', ' ').title())
        axes[i].set_ylabel("Composite Score")
    plt.tight_layout()
    plt.show()
    
    # 4. School Characteristics Analysis
    print("\n[2.4] School Characteristics Analysis")
    plt.figure(figsize=(15, 5))
    
    # Student-teacher ratio
    plt.subplot(1, 2, 1)
    sns.regplot(x='stratio', y='composite_score', data=df, 
                scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    plt.title("Scores by Student-Teacher Ratio")
    plt.xlabel("Student-Teacher Ratio")
    plt.ylabel("Composite Score")
    
    # School location
    plt.subplot(1, 2, 2)
    sns.boxplot(x='schllocation', y='composite_score', hue='schltype', data=df)
    plt.title("Scores by School Location")
    plt.xlabel("School Location")
    plt.ylabel("Composite Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# =============================
# SECTION 3: STATISTICAL TESTING
# =============================

def perform_statistical_tests(df):
    """
    Conduct all required statistical tests for hypotheses
    Args:
        df (DataFrame): Cleaned dataset
    """
    print("\n[3.1] Performing Statistical Tests...")
    
    # Hypothesis 1: School type differences
    print("\n[3.1.1] Testing Hypothesis 1: School Type Differences")
    
    # Separate private and public school students
    private_scores = df[df['schltype'] == 'Private']['composite_score']
    public_scores = df[df['schltype'] == 'Public']['composite_score']
    
    # Independent samples t-test
    t_stat, p_val = stats.ttest_ind(private_scores, public_scores, equal_var=False)
    print(f"T-test results - t-statistic: {t_stat:.3f}, p-value: {p_val:.4f}")
    
    # Effect size (Cohen's d)
    cohen_d = (private_scores.mean() - public_scores.mean()) / np.sqrt(
        (private_scores.std()**2 + public_scores.std()**2) / 2)
    print(f"Effect size (Cohen's d): {cohen_d:.3f}")
    
    # Hypothesis 2: Controlling for background factors
    print("\n[3.1.2] Testing Hypothesis 2: Controlling for Background Factors")
    
    # ANCOVA model
    ancova = pg.ancova(data=df, dv='composite_score', between='schltype',
                       covar=['hedres', 'wealth'])
    print("\nANCOVA Results (controlling for hedres and wealth):")
    print(ancova)
    
    # Hypothesis 3: Student-teacher ratio effect
    print("\n[3.1.3] Testing Hypothesis 3: Student-Teacher Ratio Effect")
    
    # Pearson correlation
    corr, p_val = stats.pearsonr(df['stratio'], df['composite_score'])
    print(f"Correlation between stratio and composite_score: r = {corr:.3f}, p = {p_val:.4f}")
    
    # Hypothesis 4: School location effect
    print("\n[3.1.4] Testing Hypothesis 4: School Location Effect")
    
    # ANOVA test
    anova = pg.anova(data=df, dv='composite_score', between='schllocation', detailed=True)
    print("\nANOVA Results for School Location:")
    print(anova)
    
    # Post-hoc tests if ANOVA is significant
    if anova['p-unc'][0] < 0.05:
        posthoc = pg.pairwise_tests(data=df, dv='composite_score', 
                                   between='schllocation', padjust='bonf')
        print("\nPost-hoc Pairwise Comparisons:")
        print(posthoc)

# =============================
# SECTION 4: ADVANCED MODELING
# =============================

def perform_advanced_modeling(df):
    """
    Perform regression modeling and feature importance analysis
    Args:
        df (DataFrame): Cleaned dataset
    """
    print("\n[4.1] Performing Advanced Modeling...")
    
    # Prepare data for modeling
    print("\n[4.1.1] Preparing data for modeling...")
    
    # Encode categorical variables
    cat_cols = ['gender', 'immig', 'homelang', 'schllocation', 'schltype']
    le = LabelEncoder()
    for col in cat_cols:
        df[f'{col}_encoded'] = le.fit_transform(df[col])
    
    # Select features and target
    features = ['hedres', 'wealth', 'stratio', 'total_resources',
                'gender_encoded', 'immig_encoded', 'homelang_encoded',
                'schllocation_encoded', 'schltype_encoded']
    X = df[features]
    y = df['composite_score']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Linear Regression
    print("\n[4.1.2] Linear Regression Results:")
    X_train_const = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train_const).fit()
    print(model.summary())
    
    # Random Forest for feature importance
    print("\n[4.1.3] Random Forest Feature Importance:")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Permutation importance
    result = permutation_importance(
        rf, X_test, y_test, n_repeats=10, random_state=42)
    
    # Display feature importance
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': result.importances_mean,
        'std': result.importances_std
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance Scores:")
    print(importance_df)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance_df, xerr=importance_df['std'])
    plt.title("Feature Importance from Random Forest")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

# =============================
# SECTION 5: COMPREHENSIVE REPORTING
# =============================

def generate_comprehensive_report(df):
    """
    Generate final report with key findings
    Args:
        df (DataFrame): Cleaned dataset
    """
    print("\n[5.1] Generating Comprehensive Report...")
    
    # 1. Key Statistics
    print("\n=== KEY STATISTICS ===")
    print(f"Total students analyzed: {len(df)}")
    print(f"Private school students: {len(df[df['schltype']=='Private'])}")
    print(f"Public school students: {len(df[df['schltype']=='Public'])}")
    
    # 2. Performance Differences
    print("\n=== PERFORMANCE DIFFERENCES ===")
    score_means = df.groupby('schltype')[['math_score', 'read_score', 'scie_score']].mean()
    print("\nAverage Scores by School Type:")
    print(score_means)
    
    # 3. Effect Sizes
    print("\n=== EFFECT SIZES ===")
    for subject in ['math_score', 'read_score', 'scie_score']:
        private_mean = df[df['schltype']=='Private'][subject].mean()
        public_mean = df[df['schltype']=='Public'][subject].mean()
        pooled_std = np.sqrt((df[df['schltype']=='Private'][subject].std()**2 + 
                            df[df['schltype']=='Public'][subject].std()**2)/2)
        cohen_d = (private_mean - public_mean) / pooled_std
        print(f"{subject.replace('_', ' ').title()}: Cohen's d = {cohen_d:.3f}")
    
    # 4. Correlation Matrix
    print("\n=== CORRELATION ANALYSIS ===")
    corr_matrix = df[['math_score', 'read_score', 'scie_score', 
                     'hedres', 'wealth', 'stratio']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title("Correlation Matrix of Key Variables")
    plt.tight_layout()
    plt.show()

# =============================
# MAIN EXECUTION
# =============================

if __name__ == "__main__":
    # Load and clean data
    filepath = "PISA-2015-GBR.csv"  # Update with your file path
    df = load_and_clean_data(filepath)
    
    # Perform EDA
    if df is not None:
        perform_eda(df)
        
        # Statistical testing
        perform_statistical_tests(df)
        
        # Advanced modeling
        perform_advanced_modeling(df)
        
        # Final report
        generate_comprehensive_report(df)
        
        print("\n[COMPLETE] Analysis finished successfully!")
    else:
        print("\n[ERROR] Analysis could not be completed due to data loading issues.")