# 🚢 Titanic Survival Prediction - Binary Classification

A **canonical binary classification dataset** predicting passenger survival on the Titanic with feature engineering, handling missing data, and multiple machine learning algorithms.

## 🎯 Overview

This project demonstrates:
- ✅ Data cleaning & missing value imputation
- ✅ Feature engineering & encoding
- ✅ Binary classification models
- ✅ Model comparison & evaluation
- ✅ ROC curves & performance metrics
- ✅ Hyperparameter tuning

## 📊 Dataset Overview

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

class TitanicLoader:
    """Load and explore Titanic dataset"""
    
    @staticmethod
    def load_data():
        """Load train/test data"""
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        
        return train_df, test_df
    
    @staticmethod
    def dataset_info(df):
        """Print dataset information"""
        print(f"Shape: {df.shape}")
        print(f"\nMissing values:\n{df.isnull().sum()}")
        print(f"\nData types:\n{df.dtypes}")
        print(f"\nBasic statistics:\n{df.describe()}")
```

## 🔧 Data Preprocessing

```python
class TitanicPreprocessor:
    """Clean and prepare data"""
    
    def __init__(self):
        self.le_sex = LabelEncoder()
        self.le_embarked = LabelEncoder()
        self.scaler = StandardScaler()
    
    def fill_missing_values(self, df):
        """Handle missing values"""
        df_copy = df.copy()
        
        # Age: Impute with mean by Sex and Pclass
        df_copy['Age'] = df_copy.groupby(['Sex', 'Pclass'])['Age'].transform(
            lambda x: x.fillna(x.median())
        )
        
        # Embarked: Fill with mode
        df_copy['Embarked'].fillna(df_copy['Embarked'].mode()[0], inplace=True)
        
        # Fare: Fill with median
        df_copy['Fare'].fillna(df_copy['Fare'].median(), inplace=True)
        
        return df_copy
    
    def engineer_features(self, df):
        """Create new features"""
        df_copy = df.copy()
        
        # Family size
        df_copy['FamilySize'] = df_copy['SibSp'] + df_copy['Parch'] + 1
        
        # Is alone
        df_copy['IsAlone'] = (df_copy['FamilySize'] == 1).astype(int)
        
        # Title extraction
        df_copy['Title'] = df_copy['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        title_mapping = {
            'Mr': 0, 'Mrs': 1, 'Miss': 1, 'Master': 2, 'Dr': 3,
            'Rev': 3, 'Col': 4, 'Major': 4, 'Mlle': 1, 'Countess': 4,
            'Ms': 1, 'Jonkheer': 4
        }
        df_copy['Title'] = df_copy['Title'].map(title_mapping)
        
        # Age groups
        df_copy['AgeGroup'] = pd.cut(df_copy['Age'], bins=[0, 12, 18, 35, 60, 100],
                                     labels=[0, 1, 2, 3, 4])
        
        # Fare groups
        df_copy['FareGroup'] = pd.qcut(df_copy['Fare'], 4, labels=[0, 1, 2, 3], duplicates='drop')
        
        return df_copy
    
    def encode_categorical(self, df):
        """Encode categorical features"""
        df_copy = df.copy()
        
        # Sex
        df_copy['Sex'] = self.le_sex.fit_transform(df_copy['Sex'])
        
        # Embarked
        df_copy['Embarked'] = self.le_embarked.fit_transform(df_copy['Embarked'])
        
        return df_copy
    
    def prepare_features(self, df):
        """Full preprocessing pipeline"""
        df = self.fill_missing_values(df)
        df = self.engineer_features(df)
        df = self.encode_categorical(df)
        
        # Select features
        feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                       'Embarked', 'FamilySize', 'IsAlone', 'Title',
                       'AgeGroup', 'FareGroup']
        
        X = df[feature_cols].copy()
        X = X.fillna(X.median())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
        
        return X_scaled
```

## 🤖 Classification Models

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

class TitanicClassifiers:
    """Multiple classifiers"""
    
    @staticmethod
    def logistic_regression():
        """Logistic Regression"""
        return LogisticRegression(max_iter=1000, random_state=42)
    
    @staticmethod
    def random_forest():
        """Random Forest"""
        return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    
    @staticmethod
    def gradient_boosting():
        """Gradient Boosting"""
        return GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    
    @staticmethod
    def svm_classifier():
        """Support Vector Machine"""
        return SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    
    @staticmethod
    def get_all_models():
        """Get all models"""
        return {
            'Logistic Regression': TitanicClassifiers.logistic_regression(),
            'Random Forest': TitanicClassifiers.random_forest(),
            'Gradient Boosting': TitanicClassifiers.gradient_boosting(),
            'SVM': TitanicClassifiers.svm_classifier()
        }
```

## 📊 Model Evaluation

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report
)

class TitanicEvaluator:
    """Evaluate models"""
    
    @staticmethod
    def evaluate_model(y_true, y_pred, y_pred_proba=None):
        """Calculate metrics"""
        results = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1': f1_score(y_true, y_pred)
        }
        
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            results['AUC'] = auc(fpr, tpr)
        
        return results
    
    @staticmethod
    def plot_roc(y_true, y_pred_proba, model_name='Model'):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', label=f'{model_name} (AUC={roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'r--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.grid()
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
```

## 🎯 Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

class TitanicTuning:
    """Hyperparameter optimization"""
    
    @staticmethod
    def tune_random_forest(X_train, y_train):
        """Tune Random Forest"""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=5,
            scoring='f1'
        )
        
        grid.fit(X_train, y_train)
        
        print(f"Best params: {grid.best_params_}")
        print(f"Best CV score: {grid.best_score_:.4f}")
        
        return grid.best_estimator_
    
    @staticmethod
    def tune_gradient_boosting(X_train, y_train):
        """Tune Gradient Boosting"""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7]
        }
        
        grid = GridSearchCV(
            GradientBoostingClassifier(random_state=42),
            param_grid,
            cv=5
        )
        
        grid.fit(X_train, y_train)
        return grid.best_estimator_
```

## 💡 Interview Talking Points

**Q: Feature engineering approach?**
```
Answer:
- Title extraction: Mr/Mrs/Miss correlates with survival
- Family size: Alone passengers had lower survival
- Age groups: Children had higher survival rates
- Fare groups: Richer passengers more likely to survive
```

**Q: Handle class imbalance?**
```
Answer:
- Class weights in models
- Stratified cross-validation
- SMOTE oversampling
- F1 score instead of accuracy
```

## 🌟 Portfolio Value

✅ EDA and data exploration
✅ Feature engineering
✅ Missing value imputation
✅ Multiple classifiers
✅ Hyperparameter tuning
✅ Model interpretation
✅ Evaluation metrics

---

**Technologies**: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

