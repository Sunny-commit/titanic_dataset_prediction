🚢 Titanic Dataset Survival Prediction
This project focuses on analyzing the Titanic dataset and building a Logistic Regression model to predict passenger survival based on various demographic and travel features.

📁 Dataset
The project works with two datasets:

train.csv – used for exploration and visualization

Seaborn’s built-in titanic dataset – used for model training and testing

📊 Exploratory Data Analysis (EDA)
The notebook performs extensive EDA using Seaborn and Matplotlib:

Distribution plots (distplot) for continuous variables like Age, Fare, SibSp, Parch

Count plots for categorical features like Survived, Pclass, Embarked

Bar plots for gender comparison

This helps to visually understand survival rates based on various features such as class, gender, and embarkation point.

🧠 Model Training
A Logistic Regression model is built using:

Features: pclass, sex, age, sibsp, parch, fare, embarked

Target: survived

Steps:

Missing Value Handling: Drops rows with nulls for selected features

One-Hot Encoding: Converts categorical variables to numerical format

Train-Test Split: 80% training, 20% testing

Model Training using LogisticRegression from sklearn

Evaluation with:

Accuracy score

Confusion matrix

Classification report (Precision, Recall, F1-score)

🛠️ Libraries Used
pandas, numpy

matplotlib.pyplot, seaborn

scikit-learn

🧪 How to Run
Ensure the Titanic dataset (train.csv and test.csv) is in the proper path or use Seaborn’s dataset directly.

Run titanic_dataset.py in a Python environment or Google Colab.

Review the EDA visualizations and model accuracy metrics.
