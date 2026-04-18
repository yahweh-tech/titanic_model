import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.datasets import fetch_openml

def main():
    print("Fetching Titanic dataset from OpenML...")
    titanic = fetch_openml('titanic', version=1, as_frame=True, parser='auto')
    df = titanic.frame
    
    # In OpenML titanic dataset, target is 'survived'
    X = df.drop('survived', axis=1)
    # Convert 'survived' to integer to avoid type issues
    y = df['survived'].astype(int)

    # Select relevant features for prediction
    numeric_features = ['age', 'fare', 'sibsp', 'parch']
    categorical_features = ['pclass', 'sex', 'embarked']

    print("Preprocessing data...")
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create the training pipeline
    model = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest model...")
    model.fit(X_train, y_train)

    # Evaluate the model
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the trained model to disk
    joblib.dump(model, 'titanic_model.pkl')
    print("Model successfully saved as 'titanic_model.pkl'")

if __name__ == '__main__':
    main()
