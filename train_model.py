
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(
    BASE_DIR / "House Price Prediction Dataset.csv"
)

print("Dataset Shape:", df.shape)

print("\nFirst Five Rows:")
print(df.head())

# --------------------------------------------------
# 2. DATA QUALITY CHECK
# --------------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

# --------------------------------------------------
# 3. FEATURE AND TARGET SELECTION
# --------------------------------------------------

X = df.drop(columns=["Id", "Price"])
y = df["Price"]

# --------------------------------------------------
# 4. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=2
)

# --------------------------------------------------
# 5. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# --------------------------------------------------

numerical_columns = X.select_dtypes(
    include="int64"
).columns.tolist()

categorical_columns = X.select_dtypes(
    include="object"
).columns.tolist()

print("\nNumerical Features:", numerical_columns)
print("Categorical Features:", categorical_columns)

# --------------------------------------------------
# 6. NUMERICAL PREPROCESSING
# --------------------------------------------------

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# --------------------------------------------------
# 7. CATEGORICAL PREPROCESSING
# --------------------------------------------------

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder())
])

# --------------------------------------------------
# 8. COMBINE PREPROCESSING PIPELINES
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "Numerical Pipeline",
            num_pipeline,
            numerical_columns
        ),
        (
            "Categorical Pipeline",
            cat_pipeline,
            categorical_columns
        )
    ]
)

# --------------------------------------------------
# 9. CREATE MACHINE LEARNING PIPELINE
# --------------------------------------------------

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

# --------------------------------------------------
# 10. TRAIN THE MODEL
# --------------------------------------------------

pipeline.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")

# --------------------------------------------------
# 11. MAKE PREDICTIONS
# --------------------------------------------------

y_pred = pipeline.predict(X_test)

results = pd.DataFrame({
    "Actual Price": y_test,
    "Predicted Price": y_pred
})

print("\nActual vs Predicted Prices:")
print(results.head(10))

# --------------------------------------------------
# 12. MODEL EVALUATION
# --------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# --------------------------------------------------
# 13. SAVE TRAINED MODEL
# --------------------------------------------------

model_path = BASE_DIR / "house_price_pipeline.pkl"

if model_path.exists():
    print("\nExisting trained model preserved.")
else:
    joblib.dump(pipeline, model_path)
    print("\nModel Saved Successfully!")

# --------------------------------------------------
# 14. LOAD AND TEST SAVED MODEL
# --------------------------------------------------

loaded_model = joblib.load(model_path)

sample_predictions = loaded_model.predict(
    X_test.head(5)
)

print("\nSample Predictions from Saved Model:")
print(sample_predictions)
