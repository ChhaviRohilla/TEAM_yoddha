import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load data
data = pd.read_csv("https://raw.githubusercontent.com/sunnysavita10/credit_card_pw_hindi/main/creditCardFraud_28011964_120214.csv")

# Preprocess data
X = data.drop(labels=["default payment next month"], axis=1)
y = data["default payment next month"]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=50)
train_scaler = StandardScaler()
scaled_train_data = train_scaler.fit_transform(x_train)

# Train Naive Bayes model
clf = GaussianNB(var_smoothing=0.5)
clf.fit(scaled_train_data, y_train)
# Save Naive Bayes model
joblib.dump(clf, 'models/naive_bayes_model.joblib')

# Train XGBoost model
xgb = XGBClassifier(max_depth=4, n_estimators=90, random_state=0)
xgb.fit(scaled_train_data, y_train)
# Save XGBoost model
joblib.dump(xgb, 'models/xgboost_model.joblib')

# Save the scaler
joblib.dump(train_scaler, 'models/train_scaler.joblib')

print("Models and scaler have been trained and saved successfully.")
