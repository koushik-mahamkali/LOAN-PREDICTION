import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix

df = pd.read_csv("loan_approval_dataset.csv")

df.columns = df.columns.str.strip()

df.drop(columns=['loan_id'],inplace=True)

df['assets'] = df.residential_assets_value + df.commercial_assets_value + df.luxury_assets_value + df.bank_asset_value

df.drop(columns=['residential_assets_value','commercial_assets_value','luxury_assets_value','bank_asset_value'],inplace=True)

df.isnull().sum()

def clean(str):
    str = str.strip()
    return str
    

df.education = df.education.apply(clean)
df['education'] = df['education'].replace(['Graduate', 'Not Graduate'],[1,0])

df.self_employed = df.self_employed.apply(clean)
df['self_employed'] = df['self_employed'].replace(['Yes', 'No'],[1,0])

df.loan_status = df.loan_status.apply(clean)
df['loan_status'] = df['loan_status'].replace(['Approved', 'Rejected'],[1,0])

label_encoders = {}
for col in ["education", "self_employed", "loan_status"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le 

X = df.drop(columns=["loan_status"])
y = df["loan_status"]

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)


logreg_model = LogisticRegression()
logreg_model.fit(X_train_scaled, y_train)

logreg_predictions_train = logreg_model.predict_proba(X_train_scaled)[:, 1]  # Get probability of class 1
logreg_predictions_test = logreg_model.predict_proba(X_test_scaled)[:, 1] 

X_train_final = np.column_stack((X_train_scaled, logreg_predictions_train))
X_test_final = np.column_stack((X_test_scaled, logreg_predictions_test))

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_final, y_train)

y_pred = xgb_model.predict(X_test_final)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

def predict_loan_approval():
    # Taking user input for features in the dataset
    education = input("Are you a graduate? (Yes/No): ")
    self_employed = input("Are you self-employed? (Yes/No): ")
    no_of_dependents = int(input("Enter the number of dependents: "))
    income_annum = float(input("Enter your annual income: "))
    loan_amount = float(input("Enter your loan amount: "))
    loan_term = int(input("Enter loan term (in years): "))
    cibil_score = int(input("Enter your CIBIL score (e.g., 600, 750): "))
    assets = int(input("Enter your total Assets which includes residential,commercial,luxury and bank assets: "))
    

    # Manually encode the inputs as 1 (Yes) or 0 (No)
    education_encoded = 1 if education.lower() == 'yes' else 0
    self_employed_encoded = 1 if self_employed.lower() == 'yes' else 0

    # Create a feature array with the provided input
    input_data = np.array([[education_encoded, self_employed_encoded, no_of_dependents,
                            income_annum, loan_amount, loan_term, cibil_score, assets]])

    # Standardize the input data
    input_data_scaled = scaler.transform(input_data)

    # Use Logistic Regression model to predict probabilities for the input data
    logreg_pred = logreg_model.predict_proba(input_data_scaled)[:, 1]  # Get probability of class 1

    # Append Logistic Regression prediction to the input data as an additional feature
    input_data_final = np.column_stack((input_data_scaled, logreg_pred))

    # Make the final prediction using XGBoost
    prediction = xgb_model.predict(input_data_final)

    # Output the result
    if prediction[0] == 1:
        print("Congratulaions, Your loan is Approved.")
    else:
        print("Sorry, Your loan is Denied.")

# Call the function to predict loan approval
predict_loan_approval()

