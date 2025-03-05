# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import pickle
import os  # Import os to handle folder creation

# Step 2: Load the Dataset
# Replace 'big data 100000 values 1 lac.xlsx' with the actual file name
df = pd.read_excel("big data 100000 values 1 lac.xlsx")

# Display the first few rows to verify
print(df.head())

# Step 3: Preprocess the Data
# Drop rows with missing values
df.dropna(inplace=True)

# Define features (X) and target (y)
# Features: Feed Rate, Depth of Cut, Spindle Speed
# Target: power(watt)
X = df[['Feed Rate (mm/rev)', 'Depth of Cut (mm)', 'Spindle Speed (RPM)']]
y = df['power(watt)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Scale the Data
# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data using the same scaler
X_test_scaled = scaler.transform(X_test)

# Step 5: Train the Linear Regression Model
# Initialize the Linear Regression model
lin_reg = LinearRegression()

# Train the model on the scaled training data
lin_reg.fit(X_train_scaled, y_train)

# Evaluate the model (optional)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = lin_reg.predict(X_test_scaled)
print("Linear Regression Metrics:")
print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"R²: {r2_score(y_test, y_pred)}")

# Step 6: Train the Logistic Regression Model
# Define a function to classify the power source
def classify_power_source(power):
    if 1600 <= power <= 2200:
        return "Solar"
    else:
        return "K-Electric"

# Apply the classification function to the target variable
df['Power_Source'] = df['power(watt)'].apply(classify_power_source)

# Update the target variable for classification
y_class = df['Power_Source'].map({"Solar": 1, "K-Electric": 0})

# Split the data for classification
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Scale the data for classification
X_train_c_scaled = scaler.transform(X_train_c)
X_test_c_scaled = scaler.transform(X_test_c)

# Initialize the Logistic Regression model
log_reg = LogisticRegression()

# Train the model on the scaled training data
log_reg.fit(X_train_c_scaled, y_train_c)

# Evaluate the model (optional)
from sklearn.metrics import accuracy_score, confusion_matrix

y_pred_c = log_reg.predict(X_test_c_scaled)
print("Logistic Regression Metrics:")
print(f"Accuracy: {accuracy_score(y_test_c, y_pred_c)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test_c, y_pred_c)}")

# Step 7: Save the Models and Scaler
# Ensure the folder exists
folder_name = "cnc_web_app"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Save Linear Regression Model
with open(os.path.join(folder_name, 'trained_model.pkl'), 'wb') as file:
    pickle.dump(lin_reg, file)

# Save Logistic Regression Model
with open(os.path.join(folder_name, 'power_classification_model.pkl'), 'wb') as file:
    pickle.dump(log_reg, file)

# Save Scaler
with open(os.path.join(folder_name, 'scaler.pkl'), 'wb') as file:
    pickle.dump(scaler, file)

print("Models and scaler saved successfully in the 'cnc_web_app' folder!")