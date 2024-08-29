import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump

# Step 1: Read data from Excel
data = pd.read_excel('chlorophyl_file.xlsx')

# Step 2: Separate target and input variables
output_variable = data['chlorophyl']  
input_variables = data.drop(columns=['chlorophyl'])  

# Step 3: Split the data into training, testing, and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(input_variables, output_variable, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 4: Use SVM for regression with hyperparameters
svr = SVR()
svr_params = {'C': [0.1, 1, 10], 'epsilon': [0.01, 0.1, 0.5], 'kernel': ['linear', 'rbf']}
svr_grid = GridSearchCV(svr, svr_params, cv=5)
svr_grid.fit(X_train, y_train)

# Evaluate SVM model
y_pred_val_svr = svr_grid.predict(X_val)
mse_svr = mean_squared_error(y_val, y_pred_val_svr)
print(f"SVM Validation MSE: {mse_svr}")

# Step 5: Use Random Forest for regression with hyperparameters
rf = RandomForestRegressor()
rf_params = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
rf_grid = GridSearchCV(rf, rf_params, cv=5)
rf_grid.fit(X_train, y_train)

# Evaluate Random Forest model
y_pred_val_rf = rf_grid.predict(X_val)
mse_rf = mean_squared_error(y_val, y_pred_val_rf)
print(f"Random Forest Validation MSE: {mse_rf}")

# Step 6: Save both models
dump(svr_grid.best_estimator_, 'svm_model.joblib')
dump(rf_grid.best_estimator_, 'random_forest_model.joblib')
