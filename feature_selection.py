import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle

from sklearn_lr_pvalues_and_ci import get_sklearn_lr_pvalues_and_ci


# Load example dataset
data = load_breast_cancer()
X, y = data.data, data.target
# only the first 5 features to make it faster, for testing purposes
n_features = 5
X = X[:,:n_features]

# STEP 1: pick the best l1_ratio using gridsearch (cv=cv) for each step of RFECV (cv=cv), 
#         to guarantee that we are using the best model possible to select the features
#########################################################################################
# Define StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define logistic regression model with ElasticNet regularization
log_reg = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=10000)

# Create a pipeline that includes StandardScaler and RFECV
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rfecv', RFECV(estimator=log_reg, step=1, cv=cv, min_features_to_select=2, scoring='accuracy'))
])

# Define parameter grid for GridSearchCV to find the best l1_ratio for RFECV
param_grid = {
    'rfecv__estimator__l1_ratio': np.linspace(0, 1, 11) 
}

# Create GridSearchCV to find the best l1_ratio
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=10, verbose=3)

# Fit the model
print("Starting grid search...")
grid_search.fit(X, y)
print("Grid search complete.")


# Output the best parameters and best score
best_l1_ratio = grid_search.best_params_['rfecv__estimator__l1_ratio']
best_score = grid_search.best_score_

# Output the selected features
rfecv = grid_search.best_estimator_.named_steps['rfecv']
selected_features = np.where(rfecv.support_)[0]

print(f"Best l1_ratio for RFECV: {best_l1_ratio}")
print(f"Best cross-validated accuracy: {best_score}")
print(f"Number of selected features: {rfecv.n_features_}")
print(f"Selected features indices: {selected_features}")

# save the pickle file to make sure we can retrieve more info later, if necessary
with open('best_model_via_gridsearch.pickle', 'wb') as handle:
    pickle.dump(grid_search, handle, protocol=pickle.HIGHEST_PROTOCOL)




# STEP 2: Compute the final model with the chosen features and best l1_ratio for the corresponding set of features
# Extract the selected features
X_selected = X[:, selected_features]

# Standardize the input data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Fit logistic regression on the selected features
log_reg_sk = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=best_l1_ratio, max_iter=10000)
log_reg_sk.fit(X=X_scaled, y=y)

# Compute the p-values
standard_errors, z_scores_sk, p_values_sk, ci_lower_sk, ci_upper_sk = get_sklearn_lr_pvalues_and_ci(log_reg_sk=log_reg_sk, X=X_scaled)
# Get coefficients
intercept_sk = log_reg_sk.intercept_[0]
coefficients_sk = log_reg_sk.coef_.flatten()

results = pd.DataFrame({
    'Feature': ['Intercept'] + data.feature_names[selected_features].tolist(),
    'coeff': np.concatenate(([intercept_sk], coefficients_sk)),
    'std_err': standard_errors,
    'z': z_scores_sk,
    'P>|z|': p_values_sk,
    '[0.025': ci_lower_sk,
    '0.975]': ci_upper_sk
})

print('\nFinal result:')
print(results)