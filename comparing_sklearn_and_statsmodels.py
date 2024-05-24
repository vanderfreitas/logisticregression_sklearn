import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, precision_recall_fscore_support
import statsmodels.api as sm

from sklearn_lr_pvalues_and_ci import get_sklearn_lr_pvalues_and_ci


# Load example dataset
data = load_breast_cancer()
X, y = data.data, data.target
# only the first 5 features to make it faster, for testing purposes
n_features = 5
X = X[:,:n_features]


# Standardize the input data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# SKLEARN: Fit logistic regression using scikit-learn ###########
log_reg_sk = LogisticRegression(penalty=None, solver='lbfgs', intercept_scaling=1, max_iter=10000)
log_reg_sk.fit(X=X_scaled, y=y)
# Compute the p-values
standard_errors, z_scores_sk, p_values_sk, ci_lower_sk, ci_upper_sk = get_sklearn_lr_pvalues_and_ci(log_reg_sk=log_reg_sk, X=X_scaled)
# Get coefficients
intercept_sk = log_reg_sk.intercept_[0]
coefficients_sk = log_reg_sk.coef_.flatten()
#################################################################


# STATSMODELS ###################################################
# Add an intercept column for statsmodels
X_scaled_with_intercept = sm.add_constant(X_scaled)
# Fit logistic regression using statsmodels
log_reg_sm = sm.Logit(endog=y, exog=pd.DataFrame(data=X_scaled_with_intercept,columns=[*['intercept'],*data.feature_names.tolist()[:n_features]]))
result_sm = log_reg_sm.fit(disp=False, method='lbfgs',maxiter=10000)
print('STATSMODELS')
print(result_sm.summary())
#################################################################

results = pd.DataFrame({
    'Feature': ['Intercept'] + data.feature_names.tolist()[:n_features],
    'coeff': np.concatenate(([intercept_sk], coefficients_sk)),
    'std_err': standard_errors,
    'z': z_scores_sk,
    'P>|z|': p_values_sk,
    '[0.025': ci_lower_sk,
    '0.975]': ci_upper_sk
})

print('\n\nSKLEARN')
print(results)



print('\n\nEvaluate model performance')

y_pred = log_reg_sk.predict(X_scaled)
#y_pred_proba = log_reg_sk.predict_proba(X_scaled)[:, 1]

# Evaluate the model's performance
accuracy = accuracy_score(y, y_pred)
#conf_matrix = confusion_matrix(y, y_pred)
#class_report = classification_report(y, y_pred)
precision, recall, fbeta_score, support = precision_recall_fscore_support(y,y_pred)
#roc_auc = roc_auc_score(y, y_pred_proba)

results = pd.DataFrame(data={
    'accuracy': [accuracy],
    'precision': [precision],
    'recall': [recall],
    'fbeta_score': [fbeta_score],
    'support': [support],
})
print(results)