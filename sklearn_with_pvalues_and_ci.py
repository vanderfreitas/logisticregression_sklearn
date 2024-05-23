import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


def compute_sklearn_lr_pvalues_and_ci(log_reg_sk, X):
    """Returns the confidence intervals and p-values associated to sklearn LogisticRegression coefficients
    
    Parameters:
    log_reg_sk: fitted sklearn LogistiticRegression model
    X: input features

    Returns:
    z_scores: z statistics
    p_values: p-values associated to coefficients
    ci_lower: lower boundary for the confidence interval (ci) 
    ci_upper: upper boundary of the ci
    """

    # Get the coefficients and the intercept
    intercept = log_reg_sk.intercept_[0]
    coefficients = log_reg_sk.coef_.flatten()

    # Add an intercept column to X_selected
    X_selected_with_intercept = np.hstack((np.ones((X.shape[0], 1)), X))

    # Compute the predicted probabilities
    pred_probs = log_reg_sk.predict_proba(X)[:, 1]

    # Compute the diagonal matrix of weights for the observations
    W = np.diag(pred_probs * (1 - pred_probs))

    # Compute the Hessian matrix (second derivative of the log-likelihood)
    XWX = X_selected_with_intercept.T @ W @ X_selected_with_intercept

    # Compute the covariance matrix as the inverse of the Hessian matrix
    cov_matrix = np.linalg.inv(XWX)

    # Extract the standard errors from the diagonal of the covariance matrix
    standard_errors = np.sqrt(np.diag(cov_matrix))

    # Compute the z-scores for each coefficient
    z_scores = coefficients / standard_errors[1:]

    # Compute the p-values from the z-scores
    p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))

    # Compute the 95% confidence intervals for each coefficient
    z_value = 1.96  # for 95% confidence
    ci_lower = coefficients - z_value * standard_errors[1:]
    ci_upper = coefficients + z_value * standard_errors[1:]

    # Include the intercept in the results
    ci_lower = np.concatenate(([intercept - z_value * standard_errors[0]], ci_lower))
    ci_upper = np.concatenate(([intercept + z_value * standard_errors[0]], ci_upper))
    #p_values = np.concatenate(([2 * (1 - norm.cdf(np.abs(intercept / standard_errors[0])))], p_values))

    return z_scores, p_values, ci_lower, ci_upper




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
z_scores_sk, p_values_sk, ci_lower_sk, ci_upper_sk = compute_sklearn_lr_pvalues_and_ci(log_reg_sk=log_reg_sk, X=X_scaled)
# Get coefficients
intercept_sk = log_reg_sk.intercept_[0]
coefficients_sk = log_reg_sk.coef_.flatten()
#################################################################


# STATSMODELS ###################################################
# Add an intercept column for statsmodels
X_scaled_with_intercept = sm.add_constant(X_scaled)
# Fit logistic regression using statsmodels
log_reg_sm = sm.Logit(endog=y, exog=pd.DataFrame(data=X_scaled_with_intercept,columns=[*['intercept'],*data.feature_names.tolist()[:n_features]]), max_iter=10000)
result_sm = log_reg_sm.fit(disp=False, method='lbfgs')
# Get coefficients
params_sm = result_sm.params
print(result_sm.summary())
#################################################################

# Ensure the coefficients are aligned
results = pd.DataFrame({
    'Feature': ['Intercept'] + data.feature_names.tolist()[:n_features],
    'coeff_sk': np.concatenate(([intercept_sk], coefficients_sk)),
    'pv_sk': np.concatenate(([np.nan], p_values_sk)),
    'cil': ci_lower_sk,
    'ciu': ci_upper_sk,
    'coeff_sm': params_sm,
    'pv_sm': result_sm.pvalues
})

print('Comparing sklearn and statsmodels')
print(results)