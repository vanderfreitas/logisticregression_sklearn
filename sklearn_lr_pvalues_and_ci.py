import numpy as np
from scipy.stats import norm


def get_sklearn_lr_pvalues_and_ci(log_reg_sk, X):
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
    coefficients = np.concatenate(([intercept], log_reg_sk.coef_.flatten())) #log_reg_sk.coef_.flatten()

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
    z_scores = coefficients / standard_errors

    # Compute the p-values from the z-scores
    p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))

    # Compute the 95% confidence intervals for each coefficient
    z_value = 1.96  # for 95% confidence
    ci_lower = coefficients - z_value * standard_errors
    ci_upper = coefficients + z_value * standard_errors

    return standard_errors, z_scores, p_values, ci_lower, ci_upper