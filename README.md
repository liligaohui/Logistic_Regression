# Regression & Logistic Regression From Scratch

This Jupyter notebook demonstrates how to implement both **linear regression** (MSE + gradient descent) and **logistic regression** (binary cross-entropy + L2-regularized gradient descent) **from first principles**, without using any high-level ML libraries. It also shows how to preprocess your data with feature normalization and inspect your model’s learned parameters and cost over training iterations.

---

1. **Data Loading**  
   - Example: load the breast cancer dataset from `sklearn.datasets`  
   - (Optionally) generate synthetic, linearly separable data

2. **Helper Functions**  
   - `normalize_features(X)` — standardize each feature to zero mean & unit variance  
   - `sigmoid(z)` — the logistic (sigma) function  

3. **Linear Regression Implementation**  
   - `compute_cost(X, y, theta)` — Mean Squared Error (MSE) cost  
   - `gradient_descent(X, y, theta, alpha, num_iters)` — batch gradient descent  

4. **Logistic Regression Implementation**  
   - `compute_cost_logistic(X, y, theta, lambda_=0.0)` — binary cross-entropy loss + L2 penalty  
   - `gradient_descent_logistic(X, y, theta, alpha, num_iters, lambda_=0.0)` — gradient descent w/ regularization  

5. **Demonstration & Results**  
   - **Linear Regression** on synthetic data (prints learned θ & final MSE)  
   - **Logistic Regression** on synthetic (or real) binary data (prints learned θ & final cross-entropy cost)  
