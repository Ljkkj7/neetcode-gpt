import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        y_hat = X @ weights # dot matrix product sum[x1 * w1, x2*w2...]
        return np.round(y_hat, 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        mse = np.sum((model_prediction - ground_truth)**2/len(ground_truth))
        return np.round(mse, 5)
