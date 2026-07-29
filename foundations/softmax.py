import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        exp_sum = sum(np.round(np.exp(z-max(z)), 4))
        output = []
        for i in z:
            output.append(float(np.round((np.exp(i-max(z))/exp_sum), 4)))
        return output
        

