import numpy as np
from scipy.stats import zscore

def get_outliers(data: np.ndarray) -> list:
    z_scores = zscore(data)
    threshold = 2
    return data[np.abs(z_scores) > threshold]
    
if __name__ == "__main__":
    data = np.array([400, 450, 440, 420, 420, 310])  # 100 is likely an outlier
    print("Starting outlier detection")
    print("Got these outliers: ", get_outliers(data))