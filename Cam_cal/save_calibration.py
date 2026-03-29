import numpy as np

# 👉 Paste your printed values here
mtx = np.array([
    [859.4008161, 0.0, 333.99953866],
    [0.0, 862.07924239, 239.1829911],
    [0.0, 0.0, 1.0]
])

dist = np.array([[-0.08819213, 1.6079674, 0.0031731, 0.00959841, -2.71853549]])  

# Save file
np.savez("calibration_data.npz", mtx=mtx, dist=dist)

print(" calibration_data.npz created!")