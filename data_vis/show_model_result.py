import numpy as np

file_path = r"D:\code\data_driven_EF\data\EFNY\results\efny_schaefer400_pls_weights_20251212_170259.npz"

data = np.load(file_path)
print(data['X_scores'])

