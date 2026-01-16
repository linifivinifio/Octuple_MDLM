import numpy as np

try:
    data = np.load("data/POP909_melody_octuple.npy", allow_pickle=True)
    sample = data[0]
    print("Sample 0 Bar IDs (first 20 rows):")
    print(sample[:20, 0])
    print("Max Bar ID in Sample 0:", sample[:, 0].max())
    
    sample = data[10]
    print("\nSample 10 Bar IDs (first 20 rows):")
    print(sample[:20, 0])
    print("Max Bar ID in Sample 10:", sample[:, 0].max())

except Exception as e:
    print(e)
