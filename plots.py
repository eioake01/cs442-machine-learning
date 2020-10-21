import matplotlib.pyplot as plt,pandas as pd
import numpy as np


path = "/home/elenipersonal/Documents/ML/cs442-machine-learning/Assignment_2/truePositives.txt"
data = pd.read_csv(path," ")
data = np.array(data)

plt.figure(1)
plt.plot(data[:, 0], data[:, 1], label="Train True Positives Rate")
plt.plot(data[:, 0], data[:, 2], label="Test True Positives")
plt.xlabel("Epochs")
plt.ylabel("True Positives Rate Values")
plt.title("True Positives Graph")
plt.legend()
plt.show()
