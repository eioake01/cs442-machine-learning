import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
xmin, xmax = -0.5, 1.5
X = np.arange(xmin, xmax, 0.1)
ax.scatter(0, 0, color="r")
ax.scatter(0, 1, color="g")
ax.scatter(1, 0, color="g")
ax.scatter(1, 1, color="r")
ax.set_xlim([xmin, xmax])
ax.set_ylim([-0.1, 1.1])
ax.set_title("Figure 2: XOR Seperation")
m = -1
ax.plot(X, m * X + 1.2, label="decision boundary", color = "k")
ax.plot(X, m * X + 0.8, label="decision boundary", color = "k")
plt.show()