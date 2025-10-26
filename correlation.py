import pandas as pd
from scipy.stats import shapiro, pearsonr, spearmanr
import matplotlib.pyplot as plt

x = [10, 12, 14, 16, 18, 20, 22]
y = [30, 45, 55, 70, 85, 95, 80]

plt.plot(x, y, marker="o", linestyle="-", color="blue")
plt.title("Відвідуваність кіносеансів протягом дня")
plt.xlabel("Час сеансу (години)")
plt.ylabel("Відвідувачі")
plt.grid(True)
plt.show()

px = shapiro(x)[1]; py = shapiro(y)[1]
if px > 0.05 and py > 0.05:
    r, p = pearsonr(x, y); method = "Pearson"
else:
    r, p = spearmanr(x, y); method = "Spearman"

print(f"{method}: r={r:.3f}, p={p:.4f}")
