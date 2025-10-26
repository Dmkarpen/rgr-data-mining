import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Дані
data = {
    "Time": [10,12,14,16,18,20,22,14,18,20],
    "Price": [80,85,90,95,100,110,90,100,120,130],
    "Visitors": [35,45,55,70,85,95,80,50,78,88]
}
df = pd.DataFrame(data)

Y = df["Visitors"]
X = df[["Time","Price"]]
X = sm.add_constant(X)

model = sm.OLS(Y,X).fit()
print(model.summary())

# Прогноз
predictions = model.predict(X)

# Графік: фактичні vs прогнозовані
plt.scatter(Y, predictions, alpha=0.7, edgecolor="k")
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], "r--", lw=2)
plt.xlabel("Фактична кількість відвідувачів")
plt.ylabel("Прогнозована кількість відвідувачів")
plt.title("Фактичні vs прогнозовані значення (лінійна регресія)")
plt.grid(True)
plt.show()
