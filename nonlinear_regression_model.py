import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

# Дані
X = np.array([10, 12, 14, 16, 18, 20, 22], dtype=float)
Y = np.array([30, 45, 55, 70, 85, 95, 80], dtype=float)

results = {}

# --- 1. Поліном 2-го ступеня ---
coeffs2 = np.polyfit(X, Y, 2)
poly2 = np.poly1d(coeffs2)
Y_pred2 = poly2(X)
results["Поліном 2-го ст."] = r2_score(Y, Y_pred2)

# --- 2. Поліном 3-го ступеня ---
coeffs3 = np.polyfit(X, Y, 3)
poly3 = np.poly1d(coeffs3)
Y_pred3 = poly3(X)
results["Поліном 3-го ст."] = r2_score(Y, Y_pred3)

# --- 3. Гіперболічна ---
def hyperbola(x, a, b):
    return a + b/x
params_hyp, _ = curve_fit(hyperbola, X, Y, maxfev=5000)
Y_pred_hyp = hyperbola(X, *params_hyp)
results["Гіперболічна"] = r2_score(Y, Y_pred_hyp)

# --- 4. Степенева ---
def power_func(x, a, b):
    return a * (x**b)
params_pow, _ = curve_fit(power_func, X, Y, maxfev=5000)
Y_pred_pow = power_func(X, *params_pow)
results["Степенева"] = r2_score(Y, Y_pred_pow)

# --- 5. Експоненціальна ---
def exp_func(x, a, b):
    return a * np.exp(b*x)
params_exp, _ = curve_fit(exp_func, X, Y, maxfev=5000)
Y_pred_exp = exp_func(X, *params_exp)
results["Експоненціальна"] = r2_score(Y, Y_pred_exp)

# Вивід коефіцієнтів
print("Поліном 2-го:", coeffs2)
print("Поліном 3-го:", coeffs3)
print("Гіпербола:", params_hyp)
print("Степенева:", params_pow)
print("Експоненціальна:", params_exp)
print("R2:", results)

# Графік
plt.scatter(X, Y, color="black", label="Дані")
x_range = np.linspace(10, 22, 200)
plt.plot(x_range, poly2(x_range), label="Поліном 2-го ст.")
plt.plot(x_range, poly3(x_range), label="Поліном 3-го ст.")
plt.plot(x_range, hyperbola(x_range, *params_hyp), label="Гіперболічна")
plt.plot(x_range, power_func(x_range, *params_pow), label="Степенева")
plt.plot(x_range, exp_func(x_range, *params_exp), label="Експоненціальна")
plt.xlabel("Час сеансу (години)")
plt.ylabel("Відвідувачі")
plt.title("Нелінійні моделі: Час vs Відвідувачі")
plt.legend()
plt.grid(True)
plt.show()