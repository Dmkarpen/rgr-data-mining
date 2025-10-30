import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# -------------------------------
# 1) Універсуми змінних
# -------------------------------
pop_univ   = np.linspace(0, 10, 101)     # x1: популярність фільму (0..10)
price_univ = np.linspace(0, 300, 121)    # x2: ціна квитка (0..300 грн)
occ_univ   = np.linspace(0, 100, 101)    # y: заповненість залу (0..100%)

# -------------------------------
# Допоміжна функція для збереження 3D поверхні
# -------------------------------
def plot_surface(X, Y, Z, xlabel, ylabel, zlabel, title, fname):
    fig = plt.figure(figsize=(8, 6), dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(fname, bbox_inches='tight')
    plt.show()

# ===========================================================
# 2) Модель MAMDANI (skfuzzy.control)
# ===========================================================
# Антецеденти та консеквент
pop_m = ctrl.Antecedent(pop_univ, 'popularity')
price_m = ctrl.Antecedent(price_univ, 'price')
occ_m = ctrl.Consequent(occ_univ, 'occupancy')

# Функції належності (інтерпретовані та масштабовані під предметну область)
# Популярність
pop_m['low']    = fuzz.trapmf(pop_univ,  [0, 0, 2.5, 4.0])
pop_m['medium'] = fuzz.trimf(pop_univ,   [3.0, 5.0, 7.0])
pop_m['high']   = fuzz.trapmf(pop_univ,  [6.0, 8.0, 10, 10])

# Ціна
price_m['low']    = fuzz.trapmf(price_univ,  [0, 0, 80, 140])
price_m['medium'] = fuzz.trimf(price_univ,   [100, 170, 240])
price_m['high']   = fuzz.trapmf(price_univ,  [200, 260, 300, 300])

# Заповненість
occ_m['low']    = fuzz.trapmf(occ_univ,  [0, 0, 25, 45])
occ_m['medium'] = fuzz.trimf(occ_univ,   [35, 55, 75])
occ_m['high']   = fuzz.trapmf(occ_univ,  [65, 85, 100, 100])

# Правила (логіка: висока популярність і низька ціна -> висока заповненість тощо)
rule1 = ctrl.Rule(antecedent=(pop_m['high'] & price_m['low']),    consequent=occ_m['high'])
rule2 = ctrl.Rule(antecedent=(pop_m['high'] & price_m['medium']), consequent=occ_m['high'])
rule3 = ctrl.Rule(antecedent=(pop_m['high'] & price_m['high']),   consequent=occ_m['medium'])

rule4 = ctrl.Rule(antecedent=(pop_m['medium'] & price_m['low']),    consequent=occ_m['high'])
rule5 = ctrl.Rule(antecedent=(pop_m['medium'] & price_m['medium']), consequent=occ_m['medium'])
rule6 = ctrl.Rule(antecedent=(pop_m['medium'] & price_m['high']),   consequent=occ_m['low'])

rule7 = ctrl.Rule(antecedent=(pop_m['low'] & price_m['low']),    consequent=occ_m['medium'])
rule8 = ctrl.Rule(antecedent=(pop_m['low'] & price_m['medium']), consequent=occ_m['low'])
rule9 = ctrl.Rule(antecedent=(pop_m['low'] & price_m['high']),   consequent=occ_m['low'])

mamdani_sys = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
mamdani_sim = ctrl.ControlSystemSimulation(mamdani_sys)

def mamdani_predict(pop_val, price_val):
    mamdani_sim.input['popularity'] = float(pop_val)
    mamdani_sim.input['price'] = float(price_val)
    mamdani_sim.compute()
    return mamdani_sim.output['occupancy']

def mamdani_surface(nx=41, ny=41):
    X = np.linspace(pop_univ.min(), pop_univ.max(), nx)
    Y = np.linspace(price_univ.min(), price_univ.max(), ny)
    XX, YY = np.meshgrid(X, Y)
    ZZ = np.zeros_like(XX, dtype=float)
    # Обчислюємо точку за точкою (ControlSystemSimulation зберігає стан)
    for i in range(YY.shape[0]):
        for j in range(XX.shape[1]):
            ZZ[i, j] = mamdani_predict(XX[0, j], YY[i, 0])
    return XX, YY, ZZ

# ===========================================================
# 3) Модель SUGENO (власна реалізація, перший порядок)
# ===========================================================
# Візьмемо ті ж самі МФ для антецедентів, що й у Mamdani
# (використаємо interp_membership для отримання ступенів належності)

def mu_pop(x):
    return {
        'low':    fuzz.interp_membership(pop_univ,   pop_m['low'].mf,    x),
        'medium': fuzz.interp_membership(pop_univ,   pop_m['medium'].mf, x),
        'high':   fuzz.interp_membership(pop_univ,   pop_m['high'].mf,   x),
    }

def mu_price(x):
    return {
        'low':    fuzz.interp_membership(price_univ, price_m['low'].mf,    x),
        'medium': fuzz.interp_membership(price_univ, price_m['medium'].mf, x),
        'high':   fuzz.interp_membership(price_univ, price_m['high'].mf,   x),
    }

# Правила Сугено (лінійні наслідки z = a*pop + b*price + c)
# Налаштовані під інтуїцію предметки (масштаби підібрані під 0..100)
def z1(pop, price):  # pop high & price low -> висока
    return 8.0*pop - 0.10*price + 40.0

def z2(pop, price):  # pop high & price high -> середня
    return 7.0*pop - 0.20*price + 30.0

def z3(pop, price):  # pop medium & price medium -> середня
    return 6.0*pop - 0.15*price + 35.0

def z4(pop, price):  # pop medium & price low -> висока
    return 7.0*pop - 0.08*price + 35.0

def z5(pop, price):  # pop low & price low -> середня
    return 4.0*pop - 0.05*price + 30.0

def z6(pop, price):  # pop low & price high -> низька (константа)
    return 10.0

# Опис правил як (умова -> функція наслідку)
# Використаємо t-норму min для активації
_sugeno_rules = [
    (('high', 'low'),    z1),
    (('high', 'high'),   z2),
    (('medium', 'medium'), z3),
    (('medium', 'low'),  z4),
    (('low', 'low'),     z5),
    (('low', 'high'),    z6),
]

def sugeno_predict(pop_val, price_val, clip_to=(0,100)):
    mup = mu_pop(pop_val)
    mupc = mu_price(price_val)

    alphas = []
    zs = []
    for (p_lab, pr_lab), zfun in _sugeno_rules:
        alpha = min(mup[p_lab], mupc[pr_lab])  # t-норма min
        if alpha > 0.0:
            alphas.append(alpha)
            zs.append(zfun(pop_val, price_val))
    if not alphas:
        return 0.0
    # Нормалізована зважена сума (класичний Sugeno)
    y = np.dot(alphas, zs) / (np.sum(alphas) + 1e-12)
    y = float(np.clip(y, clip_to[0], clip_to[1]))
    return y

def sugeno_surface(nx=41, ny=41):
    X = np.linspace(pop_univ.min(), pop_univ.max(), nx)
    Y = np.linspace(price_univ.min(), price_univ.max(), ny)
    XX, YY = np.meshgrid(X, Y)
    ZZ = np.zeros_like(XX, dtype=float)
    for i in range(YY.shape[0]):
        for j in range(XX.shape[1]):
            ZZ[i, j] = sugeno_predict(XX[0, j], YY[i, 0])
    return XX, YY, ZZ

# ===========================================================
# 4) Побудова поверхонь і приклади розрахунків
# ===========================================================
if __name__ == "__main__":
    # --- Поверхня Mamdani
    XXm, YYm, ZZm = mamdani_surface(nx=45, ny=45)
    plot_surface(XXm, YYm, ZZm,
                 xlabel='Популярність (0..10)',
                 ylabel='Ціна (грн)',
                 zlabel='Заповненість (%)',
                 title='Mamdani: поверхня "входи → вихід"',
                 fname='mamdani_surface.png')

    # --- Поверхня Sugeno
    XXs, YYs, ZZs = sugeno_surface(nx=45, ny=45)
    plot_surface(XXs, YYs, ZZs,
                 xlabel='Популярність (0..10)',
                 ylabel='Ціна (грн)',
                 zlabel='Заповненість (%)',
                 title='Sugeno: поверхня "входи → вихід"',
                 fname='sugeno_surface.png')

    # --- Контрольні точки (можеш змінити під свій звіт)
    test_points = [
        {'pop': 9.0, 'price': 80.0},
        {'pop': 8.5, 'price': 220.0},
        {'pop': 5.0, 'price': 150.0},
        {'pop': 3.0, 'price': 90.0},
        {'pop': 2.0, 'price': 260.0},
    ]
    print("\nПриклади прогнозів (Заповненість, %):")
    print("pop\tprice\tMamdani\tSugeno")
    for tp in test_points:
        y_m = mamdani_predict(tp['pop'], tp['price'])
        y_s = sugeno_predict(tp['pop'], tp['price'])
        print(f"{tp['pop']:.1f}\t{tp['price']:.0f}\t{y_m:6.2f}\t{y_s:6.2f}")

    # --- Порівняння зрізів по популярності при фіксованій ціні
    fixed_price = 150.0
    xs = np.linspace(0, 10, 101)
    ys_m = np.array([mamdani_predict(x, fixed_price) for x in xs])
    ys_s = np.array([sugeno_predict(x, fixed_price) for x in xs])

    plt.figure(figsize=(7,4), dpi=120)
    plt.plot(xs, ys_m, label='Mamdani')
    plt.plot(xs, ys_s, label='Sugeno', linestyle='--')
    plt.xlabel('Популярність (0..10)')
    plt.ylabel('Заповненість (%)')
    plt.title(f'Порівняння зрізу при ціні {fixed_price:.0f} грн')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('slice_price150.png', bbox_inches='tight')
    plt.show()

    # --- Порівняння зрізів по ціні при фіксованій популярності
    fixed_pop = 7.0
    ys_m2 = np.array([mamdani_predict(fixed_pop, p) for p in price_univ])
    ys_s2 = np.array([sugeno_predict(fixed_pop, p) for p in price_univ])

    plt.figure(figsize=(7,4), dpi=120)
    plt.plot(price_univ, ys_m2, label='Mamdani')
    plt.plot(price_univ, ys_s2, label='Sugeno', linestyle='--')
    plt.xlabel('Ціна (грн)')
    plt.ylabel('Заповненість (%)')
    plt.title(f'Порівняння зрізу при популярності {fixed_pop:.1f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('slice_pop7.png', bbox_inches='tight')
    plt.show()

    print("\nФайли графіків збережено як:")
    print(" - mamdani_surface.png")
    print(" - sugeno_surface.png")
    print(" - slice_price150.png")
    print(" - slice_pop7.png")
