import os
import io
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import lifelines

# ----- вихідна папка -----
OUTDIR = os.path.join(os.path.dirname(__file__), "outputs_cox")
os.makedirs(OUTDIR, exist_ok=True)

rng = np.random.default_rng(42)

# ---------------- 1) Генерація даних виживаності ----------------
N = 420
genres = rng.choice(["драма", "комедія", "бойовик", "мультфільм", "жахи"], size=N, p=[0.2,0.25,0.22,0.2,0.13])
start_time = rng.choice([10,12,14,16,18,20,21,22], size=N)
day = rng.choice(["Пн","Вт","Ср","Чт","Пт","Сб","Нд"], size=N, p=[0.12,0.14,0.14,0.14,0.16,0.16,0.14])
price = rng.integers(90, 160, size=N)
promo = rng.choice([0,1], size=N, p=[0.7,0.3])
capacity = rng.choice([100, 120, 150], size=N, p=[0.45,0.35,0.20])

# "справжні" коефіцієнти у лінійному предикторі ризику
b0 = -3.2
b_time = -0.10   # пізніший старт → менший ризик швидкого падіння
b_price = +0.006 # вища ціна → трохи швидше падіння
b_promo = -0.45  # промо зменшує ризик
b_capacity = +0.002

day_effect = {"Пн": +0.15, "Вт": +0.10, "Ср": +0.05, "Чт": 0.0, "Пт": -0.10, "Сб": -0.30, "Нд": -0.25}
genre_effect = {"драма": 0.0, "комедія": -0.35, "бойовик": -0.25, "мультфільм": -0.15, "жахи": +0.25}

T_MAX = 60  # днів спостереження
base_rate = rng.gamma(shape=1.4, scale=1/20.0, size=N)

linpred = (b0 + b_time * start_time + b_price * price + b_promo * promo + b_capacity * capacity
           + np.vectorize(day_effect.get)(day) + np.vectorize(genre_effect.get)(genres))
hazard_multiplier = np.exp(linpred)
rates = base_rate * hazard_multiplier
time_to_event = rng.exponential(1.0 / np.maximum(rates, 1e-9))
observed = (time_to_event <= T_MAX).astype(int)
time_obs = np.minimum(time_to_event, T_MAX).astype(float)

df = pd.DataFrame({
    "time": time_obs, "event": observed, "genre": genres, "start_time": start_time,
    "day": day, "price": price, "promo": promo, "capacity": capacity
})

# ---------------- 2) CoxPH-модель ----------------
df_model = pd.get_dummies(df, columns=["genre","day"], drop_first=True)

cph = CoxPHFitter()
cph.fit(df_model, duration_col="time", event_col="event", show_progress=False)

summary = cph.summary.copy()
summary["HR"] = np.exp(summary["coef"])
summary_out = summary[["coef","exp(coef)","HR","se(coef)","z","p","coef lower 95%","coef upper 95%"]]
summary_out.to_csv(os.path.join(OUTDIR, "cox_summary.csv"), index=True, encoding="utf-8-sig")

# Узгодженість
cindex = cph.concordance_index_

# ---------------- 3) Перевірка припущення пропорційності ----------------
print("lifelines version:", lifelines.__version__)
buf = io.StringIO()
try:
    # Новіші версії підтримують raise_on_violation
    with redirect_stdout(buf):
        cph.check_assumptions(df_model, p_value_threshold=0.05, show_plots=False, raise_on_violation=False)
except TypeError:
    # Старіші — без цього параметра
    with redirect_stdout(buf):
        cph.check_assumptions(df_model, p_value_threshold=0.05, show_plots=False)

assump_text = buf.getvalue().strip()
with open(os.path.join(OUTDIR, "assumptions.txt"), "w", encoding="utf-8") as f:
    f.write(assump_text if assump_text else "Перевірку виконано. Порушень не виявлено або вивід порожній.")

# ---------------- 4) Графіки ----------------
# 4.1 KM: вечір (≥18) vs день
km = KaplanMeierFitter()
mask_evening = df["start_time"] >= 18
T_e, E_e = df.loc[mask_evening, "time"], df.loc[mask_evening, "event"]
T_d, E_d = df.loc[~mask_evening, "time"], df.loc[~mask_evening, "event"]

plt.figure()
km.fit(T_e, E_e, label="Вечірні сеанси (≥18:00)")
ax = km.plot()
km.fit(T_d, E_d, label="Денні сеанси (<18:00)")
km.plot(ax=ax)
plt.title("Криві виживання Kaplan–Meier: вечір vs день")
plt.xlabel("Дні в прокаті")
plt.ylabel("Ймовірність «утримувати попит»")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "km_evening_vs_day.png"))
plt.close()

# Лог-ранг тест
logrank = logrank_test(T_e, T_d, event_observed_A=E_e, event_observed_B=E_d)
with open(os.path.join(OUTDIR, "logrank_evening_vs_day.txt"), "w", encoding="utf-8") as f:
    f.write(str(logrank))

# 4.2 Базова крива виживання
plt.figure()
cph.baseline_survival_.plot()
plt.title("Базова крива виживання (Cox PH)")
plt.xlabel("Час (дні)")
plt.ylabel("S0(t)")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "baseline_survival.png"))
plt.close()

# 4.3 Часткові ефекти для різних start_time
plt.figure()
for st in [14, 18, 21]:
    row = df_model.median(numeric_only=True)
    row["start_time"] = st
    surv = cph.predict_survival_function(row.to_frame().T)
    plt.plot(surv.index, surv.values, label=f"start_time={st}")
plt.title("Часткові ефекти start_time на виживання")
plt.xlabel("Час (дні)")
plt.ylabel("S(t)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "partial_effects_start_time.png"))
plt.close()

# Зберегти дані
df.head(50).to_csv(os.path.join(OUTDIR, "dataset_preview.csv"), index=False, encoding="utf-8-sig")
df.to_csv(os.path.join(OUTDIR, "dataset_full.csv"), index=False, encoding="utf-8-sig")

# ---------------- 5) Консольне резюме ----------------
print("=== COX PH: РЕЗУЛЬТАТИ ===")
print("Спостережень:", len(df), "| Частка подій:", float(df['event'].mean()))
print(f"Harrell's c-index: {cindex:.3f}")
print("\nВитяг з таблиці коефіцієнтів:")
print(summary_out.head(10).to_string())
print("\nПеревірка припущення пропорційності (див. assumptions.txt).")
print("Збережено у:", OUTDIR)
