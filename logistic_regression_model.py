
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (log_loss, roc_curve, roc_auc_score, confusion_matrix,
                             accuracy_score, precision_score, recall_score)

OUTDIR = os.path.join(os.path.dirname(__file__), "outputs_balanced")
os.makedirs(OUTDIR, exist_ok=True)

HIT_THRESH = 0.80

def generate_dataset(n=360, seed=123, boost_scale=1.0):
    rng = np.random.default_rng(seed)
    times = rng.choice([10,12,14,16,18,20,21,22], size=n)
    days  = rng.choice(["Пн","Вт","Ср","Чт","Пт","Сб","Нд"], size=n,
                       p=[0.12,0.14,0.14,0.14,0.16,0.16,0.14])
    genres = rng.choice(["драма","комедія","бойовик","мультфільм","жахи"], size=n,
                        p=[0.18,0.28,0.22,0.20,0.12])
    price = rng.integers(90,160, size=n)
    capacity = rng.choice([100,120,150], size=n, p=[0.45,0.35,0.20])

    base = (0.10
            + 0.06 * ((times >= 16) & (times < 18)) * boost_scale
            + 0.30 * (times >= 18) * boost_scale
            + 0.05 * np.isin(days, ["Пт"]) * boost_scale
            + 0.18 * np.isin(days, ["Сб","Нд"]) * boost_scale)
    genre_boost = (0.14*(genres=="комедія")*boost_scale
                   + 0.11*(genres=="бойовик")*boost_scale
                   + 0.09*(genres=="мультфільм")*boost_scale
                   - 0.02*(genres=="жахи")
                   + 0.00*(genres=="драма"))
    price_effect = -0.0010 * (price - 110)

    p_hit = np.clip(base + genre_boost + price_effect, 0.02, 0.98)
    noise = rng.normal(0, 0.035, size=n)
    fill_ratio = np.clip(p_hit + noise, 0, 1)

    visitors = np.round(fill_ratio * capacity).astype(int)
    df = pd.DataFrame({
        "time": times, "day": days, "genre": genres, "price": price,
        "capacity": capacity, "visitors": visitors,
        "fill_ratio": (visitors / capacity).round(3)
    })
    return df

def generate_target_prevalence(n=360, base_seed=500, target_low=0.25, target_high=0.35, max_tries=60):
    boost = 0.9
    for k in range(max_tries):
        df = generate_dataset(n=n, seed=base_seed + k, boost_scale=boost)
        y = (df["visitors"] >= HIT_THRESH * df["capacity"]).astype(int)
        p = y.mean()
        if target_low <= p <= target_high and y.sum() >= 30:
            df["hit"] = y
            return df, boost, p, np.bincount(y, minlength=2)
        if p < target_low:
            boost += 0.05
        else:
            boost -= 0.03
        boost = max(0.6, min(1.8, boost))
    df["hit"] = (df["visitors"] >= HIT_THRESH * df["capacity"]).astype(int)
    return df, boost, df["hit"].mean(), np.bincount(df["hit"], minlength=2)

df, used_boost, prevalence, counts = generate_target_prevalence()
print(f"Підібрано boost_scale={used_boost:.2f}; частка аншлагів ≈ {prevalence:.3f}")
print("Розмір:", df.shape, "| Розподіл класів:", {0:int(counts[0]), 1:int(counts[1])})

plt.figure()
plt.hist(df["fill_ratio"], bins=20)
plt.xlabel("Заповнюваність залу (частка)")
plt.ylabel("Кількість сеансів")
plt.title("Гістограма заповнюваності залів")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "hist_fill_ratio.png"))
plt.close()

plt.figure()
plt.scatter(df["time"], df["fill_ratio"], s=18)
plt.xlabel("Час початку сеансу (год)")
plt.ylabel("Заповнюваність залу (частка)")
plt.title("Діаграма розсіювання: час vs заповнюваність")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "scatter_time_fill.png"))
plt.close()

X = df[["time","day","genre","price"]]
y = df["hit"].values
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first"), ["day","genre"]),
    ("num", "passthrough", ["time","price"])
])
pipe = Pipeline([("pre", preprocess), ("clf", LogisticRegression(max_iter=500, solver="lbfgs"))])

import numpy as np
strat = y if (np.unique(y).size > 1 and np.bincount(y).min() >= 2) else None
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=7, stratify=strat)
pipe.fit(X_train, y_train)

proba_train = pipe.predict_proba(X_train)[:,1]
proba_test  = pipe.predict_proba(X_test)[:,1]

from sklearn.metrics import log_loss, roc_curve, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score
LL_model = -log_loss(y_train, proba_train, normalize=False)
LL_null  = -log_loss(y_train, np.full_like(proba_train, y_train.mean()), normalize=False)
minus2LL_model = -2*LL_model
minus2LL_null  = -2*LL_null
chi2_impr = minus2LL_null - minus2LL_model

n_train = y_train.shape[0]
cox_snell_r2 = 1 - np.exp((2/n_train) * (LL_model - LL_null))
nagelkerke_r2 = cox_snell_r2 / (1 - np.exp((2/n_train) * (0 - LL_null)))

y_pred_05 = (proba_test >= 0.5).astype(int)
acc_05  = accuracy_score(y_test, y_pred_05)
prec_05 = precision_score(y_test, y_pred_05, zero_division=0)
rec_05  = recall_score(y_test, y_pred_05, zero_division=0)
cm_05   = confusion_matrix(y_test, y_pred_05)

fpr, tpr, thr = roc_curve(y_test, proba_test)
auc = roc_auc_score(y_test, proba_test)
best_idx = np.argmax(tpr - fpr)
best_thr = thr[best_idx]
y_pred_best = (proba_test >= best_thr).astype(int)
acc_b  = accuracy_score(y_test, y_pred_best)
prec_b = precision_score(y_test, y_pred_best, zero_division=0)
rec_b  = recall_score(y_test, y_pred_best, zero_division=0)
cm_b   = confusion_matrix(y_test, y_pred_best)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC-крива (логістична регресія, «аншлаг» @80%)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "roc_curve.png"))
plt.close()

ohe = pipe.named_steps["pre"].named_transformers_["cat"]
features = list(ohe.get_feature_names_out(["day","genre"])) + ["time","price"]
coef = pipe.named_steps["clf"].coef_.flatten()
odds = np.exp(coef)
coef_df = pd.DataFrame({"Ознака": features, "β": np.round(coef,4), "OR": np.round(odds,4)})
coef_df.to_csv(os.path.join(OUTDIR, "coef_table.csv"), index=False, encoding="utf-8-sig")

print("\n=== РЕЗУЛЬТАТИ МОДЕЛІ (ціль 25–35% аншлагів @80%) ===")
print(f"-2LL (модель): {minus2LL_model:.3f} | -2LL (нульова): {minus2LL_null:.3f} | χ²-покращення: {chi2_impr:.3f}")
print(f"Cox & Snell R²: {cox_snell_r2:.3f} | Nagelkerke R²: {nagelkerke_r2:.3f} | AUC: {auc:.3f} | Поріг Юдена: {best_thr:.3f}")
print("Матриця @0.50:\n", cm_05, f"  Acc={acc_05:.3f} Prec={prec_05:.3f} Rec={rec_05:.3f}")
print("Матриця @Юден:\n", cm_b,  f"  Acc={acc_b:.3f} Prec={prec_b:.3f} Rec={rec_b:.3f}")
print(f"\nФайли: {OUTDIR}\\hist_fill_ratio.png, scatter_time_fill.png, roc_curve.png, coef_table.csv")
