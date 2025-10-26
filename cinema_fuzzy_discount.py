import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os

# -----------------------------
# Utility: triangular membership
# -----------------------------
def trimf(x, params):
    a, b, c = params
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x, dtype=float)
    
    idx = (a < x) & (x <= b)
    if b != a:
        y[idx] = (x[idx] - a) / (b - a)
    else:
        y[idx] = 1.0
    
    y[x == b] = 1.0
    
    idx = (b < x) & (x < c)
    if c != b:
        y[idx] = (c - x[idx]) / (c - b)
    else:
        y[idx] = 1.0
    
    if a == b:
        y[x <= b] = 1.0
    if b == c:
        y[x >= b] = 1.0
    return np.clip(y, 0.0, 1.0)

# -----------------------------
# Universes and MFs
# -----------------------------
u_demand = np.linspace(0, 100, 1001)
u_lead   = np.linspace(0, 72,  721)
u_disc   = np.linspace(0, 30,  601)

demand_mfs = {
    'low' : lambda x: trimf(x, (0, 0, 40)),
    'mid' : lambda x: trimf(x, (20, 50, 80)),
    'high': lambda x: trimf(x, (60, 100, 100))
}

lead_mfs = {
    'short': lambda x: trimf(x, (0, 0, 12)),
    'mid'  : lambda x: trimf(x, (6, 24, 48)),
    'long' : lambda x: trimf(x, (36, 72, 72))
}

disc_mfs = {
    'none'  : lambda x: trimf(x, (0, 0, 10)),
    'medium': lambda x: trimf(x, (5, 15, 25)),
    'high'  : lambda x: trimf(x, (20, 30, 30))
}

# -----------------------------
# Rule base (Mamdani)
# -----------------------------
rules = [
    ('low',  'short', 'high'),
    ('low',  'mid',   'high'),
    ('low',  'long',  'medium'),
    ('mid',  'short', 'medium'),
    ('mid',  'mid',   'medium'),
    ('mid',  'long',  'none'),
    ('high', 'short', 'none'),
    ('high', 'mid',   'none'),
    ('high', 'long',  'none'),
]

def evaluate_fis(demand_val, lead_val):
    """Evaluate the Mamdani system for crisp inputs using centroid defuzzification."""
    mu_demand = {k: demand_mfs[k](np.array([demand_val]))[0] for k in demand_mfs}
    mu_lead   = {k: lead_mfs[k](np.array([lead_val]))[0] for k in lead_mfs}

    agg = np.zeros_like(u_disc, dtype=float)
    for d_lab, l_lab, out_lab in rules:
        firing = min(mu_demand[d_lab], mu_lead[l_lab])           
        out_mf = disc_mfs[out_lab](u_disc)                       
        agg = np.maximum(agg, np.minimum(firing, out_mf))        

    if agg.sum() == 0:
        return 0.0
    return float(np.trapz(u_disc * agg, u_disc) / np.trapz(agg, u_disc))

# -----------------------------
# Plot helpers
# -----------------------------
def plot_mfs(universe, mfs_dict, title, filename):
    plt.figure(figsize=(7, 4))
    for name, mf in mfs_dict.items():
        y = mf(universe)
        plt.plot(universe, y, label=name) 
    plt.title(title)
    plt.xlabel('Universe')
    plt.ylabel('Membership')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=160)
    plt.close()

def main():
    outdir = os.path.abspath("./outputs")
    os.makedirs(outdir, exist_ok=True)

    plot_mfs(u_demand, demand_mfs, "Membership Functions — demand (0..100)", os.path.join(outdir, "mfs_demand.png"))
    plot_mfs(u_lead,   lead_mfs,   "Membership Functions — lead (0..72h)",   os.path.join(outdir, "mfs_lead.png"))
    plot_mfs(u_disc,   disc_mfs,   "Membership Functions — discount (0..30%)", os.path.join(outdir, "mfs_discount.png"))

    samples = [(25, 8), (55, 40), (85, 10), (40, 20)]
    eval_rows = []
    for d, l in samples:
        y = evaluate_fis(d, l)
        eval_rows.append({"demand": d, "lead": l, "discount_reco": round(y, 2)})
    df_eval = pd.DataFrame(eval_rows)
    df_eval.to_csv(os.path.join(outdir, "rule_viewer_samples.csv"), index=False)

    D, L = np.meshgrid(np.linspace(0, 100, 61), np.linspace(0, 72, 61))
    Z = np.zeros_like(D, dtype=float)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            Z[i, j] = evaluate_fis(D[i, j], L[i, j])

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(D, L, Z, linewidth=0, antialiased=True) 
    ax.set_title("Control surface: discount = f(demand, lead)")
    ax.set_xlabel("demand (0..100)")
    ax.set_ylabel("lead (hours)")
    ax.set_zlabel("discount (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "surface_discount.png"), dpi=160)
    plt.close()

    rng = np.random.default_rng(42)
    genres = ["Action", "Comedy", "Drama", "Horror", "SciFi"]
    slots  = ["Morning", "Day", "Prime", "Late"]
    days   = ["Mon-Thu", "Fri", "Sat", "Sun"]

    genre_pop = {g: v for g, v in zip(genres, [65, 55, 50, 40, 60])}
    slot_pop  = {"Morning": 35, "Day": 50, "Prime": 80, "Late": 45}
    day_pop   = {"Mon-Thu": 45, "Fri": 70, "Sat": 85, "Sun": 65}

    def forecast_demand(g, s, d):
        return 0.4*genre_pop[g] + 0.4*slot_pop[s] + 0.2*day_pop[d]

    rows = []
    for i in range(40):
        g = rng.choice(genres)
        s = rng.choice(slots)
        d = rng.choice(days)
        lead = float(rng.choice([6, 12, 24, 36, 48, 60, 72]))
        demand = forecast_demand(g, s, d)
        demand = float(np.clip(demand + rng.normal(0, 5), 0, 100))
        disc = evaluate_fis(demand, lead)
        rows.append({
            "show_id": i+1,
            "genre": g, "slot": s, "day_type": d,
            "lead_h": lead, "forecast_demand": round(demand, 1),
            "discount_reco_%": round(disc, 2)
        })

    df_demo = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, "cinema_discount_recommendations.csv")
    df_demo.to_csv(csv_path, index=False)

    # Console summary
    print("Outputs saved to:", outdir)
    print("Samples (Rule-Viewer-like):")
    print(df_eval)
    print("Demo CSV:", csv_path)

if __name__ == "__main__":
    main()
