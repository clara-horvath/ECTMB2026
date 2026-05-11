import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# -----------------------
# --- Hilfsfunktionen ---
# -----------------------
def sat(x, K, eps=1e-12):
    return x / (x + K + eps)

def hill(x, K, n, eps=1e-12):
    return x ** n / (x ** n + K ** n + eps)

def hilfe(A, B, C, D, eps=1e-12):
    return (A + B) / (C + D + eps)

def bruch(A, B, eps=1e-8):
    return A / (B + eps)

# -----------------------
# --- ODE ---
# -----------------------
def hpo_ode(t, x, p):
    # Voraussetzung
    N = 10

    # Variablen
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = x

    # Parameter zum dimensionslos machen
    d1 = p["a"]; d2 = p["b"]; d3 = p["c"]; d4 = p["d"]; d5 = p["e"]
    d6 = p["f"]; d7 = p["g"]; d8 = p["h"]; d9 = p["i"]; d10 = p["j"]

    # fixed Parameter
    f1 = p["V"]; f2 = p["delta_F"]; f3 = p["delta_L"]; f4 = p["delta_E"]; f5 = p["delta_P"]

    # restliche Parameter
    v1 = p["v_F"]; v2 = p["v_0L"]; v3 = p["v_1L"]
    k1 = p["k_IFI"]; k2 = p["k_F"]; k3 = p["k_iLP"]; k4 = p["k_mL"]; k5 = p["k_L"]
    c1 = p["c_FI"]; c2 = p["c_FP"]; c3 = p["c_FE"]; c4 = p["c_LP"]; c5 = p["c_LE"]
    p1 = p["f_1"]; p2 = p["f_2"]; p3 = p["h_1"]; p4 = p["h_2"]; p5 = p["w"]
    p6 = p["ell"]; p7 = p["s_hat"]; p8 = p["delta_s"]; p9 = p["e_0"]
    p10 = p["t_g1"]; p11 = p["eta"]; p12 = p["kappa_2"]; p13 = p["p"]; p14 = p["h_s"]

    # Gleichungen
    g1 = d1 * v1 * hilfe(k1 * d7 * d8, x8 * x7, k1 * d7 * d8, (1 + c1) * x8 * x7) - bruch(k2 * d9**2, d10) * hilfe(d10, c2 * x10, d9**2, c3 * x9**2) * x1
    g2 = bruch(d2 * k2 * d9**2, f1 * d1 * d10) * hilfe(d10, c2 * x10, d9**2, c3 * x9**2) * x1 - f2 * x2
    g3 = (v2 + v3 * hill(x9, d9 * k4, N)) * bruch(d3 * k3 * d10, k3 * d10 + x10) - bruch(k5 * d9**2, d10) * hilfe(d10, c4 * x10, d9, c5 * x9) * x3
    g4 = bruch(d4 * k5 * d9, d3 * f1 * d10) * hilfe(d10, c4 * x10, d9, c5 * x9) * x3 - f3 * x4
    g5 = (p1 * hill(x2, p3 * d2, 2) - p2 * hill(x4, p4 * d4, 2)) * x5
    g6 = bruch(d6 * p2, d5) * hill(x4, p4 * d4, 2) * x5 - bruch(p5, d8) * x8 * x6
    g7 = bruch(d7 * p5, d6 * d8) * x8 * x6 - p6 * x7 + bruch(p6, d8) * x8 * x7
    g8 = d8 * p7 * hill(x4, p14 * d4, 4) - p7 * hill(x4, p14 * d4, 4) * x8 - p8 * x8
    g9 = d9 * p9 - f4 * x9 + bruch(p10 * d9, d5 * d7 * d8) * (d7 * d8 * x5 + p11 * d5 *x7 * x8) * sat(x4, p12 * d4)
    g10 = - f5 * x10 + bruch(p13 * d10, d7 * d8) * x7 * x8

    return np.array([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10])

x0 = np.ones(10)
x0[0] = 3
x0[1] = 3
x0[5] = 10
x0[8] = 0.01

# -----------------------
# --- Parameter ---
# -----------------------
params = {"a": 1, "b": 1, "c": 1, "d": 1, "e": 1, "f": 1, "g": 1, "h": 1, "i": 1, "j": 1,
          "V": 2.5, "delta_F": 8.21, "delta_L": 14, "delta_E": 1.1, "delta_P": 0.5,
          "v_F": 3219.9, "v_0L": 308.35, "v_1L": 44700,
          "k_IFI": 149.76, "k_F": 3.0212, "k_iLP": 3.2279, "k_mL": 226.37, "k_L": 0.67146,
          "c_FI": 3.0188, "c_FP": 65.229, "c_FE": 0.0024047, "c_LP": 0.015844, "c_LE": 0.00068867,
          "f_1": 1.0958, "f_2": 46.225, "h_1": 146.31, "h_2": 798.39, "w": 0.23497,
          "ell": 0.64178, "s_hat": 2.6338, "delta_s": 0.38256, "e_0": 9.6377,
          "t_g1": 6.3594, "h_s": 11.691, "eta": 0.81426, "kappa_2": 8.276, "p": 0.22851}

# -----------------------
# --- Simulation ---
# -----------------------

# Hilfsfunktion damit solve_IVP funktioniert (problematisch wegen Input-Anzahl)
def f(t, x):
    return hpo_ode(t, x, params)

# --- Stiffness check: compare RK45 vs Radau ---
tol = dict(rtol=1e-6, atol=1e-8)

t0 = time.perf_counter()
sol_rk = solve_ivp(f, [0, 90], x0, method="RK45", **tol)
dt_rk = time.perf_counter() - t0

t0 = time.perf_counter()
sol_rad = solve_ivp(f, [0, 90], x0, method="Radau", **tol)
dt_rad = time.perf_counter() - t0

print(f"RK45  : {len(sol_rk.t):>6} steps,  {dt_rk:.3f} s")
print(f"Radau : {len(sol_rad.t):>6} steps,  {dt_rad:.3f} s")
print("(Many more RK45 steps than Radau → system is stiff)")

# Use Radau solution for plotting
sol = sol_rad
if not sol.success:
    raise RuntimeError(f"solve_ivp failed: {sol.message}")

t = sol.t
x = sol.y.T

# Optional (empfohlen): Namen der Zustände
state_names = [
    r"$FSH_\rho$",
    r"$FSH$",
    r"$LH_\rho$",
    r"$LH$",
    r"$\Phi$",
    r"$\Omega$",
    r"$\Lambda$",
    r"$S$",
    r"$E_2$",
    r"$P_4$"
]

# Minimum height a peak must exceed to be counted (None = no threshold)
peak_thresholds = [1000, 300, 10000, 200, 40, 40,30, 0.7,200, 12]

# --- Peak detection ---
peak_indices = []
peak_times = []
for i in range(10):
    idx, _ = find_peaks(x[:, i], height=peak_thresholds[i])
    peak_indices.append(idx)
    peak_times.append(t[idx] if len(idx) >= 1 else np.array([]))

# Print inter-peak interval comparison
print(f"\n{'State':<12} {'# peaks':>8} {'mean interval':>15} {'std interval':>14}")
print("-" * 52)
for i, name in enumerate(state_names):
    pt = peak_times[i]
    if len(pt) >= 2:
        intervals = np.diff(pt)
        print(f"{name:<12} {len(pt):>8} {np.mean(intervals):>15.4f} {np.std(intervals):>14.4f}")
    else:
        print(f"{name:<12} {len(pt):>8} {'—':>15} {'—':>14}")

# --- Plot: Alle 10 Zustände mit markierten Peaks ---
fig, axes = plt.subplots(5, 2, sharex=True, figsize=(12, 10))
axes = axes.ravel()
for i in range(10):
    axes[i].plot(t, x[:, i])
    idx = peak_indices[i]
    if len(idx) >= 1:
        axes[i].plot(t[idx], x[idx, i], "x", color="red")
    axes[i].set_ylabel(state_names[i])
    axes[i].grid(True, alpha=0.3)

axes[-2].set_xlabel("t")
axes[-1].set_xlabel("t")
fig.tight_layout()
plt.show()

