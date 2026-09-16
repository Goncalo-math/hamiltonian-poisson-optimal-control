import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.integrate import solve_ivp

# Ferry Problem
# Minimum-time navigation across a river.

# Parameters
x0 = {"x1": 0.0, "x2": 0.0}

params = {
    "L": 1.0,
    "K": np.pi,
    "S": 1.0,
    "method": "HPI",
    "N": 20,
    "J": np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
    ], dtype=float)
}


def hamiltonian(z, params):
    x2, y1, y2 = z[1], z[2], z[3]
    return 1 + y1*np.sin(x2) - params["S"]*np.hypot(y1, y2)


def grad_hamiltonian(z, params):
    x2, y1, y2 = z[1], z[2], z[3]
    r = np.hypot(y1, y2)
    if r < 1e-14:
        raise ValueError(
            "Costate norm is too close to zero; theta* is undefined.")
    return np.array([
        0.0,
        y1*np.cos(x2),
        np.sin(x2) - params["S"]*y1/r,
        -params["S"]*y2/r,
    ])


def canonical_ode(t, z, params):
    return params["J"] @ grad_hamiltonian(z, params)


def poisson_step(z, dt, params):
    Y = z.copy()
    tol = 1e-12
    max_iter = 100
    converged = False

    for _ in range(max_iter):
        g = grad_hamiltonian(Y, params)
        Ynew = z + 0.5*dt*(params["J"] @ g)
        if np.linalg.norm(Ynew - Y, ord=np.inf) < tol*(1 + np.linalg.norm(Ynew, ord=np.inf)):
            Y = Ynew
            converged = True
            break
        Y = Ynew

    if not converged:
        print(
            f"Warning: fixed-point iteration did not converge in {max_iter} iterations.")

    g = grad_hamiltonian(Y, params)
    return Y + 0.5*dt*(params["J"] @ g)


def integrate_arc_phi(z0, dt, N, params):
    t = np.zeros(N+1)
    y = np.zeros((N+1, 4))
    y[0] = z0
    z = z0.copy()

    for k in range(N):
        z = poisson_step(z, dt, params)
        y[k+1] = z
        t[k+1] = t[k] + dt
    return t, y


def integrate_trajectory_phi(lam0, tf, params, x0):
    N = params["N"]
    dt = tf/N
    z0 = np.array([x0["x1"], x0["x2"], lam0[0], lam0[1]], dtype=float)
    return integrate_arc_phi(z0, dt, N, params)


def shooting_residual_phi(z, params, x0):
    lam0 = z[:2]
    tf = z[2]
    if tf <= 0:
        return np.ones(3)*1e3*(1 - tf)

    _, y = integrate_trajectory_phi(lam0, tf, params, x0)
    yf = y[-1]
    return np.array([
        yf[0] - params["L"],
        yf[1] - params["K"],
        hamiltonian(yf, params),
    ])


def rk2_step(z, h, params):
    k1 = params["J"] @ grad_hamiltonian(z, params)

    k2 = params["J"] @ grad_hamiltonian(
        z + 0.5*h*k1,
        params
    )

    return z + h*k2


def integrate_trajectory_rk2(lam0, tf, params, x0):

    N = params["N"]
    dt = tf / N

    z0 = np.array([
        x0["x1"],
        x0["x2"],
        lam0[0],
        lam0[1]
    ], dtype=float)

    t = np.linspace(0.0, tf, N + 1)

    y = np.zeros((N + 1, 4))
    y[0] = z0

    for k in range(N):
        y[k + 1] = rk2_step(
            y[k],
            dt,
            params
        )

    return t, y


def shooting_residual_ode(z, params, x0):

    lam0 = z[:2]
    tf = z[2]

    # Prevent nonphysical negative final times
    if tf <= 0:
        return np.ones(3)*1e3*(1 - tf)

    _, y = integrate_trajectory_rk2(
        lam0,
        tf,
        params,
        x0
    )

    yf = y[-1]

    return np.array([
        yf[0] - params["L"],
        yf[1] - params["K"],
        hamiltonian(yf, params),
    ])


# Shooting
z0 = np.array([1.0, -1.0, 1.0])

sol_phi = root(lambda z: shooting_residual_phi(z, params, x0), z0, method="hybr",
               options={"xtol": 1e-11, "maxfev": 5000})
z_phi = sol_phi.x
F_phi = shooting_residual_phi(z_phi, params, x0)

if z_phi[2] <= 0:
    raise RuntimeError(
        "The HPI shooting method returned a non-positive final time.")

sol_ode = root(lambda z: shooting_residual_ode(z, params, x0), z0, method="hybr",
               options={"xtol": 1e-11, "maxfev": 5000})
z_ode = sol_ode.x
F_ode = shooting_residual_ode(z_ode, params, x0)

if z_ode[2] <= 0:
    raise RuntimeError(
        "The RK2 shooting method returned a non-positive final time.")

# HPI optimal trajectory
y10_phi, y20_phi, Topt_phi = z_phi
t_phi, y_phi = integrate_trajectory_phi(
    [y10_phi, y20_phi], Topt_phi, params, x0)

x1_phi, x2_phi, y1_phi, y2_phi = y_phi.T
theta_phi = np.arctan2(-y2_phi, -y1_phi)
H_phi = np.array([hamiltonian(row, params) for row in y_phi])
A_phi = params["S"]/np.cos(theta_phi) + np.sin(x2_phi)

# ODE comparison
y10_ode, y20_ode, Topt_ode = z_ode
Y0 = np.array([x0["x1"], x0["x2"], y10_ode, y20_ode])
t_eval = np.linspace(0, Topt_ode, params["N"]+1)

t_ode, y_ode = integrate_trajectory_rk2(
    [y10_ode, y20_ode], Topt_ode, params, x0)
x1_ode, x2_ode, y1_ode, y2_ode = y_ode.T
theta_ode = np.arctan2(-y2_ode, -y1_ode)
H_ode = np.array([hamiltonian(row, params) for row in y_ode])
A_ode = params["S"]/np.cos(theta_ode) + np.sin(x2_ode)

# Results
print("="*60)
print("FERRY PROBLEM RESULTS")
print("="*60)
print(f"L                          = {params['L']:.12f}")
print(f"K                          = pi = {params['K']:.12f}")
print(f"S                          = {params['S']:.12f}")
print(f"y1(0) (HPI)                = {y10_phi:.12f}")
print(f"y2(0) (HPI)                = {y20_phi:.12f}")
print(f"y1(0) (RK2)                = {y10_ode:.12f}")
print(f"y2(0) (RK2)                = {y20_ode:.12f}")
print(f"Optimal final time T (HPI) = {Topt_phi:.12f}")
print(f"Optimal final time T (RK2) = {Topt_ode:.12f}")
print(f"Function evaluations (HPI) = {sol_phi.nfev}")
print(f"Function evaluations (RK2) = {sol_ode.nfev}")
print(f"||HPI residual||inf        = {np.linalg.norm(F_phi, ord=np.inf):.3e}")
print(f"||RK2 residual||inf        = {np.linalg.norm(F_ode, ord=np.inf):.3e}")

A_theory = -1/y10_phi
print(f"A = -1/y1(0)               = {A_theory:.12f}")

# Plots
plt.figure()
plt.plot(x1_phi, x2_phi, "o-", markersize=3, linewidth=1.0, label="HPI")
plt.plot(x1_ode, x2_ode, "--", linewidth=1.4, label="RK2")
plt.plot(x0["x1"], x0["x2"], "ks", markersize=7, label="Start")
plt.plot(params["L"], params["K"], "kp", markersize=9, label="Target")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Optimal ferry trajectory")
plt.grid(True)
plt.legend()

# plt.figure()
# plt.plot(t_phi, theta_phi, "o-", markersize=3, linewidth=1.0, label="HPI")
# plt.plot(t_ode, theta_ode, "--", linewidth=1.4, label="RK23")
# plt.xlabel("t")
# plt.ylabel("theta*(t) [rad]")
# plt.title("Optimal heading angle")
# plt.grid(True)
# plt.legend()

# plt.figure()
# plt.plot(t_phi, y1_phi, label="y1 HPI")
# plt.plot(t_phi, y2_phi, label="y2 HPI")
# plt.plot(t_ode, y1_ode, "--", label="y1 RK23")
# plt.plot(t_ode, y2_ode, "--", label="y2 RK23")
# plt.xlabel("t")
# plt.ylabel("Costates")
# plt.title("Adjoint variables")
# plt.grid(True)
# plt.legend()

plt.figure()
plt.plot(t_phi, H_phi-H_phi[0], label="HPI")
plt.plot(t_ode, H_ode-H_ode[0], "--", label="RK2")
plt.xlabel("t")
plt.ylabel("Hamiltonian Error")
plt.title("Hamiltonian conservation error")
plt.grid(True)
plt.legend()

plt.figure()
plt.plot(t_phi, A_phi, label="HPI")
plt.plot(t_ode, A_ode, "--", label="RK2")
plt.axhline(A_theory, linestyle=":", label="Theory")
plt.xlabel("t")
plt.ylabel("A(t)")
plt.title("A = S/cos(theta) + sin(x2)")
plt.grid(True)
plt.legend()

plt.show()
