import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.integrate import solve_ivp

# Insects as Optimizers

T = 6.0
x0 = {"w": 1.0, "q": 0.0}

params = {
    "b": 0.6,
    "c": 0.4,
    "mu": 0.2,
    "J": np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
    ], dtype=float)
}


def Hamiltonian(y, u, params):
    w, q, yw, yq = y
    return yw*(params["b"]*u*w - params["mu"]*w) + yq*params["c"]*(1-u)*w


def grad_hamiltonian(y, u, params):
    w, q, yw, yq = y
    dH_dw = (params["b"]*u - params["mu"])*yw + params["c"]*(1-u)*yq
    dH_dq = 0.0
    dH_dyw = params["b"]*u*w - params["mu"]*w
    dH_dyq = params["c"]*(1-u)*w
    return np.array([dH_dw, dH_dq, dH_dyw, dH_dyq])


def rk2_step(y, h, params, u):

    k1 = params["J"] @ grad_hamiltonian(y, u, params)
    k2 = params["J"] @ grad_hamiltonian(y + 0.5*h*k1, u, params)
    return y + h*k2


def RK2(y0, dt, n, u, params):
    y = np.zeros((n, 4))
    y[0] = y0

    for k in range(n-1):
        y[k+1] = rk2_step(y[k], dt, params, u)

    return y


def poisson_step(x, dt, params, u):
    y = x.copy()
    tol = 1e-12
    max_iter = 100
    converged = False

    for _ in range(max_iter):
        g = grad_hamiltonian(y, u, params)
        ynew = x + 0.5*dt*(params["J"] @ g)
        if np.linalg.norm(ynew-y, ord=np.inf) < tol*(1 + np.linalg.norm(ynew, ord=np.inf)):
            y = ynew
            converged = True
            break
        y = ynew

    if not converged:
        print("Warning: fixed-point iteration did not converge.")

    g = grad_hamiltonian(y, u, params)
    return y + 0.5*dt*(params["J"] @ g)


def integrate_arc(x0_vec, dt, N, params, u):
    t = np.zeros(N+1)
    y = np.zeros((N+1, 4))
    y[0] = x0_vec
    x = x0_vec.copy()

    for k in range(N):
        x = poisson_step(x, dt, params, u)
        y[k+1] = x
        t[k+1] = t[k] + dt
    return t, y


def rhs(t, y, u, params):
    w, q, yw, yq = y
    return np.array([
        params["b"]*u*w - params["mu"]*w,
        params["c"]*(1-u)*w,
        -((params["b"]*u-params["mu"])*yw + params["c"]*(1-u)*yq),
        0.0
    ])


def integrate_trajectory(lam0, ts, tf, params, x0, method):
    N1 = 10
    N2 = 10
    dt1 = ts/N1
    dt2 = (tf-ts)/N2
    x = np.array([x0["w"], x0["q"], lam0[0], lam0[1]], dtype=float)

    if method == "HPI":
        t1, y1 = integrate_arc(x, dt1, N1, params, 1.0)
        x = y1[-1].copy()
        t2, y2 = integrate_arc(x, dt2, N2, params, 0.0)
        t2 = t2 + ts
        t = np.concatenate([t1, t2[1:]])
        y = np.vstack([y1, y2[1:]])
    elif method == "RK2":
        t1 = np.linspace(0, ts, N1+1)
        sol1 = RK2(x, dt1, N1+1, 1.0, params)

        x = sol1[-1].copy()

        t2 = np.linspace(ts, tf, N2+1)
        sol2 = RK2(x, dt2, N2+1, 0.0, params)

        t = np.concatenate([t1, t2[1:]])
        y = np.vstack([sol1, sol2[1:]])
    else:
        raise ValueError("Wrong integrator.")
    return t, y


def shooting_residual(z, params, x0, ts, T, method):
    _, y = integrate_trajectory(z, ts, T, params, x0, method)
    yf = y[-1]
    return np.array([yf[2], yf[3] - 1.0])

def diff_Hamiltonian(y, params, ts, t):

    H_diff = np.zeros(len(t))

    # index of switching point
    i_switch = np.argmin(np.abs(t - ts))

    # =========================
    # ARC 1 : u = 1
    # =========================
    H1 = np.array([
        Hamiltonian(yi, 1.0, params)
        for yi in y[:i_switch + 1]
    ])

    H1_ref = H1[0]

    H_diff[:i_switch + 1] = np.abs(
        H1 - H1_ref
    )

    # =========================
    # ARC 2 : u = 0
    # =========================
    H2 = np.array([
        Hamiltonian(yi, 0.0, params)
        for yi in y[i_switch + 1:]
    ])

    if len(H2) > 0:

        # Evaluate reference for arc 2 at switching state
        H2_ref = Hamiltonian(
            y[i_switch],
            0.0,
            params
        )

        H_diff[i_switch + 1:] = np.abs(
            H2 - H2_ref
        )

    return H_diff


# Shooting
z0 = np.array([0.0, 1.0])
ts = T + (1/params["mu"])*np.log(1 - params["mu"]/params["b"])

sol_phi = root(lambda z: shooting_residual(z, params, x0, ts, T, "HPI"), z0)
sol_ode = root(lambda z: shooting_residual(z, params, x0, ts, T, "RK2"), z0)

z_phi = sol_phi.x
z_ode = sol_ode.x

t_phi, y_phi = integrate_trajectory(z_phi, ts, T, params, x0, "HPI")
t_ode, y_ode = integrate_trajectory(z_ode, ts, T, params, x0, "RK2")

# Hamiltonian difference along the flow
H_diff_phi = diff_Hamiltonian(y_phi, params, ts, t_phi)
H_diff_ode = diff_Hamiltonian(y_ode, params, ts, t_ode)

print(f"Shooting HPI converged: {sol_phi.success}")
print(f"Shooting RK2 converged: {sol_ode.success}")
print(f"Recovered (HPI) y_w(0) = {z_phi[0]:.6f}, y_q(0) = {z_phi[1]:.6f}")
print(f"Recovered (RK2) y_w(0) = {z_ode[0]:.6f}, y_q(0) = {z_ode[1]:.6f}")
print(f"HPI Final w(T) = {y_phi[-1, 0]:.6f}")
print(f"HPI Final q(T) = {y_phi[-1, 1]:.6f}")
print(f"RK2 Final w(T) = {y_ode[-1, 0]:.6f}")
print(f"RK2 Final q(T) = {y_ode[-1, 1]:.6f}")
print(f"Switch time t1 = {ts:.6f} s, Final time tf = {T:.6f} s")
print(f"Function evaluations (HPI) = {sol_phi.nfev}, (RK2) = {sol_ode.nfev}")


fig, ax = plt.subplots(3, 1, sharex=True)
labels = ["w(t)", "q(t)", r"$y_w(t)$"]
for i in range(3):
    ax[i].plot(t_phi, y_phi[:, i], label="HPI")
    ax[i].plot(t_ode, y_ode[:, i], "--", label="RK2")
    # ax[i].plot(t_ode, y_ode[:,i], "--", label="RK2")
    ax[i].set_ylabel(labels[i])
    ax[i].grid(True)
ax[-1].set_xlabel("time")
ax[-1].legend()
fig.suptitle("Optimal trajectories")
plt.show()

fig, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(t_phi, H_diff_phi, '-', label='HPI')
ax2.plot(t_ode, H_diff_ode, '--', label='RK2')
ax2.set_ylabel('Hamiltonian Error')
ax2.set_xlabel('time')
ax2.legend()
plt.title("Hamiltonian Error Insects")
plt.tight_layout()
plt.show()