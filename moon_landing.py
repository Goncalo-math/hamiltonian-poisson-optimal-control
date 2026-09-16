import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.integrate import solve_ivp

# Fuel-optimal lunar soft landing:
# RK2/event method versus HPI/Poisson method.

params = {
    "g": 1.62,
    "Isp": 311.0,
    "g0": 9.80665,
    "Tmax": 28000.0,
    "mdry": 5000.0,
}
params["c"] = params["Isp"] * params["g0"]
params["J"] = np.array([
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
    [-1, 0, 0, 0, 0, 0],
    [0, -1, 0, 0, 0, 0],
    [0, 0, -1, 0, 0, 0],
], dtype=float)

x0 = {"h": 15700.0, "v": -153.0, "m": 10000.0}

def switching_function(y, params):
    m = y[2]
    lambda_v = y[4]
    lambda_m = y[5]
    return lambda_v/m - lambda_m/params["c"]

def switching_function_history(y, params):
    return y[:,4]/y[:,2] - y[:,5]/params["c"]

def full_odes(t, y, params, T):
    v = y[1]
    m = y[2]
    lambda_h = y[3]
    lambda_v = y[4]

    hdot = v
    vdot = -params["g"] + T/m
    mdot = -T/params["c"]
    lambda_h_dot = 0.0
    lambda_v_dot = -lambda_h
    lambda_m_dot = lambda_v*T/m**2

    return np.array([
        hdot, vdot, mdot,
        lambda_h_dot, lambda_v_dot, lambda_m_dot
    ])

def switch_event(t, y, params):
    return switching_function(y, params)

switch_event.terminal = True
switch_event.direction = 1


def residual_ode(z, params, x0):
    tf = z[3]

    if tf <= 0:
        return np.ones(4)*1e6

    _, y, _ = integrate_trajectory_ode(
        z,
        params,
        x0,
        N=200
    )

    yf = y[-1]

    h_f, v_f, m_f = yf[:3]
    lamh_f, lamv_f, lamm_f = yf[3:]

    T_f = params["Tmax"]

    H_f = Hamiltonian(
        yf,
        params,
        T_f
    )

    return np.array([
        h_f,
        v_f,
        lamm_f - 1.0,
        H_f
    ])

def Hamiltonian(y, params, T):
    m = y[2]
    lambda_h = y[3]
    lambda_v = y[4]
    lambda_m = y[5]

    return (
        lambda_h*y[1]
        + lambda_v*(T/m - params["g"])
        - lambda_m*(T/params["c"])
    )

def grad_hamiltonian(y, T, params):
    m = y[2]
    lambda_h = y[3]
    lambda_v = y[4]

    return np.array([
        0.0,
        lambda_h,
        -lambda_v*T/m**2,
        y[1],
        T/m - params["g"],
        -T/params["c"]
    ])

def rk2_step(y, h, params, T):
    k1 = params["J"] @ grad_hamiltonian(y, T, params)
    k2 = params["J"] @ grad_hamiltonian(
        y + 0.5*h*k1,
        T,
        params
    )

    return y + h*k2

def RK2(y0, dt, n):
    y = np.zeros((n, 4))
    y[0] = y0

    for k in range(n-1):
        y[k+1] = rk2_step(y[k], dt, params, 0.0)

    return y

def integrate_trajectory_ode(z, params, x0, N=100):
    """
    Integrate the lunar-landing trajectory using fixed-step RK2.

    Arc 1:
        T = 0

    Switching condition:
        switching_function(y) crosses zero from negative to positive

    Arc 2:
        T = Tmax

    Parameters
    ----------
    z : array_like
        z = [lambda_h(0), lambda_v(0), lambda_m(0), tf]

    params : dict
        Problem parameters.

    x0 : dict
        Initial state.

    N : int
        Number of nominal RK2 steps over [0, tf].

    Returns
    -------
    t : ndarray
        Complete time trajectory.

    y : ndarray
        Complete state/costate trajectory.

    t_switch : float
        Switching time.
    """

    # -----------------------------------
    # Initial conditions
    # -----------------------------------

    lam0 = z[:3]
    tf = z[3]

    y0 = np.array([
        x0["h"],
        x0["v"],
        x0["m"],
        lam0[0],
        lam0[1],
        lam0[2]
    ], dtype=float)

    # Fixed nominal time step
    dt = tf / N

    # Store trajectory dynamically because we insert
    # the switching point explicitly.
    t_list = [0.0]
    y_list = [y0.copy()]  

    t = 0.0
    y = y0.copy()

    # -----------------------------------
    # ARC 1 : coast, T = 0
    # -----------------------------------

    T = 0.0

    phi_old = switching_function(y, params)

    switched = False
    t_switch = tf

    while t < tf:

        h = min(dt, tf - t)

        # Candidate RK2 step on coast arc
        y_new = rk2_step(
            y,
            h,
            params,
            T
        )

        t_new = t + h

        phi_new = switching_function(
            y_new,
            params
        )

        # --------------------------------
        # Detect switching-function crossing
        #
        # event.direction = +1 equivalent:
        #
        # phi_old < 0
        # phi_new >= 0
        # --------------------------------

        if phi_old < 0.0 and phi_new >= 0.0:

            # Linear interpolation of the switching location
            alpha = (
                -phi_old /
                (phi_new - phi_old)
            )

            alpha = np.clip(alpha, 0.0, 1.0)

            # Switching time
            t_switch = t + alpha*h

            # Approximate switching state
            #
            # Better than simply interpolating y:
            # integrate from y for the fractional
            # RK2 step.
            h_switch = alpha*h

            y_switch = rk2_step(
                y,
                h_switch,
                params,
                0.0
            )

            # Store exact switching point
            t_list.append(t_switch)
            y_list.append(y_switch.copy())

            y = y_switch
            t = t_switch

            switched = True

            break

        # No switch yet
        t_list.append(t_new)
        y_list.append(y_new.copy())

        y = y_new
        t = t_new
        phi_old = phi_new

    # -----------------------------------
    # ARC 2 : powered descent
    # T = Tmax
    # -----------------------------------

    if switched:

        T = params["Tmax"]

        while t < tf:

            h = min(dt, tf - t)

            y = rk2_step(
                y,
                h,
                params,
                T
            )

            t += h

            t_list.append(t)
            y_list.append(y.copy())

    # -----------------------------------
    # Convert lists to arrays
    # -----------------------------------

    t_array = np.array(t_list)
    y_array = np.vstack(y_list)

    return t_array, y_array, t_switch

def poisson_step(x, T, dt, params):
    y = x.copy()
    tol = 1e-12
    max_iter = 100
    converged = False

    for _ in range(max_iter):
        g = grad_hamiltonian(y, T, params)
        ynew = x + 0.5*dt*(params["J"] @ g)
        if np.linalg.norm(ynew-y, ord=np.inf) < tol*(1 + np.linalg.norm(ynew, ord=np.inf)):
            y = ynew
            converged = True
            break
        y = ynew

    if not converged:
        print("Warning: fixed-point iteration did not converge.")

    g = grad_hamiltonian(y, T, params)
    return y + 0.5*dt*(params["J"] @ g)

def integrate_arc_phi(x0_vec, T, dt, N, params):
    t = np.zeros(N+1)
    y = np.zeros((N+1, 6))
    y[0] = x0_vec
    x = x0_vec.copy()

    for k in range(N):
        x = poisson_step(x, T, dt, params)
        y[k+1] = x
        t[k+1] = t[k] + dt
    return t, y

def integrate_trajectory_phi(lam0, ts, tf, N1, N2, params, x0):
    dt1 = ts/N1
    dt2 = (tf-ts)/N2
    x_init = np.array([x0["h"], x0["v"], x0["m"], *lam0], dtype=float)

    t1, y1 = integrate_arc_phi(x_init, 0.0, dt1, N1, params)
    x_switch = y1[-1].copy()
    t2, y2 = integrate_arc_phi(x_switch, params["Tmax"], dt2, N2, params)
    t2 = t2 + ts

    t = np.concatenate([t1, t2[1:]])
    y = np.vstack([y1, y2[1:]])
    return t, y

def residual_phi(z, N1, N2, params, x0):
    lam0 = z[:3]
    ts = z[3]
    tf = z[4]

    if ts <= 0 or tf <= 0 or ts >= tf:
        return np.ones(5)*1e6

    _, y = integrate_trajectory_phi(lam0, ts, tf, N1, N2, params, x0)
    yf = y[-1]
    ys = y[N1]

    h_f, v_f, m_f = yf[:3]
    lamh_f, lamv_f, lamm_f = yf[3:]

    T_f = params["Tmax"]
    H_f = (
        lamh_f*v_f
        + lamv_f*(T_f/m_f - params["g"])
        - lamm_f*(T_f/params["c"])
    )

    phi_s = switching_function(ys, params)
    return np.array([h_f, v_f, lamm_f - 1.0, H_f, phi_s])

def hamiltonian_difference(t, y, ts, params):
    """
    Compute Hamiltonian conservation error separately on each arc.

    Arc 1:
        T = 0
        reference = H(t=0)

    Arc 2:
        T = Tmax
        reference = H(t=ts+) evaluated at the switching state

    Parameters
    ----------
    t : ndarray, shape (N,)
        Time vector.

    y : ndarray, shape (N, 6)
        State/costate trajectory.

    ts : float
        Switching time.

    params : dict
        Problem parameters.

    Returns
    -------
    H_diff : ndarray, shape (N,)
        Complete vector of Hamiltonian errors.
    """

    H_diff = np.zeros(len(t))

    # Find the trajectory point corresponding to the switch
    i_switch = np.argmin(np.abs(t - ts))

    # -------------------------
    # ARC 1 : T = 0
    # -------------------------
    H1 = np.array([
        Hamiltonian(yi, params, 0.0)
        for yi in y[:i_switch + 1]
    ])

    H1_ref = H1[0]

    H_diff[:i_switch + 1] = np.abs(H1 - H1_ref)

    # -------------------------
    # ARC 2 : T = Tmax
    # -------------------------

    # Hamiltonian immediately after switching, evaluated
    # using the SAME switching state but T = Tmax
    H2_ref = Hamiltonian(
        y[i_switch],
        params,
        params["Tmax"]
    )

    if i_switch + 1 < len(t):

        H2 = np.array([
            Hamiltonian(yi, params, params["Tmax"])
            for yi in y[i_switch + 1:]
        ])

        H_diff[i_switch + 1:] = np.abs(H2 - H2_ref)

    return H_diff


# Method A: RK2 + event-detected switch
# A numerically consistent initial guess is important for SciPy fsolve.
# Unknowns: [lambda_h(0), lambda_v(0), lambda_m(0), tf]
z0_ode = np.array([-2, 4, 2, 200])
sol_ode = root(lambda z: residual_ode(z, params, x0), z0_ode,
               method="hybr", options={"xtol": 1e-10, "maxfev": 5000})
z_ode = sol_ode.x

res_ode = residual_ode(z_ode, params, x0)

lambda0_ode = z_ode[:3]
tf_ode = z_ode[3]

t_ode, y_ode, ts_ode = integrate_trajectory_ode(z_ode, params, x0)
h_ode, v_ode, m_ode = y_ode[:,0], y_ode[:,1], y_ode[:,2]
T_ode = params["Tmax"]*(t_ode >= ts_ode)
phi_ode = switching_function_history(y_ode, params)
fuel_ode = x0["m"] - m_ode[-1]

# Method B: HPI / Poisson
N1 = 100
N2 = 100
z0_phi = np.array([*lambda0_ode, ts_ode, tf_ode])

sol_phi = root(lambda z: residual_phi(z, N1, N2, params, x0), z0_phi,
               method="hybr", options={"xtol": 1e-10, "maxfev": 5000})
z_phi = sol_phi.x
res_phi = residual_phi(z_phi, N1, N2, params, x0)

lambda0_phi = z_phi[:3]
ts_phi = z_phi[3]
tf_phi = z_phi[4]

t_phi, y_phi = integrate_trajectory_phi(lambda0_phi, ts_phi, tf_phi, N1, N2, params, x0)
h_phi, v_phi, m_phi = y_phi[:,0], y_phi[:,1], y_phi[:,2]
T_phi = params["Tmax"]*(t_phi >= ts_phi)
phi_phi = switching_function_history(y_phi, params)
fuel_phi = x0["m"] - m_phi[-1]

## Hamiltonian conservation check
H_diff_ode = hamiltonian_difference(t_ode, y_ode, ts_ode, params)
H_diff_phi = hamiltonian_difference(t_phi, y_phi, ts_phi, params)


print("="*60)
print("RK2 / EVENT METHOD")
print("="*60)
print(f"Shooting converged: {sol_ode.success}")
print(f"lambda_h(0) = {lambda0_ode[0]:.10f}")
print(f"lambda_v(0) = {lambda0_ode[1]:.10f}")
print(f"lambda_m(0) = {lambda0_ode[2]:.10f}")
print(f"Function evaluations = {sol_ode.nfev}")
print(f"Switch time ts = {ts_ode:.10f} s")
print(f"Final time  tf = {tf_ode:.10f} s")
print(f"Final h(tf) = {h_ode[-1]:.10e} m")
print(f"Final v(tf) = {v_ode[-1]:.10e} m/s")
print(f"Final m(tf) = {m_ode[-1]:.10f} kg")
print(f"Fuel used   = {fuel_ode:.10f} kg")
print(f"||residual||_2 = {np.linalg.norm(res_ode):.6e}")

print("="*60)
print("HPI / POISSON METHOD")
print("="*60)
print(f"Shooting converged: {sol_phi.success}")
print(f"lambda_h(0) = {lambda0_phi[0]:.10f}")
print(f"lambda_v(0) = {lambda0_phi[1]:.10f}")
print(f"lambda_m(0) = {lambda0_phi[2]:.10f}")
print(f"Function evaluations = {sol_phi.nfev}")
print(f"Switch time ts = {ts_phi:.10f} s")
print(f"Final time  tf = {tf_phi:.10f} s")
print(f"Final h(tf) = {h_phi[-1]:.10e} m")
print(f"Final v(tf) = {v_phi[-1]:.10e} m/s")
print(f"Final m(tf) = {m_phi[-1]:.10f} kg")
print(f"Fuel used   = {fuel_phi:.10f} kg")
print(f"Phi(ts)     = {switching_function(y_phi[N1], params):.10e}")
print(f"||residual||_2 = {np.linalg.norm(res_phi):.6e}")

fig, ax = plt.subplots(4, 1, sharex=True)
ax[0].plot(t_phi, h_phi, label="h_HPI")
ax[0].plot(t_ode, h_ode, "--", label="h_RK2")
ax[0].axhline(0, linestyle=":")
ax[0].set_ylabel("h(t) [m]")
ax[0].grid(True)
ax[0].legend()

ax[1].plot(t_phi, v_phi, label="v_HPI")
ax[1].plot(t_ode, v_ode, "--", label="v_RK2")
ax[1].axhline(0, linestyle=":")
ax[1].set_ylabel("v(t) [m/s]")
ax[1].grid(True)
ax[1].legend()

ax[2].plot(t_phi, m_phi, label="m_HPI")
ax[2].plot(t_ode, m_ode, "--", label="m_RK2")
ax[2].axhline(params["mdry"], linestyle=":", label="dry mass")
ax[2].set_ylabel("m(t) [kg]")
ax[2].grid(True)
ax[2].legend()

ax[3].plot(t_phi, T_phi, label="T_HPI")
ax[3].plot(t_ode, T_ode, "--", label="T_RK2")
ax[3].set_ylabel("T(t) [N]")
ax[3].set_xlabel("time [s]")
ax[3].grid(True)
ax[3].legend()

fig.suptitle("Fuel-optimal lunar landing: HPI/Poisson vs RK2/Event")

plt.figure()
plt.plot(t_phi, phi_phi, label="Phi_HPI")
plt.plot(t_ode, phi_ode, "--", label="Phi_RK2")
plt.axhline(0, linestyle=":")
plt.axvline(ts_phi, linestyle=":", label="ts_HPI")
plt.axvline(ts_ode, linestyle="--", label="ts_RK2")
plt.xlabel("time [s]")
plt.ylabel("Phi(t)")
plt.title("Switching function comparison")
plt.grid(True)
plt.legend()

plt.show()

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(t_phi, H_diff_phi, label="HPI")
ax2.plot(t_ode, H_diff_ode, "--", label="RK2")
ax2.set_ylabel('Hamiltonian Error')
ax2.legend()
ax2.set_title('Hamiltonian Error Moon Landing (switching method)')
plt.tight_layout()
plt.show()
