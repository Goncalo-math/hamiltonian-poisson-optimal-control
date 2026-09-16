import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root, brentq
from scipy.integrate import solve_ivp

# Harvesting for profit

T = 6.0
x0 = 150.0

params = {
    "r": 0.5,
    "K": 1000.0,
    "q": 0.01,
    "E": 50.0,
    "p": 10.0,
    "c": 2.0,
    "J": np.array([[0, 1], [-1, 0]], dtype=float),
    "N": 10
}

def Hamiltonian(z, u, params):
    x, y = z

    running_profit = (
        params["p"] * params["q"] * x - params["c"]
    ) * u

    dynamics = (
        params["r"] * x * (1 - x / params["K"])
        - params["q"] * x * u
    )

    return running_profit + y * dynamics

def grad_hamiltonian(dyn, u, params):
    x, y = dyn
    dH_dx = params["p"]*params["q"]*u + (
        params["r"]*(1 - 2*x/params["K"]) - params["q"]*u
    )*y
    dH_dy = params["r"]*x*(1 - x/params["K"]) - params["q"]*x*u
    return np.array([dH_dx, dH_dy])

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
    y = np.zeros((N+1, 2))
    y[0] = x0_vec
    x = x0_vec.copy()

    for k in range(N):
        x = poisson_step(x, dt, params, u)
        y[k+1] = x
        t[k+1] = t[k] + dt
    return t, y

def rhs(t, dyn, u, params):
    x, y = dyn
    return np.array([
        params["r"]*x*(1-x/params["K"]) - params["q"]*x*u,
        -(params["p"]*params["q"]*u
          + (params["r"]*(1 - 2*x/params["K"]) - params["q"]*u)*y)
    ])

def switch_time(x0, params, T):
    """
    Compute the continuous PMP switching time for the assumed policy

        u(t) = 0,  0 <= t < ts,
        u(t) = E,  ts <= t <= T.

    The switching time satisfies

        q*x(ts)*(p - y(ts)) - c = 0,

    where y(T) = 0.
    """

    r = params["r"]
    K = params["K"]
    q = params["q"]
    E = params["E"]
    price = params["p"]
    cost = params["c"]

    def switching_residual(ts):
        # Arc 1: no harvesting, u = 0
        state_arc_1 = solve_ivp(
            lambda t, x: r * x * (1.0 - x / K),
            (0.0, ts),
            [x0],
            rtol=1e-10,
            atol=1e-12
        )

        xs = state_arc_1.y[0, -1]

        # Arc 2: maximum harvesting, u = E
        state_arc_2 = solve_ivp(
            lambda t, x: (
                r * x * (1.0 - x / K)
                - q * E * x
            ),
            (ts, T),
            [xs],
            dense_output=True,
            rtol=1e-10,
            atol=1e-12
        )

        if not state_arc_2.success:
            raise RuntimeError(state_arc_2.message)

        # Integrate the costate backward on the harvesting arc.
        def adjoint_on_harvest(t, y):
            x = float(state_arc_2.sol(t)[0])

            return [
                -price * q * E
                - y[0] * (
                    r * (1.0 - 2.0 * x / K)
                    - q * E
                )
            ]

        costate_arc_2 = solve_ivp(
            adjoint_on_harvest,
            (T, ts),
            [0.0],               # y(T) = 0
            rtol=1e-10,
            atol=1e-12
        )

        if not costate_arc_2.success:
            raise RuntimeError(costate_arc_2.message)

        ys = costate_arc_2.y[0, -1]

        # Switching condition at t = ts
        return q * xs * (price - ys) - cost

    # Search for an interval in which the switching function changes sign.
    eps = max(1e-10, 1e-10 * T)
    grid = np.linspace(eps, T - eps, 201)

    values = np.array([
        switching_residual(t)
        for t in grid
    ])

    sign_changes = np.where(
        values[:-1] * values[1:] <= 0.0
    )[0]

    if len(sign_changes) == 0:
        raise RuntimeError(
            "No interior 0-to-E switching time was found."
        )

    if len(sign_changes) > 1:
        print(
            "Warning: multiple candidate switching times were found; "
            "using the first one."
        )

    i = sign_changes[0]

    ts = brentq(
        switching_residual,
        grid[i],
        grid[i + 1],
        xtol=1e-12,
        rtol=1e-12
    )

    print(f"Switching time ts = {ts:.9f}")
    print(f"S(ts) = {switching_residual(ts):.3e}")

    return ts

def rk2_step(y, h, params, u):
    k1 = params["J"] @ grad_hamiltonian(y, u, params)

    k2 = params["J"] @ grad_hamiltonian(
        y + 0.5*h*k1,
        u,
        params
    )

    return y + h*k2

def integrate_arc_rk2(x0_vec, dt, N, params, u):
    t = np.zeros(N + 1)
    y = np.zeros((N + 1, 2))

    y[0] = x0_vec
    x = x0_vec.copy()

    for k in range(N):
        x = rk2_step(
            x,
            dt,
            params,
            u
        )

        y[k + 1] = x
        t[k + 1] = t[k] + dt

    return t, y

def integrate_trajectory(lam0, ts, tf, params, x0, method):

    N1 = params["N"]
    N2 = params["N"]

    dt1 = ts / N1
    dt2 = (tf - ts) / N2

    x = np.array([
        x0,
        float(np.atleast_1d(lam0)[0])
    ])

    # ==================================================
    # HPI
    # ==================================================
    if method == "HPI":

        # Arc 1: u = 0
        t1, dyn1 = integrate_arc(
            x,
            dt1,
            N1,
            params,
            0.0
        )

        x = dyn1[-1].copy()

        # Arc 2: u = E
        t2, dyn2 = integrate_arc(
            x,
            dt2,
            N2,
            params,
            params["E"]
        )

        # integrate_arc starts time from zero
        t2 = t2 + ts

        t = np.concatenate([
            t1,
            t2[1:]
        ])

        y = np.vstack([
            dyn1,
            dyn2[1:]
        ])

    # ==================================================
    # RK2
    # ==================================================
    elif method == "RK2":

        # Arc 1: u = 0
        t1, dyn1 = integrate_arc_rk2(
            x,
            dt1,
            N1,
            params,
            0.0
        )

        x = dyn1[-1].copy()

        # Arc 2: u = E
        t2, dyn2 = integrate_arc_rk2(
            x,
            dt2,
            N2,
            params,
            params["E"]
        )

        # RK2 arc also starts from zero
        t2 = t2 + ts

        t = np.concatenate([
            t1,
            t2[1:]
        ])

        y = np.vstack([
            dyn1,
            dyn2[1:]
        ])

    else:
        raise ValueError("Wrong integrator.")

    return t, y

def shooting_residual(z, params, x0, ts, T, method):
    _, d = integrate_trajectory(z, ts, T, params, x0, method)
    return np.array([d[-1, 1]])

def diff_Hamiltonian(y, t, ts, params):
    """
    Compute Hamiltonian conservation error separately on each arc.

    Arc 1:
        u = 0

    Arc 2:
        u = E

    Returns
    -------
    H_diff : ndarray
        Complete Hamiltonian error vector.
    """

    H_diff = np.zeros(len(t))

    # Index corresponding to switching time
    i_switch = np.argmin(np.abs(t - ts))

    # =====================================
    # ARC 1 : u = 0
    # =====================================
    H1 = np.array([
        Hamiltonian(yi, 0.0, params)
        for yi in y[:i_switch + 1]
    ])

    H1_ref = H1[0]

    H_diff[:i_switch + 1] = np.abs(
        H1 - H1_ref
    )

    # =====================================
    # ARC 2 : u = E
    # =====================================

    # Reference Hamiltonian for second arc,
    # evaluated at the switching state
    H2_ref = Hamiltonian(
        y[i_switch],
        params["E"],
        params
    )

    H2 = np.array([
        Hamiltonian(yi, params["E"], params)
        for yi in y[i_switch + 1:]
    ])

    H_diff[i_switch + 1:] = np.abs(
        H2 - H2_ref
    )

    return H_diff

# Shooting
z0 = np.array([1.0])
ts = switch_time(x0, params, T)

sol_phi = root(lambda z: shooting_residual(z, params, x0, ts, T, "HPI"), z0)
sol_ode = root(lambda z: shooting_residual(z, params, x0, ts, T, "RK2"), z0)

z_phi = sol_phi.x[0]
z_ode = sol_ode.x[0]

t_phi, y_phi = integrate_trajectory(z_phi, ts, T, params, x0, "HPI")
t_ode, y_ode = integrate_trajectory(z_ode, ts, T, params, x0, "RK2")

ts_T = np.linspace(ts, T, 100);

idx_phi = np.argmin(np.abs(t_phi - ts))
idx_rk2 = np.argmin(np.abs(t_ode - ts))


# Profit integrand on harvesting arc: u = E
profit_phi = (
    params["p"] * params["q"] * params["E"] * y_phi[idx_phi:, 0]
    - params["c"] * params["E"]
)

profit_rk2 = (
    params["p"] * params["q"] * params["E"] * y_ode[idx_rk2:, 0]
    - params["c"] * params["E"]
)

# Integrate from ts to T
J_HPI = np.trapezoid(
    profit_phi,
    t_phi[idx_phi:]
)

J_RK2 = np.trapezoid(
    profit_rk2,
    t_ode[idx_rk2:]
)

# Hamiltonian difference along the flow
H_diff_phi = diff_Hamiltonian(
    y_phi,
    t_phi,
    ts,
    params
)

H_diff_rk2 = diff_Hamiltonian(
    y_ode,
    t_ode,
    ts,
    params
)

print(f"Shooting HPI converged: {sol_phi.success}")
print(f"Shooting RK2 converged: {sol_ode.success}")
print(f"Recovered (HPI) y(0) = {z_phi:.6f}")
print(f"Recovered (RK2) y(0) = {z_ode:.6f}")
print(f"HPI Final y(T) = {y_phi[-1,1]:.6f}")
print(f"RK2 Final y(T) = {y_ode[-1,1]:.6f}")
print(f"Switch time t1 = {ts:.6f} s, Final time tf = {T:.6f} s")
print(f"Function evaluations (HPI) = {sol_phi.nfev}, (RK2) = {sol_ode.nfev}")
print(f"Profit (HPI) = {J_HPI:.6f}, Profit (RK2) = {J_RK2:.6f}")

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(t_phi, y_phi[:,0], label="HPI")
ax[0].plot(t_ode, y_ode[:,0], "--", label="RK2")
ax[0].set_ylabel("x(t)")
ax[0].grid(True)
ax[0].legend()

ax[1].plot(t_phi, y_phi[:,1], label="HPI")
ax[1].plot(t_ode, y_ode[:,1], "--", label="RK2")
ax[1].set_ylabel("y(t)")
ax[1].set_xlabel("time")
ax[1].grid(True)
ax[1].legend()

fig.suptitle("Optimal trajectories")
plt.show()

fig, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(t_phi, H_diff_phi, '-', label='HPI')
ax2.plot(t_ode, H_diff_rk2, '--', label='RK2')
ax2.set_ylabel('Hamiltonian Error')
ax2.set_xlabel('time')
ax2.legend()
plt.title("Hamiltonian Error Harvesting")  
plt.tight_layout()
plt.show()
