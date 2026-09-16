## Minimum-energy swing-up of a pendulum via Pontryagin's Minimum Principle
# Solved with a shooting method (ode23 + fsolve).
#
# State:    theta (angle, 0 = down, pi = up), omega (angular velocity)
# Control:  u (torque)
# Dynamics: theta' = omega
#           omega' = -sin(theta) + u
# Cost:     J = integral_0^T (1/2) u^2 dt
#
# Boundary conditions:
#   theta(0) = 0, omega(0) = 0      (start at rest, hanging down)
#   theta(T) = pi, omega(T) = 0     (end at rest, balanced up)
#
# Hamiltonian:
#   H = 1/2 u^2 + p1*omega + p2*(-sin(theta) + u)
#
# Optimality condition (dH/du = 0):
#   u* = -p2
#
# Canonical (Hamiltonian) system after substituting u*:
#   theta' = omega
#   omega' = -sin(theta) - p2
#   p1'    = p2*cos(theta)
#   p2'    = -p1
#
# Unknowns to shoot for: p1(0), p2(0)

import numpy as np
import scipy.optimize as opt
import scipy.integrate as integrate
import matplotlib.pyplot as plt

optimization_options = {'xtol': 1e-12, 'maxfev': 1000}

Final = np.array([np.pi, 0])  # final state (theta, omega)

T = 6  # final time
n = 20 # number of time steps

theta_0 = 0
omega_0 = 0

tspan = np.linspace(0, T, n)
dt = tspan[1] - tspan[0]

def Hamiltonian(y):
    theta, omega, p1, p2 = y
    H = 0.5 * p2**2 + p1 * omega + p2 * (-np.sin(theta)-p2)
    return H

def  grad_Hamiltonian(y):
    theta, omega, p1, p2 = y
    grad_H = np.array([-p2 * np.cos(theta), p1, omega, -np.sin(theta) - p2])  # gradient of H with respect to [theta, omega, p1, p2]

    return grad_H

def poissonStep(x, dt):
    """Perform one step of the Poisson Hamiltonian Integrator.

    Parameters
    ----------
    y : array_like
        Current state and costate [theta, omega, p1, p2].
    dt : float
        Time step size.

    Returns
    -------
    y_next : ndarray
        Next state and costate after one integration step.
    """
    J = np.array([
    [ 0,  0, 1, 0],
    [ 0,  0, 0, 1],
    [-1,  0, 0, 0],
    [ 0, -1, 0, 0]
], dtype=float)

    y = x
    tol = 1e-12
    max_iter = 100
    converged = False

    for k in range(max_iter):
        # Compute the Hamiltonian vector field X_H = J * grad H
        theta, omega, p1, p2 = y
        grad_H = grad_Hamiltonian(y)

        y_new = x + 0.5 * dt * (J @ grad_H) 

        if np.linalg.norm(y_new - y, ord=np.inf) < tol * (1 + np.linalg.norm(y_new, ord=np.inf)):
            y = y_new
            converged = True
            break

        y = y_new


    if not converged:
        raise RuntimeError("Poisson step did not converge within the maximum number of iterations.")

    g = grad_Hamiltonian(y)
    y_next = y + 0.5 * dt * (J @ g) 

    return y_next
            
def PHI(y0, dt, n):
    """Integrate the canonical system using PHI (Poisson Hamiltonian Integrator).

    Parameters
    ----------
    y0 : array_like
        Initial state and costate [theta(0), omega(0), p1(0), p2(0)].
    dt : float
        Time step size.
    """
    y = np.zeros((n, 4))
    y[0, :] = y0

    for k in range(n - 1):
        y[k + 1, :] = poissonStep(y[k, :], dt)

    return y

def residual_PHI(p0, dt, n, Final, initial_cond):
    y0 = np.array([
        initial_cond[0],
        initial_cond[1],
        p0[0],
        p0[1]
    ], dtype=float)

    y = PHI(y0, dt, n)

    return y[-1, :2] - Final

def rk2_step(y, h):
    J = np.array([
        [ 0,  0, 1, 0],
        [ 0,  0, 0, 1],
        [-1,  0, 0, 0],
        [ 0, -1, 0, 0]
        ], dtype=float)
    k1 = J @ grad_Hamiltonian(y)
    k2 = J @ grad_Hamiltonian(y + 0.5*h*k1)
    return y + h*k2

def RK2(y0, dt, n):
    y = np.zeros((n, 4))
    y[0] = y0

    for k in range(n-1):
        y[k+1] = rk2_step(y[k], dt)

    return y

def residual_ode23(p0, dt, n, Final, initial_cond):
    y0 = np.array([
        initial_cond[0],
        initial_cond[1],
        p0[0],
        p0[1]
    ], dtype=float)


    tspan = np.linspace(0, dt * (n - 1), n)
    y = RK2(y0, dt, n)

    return y[-1, :2] - Final


## Shooting: find p0 = [p1(0), p2(0)] such that theta(T)=pi, omega(T)=0

p0_guess = np.array([1, 1])  # initial guess for p0

p0_optimal_PHI, info, ier, mesg = opt.fsolve(
    residual_PHI,
    p0_guess,
    args=(dt, n, Final, [theta_0, omega_0]),
    full_output=True,
    **optimization_options
)

print("HPI Solution:", p0_optimal_PHI)
print("HPI Func-count:", info["nfev"])
print("HPI Message:", mesg)

p0_optimal_ode23, info_ode23, ier_ode23, mesg_ode23 = opt.fsolve(
    residual_ode23,
     p0_guess, 
     args=(dt, n, Final, [theta_0, omega_0]), 
     full_output=True, 
     **optimization_options
)

print("\n")

print("RK2 Solution:", p0_optimal_ode23)
print("RK2 Func-count:", info_ode23["nfev"])
print("RK2 Message:", mesg_ode23)


## Re-integrate the canonical system with the optimal p0 to get the full trajectory

J = np.array([
    [ 0,  0, 1, 0],
    [ 0,  0, 0, 1],
    [-1,  0, 0, 0],
    [ 0, -1, 0, 0]
    ], dtype=float)

optimal_y_PHI = PHI(np.array([theta_0, omega_0, p0_optimal_PHI[0], p0_optimal_PHI[1]]), dt, n)
optimal_y_ode23 = RK2(np.array([theta_0, omega_0, p0_optimal_ode23[0], p0_optimal_ode23[1]]), dt, n)

print("\n")
print("Targets: ", Final)
print("HPI Final State: ", optimal_y_PHI[-1, :2])
print("RK2 Final State: ", optimal_y_ode23[-1, :2])

print("\n")
J_PHI = np.trapezoid(0.5*optimal_y_PHI[:, 3]**2, tspan)
J_RK2 = np.trapezoid(0.5*optimal_y_ode23[:, 3]**2, tspan)

print("HPI Cost: ", J_PHI)
print("RK2 Cost: ", J_RK2)

## Hamiltonian along flow

H_PHI = [Hamiltonian(y) for y in optimal_y_PHI]
H_RK2 = [Hamiltonian(y) for y in optimal_y_ode23]

H0_PHI = H_PHI[0]
H0_RK2 = H_RK2[0]

diff_PHI = [abs(H - H0_PHI) for H in H_PHI]
diff_RK2 = [abs(H - H0_RK2) for H in H_RK2]


## Plot the results

fig, ax = plt.subplots(3, 1, figsize=(8, 9))

# theta and omega trajectories
ax[0].plot(tspan, optimal_y_PHI[:, 0],'-' ,label='HPI')
ax[0].plot(tspan, optimal_y_ode23[:, 0], '--', label='RK2')
ax[0].set_ylabel(r'$\theta(t)$')
ax[0].legend()

ax[1].plot(tspan, optimal_y_PHI[:, 1],'-' ,label='HPI')
ax[1].plot(tspan, optimal_y_ode23[:, 1], '--', label='RK2')
ax[1].set_ylabel(r'$\omega(t)$')
ax[1].legend()

# costate trajectories
ax[2].plot(tspan, -optimal_y_PHI[:, 3],'-', label='HPI')
ax[2].plot(tspan, -optimal_y_ode23[:, 3], '--', label='RK2')
ax[2].set_ylabel('u (control)')
ax[2].legend()


fig.suptitle('Pendulum swing-up: optimal trajectory (shooting method)')

plt.tight_layout()
plt.show()

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(tspan, diff_PHI, '-', label='HPI')
ax2.plot(tspan, diff_RK2, '--', label='RK2')
ax2.set_ylabel('Hamiltonian Error')
ax2.legend()
ax2.set_title('Pendulum swing-up: Hamiltonian Error (shooting method)')
plt.tight_layout()
plt.show()



