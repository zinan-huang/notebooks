"""
α^β via Bounded GPAC — Full Simulation Code

Construction: given upstream GPACs computing x(t) → α and y(t) → β,
the system
    x₁' = (x - 1) - x₁
    u'  = (1 - v) x₁'
    v'  = (1 - v)² x₁'
    z'  = z (y' u + y (1-v) x₁')
with x₁(0) = u(0) = v(0) = 0, z(0) = 1
computes z(t) → α^β.

Key identity: z(t) = exp(y(t) · u(t)), so z → exp(β ln α) = α^β.

Run this script or open it in Google Colab / Jupyter.

Requirements: numpy, scipy, matplotlib
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def simulate_alpha_beta(alpha_func, alpha_deriv, beta_func, beta_deriv,
                        alpha_star, beta_star, T=20):
    """
    Simulate the α^β bounded GPAC construction.

    Parameters
    ----------
    alpha_func : callable
        x(t) → α as t → ∞
    alpha_deriv : callable
        x'(t)
    beta_func : callable
        y(t) → β as t → ∞
    beta_deriv : callable
        y'(t)
    alpha_star, beta_star : float
        Target values
    T : float
        Simulation time

    Returns
    -------
    t : array
    x1, u, v, z : arrays (internal variables)
    target : float (α^β)
    """
    target = alpha_star ** beta_star

    def rhs(t, state):
        x1, u, v, z = state
        x_t = alpha_func(t)
        y_t = beta_func(t)
        dy = beta_deriv(t)

        x1_dot = (x_t - 1) - x1
        u_dot = (1 - v) * x1_dot
        v_dot = (1 - v)**2 * x1_dot
        z_dot = z * (dy * u + y_t * (1 - v) * x1_dot)

        return [x1_dot, u_dot, v_dot, z_dot]

    sol = solve_ivp(rhs, [0.001, T], [0, 0, 0, 1],
                    max_step=0.005, dense_output=True,
                    rtol=1e-12, atol=1e-14)
    t = np.linspace(0.001, T, 3000)
    y = sol.sol(t)
    return t, y[0], y[1], y[2], y[3], target


def plot_full_system(t, x1, u, v, z, target,
                     alpha_func, beta_func,
                     alpha_star, beta_star, title=""):
    """Plot all four panels for a single α^β computation."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    T = t[-1]

    # (a) Inputs
    ax = axes[0, 0]
    ax.plot(t, alpha_func(t), 'C0', lw=2, label=f'α(t) → {alpha_star:.4f}')
    ax.plot(t, beta_func(t), 'C1', lw=2, label=f'β(t) → {beta_star:.4f}')
    ax.axhline(y=alpha_star, color='C0', ls='--', alpha=0.3)
    ax.axhline(y=beta_star, color='C1', ls='--', alpha=0.3)
    ax.set_xlabel('Time t'); ax.set_title('(a) Inputs'); ax.legend()

    # (b) Internal variables
    ax = axes[0, 1]
    ax.plot(t, x1, 'C2', lw=2, label=f'x₁ → {alpha_star-1:.4f}')
    ax.plot(t, u, 'C3', lw=2, label=f'u → ln(α) = {np.log(alpha_star):.4f}')
    ax.plot(t, v, 'C4', lw=2, label=f'v → (α-1)/α = {(alpha_star-1)/alpha_star:.4f}')
    ax.set_xlabel('Time t'); ax.set_title('(b) Internal variables')
    ax.legend(fontsize=8)

    # (c) Output
    ax = axes[1, 0]
    ax.plot(t, z, 'C3', lw=2.5, label=f'z(t) → α^β ≈ {target:.4f}')
    ax.axhline(y=target, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Time t'); ax.set_ylabel('z(t)')
    ax.set_title('(c) Output'); ax.legend()

    # (d) Error
    ax = axes[1, 1]
    err = np.abs(z - target)
    ax.semilogy(t, np.maximum(err, 1e-15), 'C3', lw=2)
    ax.set_xlabel('Time t'); ax.set_ylabel('|z(t) - α^β|')
    ax.set_title('(d) Error (log scale)'); ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


# ==============================
# Example 1: (π/4)^e
# ==============================
pi4 = lambda t: np.pi/4 * (1 - np.exp(-t))
pi4d = lambda t: np.pi/4 * np.exp(-t)
efn = lambda t: np.e * (1 - np.exp(-t))
efd = lambda t: np.e * np.exp(-t)

t, x1, u, v, z, tgt = simulate_alpha_beta(pi4, pi4d, efn, efd, np.pi/4, np.e, T=15)
fig1 = plot_full_system(t, x1, u, v, z, tgt, pi4, efn, np.pi/4, np.e,
                        title='Computing (π/4)^e via Bounded GPAC')
print(f"(π/4)^e: z(T) = {z[-1]:.8f}, target = {tgt:.8f}, error = {abs(z[-1]-tgt):.2e}")

# ==============================
# Example 2: e^π
# ==============================
pifn = lambda t: np.pi * (1 - np.exp(-t))
pifd = lambda t: np.pi * np.exp(-t)

t2, x12, u2, v2, z2, tgt2 = simulate_alpha_beta(efn, efd, pifn, pifd, np.e, np.pi, T=15)
fig2 = plot_full_system(t2, x12, u2, v2, z2, tgt2, efn, pifn, np.e, np.pi,
                        title='Computing e^π via Bounded GPAC')
print(f"e^π: z(T) = {z2[-1]:.8f}, target = {tgt2:.8f}, error = {abs(z2[-1]-tgt2):.2e}")

plt.show()
