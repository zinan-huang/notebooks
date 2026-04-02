#!/usr/bin/env python3
"""
Full α^β simulation using ACTUAL DNA 25 ODE constructions for e and π/4.

e construction (Theorem 4, Huang-Klinge-Lathrop 2019):
  x' = -x,  y' = -xy,  x(0)=1, y(0)=1
  → x(t) = e^{-t},  y(t) = e^{1-e^{-t}} → e

π/4 construction (Theorem 5):
  w' = -w,  x' = -2wxy,  y' = wx² - wy²,  z' = wx
  w(0)=x(0)=1, y(0)=z(0)=0
  → z(t) = arctan(1-e^{-t}) → π/4

Then feed these into the α^β construction.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'text.usetex': False,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 300,
})


def full_system_pi4_e(t, state):
    """
    Combined system: DNA25 constructions for π/4 and e,
    feeding into the α^β construction.

    State vector (13 variables):
      [0] w_pi  — π/4 construction: w' = -w
      [1] x_pi  — π/4 construction: x' = -2wxy
      [2] y_pi  — π/4 construction: y' = wx² - wy²
      [3] z_pi  — π/4 construction: z' = wx  (→ π/4)
      [4] x_e   — e construction: x' = -x
      [5] y_e   — e construction: y' = -xy  (→ e)
      [6] x1    — α^β: x1' = (α - 1) - x1
      [7] u     — α^β: u' = (1-v) x1'
      [8] v     — α^β: v' = (1-v)² x1'
      [9] z_out — α^β: z' = z(y'u + y(1-v)x1')  (→ (π/4)^e)
    """
    w_pi, x_pi, y_pi, z_pi, x_e, y_e, x1, u, v, z_out = state

    # --- DNA25 π/4 construction ---
    dw_pi = -w_pi
    dx_pi = -2 * w_pi * x_pi * y_pi
    dy_pi = w_pi * x_pi**2 - w_pi * y_pi**2
    dz_pi = w_pi * x_pi  # z_pi → π/4

    # --- DNA25 e construction ---
    # y(t) = e^{1-e^{-t}} → e (Theorem 4, Huang-Klinge-Lathrop 2019)
    dx_e = -x_e
    dy_e = x_e * y_e  # y_e → e

    # --- α^β construction with α = z_pi (→π/4), β = y_e (→e) ---
    alpha_t = z_pi
    beta_t = y_e
    dalpha = dz_pi
    dbeta = dy_e

    dx1 = (alpha_t - 1) - x1
    du = (1 - v) * dx1
    dv = (1 - v)**2 * dx1
    dz_out = z_out * (dbeta * u + beta_t * (1 - v) * dx1)

    return [dw_pi, dx_pi, dy_pi, dz_pi, dx_e, dy_e, dx1, du, dv, dz_out]


def full_system_e_pi(t, state):
    """
    Combined system for e^π.
    α = y_e (→ e), β = 4*z_pi (→ π).  Since RRT is a field,
    4*z_pi → π.  But simpler: just use z_pi → π/4 and
    compute e^(4*z_pi) by adjusting β.

    Actually, let's build a separate π construction:
    same as π/4 but multiply by 4.  Or just compute e^π directly
    by using β = 4 * z_pi.
    """
    w_pi, x_pi, y_pi, z_pi, x_e, y_e, x1, u, v, z_out = state

    # DNA25 constructions (same as above)
    dw_pi = -w_pi
    dx_pi = -2 * w_pi * x_pi * y_pi
    dy_pi = w_pi * x_pi**2 - w_pi * y_pi**2
    dz_pi = w_pi * x_pi

    dx_e = -x_e
    dy_e = x_e * y_e  # y_e → e

    # α = y_e → e,  β = 4*z_pi → π
    alpha_t = y_e
    beta_t = 4 * z_pi
    dalpha = dy_e
    dbeta = 4 * dz_pi

    dx1 = (alpha_t - 1) - x1
    du = (1 - v) * dx1
    dv = (1 - v)**2 * dx1
    dz_out = z_out * (dbeta * u + beta_t * (1 - v) * dx1)

    return [dw_pi, dx_pi, dy_pi, dz_pi, dx_e, dy_e, dx1, du, dv, dz_out]


T = 25
t_eval = np.linspace(0.001, T, 5000)

# Initial conditions:
# π/4: w=1, x=1, y=0, z=0
# e:   x=1, y=1
# α^β: x1=0, u=0, v=0, z=1
y0 = [1, 1, 0, 0,  1, 1,  0, 0, 0, 1]

# ============================================================
# (π/4)^e
# ============================================================
sol1 = solve_ivp(full_system_pi4_e, [0.001, T], y0,
                 t_eval=t_eval, max_step=0.01,
                 rtol=1e-12, atol=1e-14)
target1 = (np.pi/4)**np.e

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# (a) DNA25 inputs
ax = axes[0, 0]
ax.plot(sol1.t, sol1.y[3], 'C0', lw=2, label='$z_{\\pi}(t) \\to \\pi/4$ (DNA25)')
ax.plot(sol1.t, sol1.y[5], 'C1', lw=2, label='$y_e(t) \\to e$ (DNA25)')
ax.axhline(y=np.pi/4, color='C0', ls='--', alpha=0.3)
ax.axhline(y=np.e, color='C1', ls='--', alpha=0.3)
ax.set_xlabel('Time $t$'); ax.set_title('(a) DNA25 inputs'); ax.legend()
ax.set_xlim(0, T)

# (b) Internal variables
ax = axes[0, 1]
ax.plot(sol1.t, sol1.y[6], 'C2', lw=2,
        label=f'$x_1 \\to \\pi/4 - 1 = {np.pi/4-1:.4f}$')
ax.plot(sol1.t, sol1.y[7], 'C3', lw=2,
        label=f'$u \\to \\ln(\\pi/4) = {np.log(np.pi/4):.4f}$')
ax.plot(sol1.t, sol1.y[8], 'C4', lw=2,
        label=f'$v \\to {(np.pi/4-1)/(np.pi/4):.4f}$')
ax.set_xlabel('Time $t$'); ax.set_title('(b) Internal variables')
ax.legend(fontsize=8); ax.set_xlim(0, T)

# (c) Output
ax = axes[1, 0]
ax.plot(sol1.t, sol1.y[9], 'C3', lw=2.5,
        label=f'$z(t) \\to (\\pi/4)^e \\approx {target1:.4f}$')
ax.axhline(y=target1, color='gray', ls='--', alpha=0.5)
ax.set_xlabel('Time $t$'); ax.set_ylabel('$z(t)$')
ax.set_title('(c) Output: $(\\pi/4)^e$'); ax.legend()
ax.set_xlim(0, T); ax.set_ylim(0, 1.15)

# (d) Error
ax = axes[1, 1]
err1 = np.abs(sol1.y[9] - target1)
ax.semilogy(sol1.t, np.maximum(err1, 1e-15), 'C3', lw=2)
ax.set_xlabel('Time $t$'); ax.set_ylabel('Error')
ax.set_title('(d) Error decay (log scale)')
ax.set_xlim(0, T); ax.grid(True, alpha=0.3)

plt.suptitle('Computing $(\\pi/4)^e$ — Pure GPAC (DNA25 inputs)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('/Users/huangx/repos/zinan-blog/static/images/alpha_beta_pi4_e.png',
            bbox_inches='tight', dpi=200)
print(f'(π/4)^e: z(T)={sol1.y[9,-1]:.8f}, target={target1:.8f}, err={abs(sol1.y[9,-1]-target1):.2e}')

# ============================================================
# e^π
# ============================================================
sol2 = solve_ivp(full_system_e_pi, [0.001, T], y0,
                 t_eval=t_eval, max_step=0.01,
                 rtol=1e-12, atol=1e-14)
target2 = np.e**np.pi

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

ax = axes[0, 0]
ax.plot(sol2.t, sol2.y[5], 'C0', lw=2, label='$y_e(t) \\to e$ (DNA25)')
ax.plot(sol2.t, 4*sol2.y[3], 'C1', lw=2, label='$4z_{\\pi}(t) \\to \\pi$ (DNA25)')
ax.axhline(y=np.e, color='C0', ls='--', alpha=0.3)
ax.axhline(y=np.pi, color='C1', ls='--', alpha=0.3)
ax.set_xlabel('Time $t$'); ax.set_title('(a) DNA25 inputs'); ax.legend()
ax.set_xlim(0, T)

ax = axes[0, 1]
ax.plot(sol2.t, sol2.y[6], 'C2', lw=2, label=f'$x_1 \\to e-1 = {np.e-1:.4f}$')
ax.plot(sol2.t, sol2.y[7], 'C3', lw=2, label='$u \\to \\ln(e) = 1$')
ax.plot(sol2.t, sol2.y[8], 'C4', lw=2, label=f'$v \\to (e-1)/e = {(np.e-1)/np.e:.4f}$')
ax.set_xlabel('Time $t$'); ax.set_title('(b) Internal variables')
ax.legend(fontsize=8); ax.set_xlim(0, T)

ax = axes[1, 0]
ax.plot(sol2.t, sol2.y[9], 'C0', lw=2.5,
        label=f'$z(t) \\to e^\\pi \\approx {target2:.4f}$')
ax.axhline(y=target2, color='gray', ls='--', alpha=0.5)
ax.set_xlabel('Time $t$'); ax.set_ylabel('$z(t)$')
ax.set_title('(c) Output: $e^\\pi$'); ax.legend()
ax.set_xlim(0, T)

ax = axes[1, 1]
err2 = np.abs(sol2.y[9] - target2)
ax.semilogy(sol2.t, np.maximum(err2, 1e-15), 'C0', lw=2)
ax.set_xlabel('Time $t$'); ax.set_ylabel('Error')
ax.set_title('(d) Error decay (log scale)')
ax.set_xlim(0, T); ax.grid(True, alpha=0.3)

plt.suptitle('Computing $e^\\pi$ — Pure GPAC (DNA25 inputs)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('/Users/huangx/repos/zinan-blog/static/images/alpha_beta_e_pi.png',
            bbox_inches='tight', dpi=200)
print(f'e^π: z(T)={sol2.y[9,-1]:.8f}, target={target2:.8f}, err={abs(sol2.y[9,-1]-target2):.2e}')

print('Done.')
