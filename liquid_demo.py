"""
Liquid Neural Network (LTC) demo — from scratch in PyTorch.

Implements the Liquid Time-Constant neuron from Hasani et al. 2021
("Liquid Time-constant Networks", AAAI):

    dx/dt = -[ 1/tau + f(u, x) ] * x  +  f(u, x) * A

where f is a small learned gate (sigmoid). Two things make it "liquid":
  1. The effective time constant  tau_eff = 1 / (1/tau + f(u,x))
     depends on the CURRENT INPUT — the neuron speeds up or slows down
     its own dynamics in response to what it sees.
  2. Time is explicit: state is advanced by integrating an ODE over a
     real time interval dt, so irregular sampling is handled natively.

Part A: simulate one liquid neuron, visualize tau_eff reacting to input.
Part B: train an LTC network vs a GRU on irregularly-sampled signal
        prediction, then test at sampling rates never seen in training.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)

BLUE, ORANGE, GRAY = "#2563EB", "#EA580C", "#6B7280"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "font.size": 10,
})


# ----------------------------------------------------------------------
# The LTC cell — the whole idea is in solver_step()
# ----------------------------------------------------------------------
class LTCCell(nn.Module):
    """Liquid time-constant cell, fused semi-implicit Euler solver."""

    def __init__(self, in_dim, hidden, solver_steps=4):
        super().__init__()
        self.hidden = hidden
        self.solver_steps = solver_steps
        # the gate f(u, x): input- and state-dependent conductance
        self.gate = nn.Linear(in_dim + hidden, hidden)
        # per-neuron base time constant tau > 0 and resting target A
        self.log_tau = nn.Parameter(torch.zeros(hidden))
        self.A = nn.Parameter(torch.randn(hidden) * 0.3)

    def solver_step(self, x, u, dt):
        """Advance state x over a real time interval dt (semi-implicit).

        Discretizing  dx/dt = -(1/tau + f)x + f*A  gives
            x_next = (x + dt*f*A) / (1 + dt*(1/tau + f))
        which is unconditionally stable (state stays bounded).
        """
        tau = torch.exp(self.log_tau)
        h = dt / self.solver_steps
        for _ in range(self.solver_steps):
            f = torch.sigmoid(self.gate(torch.cat([u, x], -1)))
            x = (x + h * f * self.A) / (1.0 + h * (1.0 / tau + f))
        return x

    def tau_eff(self, x, u):
        """Effective time constant right now — the 'liquid' quantity."""
        f = torch.sigmoid(self.gate(torch.cat([u, x], -1)))
        return 1.0 / (1.0 / torch.exp(self.log_tau) + f)


class LTCNet(nn.Module):
    """LTC cell + linear readout, run over a sequence of (u_t, dt_t)."""

    def __init__(self, in_dim=1, hidden=32):
        super().__init__()
        self.cell = LTCCell(in_dim, hidden)
        self.readout = nn.Linear(hidden, 1)

    def forward(self, u, dt):
        # u: (B, T, 1) observations; dt: (B, T, 1) time since last sample
        B, T, _ = u.shape
        x = torch.zeros(B, self.cell.hidden)
        outs = []
        for t in range(T):
            x = self.cell.solver_step(x, u[:, t], dt[:, t])
            outs.append(self.readout(x))
        return torch.stack(outs, 1)


class GRUNet(nn.Module):
    """Baseline: GRU that receives dt as an extra input feature."""

    def __init__(self, in_dim=1, hidden=32):
        super().__init__()
        self.cell = nn.GRUCell(in_dim + 1, hidden)
        self.readout = nn.Linear(hidden, 1)
        self.hidden = hidden

    def forward(self, u, dt):
        B, T, _ = u.shape
        x = torch.zeros(B, self.hidden)
        outs = []
        for t in range(T):
            x = self.cell(torch.cat([u[:, t], dt[:, t]], -1), x)
            outs.append(self.readout(x))
        return torch.stack(outs, 1)


# ----------------------------------------------------------------------
# Part A — watch one liquid neuron adapt its own time constant
# ----------------------------------------------------------------------
def part_a():
    cell = LTCCell(in_dim=1, hidden=1, solver_steps=1)
    with torch.no_grad():  # hand-set weights for a clean illustration
        cell.gate.weight[:] = torch.tensor([[3.0, 0.0]])
        cell.gate.bias[:] = torch.tensor([-2.0])
        cell.log_tau[:] = np.log(2.0)   # slow baseline: tau = 2 s
        cell.A[:] = 1.0                  # excited target state

    T, dt = 1200, 0.01
    t = np.arange(T) * dt
    u = np.zeros(T, dtype=np.float32)
    u[(t > 1) & (t < 3)] = 0.8           # weak pulse
    u[(t > 5) & (t < 7)] = 3.0           # strong pulse

    xs, taus, x = [], [], torch.zeros(1, 1)
    with torch.no_grad():
        for k in range(T):
            uk = torch.tensor([[u[k]]])
            taus.append(cell.tau_eff(x, uk).item())
            x = cell.solver_step(x, uk, torch.tensor(dt))
            xs.append(x.item())

    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    for ax, y, c, label in [
        (axes[0], u, GRAY, "input u(t)"),
        (axes[1], xs, BLUE, "neuron state x(t)"),
        (axes[2], taus, ORANGE, "effective time constant τ_eff(t)  [s]"),
    ]:
        ax.plot(t, y, color=c, lw=2)
        ax.set_ylabel(label)
    axes[0].set_title("One liquid neuron: strong input → smaller τ_eff → faster dynamics")
    axes[2].set_xlabel("time [s]")
    axes[1].annotate("slow rise\n(weak input)", xy=(2.2, 0.25), color=BLUE)
    axes[1].annotate("fast rise\n(strong input)", xy=(5.5, 0.55), color=BLUE)
    fig.tight_layout()
    fig.savefig("liquid_neuron.png", dpi=140)
    print(f"[Part A] tau_eff at rest: {taus[0]:.2f}s | during weak pulse: "
          f"{taus[250]:.2f}s | during strong pulse: {taus[650]:.2f}s")


# ----------------------------------------------------------------------
# Part B — irregular-time prediction, tested outside the training regime
# ----------------------------------------------------------------------
def signal(t):
    return np.sin(t) + 0.5 * np.sin(2.7 * t)


def make_batch(n_seq, seq_len, dt_lo, dt_hi):
    """Sequences of the signal sampled at IRREGULAR intervals.
    Model sees y(t_k) and dt_k, must predict y(t_{k+1})."""
    t0 = np.random.uniform(0, 20, (n_seq, 1))
    dts = np.random.uniform(dt_lo, dt_hi, (n_seq, seq_len + 1))
    times = t0 + np.cumsum(dts, axis=1)
    y = signal(times)
    u = torch.tensor(y[:, :-1, None], dtype=torch.float32)
    target = torch.tensor(y[:, 1:, None], dtype=torch.float32)
    dt = torch.tensor(dts[:, 1:, None], dtype=torch.float32)
    return u, dt, target, times


def part_b():
    TRAIN_DT = (0.10, 0.30)   # sampling intervals seen in training
    TEST_DT = (0.45, 0.75)    # 2.5x slower — never seen in training

    ltc, gru = LTCNet(), GRUNet()
    print(f"[Part B] params — LTC: {sum(p.numel() for p in ltc.parameters())}, "
          f"GRU: {sum(p.numel() for p in gru.parameters())}")

    for name, model in [("LTC", ltc), ("GRU", gru)]:
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        for step in range(400):
            u, dt, target, _ = make_batch(64, 40, *TRAIN_DT)
            loss = nn.functional.mse_loss(model(u, dt), target)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 100 == 0:
                print(f"  {name} step {step:3d}  train mse {loss.item():.4f}")

    def eval_mse(model, dt_range, n=256):
        with torch.no_grad():
            u, dt, target, _ = make_batch(n, 40, *dt_range)
            return nn.functional.mse_loss(model(u, dt), target).item()

    results = {}
    for name, model in [("LTC", ltc), ("GRU", gru)]:
        results[name] = (eval_mse(model, TRAIN_DT), eval_mse(model, TEST_DT))
        print(f"  {name}: mse @ train rate {results[name][0]:.4f} | "
              f"@ UNSEEN slow rate {results[name][1]:.4f}")

    # --- plot: one test sequence at the unseen rate + mse comparison ---
    u, dt, target, times = make_batch(1, 60, *TEST_DT)
    with torch.no_grad():
        p_ltc, p_gru = ltc(u, dt).squeeze(), gru(u, dt).squeeze()
    tt = times[0, 1:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), width_ratios=[2.4, 1])
    ax1.plot(tt, target.squeeze(), color=GRAY, lw=2, ls="--", label="ground truth")
    ax1.plot(tt, p_ltc, color=BLUE, lw=2, label="LTC (liquid)")
    ax1.plot(tt, p_gru, color=ORANGE, lw=2, label="GRU baseline")
    ax1.set_title(f"Next-value prediction at UNSEEN sampling rate "
                  f"(Δt {TEST_DT[0]}–{TEST_DT[1]}s; trained on {TRAIN_DT[0]}–{TRAIN_DT[1]}s)")
    ax1.set_xlabel("time [s]"); ax1.set_ylabel("signal")
    ax1.legend(frameon=False)

    labels = ["train rate", "unseen rate"]
    xpos = np.arange(2)
    ax2.bar(xpos - 0.18, [results["LTC"][0], results["LTC"][1]], 0.32,
            color=BLUE, label="LTC")
    ax2.bar(xpos + 0.18, [results["GRU"][0], results["GRU"][1]], 0.32,
            color=ORANGE, label="GRU")
    for i, name in enumerate(["LTC", "GRU"]):
        for j in range(2):
            v = results[name][j]
            ax2.text(xpos[j] + (-0.18 if i == 0 else 0.18), v, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=8, color="#111827")
    ax2.set_xticks(xpos, labels); ax2.set_ylabel("test MSE")
    ax2.set_title("Generalization across timescales")
    ax2.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("liquid_vs_gru.png", dpi=140)


if __name__ == "__main__":
    part_a()
    part_b()
    print("saved: liquid_neuron.png, liquid_vs_gru.png")
