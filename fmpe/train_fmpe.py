# Train Flow-Matching Posterior Estimation (FMPE) on SBIBM "TwoMoons"
# and plot posterior samples for a few test observations.
# Default settings here should be sufficient to get ~good (but not sharp) posteriors.

import os
import random
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
from tqdm import tqdm


@dataclass
class Config:
    seed: int = 123
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Simulation / dataset
    task_name: str = "two_moons"
    num_sims: int = 100_000
    val_frac: float = 0.05
    batch_size: int = 512

    # FMPE training
    lr: float = 2e-4
    epochs: int = 20
    sigma_min: float = 0.01  # tighten path near t=1
    alpha_t_prior: float = 2.0   # p(t) ∝ t^{alpha/(1+alpha)} bias toward t≈1
    clip_grad: float = 5.0

    # ODE integration for inference
    ode_steps: int = 600
    ode_solver: str = "rk4"        # "euler" or "rk4"
    num_posterior_samples: int = 10_000

    # Plotting
    n_test_obs: int = 3
    outdir: str = "./fmpe_twomoons_outputs"


cfg = Config()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(cfg.seed)


def build_twomoons_data(num_sims, device):
    """
    Returns:
        train_ds (TensorDataset): (theta, x) pairs for training
        val_ds (TensorDataset):   (theta, x) pairs for validation
        task: sbibm task object (for observations & references)
        observations: list of 10 canonical observations (tensors)
    """
    import sbibm
    task = sbibm.get_task(cfg.task_name)

    simulator = task.get_simulator()
    prior = task.get_prior()                     # distribution over theta (2D)
    # Generate simulation pairs for training
    thetas = prior(num_samples=num_sims)         # <-- keep this API as requested
    xs = simulator(thetas)                       # [N, 2]

    # Fixed benchmark observations for testing/posterior plots
    observations = [task.get_observation(i) for i in range(1, 11)]  # 10 canonical obs

    # Shuffle & split
    idx = torch.randperm(num_sims)
    n_val = int(cfg.val_frac * num_sims)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    train_ds = TensorDataset(thetas[tr_idx].to(device), xs[tr_idx].to(device))
    val_ds   = TensorDataset(thetas[val_idx].to(device), xs[val_idx].to(device))
    return train_ds, val_ds, task, observations


class FMPEVectorField(nn.Module):
    """
    Concatenation MLP: input [t, theta_t, x] -> velocity v in R^2.
    Works well for low-dim x (TwoMoons: x_dim=2).
    """
    def __init__(self, x_dim=2, theta_dim=2, hidden=256, depth=5):
        super().__init__()
        d_in = 1 + theta_dim + x_dim
        layers = [nn.Linear(d_in, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, theta_dim)

    def forward(self, t, x, theta_t):
        # t: [B,1], x: [B,2], theta_t: [B,2] -> v: [B,2]
        z = torch.cat([t, theta_t, x], dim=-1)
        return self.head(self.net(z))


def sample_t_powerlaw(batch_size, k_exponent: float, device):
    """
    Sample t ~ C * t^k on [0,1] with inverse-CDF: T = U^{1/(k+1)}.
    k >= 0 biases draws toward t -> 1.
    """
    u = torch.rand(batch_size, 1, device=device)
    return u.pow(1.0 / (k_exponent + 1.0))


def ot_gaussian_sigma(t, sigma_min):
    """\sigma_t = 1 - (1 - \sigma_min) * t  (from 1 at t=0 to \sigma_min at t=1)."""
    return 1.0 - (1.0 - sigma_min) * t


def sample_theta_t(theta1, t, sigma_min):
    """
    θ_t ~ N( t * θ1, \sigma_t^2 I ), with \sigma_t from ot_gaussian_sigma.
    Shapes: theta1 [B, D], t [B,1]
    """
    sigma_t = ot_gaussian_sigma(t, sigma_min)  # [B,1]
    eps = torch.randn_like(theta1)
    return t * theta1 + sigma_t * eps


def u_target(theta_t, theta1, t, sigma_min):
    """
    u_t(θ_t | θ1) = (θ1 - (1 - \sigma_min) * θ_t) / (1 - (1 - \sigma_min) * t)
    Shapes: theta_t, theta1 [B,D], t [B,1]
    """
    denom = 1.0 - (1.0 - sigma_min) * t  # [B,1]
    return (theta1 - (1.0 - sigma_min) * theta_t) / denom


def train_fmpe(model, train_ds, val_ds, cfg):
    model = model.to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    best_val = float("inf")
    os.makedirs(cfg.outdir, exist_ok=True)

    k_exp = cfg.alpha_t_prior / (1.0 + cfg.alpha_t_prior)  # power-law exponent for t

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for theta1, x in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False):
            B = theta1.size(0)
            t = sample_t_powerlaw(B, k_exp, cfg.device)      # [B,1]
            theta_t = sample_theta_t(theta1, t, cfg.sigma_min)
            u = u_target(theta_t, theta1, t, cfg.sigma_min)

            v = model(t, x, theta_t)
            loss = F.mse_loss(v, u)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            opt.step()

            running += loss.item() * B

        train_loss = running / len(train_loader.dataset)

        # validation with same t-prior
        model.eval()
        with torch.no_grad():
            running = 0.0
            count = 0
            for theta1, x in val_loader:
                B = theta1.size(0)
                t = sample_t_powerlaw(B, k_exp, cfg.device)
                theta_t = sample_theta_t(theta1, t, cfg.sigma_min)
                u = u_target(theta_t, theta1, t, cfg.sigma_min)
                v = model(t, x, theta_t)
                loss = F.mse_loss(v, u, reduction="sum")
                running += loss.item()
                count += B
            val_loss = running / count

        print(f"[Epoch {epoch}] train={train_loss:.6f}  val={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__},
                       os.path.join(cfg.outdir, "fmpe_twomoons_best.pt"))
    return model


@torch.no_grad()
def integrate_ode(model, x, num_samples, steps=600, solver="rk4", device="cpu"):
    model.eval()
    theta_dim = 2
    t0, t1 = 0.0, 1.0
    dt = (t1 - t0) / steps

    # Base samples
    theta = torch.randn(num_samples, theta_dim, device=device)

    # Robustly normalize x to shape [2] then broadcast to [num_samples, 2]
    x_vec = x.to(device)
    # squeeze leading singleton dims (handles [1,1,2], [1,2], etc.)
    x_vec = x_vec.squeeze()
    # if still not 1D of length 2, reshape using the last dim as 2
    if x_vec.dim() != 1 or x_vec.shape[-1] != 2:
        x_vec = x_vec.reshape(-1, 2)[0]  # take the first row if shape is [*,2]
    x_batch = x_vec.unsqueeze(0).expand(num_samples, -1)

    for i in range(steps):
        t_curr = t0 + i * dt
        t_tensor = torch.full((num_samples, 1), t_curr, device=device)

        if solver == "euler":
            v = model(t_tensor, x_batch, theta)
            theta = theta + v * dt
        elif solver == "rk4":
            k1 = model(t_tensor,               x_batch, theta)
            k2 = model(t_tensor + 0.5*dt,      x_batch, theta + 0.5*dt*k1)
            k3 = model(t_tensor + 0.5*dt,      x_batch, theta + 0.5*dt*k2)
            k4 = model(t_tensor + dt,          x_batch, theta + dt*k3)
            theta = theta + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        else:
            raise ValueError("solver must be 'euler' or 'rk4'.")

    return theta


def plot_posteriors(samples_list, obs_list, ref_list, outpath):
    """
    samples_list: list of [Ns,2] tensors (FMPE posterior samples)
    obs_list:     list of [2] tensors (observations)
    ref_list:     list of [Nr,2] tensors (reference posterior samples)
    """
    n = len(samples_list)
    cols = n
    fig, axes = plt.subplots(1, cols, figsize=(4*cols, 4), squeeze=False)
    for j, (samps, refs, x) in enumerate(zip(samples_list, ref_list, obs_list)):
        ax = axes[0, j]
        samps = samps.cpu()
        refs = refs.cpu()
        # FMPE posterior
        ax.scatter(samps[:,0], samps[:,1], s=2, alpha=0.25, label="FMPE")
        # Reference posterior overlay
        ax.scatter(refs[:,0], refs[:,1], s=5, c="black", alpha=0.5, label="Reference")
        ax.set_title(f"TwoMoons posterior\nobs idx {j+1}")
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")
        ax.axis("equal")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right", markerscale=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_sims", type=int, default=cfg.num_sims)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--samples", type=int, default=cfg.num_posterior_samples)
    parser.add_argument("--n_test_obs", type=int, default=cfg.n_test_obs)
    args = parser.parse_args([])  # Jupyter-safe; replace with None when running as script

    cfg.num_sims = args.num_sims
    cfg.epochs = args.epochs
    cfg.num_posterior_samples = args.samples
    cfg.n_test_obs = args.n_test_obs

    os.makedirs(cfg.outdir, exist_ok=True)

    # Build data (training sims) and get fixed benchmark observations
    train_ds, val_ds, task, observations = build_twomoons_data(cfg.num_sims, cfg.device)

    # Model
    model = FMPEVectorField(x_dim=2, theta_dim=2, hidden=256, depth=5).to(cfg.device)

    # Train
    print("Training FMPE on TwoMoons …")
    model = train_fmpe(model, train_ds, val_ds, cfg)

    # Reload best
    ckpt = torch.load(os.path.join(cfg.outdir, "fmpe_twomoons_best.pt"), map_location=cfg.device)
    model.load_state_dict(ckpt["model"])

    # Inference on a few canonical observations (ref SBIBM posteriors)
    import sbibm
    print("Sampling posteriors for a few test observations …")
    obs_to_plot, samples_to_plot, ref_to_plot = [], [], []

    for k in range(cfg.n_test_obs):  # 0..n-1
        x = observations[k]  # torch.tensor shape [2]

        # FMPE posterior samples
        thetas = integrate_ode(
            model, x.to(cfg.device),
            num_samples=cfg.num_posterior_samples,
            steps=cfg.ode_steps,
            solver=cfg.ode_solver,
            device=cfg.device,
        )

        # Reference posterior samples for the corresponding canonical observation
        ref_samples = task.get_reference_posterior_samples(
            num_observation=k+1,  # 1-indexed... goes from 1 to 10
        )  # [Ns, 2]

        # Collect
        obs_to_plot.append(x)
        samples_to_plot.append(thetas)
        ref_to_plot.append(ref_samples)

    plot_path = os.path.join(cfg.outdir, "posteriors_twomoons.png")
    plot_posteriors(samples_to_plot, obs_to_plot, ref_to_plot, plot_path)
    print(f"Saved posterior plots → {plot_path}")


if __name__ == "__main__":
    main()
