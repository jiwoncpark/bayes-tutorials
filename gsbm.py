import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gaussian_sigma = 0.25
eps = 1.e-3

# Sampler for mixtures of two diagonal Gaussians
def sample_mixture(batch_size, means):
    idx = torch.randint(0, len(means), (batch_size,), device=device)
    means = torch.tensor(means, dtype=torch.float32, device=device)
    chosen = means[idx]  # (batch_size, 2)
    return chosen + gaussian_sigma*torch.randn(batch_size, 2, device=device)  # unit diagonal covariance

# Drift network
class DriftNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),  # [t, x1, x2]
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )
    def forward(self, t, x):
        # t: (B,1), x: (B,2)
        inp = torch.cat([t, x], dim=1)
        return self.net(inp)

# Spline-based conditional bridge params
class SplineBridge(nn.Module):
    def __init__(self, K=10, sigma=1.0):
        super().__init__()
        self.K = K
        # Uniform knots in [0,1]
        self.register_buffer('t_knots', torch.linspace(0, 1, K))
        # m(t) knots init = t_knots
        self.m_knots = nn.Parameter(self.t_knots.clone())
        # log(gamma) knots init = log(sigma*sqrt(t(1-t)))
        init_g = sigma * torch.sqrt(self.t_knots * (1 - self.t_knots) + 1e-6)
        self.logg_knots = nn.Parameter(torch.log(init_g + 1e-6))
        self.sigma = sigma

    def interp(self, params, t):
        # params: (K,), t: (B,1)
        B = t.shape[0]
        t_scaled = t.squeeze(-1) * (self.K - 1)  # (B,)
        idx = torch.floor(t_scaled).long().clamp(0, self.K - 2)  # (B,)
        w = (t_scaled - idx.float()).unsqueeze(-1)  # (B,1)
        p0 = params[idx].unsqueeze(-1)            # (B,1)
        p1 = params[idx + 1].unsqueeze(-1)        # (B,1)
        return (1 - w) * p0 + w * p1, idx

    def forward(self, t, x0, x1):
        # Interpolate m(t) and gamma(t) using splines
        m_t, idx = self.interp(self.m_knots, t)           # (B,1), indices
        lg_t, _   = self.interp(self.logg_knots, t)       # (B,1)
        g_t = torch.exp(lg_t)
        g_t = g_t.clamp(min=eps)

        # Mean interpolation: mu_t = (1 - m_t)*x0 + m_t*x1
        mu_t = (1 - m_t) * x0 + m_t * x1                  # (B,2)

        # Sample x_t on bridge
        x_t = mu_t + g_t * torch.randn_like(mu_t)         # (B,2)

        # Compute time-derivatives via piecewise constants
        # m_dot ≈ (m_{i+1} - m_i) * (K-1)
        m_diff = (self.m_knots[idx+1] - self.m_knots[idx]).unsqueeze(-1) * (self.K - 1)
        # logg_dot ≈ (logg_{i+1} - logg_i) * (K-1)
        lg_diff = (self.logg_knots[idx+1] - self.logg_knots[idx]).unsqueeze(-1) * (self.K - 1)
        # gamma_dot = g_t * logg_dot
        g_dot = g_t * lg_diff

        # Compute conditional drift: u_cond = m_dot*(x1 - x0) + a_t*(x_t - mu_t)
        a_t = (g_dot - self.sigma**2 / (2 * g_t)) / g_t
        u_cond = m_diff * (x1 - x0) + a_t * (x_t - mu_t)

        return x_t, mu_t, g_t, u_cond

# Hyperparameters
a = 0.0
sigma = 0.1
lr = 1e-4
batch_size = 256
epochs = 8000
K = 30

# Hyperparameters
stage1_steps = 4      # max inner loop iterations
stage1_tol   = 1e-4    # early stopping threshold on loss1
stage2_steps = 4      # max inner loop iterations
stage2_tol   = 1e-4    # early stopping threshold on loss2

# Determine whether to use spline-based bridge
use_spline = (a != 0.0)

# Initialize bridge optimizer if needed
if use_spline:
    bridge = SplineBridge(K=K, sigma=sigma).to(device)
    opt_bridge = torch.optim.Adam(bridge.parameters(), lr=lr)

# Model & optimizers
u_theta = DriftNet().to(device)
opt_u = torch.optim.Adam(u_theta.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    # Stage 1: Fit drift to current bridge approx
    x0 = sample_mixture(batch_size, [(-1, -1), (-1, 1)])
    x1 = sample_mixture(batch_size, [(1, -1), (1, 1)])
    t = torch.rand(batch_size, 1, device=device) * (1 - 2*eps) + eps

    # Drift matching loss: regress u_theta to u_cond
    for _ in range(stage1_steps):
        if use_spline:
            # Use learned spline bridge to get u_cond
            x_t, mu_t, g_t, u_cond = bridge(t, x0, x1)
        else:
            # Use closed-form Brownian bridge
            mu_t    = (1 - t) * x0 + t * x1
            gamma_t = sigma * torch.sqrt(t * (1 - t) + 1e-6)
            x_t     = mu_t + gamma_t * torch.randn_like(mu_t)
            mu_dot  = x1 - x0
            gamma_dot = sigma * (1 - 2*t) / (2 * torch.sqrt(t * (1 - t) + 1e-6))
            a_t     = (gamma_dot - sigma**2 / (2 * gamma_t)) / gamma_t
            u_cond  = mu_dot + a_t * (x_t - mu_t)

        u_pred = u_theta(t, x_t)
        loss1 = F.mse_loss(u_pred, u_cond)
        opt_u.zero_grad()
        loss1.backward()
        opt_u.step()
        if loss1.item() < stage1_tol:
            break

    # Stage 2: Fit spline bridge on same x0, x1 until convergence
    if use_spline:
        for _ in range(stage2_steps):
            t2 = torch.rand(batch_size, 1, device=device)
            x_t2, mu_t2, g_t2, u_cond2 = bridge(t2, x0, x1)
            V2 = torch.where((x_t2.pow(2).sum(-1) < 0.5**2),  a,  0.0).unsqueeze(-1)
            # loss = 0.5 ||u_cond2||^2 + V2
            loss2 = (0.5*u_cond2.pow(2).sum(dim=1, keepdim=True) + V2).mean()

            opt_bridge.zero_grad()
            loss2.backward()
            opt_bridge.step()

            with torch.no_grad():
                bridge.m_knots.data[0]   = 0.0
                bridge.m_knots.data[-1]  = 1.0
                bridge.logg_knots.data[0]  = torch.log(torch.tensor(eps))
                bridge.logg_knots.data[-1] = torch.log(torch.tensor(eps))

            if loss2.item() < stage2_tol:
                break

    # Logging
    if epoch % 400 == 0:
        msg = f"Epoch {epoch}, loss1={loss1:.4f}"
        if use_spline:
            msg += f", loss2={loss2:.4f}"
        print(msg)

# Parameters
N = 16      # number of test samples
M = 1       # number of trajectories per sample
steps = 64 # discretization of [0,1]
dt = 1.0 / steps

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x0_test = sample_mixture(N, [(-1, -1), (-1, 1)])

# Function to simulate one bridge trajectory, returning full path
def simulate_path(x0):
    x = x0.clone().unsqueeze(0).to(device)  # shape [1,2]
    path = [x.cpu().numpy().flatten()]
    for i in range(steps):
        t = torch.full((1,1), i * dt, device=device)
        with torch.no_grad():
            drift = u_theta(t, x)
        noise = sigma * torch.sqrt(torch.tensor(dt, device=device)) * torch.randn_like(x)
        x = x + drift * dt + noise
        path.append(x.cpu().numpy().flatten())
    return path

# Plotting
plt.figure(figsize=(6,6))

def bivariate_gaussian(pos, mean, cov):
    diff = pos - mean
    inv_cov = torch.inverse(cov)
    exponent = -0.5 * torch.einsum('...i,ij,...j->...', diff, inv_cov, diff)
    return torch.exp(exponent) / (2 * torch.pi * torch.sqrt(torch.det(cov)))

# Plot contours of p0 and p1
x, y = torch.meshgrid(torch.linspace(-1.5, 1.5, 100), torch.linspace(-1.5, 1.5, 100))
pos = torch.stack([x, y], dim=-1)
mean1 = torch.tensor([-1.0, -1.0])
mean2 = torch.tensor([-1.0, 1.0])
mean3 = torch.tensor([1.0, -1.0])
mean4 = torch.tensor([1.0, 1.0])
cov = torch.eye(2) * gaussian_sigma**2

z1 = bivariate_gaussian(pos, mean1, cov)
z2 = bivariate_gaussian(pos, mean2, cov)
z3 = bivariate_gaussian(pos, mean3, cov)
z4 = bivariate_gaussian(pos, mean4, cov)

# Gaussian mixture p0
plt.contour(x.numpy(), y.numpy(), z1.numpy(), levels=3, colors='tab:red', alpha=0.5)
plt.contour(x.numpy(), y.numpy(), z2.numpy(), levels=3, colors='tab:red', alpha=0.5)
# Gaussian mixture p1
plt.contour(x.numpy(), y.numpy(), z3.numpy(), levels=3, colors='tab:blue', alpha=0.5)
plt.contour(x.numpy(), y.numpy(), z4.numpy(), levels=3, colors='tab:blue', alpha=0.5)

colors = plt.cm.jet(torch.linspace(0, 1, len(x0_test)))  # Generate distinct colors for each test point

for i, x0 in enumerate(x0_test):
    start = x0.cpu().numpy()
    
    # Simulate M trajectories for this start
    for _ in range(M):
        path = simulate_path(x0)
        xs, ys = zip(*path)
        plt.plot(xs, ys, alpha=0.5, color=colors[i])  # trajectory with unique color
        plt.scatter(xs[0], ys[0], marker='x', color=colors[i])  # start point
        plt.scatter(xs[-1], ys[-1], marker='o', color=colors[i])  # end point

plt.title("Bridge paths from x0 to x1")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.savefig(f"bridge_paths_a_{a:.2f}.png")

# Generate test trajectory
x_test = x0_test[0].unsqueeze(0)  # pick first test sample (shape [1,2])
trajectory = simulate_path(x_test.squeeze(0))

# Prepare grid
steps = 64
dt = 1.0 / steps
grid_size = 100
xmin, xmax = -2.0, 2.0
ymin, ymax = -2.0, 2.0

xs = np.linspace(xmin, xmax, grid_size)
ys = np.linspace(ymin, ymax, grid_size)
xx, yy = np.meshgrid(xs, ys)
grid_points = np.stack([xx.ravel(), yy.ravel()], axis=-1)
grid_tensor = torch.tensor(grid_points, dtype=torch.float32, device=device)

# Evaluate field at each time step
fig, axs = plt.subplots(4, 4, figsize=(12, 12))
for i, pos in enumerate(trajectory[::(steps // 16)]):  # take 4 points evenly spaced
    if i >= 16:
        break
    t_val = i * dt
    t_tensor = torch.full((grid_tensor.shape[0], 1), t_val, device=device)
    
    # Compute u_theta on grid
    with torch.no_grad():
        u_vals = u_theta(t_tensor, grid_tensor)
    u_norm2 = (u_vals.pow(2).sum(dim=1)).cpu().numpy()
    
    # Compute V on grid: penalty inside radius a
    dist = np.linalg.norm(grid_points, axis=1)
    V_vals = (dist < 0.5).astype(float) * a
    
    # Compute field and reshape
    f_vals = 0.5 * u_norm2 + V_vals
    f_grid = f_vals.reshape(grid_size, grid_size)
    
    ax = axs.flat[i]
    im = ax.imshow(f_grid, extent=(xmin, xmax, ymin, ymax), origin='lower')
    ax.scatter(pos[0], pos[1], marker='o', color="k")  # overlay trajectory point
    ax.set_title(f"t={t_val:.2f}")
    ax.set_xticks([])
    ax.set_yticks([])

# Colorbar
fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6)
plt.tight_layout()
plt.savefig(f"field_evolution_a_{a:.2f}.png")