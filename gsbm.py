import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.autograd.functional as AF

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gaussian_sigma = 0.25
eps = 1e-3

# Sampler for mixtures of two diagonal Gaussians
def sample_mixture(batch_size, means):
    idx = torch.randint(0, len(means), (batch_size,), device=device)
    means = torch.tensor(means, dtype=torch.float32, device=device)
    chosen = means[idx]
    return chosen + gaussian_sigma * torch.randn(batch_size, 2, device=device)

# Drift network
class DriftNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )
    def forward(self, t, x):
        inp = torch.cat([t, x], dim=1)
        return self.net(inp)

# Spline-based conditional bridge with JVP helper methods
class SplineBridge(nn.Module):
    def __init__(self, K=30, sigma=1.0):
        super().__init__()
        self.K = K
        self.register_buffer('t_knots', torch.linspace(0, 1, K))
        self.m_knots    = nn.Parameter(self.t_knots.clone())
        init_g          = sigma * torch.sqrt(self.t_knots * (1 - self.t_knots) + 1e-6)
        self.logg_knots = nn.Parameter(torch.log(init_g + 1e-6))
        self.sigma      = sigma

    def interp(self, params, t):
        B = t.size(0)
        scaled = t.squeeze(-1) * (self.K - 1)
        idx    = scaled.floor().long().clamp(0, self.K - 2)
        w      = (scaled - idx.float()).unsqueeze(-1)
        p0     = params[idx].unsqueeze(-1)
        p1     = params[idx+1].unsqueeze(-1)
        return (1 - w)*p0 + w*p1, idx

    def forward(self, t, x0, x1):
        # Stage 1 drift-matching support
        m_t, idx   = self.interp(self.m_knots, t)
        lg_t, _    = self.interp(self.logg_knots, t)
        g_t        = torch.exp(lg_t).clamp(min=eps)
        mu_t       = (1 - m_t)*x0 + m_t*x1
        x_t        = mu_t + g_t*torch.randn_like(mu_t)
        # finite‐diff drift
        m_diff      = (self.m_knots[idx+1] - self.m_knots[idx]).unsqueeze(-1)*(self.K-1)
        lg_diff     = (self.logg_knots[idx+1] - self.logg_knots[idx]).unsqueeze(-1)*(self.K-1)
        g_dot       = g_t * lg_diff
        a_t         = (g_dot - 0.5*self.sigma**2/g_t)/g_t
        u_cond      = m_diff*(x1-x0) + a_t*(x_t-mu_t)
        return x_t, mu_t, g_t, u_cond

    def mean(self, t, x0_b, x1_b):
        # t: (T2,), x0_b,x1_b: (B,1,2)
        B = x0_b.size(0); T2 = t.size(0)
        # expand t to (B*T2,1)
        t_rep = t.view(T2,1).expand(T2,B).t().reshape(B*T2,1)
        m_flat, _ = self.interp(self.m_knots, t_rep)
        m_t = m_flat.view(B, T2, 1)
        return (1-m_t)*x0_b + m_t*x1_b  # (B,T2,2)

    def std(self, t):
        # t: (T2,) or (T2,1)
        if t.dim()==1:
            t = t.unsqueeze(-1)   # (T2,1)
        lg_flat, _ = self.interp(self.logg_knots, t)   # (T2,1)
        std = torch.exp(lg_flat).clamp(min=eps)         # (T2,1)
        return std  # (T2,1)

# Vectorized state-cost
def V(xt):
    dist = xt.norm(dim=-1)
    return (dist < a).float() * a

# Hyperparams
a = 10.0
sigma = 0.25
lr = 1e-3
batch_size = 256
epochs = 8000
K = 30

stage1_steps = 16
stage1_tol   = 1e-4
stage2_steps = 16
stage2_tol   = 1e-4

# Instantiate models
u_theta   = DriftNet().to(device)
opt_u     = torch.optim.Adam(u_theta.parameters(), lr=lr)
bridge    = SplineBridge(K=K, sigma=sigma).to(device)
opt_bridge= torch.optim.Adam(bridge.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    # Stage 1: Drift matching
    x0 = sample_mixture(batch_size, [(-1,-1),(-1,1)])
    x1 = sample_mixture(batch_size, [(1,-1),(1,1)])
    t  = torch.rand(batch_size,1,device=device)*(1-2*eps)+eps

    for _ in range(stage1_steps):
        x_t, mu_t, g_t, u_cond = bridge(t, x0, x1)
        u_pred = u_theta(t, x_t)
        loss1  = F.mse_loss(u_pred, u_cond)
        opt_u.zero_grad(); loss1.backward(); opt_u.step()
        if loss1.item()<stage1_tol: break

    # Stage 2: SplineOpt via JVP multi-time
    T2 = 16
    N2 = 4
    t2 = torch.linspace(eps, 1-eps, T2, device=device)  # (T2,)
    x0_b = x0.unsqueeze(1)  # (B,1,2)
    x1_b = x1.unsqueeze(1)

    for _ in range(stage2_steps):
        eps_noise = torch.randn(batch_size, N2, T2, 2, device=device)

        # JVP for mean
        t2_req = t2.requires_grad_(True)
        mean_BT2_D, dmean_BT2_D = AF.jvp(
            lambda tt: bridge.mean(tt, x0_b, x1_b),
            (t2_req,), (torch.ones_like(t2_req),),
            create_graph=True
        )
        # JVP for std
        std_T2_1, dstd_T2_1 = AF.jvp(
            lambda tt: bridge.std(tt),
            (t2_req,), (torch.ones_like(t2_req),),
            create_graph=True
        )

        # reshape and expand
        mean_BT2_D  = mean_BT2_D.unsqueeze(1)  # (B,1,T2,2)
        dmean_BT2_D = dmean_BT2_D.unsqueeze(1)
        std_BT2_1   = std_T2_1.clamp(min=eps).unsqueeze(0).expand(batch_size, T2, 1).unsqueeze(1)
        dstd_BT2_1  = dstd_T2_1.unsqueeze(0).expand(batch_size, T2, 1).unsqueeze(1)

        # sample MC
        xt    = mean_BT2_D + std_BT2_1 * eps_noise  # (B,N2,T2,2)
        a_BT2 = (dstd_BT2_1 - 0.5*sigma**2/std_BT2_1) / std_BT2_1

        u_cond = dmean_BT2_D + a_BT2 * (xt - mean_BT2_D)

        # compute loss
        V2       = V(xt)                             # (B,N2,T2)
        control  = 0.5/sigma**2 * (u_cond.pow(2).sum(-1))  # (B,N2,T2)
        loss2    = (control + V2).mean()

        opt_bridge.zero_grad(); loss2.backward(); opt_bridge.step()
        if loss2.item()<stage2_tol: break

    if epoch % 400 == 0:
        print(f"Epoch {epoch:04d}, loss1={loss1:.4f}, loss2={loss2:.4f}")



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