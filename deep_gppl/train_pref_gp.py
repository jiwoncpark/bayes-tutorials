# Train preference GP (pairwise probit) on GB1 with evidence maximization (SVGP ELBO).
# Structure mirrors epik/model.py: build kernel -> build model -> build objective -> optimize hyperparams.

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import gpytorch
import wandb
# from torch.distributions.normal import Normal
from normal_logcdf import Normal


AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY"); AA_TO_IDX = {a:i for i,a in enumerate(AA_ALPHABET)}
def one_hot_encode(seqs):
    L = len(seqs[0])
    X = np.zeros((len(seqs), L, 20), dtype=np.float32)
    for i, s in enumerate(seqs):
        if len(s) != L:
            assert len(s) == L, "variable length not supported"
        for p, ch in enumerate(s):
            X[i, p, AA_TO_IDX.get(ch, 0)] = 1.0
    return X.reshape(len(seqs), -1)


def load_split(d,n):
    df = pd.read_csv(Path(d)/f"{n}_instances.csv", keep_default_na=False, na_values=['_'])
    return df["sequence"].astype(str).tolist(), df["y"].to_numpy()


def load_pairs(d,n):
    npz = np.load(Path(d)/f"{n}_pairs.npz")
    return npz["V"].astype(np.int64), npz["U"].astype(np.int64)

# ------------------------------
# Pairwise probit likelihood (Chu & Ghahramani)
# ------------------------------
class PairwiseProbitLikelihood(gpytorch.likelihoods.Likelihood):
    """
    P(v ≻ u | f) = Phi((f[v] - f[u]) / (sqrt(2)*sigma_obs))
    We optimize sigma_obs along with kernel hyperparameters.
    """
    def __init__(self, init_sigma_obs=0.1):
        super().__init__()
        self.log_sigma_obs = torch.nn.Parameter(torch.log(torch.tensor(init_sigma_obs)))
        self.standard_normal = Normal(0, 1)

    @property
    def sigma_obs(self):
        return torch.clamp(self.log_sigma_obs.exp(), min=1e-5)

    # We'll use this inside our ELBO to compute E_q[log p(y|f)] for a mini-batch of pairs
    def log_prob_pairs(self, f_batch_samples, Vb, Ub):
        """
        f_batch_samples: Tensor of shape (S, nb) OR (nb,), latent utilities at the subset of instances
        Vb, Ub: integer indices (into that subset) giving pairs
        returns: per-sample-per-pair log CDF (averaged over S outside)
        """
        if f_batch_samples.dim() == 1:
            diffs = f_batch_samples[Vb] - f_batch_samples[Ub]
        else:
            diffs = f_batch_samples[..., Vb] - f_batch_samples[..., Ub]  # (S, bsz)
        z = diffs / (self.sigma_obs * (2.0**0.5))
        return self.standard_normal.logcdf(z)


    def forward(self, *args, **kwargs):
        # Not used; we compute log_prob_pairs directly inside the ELBO.
        return None

# ------------------------------
# Kernel factory (Jenga or RBF)
# ------------------------------
def build_kernel(name, input_dim, device):
    name = name.lower()
    if name == "jenga":
        # Keep user's epik import path; configure per your analysis setup
        from epik.kernel import JengaKernel
        k = JengaKernel(n_alleles=20, seq_length=input_dim//20)
        print(f"Using the Jenga kernel")
        return k.to(device)
    elif name == "connectedness":
        from epik.kernel import ConnectednessKernel
        k = ConnectednessKernel(n_alleles=20, seq_length=input_dim//20)
        print(f"Using the Connectedness kernel")
        return k.to(device)
    elif name == "vc":
        from epik.kernel import VarianceComponentKernel
        k = VarianceComponentKernel(n_alleles=20, seq_length=input_dim//20)
        print(f"Using the Variance Component kernel")
        return k.to(device)
    elif name == "additive":
        from epik.kernel import AdditiveKernel
        k = AdditiveKernel(n_alleles=20, seq_length=input_dim//20)
        print(f"Using the Additive kernel")
        return k.to(device)
    elif name == "pairwise":
        from epik.kernel import PairwiseKernel
        k = PairwiseKernel(n_alleles=20, seq_length=input_dim//20)
        print(f"Using the Pairwise kernel")
        return k.to(device)
    return gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)).to(device)

# ------------------------------
# Variational GP over instance utilities f (SVGP)
# ------------------------------
class VariationalInstanceModel(gpytorch.models.ApproximateGP):
    """
    Prior: f ~ GP(0, K + sigma_f^2 I)  (latent noise on f is learned)
    Variational family q(u) with learnable inducing locations (Type-II ML on kernel + sigma_f via ELBO).
    """
    def __init__(self, inducing_points, base_kernel, init_sigma_f=0.1):
        M = inducing_points.size(0)
        variational_dist = gpytorch.variational.CholeskyVariationalDistribution(M)
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_dist, learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.base_kernel = base_kernel
        self.log_sigma_f = torch.nn.Parameter(torch.log(torch.tensor(init_sigma_f)))

    @property
    def sigma_f(self):
        return torch.clamp(self.log_sigma_f.exp(), 1e-5, 5.0)

    def forward(self, X):
        mean = self.mean_module(X)
        cov  = self.base_kernel(X)
        cov  = cov.add_jitter(self.sigma_f**2 + 1.e-5)  # K -> K + sigma_f^2 I
        return gpytorch.distributions.MultivariateNormal(mean, cov)

# ------------------------------
# (Optional) kmeans++ seeding for inducing points
# ------------------------------
def kmeanspp_init(X_np, M, rng):
    # X_np: (n, n_alleles x seq_length)
    n = X_np.shape[0]
    M = min(M, n)
    centers = np.empty((M, X_np.shape[1]), dtype=X_np.dtype)
    i0 = rng.integers(0, n)
    centers[0] = X_np[i0]
    d2 = np.sum((X_np - centers[0])**2, axis=1)
    for m in range(1, M):
        probs = d2 / (d2.sum() + 1e-12)
        idx = rng.choice(n, p=probs)
        centers[m] = X_np[idx]
        d2 = np.minimum(d2, np.sum((X_np - centers[m])**2, axis=1))
    return centers

# ------------------------------
# ELBO helper (Type-II ML / evidence)
# ------------------------------
def stochastic_elbo_on_pair_batch(model, likelihood, X_train, Vb, Ub, S=8):
    """
    Computes a stochastic ELBO estimate on a mini-batch of pairs (Vb, Ub):
        E_q[log p(y|f)] - KL[q(u)||p(u)]
    where y are implicit "1"s (V ≻ U) for all batch pairs.

    Implementation detail:
      - Restrict q(f) to the subset of DISTINCT instance indices touched by this batch for efficiency.
      - Map from original instance indices -> positions in the subset to index f[Vb], f[Ub].
    """
    # Distinct instance indices referenced by this pair batch
    idx_inst = torch.unique(torch.cat([Vb, Ub], dim=0))
    Xb = X_train.index_select(0, idx_inst)
    q_fb = model(Xb)                                 # q(f_b) as MVN
    # Sample S draws of f_b
    f_s = q_fb.rsample(sample_shape=torch.Size([S])) # (S, nb)
    # Build map from global instance id -> position in idx_inst
    pos = torch.empty(X_train.size(0), dtype=torch.long, device=idx_inst.device).fill_(-1)
    pos[idx_inst] = torch.arange(idx_inst.numel(), device=idx_inst.device)
    Vpos = pos[Vb]
    Upos = pos[Ub]
    # Likelihood term (average over samples and pairs)
    loglik_samples = likelihood.log_prob_pairs(f_s, Vpos, Upos)  # (S, bsz)
    elp = loglik_samples.mean(dim=0).mean()                      # avg over S and pairs
    # KL term from the variational strategy
    kl = model.variational_strategy.kl_divergence().sum()
    # Combine (you can normalize KL by an estimate of num_batches if desired; here we keep it unnormalized
    # and rely on LR / pair batch size to balance)
    # print(f"Loss elp: {elp}, KL: {kl}")
    return elp - kl

# ------------------------------
# Main (mirrors epik/model.py flow)
# ------------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--out_dir", default="prefgp_ckpt")
    ap.add_argument("--kernel", default="vc", choices=["rbf", "jenga", "connectedness", "vc", "additive", "pairwise"])
    ap.add_argument("--cpu", action="store_true")

    # Evidence-maximization run control
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1.e-3)
    ap.add_argument("--seed", type=int, default=0)

    # Likelihood/prior noise
    ap.add_argument("--init_sigma_f", type=float, default=0.05, help="Init latent noise on f")
    ap.add_argument("--init_sigma_obs", type=float, default=0.5, help="Init probit observation noise")

    # Variational (SVGP) settings
    ap.add_argument("--num_inducing", type=int, default=128)
    ap.add_argument("--inducing_init", type=str, default="random", choices=["kmeanspp","random"])
    ap.add_argument("--n_mc", type=int, default=8, help="MC samples for E_q[log p(y|f)] in ELBO")

    # Pair subsetting / minibatching
    ap.add_argument("--max_train_pairs", type=int, default=512,
                    help="Cap number of training pairs (subset uniformly at random). Use 0 or -1 for all.")
    ap.add_argument("--pair_batch_size", type=int, default=1024,
                    help="Mini-batch size (#pairs) per training step.")
    ap.add_argument("--resample_pair_subset_each_epoch", action="store_true",
                    help="If set, reshuffle the pair subset periodically.")
    ap.add_argument("--resample_every", type=int, default=-1,
                    help="If >0, resample the subset every this many steps (otherwise ~10%% cadence).")
    args=ap.parse_args()

    wandb.init(project="pref_gp", name=f"{args.kernel}_seed{args.seed}", entity="prescient-design")
    wandb.config.update(args)

    # Repro
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device="cuda" if torch.cuda.is_available() and (not args.cpu) else "cpu"

    # ---- Load data (instances + pairs)
    train_seqs, _ = load_split(args.processed_dir,"train")
    val_seqs, _ = load_split(args.processed_dir,"val")
    # V is the index of the better sequence, U is the index of the worse sequence
    V_train_np, U_train_np = load_pairs(args.processed_dir,"train")
    V_val_np,   U_val_np   = load_pairs(args.processed_dir,"val")

    # Optional global pair subsetting to cap memory/compute
    if args.max_train_pairs and args.max_train_pairs > 0 and len(V_train_np) > args.max_train_pairs:
        # sel = np.random.permutation(len(V_train_np))[:args.max_train_pairs]
        V_train_np, U_train_np = V_train_np[:args.max_train_pairs], U_train_np[:args.max_train_pairs]

    # Pools on device
    V_pool = torch.as_tensor(V_train_np, dtype=torch.long, device=device)
    U_pool = torch.as_tensor(U_train_np, dtype=torch.long, device=device)
    num_pairs_pool = V_pool.numel()
    print(f"[info] training pair pool size: {num_pairs_pool:,}")
    # Encode instances (one-hot)
    X_train_np = one_hot_encode(train_seqs)
    X_val_np   = one_hot_encode(val_seqs)
    X_train = torch.from_numpy(X_train_np).to(torch.float32).to(device)
    X_val   = torch.from_numpy(X_val_np).to(torch.float32).to(device)

    # ---- Kernel
    print(f"X_train shape: {X_train.shape}")
    base_kernel = build_kernel(args.kernel, X_train.shape[1], device=device)

    # ---- Likelihood
    likelihood = PairwiseProbitLikelihood(init_sigma_obs=args.init_sigma_obs).to(device)

    # ---- Variational model (SVGP)
    # Inducing points in instance space
    M = args.num_inducing
    if args.inducing_init == "kmeanspp":
        rng = np.random.default_rng(args.seed)
        Z_np = kmeanspp_init(X_train_np, M, rng)
        Z = torch.from_numpy(Z_np).to(torch.float32).to(device)
    else:
        n = X_train.size(0)
        M = min(args.num_inducing, n)
        sel = torch.randperm(n, device=X_train.device)[:M]
        Z = X_train.index_select(0, sel).contiguous()
        # sel = np.random.permutation(X_train_np.shape[0])[:M]
        # Z_np = X_train_np[sel]

    model = VariationalInstanceModel(Z, base_kernel, init_sigma_f=args.init_sigma_f).to(device)
    model.train(); likelihood.train()

    # ---- Optimizer (evidence maximization: maximizes ELBO over kernel hypers + sigma_f + inducing locs + sigma_obs)
    opt = torch.optim.Adam([
        {'params': model.parameters(), 'lr': args.lr*0.1},
        {'params': likelihood.parameters(), 'lr': args.lr},
    ])

    # Pair subset shuffler (perm of pool), similar cadence to epik's epoch stepping
    def resample_subset():
        nonlocal V_sub, U_sub, num_pairs_sub, perm_ptr
        perm = torch.randperm(num_pairs_pool, device=device)
        V_sub = V_pool[perm]
        U_sub = U_pool[perm]
        num_pairs_sub = V_sub.numel()
        perm_ptr = 0

    V_sub = U_sub = None
    num_pairs_sub = 0
    perm_ptr = 0
    resample_subset()

    resample_period = args.resample_every if args.resample_every>0 else (max(1, args.steps//10) if args.resample_pair_subset_each_epoch else 0)

    # ---- Training loop: stochastic ELBO over pair mini-batches
    for t in range(args.steps):
        if resample_period and (t % resample_period == 0) and (t > 0):
            resample_subset()

        opt.zero_grad()

        # take a contiguous chunk to cover the subset more evenly
        bsz = min(args.pair_batch_size, num_pairs_sub)
        if perm_ptr + bsz > num_pairs_sub:
            perm_ptr = 0
        Vb = V_sub[perm_ptr:perm_ptr+bsz]
        Ub = U_sub[perm_ptr:perm_ptr+bsz]
        perm_ptr += bsz

        elbo = stochastic_elbo_on_pair_batch(model, likelihood, X_train, Vb, Ub, S=args.n_mc)
        loss = -elbo

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

        opt.step()

        # Compute condition number of the covariance matrix
        with torch.no_grad():
            # Get covariance matrix from the model at inducing points
            Z = model.variational_strategy.inducing_points
            prior_dist = model(Z)
            K = prior_dist.covariance_matrix
            # Compute only max and min eigenvalues for efficiency
            M = K.shape[0]
            X_init = torch.randn(M, 1, dtype=K.dtype, device=K.device)
            # Largest eigenvalue
            eigval_max, _ = torch.lobpcg(K, k=1, X=X_init, largest=True)
            # Smallest eigenvalue (largest of -K, then negate)
            eigval_min_neg, _ = torch.lobpcg(-K, k=1, X=X_init, largest=True)
            eigval_min = -eigval_min_neg
            cond_num = (eigval_max / eigval_min).item()

        wandb.log({
            "train/sigma_f": model.sigma_f.item(),
            "train/sigma_obs": likelihood.sigma_obs.item(),
            "train/cond_num": cond_num,
        }, step=t)

        # Periodic validation-style proxy: rank accuracy on val pairs using q(f) mean
        if (t+1) % max(1, (args.steps // 100)) == 0:
            with torch.no_grad():
                q_fval = model(X_val)
                mu_val = q_fval.mean
                Vv = torch.as_tensor(V_val_np, dtype=torch.long, device=device)
                Uv = torch.as_tensor(U_val_np, dtype=torch.long, device=device)
                acc = (mu_val[Vv] > mu_val[Uv]).float().mean().item()
            print(f"[train] step {t+1:04d}/{args.steps} | ELBO={elbo.item():.4f} | val_pair_acc={acc:.3f} | "
                  f"batch={bsz} | pool={num_pairs_pool:,} | M={model.variational_strategy.inducing_points.size(0)}")
            wandb.log({
                "val/ELBO": elbo.item(),
                "val/val_pair_acc": acc,
            }, step=t)

    # ---- Save artifacts (as in epik/model.py save flow)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "kernel_name": args.kernel,
        "input_dim": X_train.shape[1],
        "sigma_f": float(model.sigma_f.item()),
        "sigma_obs": float(likelihood.sigma_obs.item()),
        "alphabet": AA_ALPHABET,
        "train_sequences": train_seqs,
        "num_inducing": int(model.variational_strategy.inducing_points.size(0)),
        "objective": "elbo"
    }, out/"prefgp_model.pt")
    torch.save({"X_train": X_train.cpu()}, out/"train_design.pt")
    with open(out/"train_config.json","w") as f: json.dump(vars(args), f, indent=2)
    print("Saved model to", out/"prefgp_model.pt")

    wandb.finish()


if __name__=="__main__":
    main()
