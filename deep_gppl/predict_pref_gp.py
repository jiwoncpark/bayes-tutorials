
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Predict ranking on held-out test set.

import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch, gpytorch
from scipy.stats import spearmanr, kendalltau

AA_ALPHABET=list("ACDEFGHIKLMNPQRSTVWY"); AA_TO_IDX={a:i for i,a in enumerate(AA_ALPHABET)}
def one_hot_encode(seqs):
    L=len(seqs[0]); X=np.zeros((len(seqs),L,20),dtype=np.float32)
    for i,s in enumerate(seqs):
        assert len(s)==L
        for p,ch in enumerate(s): X[i,p,AA_TO_IDX.get(ch,0)]=1.0
    return X.reshape(len(seqs),-1)

def load_test(d):
    df=pd.read_csv(Path(d)/"test_instances.csv")
    return df, df["sequence"].astype(str).tolist(), df["y"].to_numpy()

class ShellModel(gpytorch.models.ExactGP):
    def __init__(self, X_codes, y_dummy, input_dim:int):
        super().__init__(X_codes, y_dummy, gpytorch.likelihoods.GaussianLikelihood())
        self.mean_module=gpytorch.means.ZeroMean()
        self.base_kernel=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=input_dim))
        self.log_sigma_f=torch.nn.Parameter(torch.log(torch.tensor(0.1)))
    @property
    def sigma_f(self): return torch.clamp(self.log_sigma_f.exp(), min=1e-6)
    def forward(self, X_codes):
        mean=self.mean_module(X_codes); cov=self.base_kernel(X_codes).add_jitter(self.sigma_f**2)
        return gpytorch.distributions.MultivariateNormal(mean, cov)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--processed_dir",required=True); ap.add_argument("--ckpt_dir",required=True); ap.add_argument("--out_dir",default="pred_outputs")
    ap.add_argument("--cpu",action="store_true")
    args=ap.parse_args()

    state=torch.load(Path(args.ckpt_dir)/"prefgp_model.pt", map_location="cpu")
    train_design=torch.load(Path(args.ckpt_dir)/"train_design.pt", map_location="cpu")
    X_train=train_design["X_train"]; input_dim=int(state["input_dim"])

    y_dummy=torch.zeros(X_train.size(0))
    model=ShellModel(X_train, y_dummy, input_dim=input_dim)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.eval()

    df_test, seqs_test, y_test = load_test(args.processed_dir)
    X_test=torch.from_numpy(one_hot_encode(seqs_test)).to(torch.float32)

    with torch.no_grad():
        X_all=torch.cat([X_train, X_test], dim=0)
        mvn_all=model(X_all); mu_all=mvn_all.mean
        mu_test=mu_all[-len(X_test):].cpu().numpy()

    rho,_=spearmanr(y_test, mu_test); tau,_=kendalltau(y_test, mu_test)

    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({"sequence":seqs_test,"y":y_test,"mu":mu_test}).to_csv(out/"test_scores.csv", index=False)
    with open(out/"ranking_metrics.json","w") as f: json.dump({"spearman_rho":float(rho),"kendall_tau":float(tau)}, f, indent=2)
    print(json.dumps({"spearman_rho":float(rho),"kendall_tau":float(tau)}, indent=2))
    print("Wrote predictions to", out)

if __name__=="__main__":
    main()
