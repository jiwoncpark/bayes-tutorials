# sketch
import numpy as np

# Suppose y is (n,) array of measured phenotype (e.g., GB1 log-enrichment)
# Build all pairs or a sampled subset:
indices = np.arange(len(y))
V, U = [], []
for i in indices:
    # sample a few j with lower y (preference i ≻ j)
    lowers = np.where(y < y[i])[0]
    if len(lowers) == 0: 
        continue
    js = np.random.choice(lowers, size=min(20, len(lowers)), replace=False)
    V.extend([i]*len(js))
    U.extend(js)
V = np.asarray(V, dtype=np.int64)
U = np.asarray(U, dtype=np.int64)

# Optionally, weight pairs by |y_i - y_j| or replicate to reflect confidence.
