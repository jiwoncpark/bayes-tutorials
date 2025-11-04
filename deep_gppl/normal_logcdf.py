import math
import torch
from torch.distributions.normal import Normal

# ---- Core: stable log Phi(z) for z ~ N(0,1) ---------------------------------
def _standard_normal_logcdf(z: torch.Tensor) -> torch.Tensor:
    """
    Stable log CDF for standard normal.
    Uses:
      log Phi(z) = log(0.5 * erfc(-z/sqrt(2)))                [general]
    For large negative z (i.e., t = -z/sqrt(2) >> 1), computes:
      log Phi(z) = log(0.5) + log(erfcx(t)) - t^2             [erfcx trick]
    which avoids underflow in erfc(t).

    Works with broadcasting and preserves gradients.
    """
    t = -z / math.sqrt(2.0)               # t >= 0 when z <= 0
    log_half = math.log(0.5)

    # Choose a threshold where erfcx is numerically preferable.
    # t>5 (~ z < -7.07) is a conservative, very-safe cutoff.
    use_erfcx = t > 5

    # Fast path (covers most inputs): direct erfc
    general = torch.log(0.5 * torch.erfc(t))

    # Stable path for far left tail: erfcx(t) * exp(-t^2)
    if hasattr(torch.special, "erfcx"):
        left_tail = log_half + torch.log(torch.special.erfcx(t)) - t * t
    else:
        # Fallback if torch.special.erfcx is unavailable:
        # Asymptotic expansion via Mills' ratio (accurate for very negative z)
        # log Phi(z) ≈ -0.5 z^2 - log(-z) - 0.5 log(2π)
        # Only used when t is large (z very negative).
        two_pi = 2.0 * math.pi
        eps = torch.finfo(z.dtype).tiny
        left_tail = -0.5 * z * z - torch.log(-z + eps) - 0.5 * math.log(two_pi)

    return torch.where(use_erfcx, left_tail, general)

# ---- Monkey patch: Normal.logcdf(x) ------------------------------------------
def _normal_logcdf(self: Normal, value: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable log CDF for a general Normal(mu, sigma).
    """
    z = (value - self.loc) / self.scale
    return _standard_normal_logcdf(z)

# Attach as a method (monkey-patch)
setattr(Normal, "logcdf", _normal_logcdf)

# ---- Example usage -----------------------------------------------------------
if __name__ == "__main__":
    dist = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    x = torch.tensor([-20.0, -10.0, -2.0, 0.0, 2.0, 10.0, 20.0])
    print(dist.logcdf(x))  # should be finite and stable across the entire range
