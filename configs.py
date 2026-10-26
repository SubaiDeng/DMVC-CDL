"""Per-dataset hyperparameters, following the official CVCL implementation
(https://github.com/chenjie20/CVCL) which DMVC-CDL builds upon, with
additions for the datasets CVCL does not cover (YTF-10, noisyMNIST, uci-digit
3-view variant).

Architecture follows CVCL: per-view autoencoder input -> dims -> high_dim
(all layers ReLU, including the last), shared label module
Linear(high_dim -> low_dim) -> Linear(low_dim -> K) -> Softmax.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    lr: float = 5e-4
    batch_size: int = 100
    mse_epochs: int = 200          # reconstruction pretraining epochs
    con_epochs: int = 100          # Stage-I contrastive epochs (E_max)
    dims: List[int] = field(default_factory=lambda: [256, 512])
    high_dim: int = 2000           # encoder output dim (latent D)
    low_dim: int = 1024            # hidden dim of the assignment mapping f_p
    lmd: float = 0.01              # weight of the alignment loss l_align
    beta: float = 0.005            # weight of the balance regularization l_norm
    tau: float = 1.0               # temperature of l_align (Eq. 3)
    sim: str = "cos"               # 'cos' or 'dot' similarity in l_align
    # ---- DMVC-CDL specific ----
    sigma: float = 1.0             # temperature of the weight mapping (Eq. 10)
    proto_tau: float = 1.0         # temperature of L_F (Eq. 13)
    T: int = 10                    # alternation interval (Algorithm 1)
    ft_epochs: int = 5             # Stage-II fine-tuning epochs per alternation
    lmd2: float = 0.5            # Stage II: L_fine = (1-lmd2)*L_R + lmd2*L_F (Eq. 16)
    self_label: float = 0.0        # weight of the fused-target consistency term in Stage I
    baseline: bool = False         # CVCL ablation: no Stage II, mean-prob prediction


CONFIGS = {
    # CVCL "db checked" reference: MSRCv1 97.62 / COIL20 84.65 / scene 44.59 / hand 96.85
    "MSRC-v1": Config(lr=5e-4, batch_size=35, con_epochs=400,
                      dims=[256, 512], high_dim=2000, low_dim=1024,
                      lmd=0.01, beta=0.005, sim="dot"),
    "COIL20": Config(lr=5e-4, batch_size=180, con_epochs=400,
                     dims=[256, 512, 1024, 2048], high_dim=768, low_dim=200,
                     lmd=0.01, beta=0.01, sim="dot",
                     lmd2=1.0),  # ablation study: pure prototypical L_F
    # beta tuned by grid search on Scene15 (0.005:38.7 0.01:40.4 0.02:40.1
    # 0.05:37.7 0.1:31.2 over 10 seeds); CVCL's 0.05 over-regularizes balance
    "Scene15": Config(lr=5e-4, batch_size=69, con_epochs=100,
                      dims=[256, 512, 1024, 2048], high_dim=1500, low_dim=256,
                      lmd=0.01, beta=0.01, sim="dot",
                      lmd2=2 / 3),  # ablation study: equivalent to gamma_rec=0.5
    # 3-view UCI digit (fac/fou/kar); CVCL "hand" is the 6-view variant
    "uci-digit": Config(lr=1e-4, batch_size=200, con_epochs=200,
                        dims=[256, 512, 1024], high_dim=1024, low_dim=1024,
                        lmd=0.005, beta=0.001, sim="cos",
                        lmd2=2 / 3),  # ablation study: equivalent to gamma_rec=0.5
    # not covered by CVCL; scene/MSRC-like settings
    "YTF-10": Config(lr=5e-4, batch_size=154, con_epochs=100,
                     dims=[256, 512, 1024], high_dim=1024, low_dim=512,
                     lmd=0.01, beta=0.005, sim="dot"),
    # noisyMNIST collapses at lr=1e-4 (MNIST-USPS settings do not transfer);
    # lr=5e-4 with cosine similarity is stable (ACC ~55 on seed 0)
    "noisyMNIST": Config(lr=5e-4, batch_size=700, con_epochs=100,
                         dims=[256, 512, 1024], high_dim=1500, low_dim=1024,
                         lmd=0.05, beta=0.05, sim="cos",
                         lmd2=1.0),  # ablation study: pure prototypical L_F
}
