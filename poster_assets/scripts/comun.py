"""
Utilidades compartidas por los scripts del poster.

Leen el checkpoint directamente (torch.load) y evaluan los polinomios de
Chebyshev en CPU. No importan gs2d_video, asi que funcionan en una maquina
sin la extension CUDA compilada. El unico script que si necesita el
rasterizador real es 01_construccion_progresiva.py.
"""
from pathlib import Path
import numpy as np
import torch

# Experimento elegido para el poster (ver BRIEF_POSTER.md, seccion "Experimento").
EXP_DIR = Path("/Users/jleandrojm/TESIS/MBB-GS/outputs_khipu/musical_20s_1600ep")
CLIP_DIR = Path("/Users/jleandrojm/TESIS/MBB-GS/data/clips/musical_20s")
CKPT = EXP_DIR / "checkpoints" / "checkpoint_final.pt"
OUT = Path(__file__).resolve().parents[1] / "img"


def cargar(ckpt_path=CKPT):
    """Devuelve (coefs, meta) con los tensores en CPU float32."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict_coefs"]
    meta = {k: sd[k] for k in ("grados", "N", "H", "W", "n_frames")}
    coefs = {k: v.float() for k, v in sd.items() if torch.is_tensor(v)}
    return coefs, meta


def matriz_chebyshev(n_frames, grado, taus=None):
    """B[j,k] = T_k(tau_j), construida en float64 igual que el codigo de entrenamiento."""
    if taus is None:
        j = np.arange(n_frames, dtype=np.float64)
        taus = 2.0 * j / (n_frames - 1) - 1.0
    taus = np.asarray(taus, dtype=np.float64)
    B = np.empty((taus.size, grado + 1), dtype=np.float64)
    B[:, 0] = 1.0
    if grado >= 1:
        B[:, 1] = taus
    for k in range(2, grado + 1):
        B[:, k] = 2.0 * taus * B[:, k - 1] - B[:, k - 2]
    return torch.from_numpy(B).float()


def evaluar(coefs, meta, atributo, taus=None):
    """
    Evalua un atributo en los instantes pedidos.
    Devuelve un tensor (T, N, dim) con el valor CRUDO, sin activacion.
    """
    a0 = coefs[f"{atributo}_a0"]
    hi = coefs[f"{atributo}_high"]
    if a0.dim() == 2:                      # opacity, theta, depth: (N,1)/(N,K)
        a0 = a0.unsqueeze(1)               # -> (N,1,1)
        hi = hi.unsqueeze(1)               # -> (N,1,K)
    full = torch.cat([a0, hi], dim=2)      # (N, dim, K+1)
    grado = meta["grados"][atributo]
    B = matriz_chebyshev(meta["n_frames"], grado, taus)   # (T, K+1)
    # (N,dim,K+1) x (K+1,T) -> (N,dim,T) -> (T,N,dim)
    return torch.einsum("ndk,tk->tnd", full, B)


def activar(atributo, crudo, H, W):
    """Activaciones del modelo: sigmoide, exponencial o identidad."""
    if atributo in ("opacity", "color"):
        return torch.sigmoid(crudo)
    if atributo == "scale":
        lo, hi = np.log(0.5), np.log(float(max(H, W)))
        return torch.exp(crudo.clamp(lo, hi))
    return crudo


def guardar(fig, nombre, dpi=200, transparente=False):
    OUT.mkdir(parents=True, exist_ok=True)
    ruta = OUT / nombre
    fig.savefig(ruta, dpi=dpi, bbox_inches="tight", pad_inches=0,
                transparent=transparente,
                facecolor="none" if transparente else "white")
    print(f"  guardado {ruta}")
    return ruta
