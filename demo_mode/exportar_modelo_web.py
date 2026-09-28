#!/usr/bin/env python3
"""
Exporta un checkpoint de MBB-GS al formato binario que consume el visor web.

El formato esta especificado en demo_mode/FORMATO_MODELO_WEB.md. Este script es
la implementacion de referencia de esa especificacion: si los dos discrepan,
manda el documento.

No modifica nada del proyecto. Importa la logica existente:
  - gs2d_video.io.checkpoints.cargar_modelo_desde_checkpoint  (carga)
  - gs2d_video.core.bases.construir_matriz_chebyshev          (base temporal)
  - GaussianasPolinomial2D.evaluar_en_frame                   (activaciones)

Determinismo
------------
Mismo checkpoint y mismos argumentos producen un archivo identico byte a byte:
  - el JSON de cabecera se serializa con sort_keys y separadores fijos,
  - no se escriben marcas de tiempo ni rutas absolutas,
  - el orden por importancia desempata por indice original (sort estable),
  - todo el calculo es en CPU float32/float64, sin no-determinismo de CUDA.

Uso
---
    python demo_mode/exportar_modelo_web.py \
        --checkpoint .../checkpoint_final.pt \
        --salida demo_mode/export_web/musical-20s.bin \
        --top_k 30000 --fps 24
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import torch

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

from gs2d_video.core.bases import construir_matriz_chebyshev  # noqa: E402
from gs2d_video.io.checkpoints import cargar_modelo_desde_checkpoint  # noqa: E402

MAGIC = b"MBBGSW01"
VERSION = 1
ATRIBUTOS = ["mu", "opacity", "color", "scale", "theta", "depth"]
DIMS = {"mu": 2, "opacity": 1, "color": 3, "scale": 2, "theta": 1, "depth": 1}
TEX_WIDTH = 2048          # ver FORMATO_MODELO_WEB.md, seccion "Subida a texturas"


# --------------------------------------------------------------------------
# importancia
# --------------------------------------------------------------------------
def calcular_importancia(modelo, n_muestras: int = 32) -> torch.Tensor:
    """
    Importancia_i = promedio_t [ alpha_i(t) * s1_i(t) * s2_i(t) ]

    alpha es la opacidad ya activada (sigmoide) y s1*s2 es proporcional al area
    de la elipse, asi que el producto es proporcional a la masa de alpha que la
    gaussiana deposita en la imagen: la integral de una gaussiana 2D es
    2*pi*alpha*s1*s2 y la constante no altera el orden.

    Los n_muestras instantes se reparten uniformemente en tau in [-1, 1]
    incluyendo ambos extremos, construyendo la matriz de Chebyshev del proyecto
    con n_frames = n_muestras.
    """
    grados = modelo.grados
    matrices = {
        g: construir_matriz_chebyshev(n_muestras, g, device="cpu", dtype=torch.float32)
        for g in set(grados.values())
    }
    acc = torch.zeros(modelo.N, dtype=torch.float64)
    with torch.no_grad():
        for j in range(n_muestras):
            p = modelo.evaluar_en_frame(j, matrices)     # ya activado
            alpha = p["opacity"].reshape(-1).double()
            s = p["scale"].double()
            acc += alpha * s[:, 0] * s[:, 1]
    return (acc / n_muestras).float()


def orden_por_importancia(imp: torch.Tensor) -> torch.Tensor:
    """Orden descendente, estable, con desempate por indice original."""
    return torch.argsort(imp, descending=True, stable=True)


# --------------------------------------------------------------------------
# escritura
# --------------------------------------------------------------------------
def _coefs_por_atributo(modelo, nombre: str) -> torch.Tensor:
    """
    Devuelve (N, dim, K+1) float32 concatenando a0 y a_high, que es el mismo
    orden en el que el modelo los evalua.
    """
    a0 = getattr(modelo, f"{nombre}_a0").detach().cpu()
    hi = getattr(modelo, f"{nombre}_high").detach().cpu()
    if a0.dim() == 2:                        # opacity / theta / depth: (N,1), (N,K)
        a0 = a0.unsqueeze(1)
        hi = hi.unsqueeze(1)
    return torch.cat([a0, hi], dim=2).contiguous().float()


def exportar(
    ruta_checkpoint: str,
    ruta_salida: Path,
    top_k: int | None,
    fps: float,
    grados_override: dict[str, int] | None,
    n_muestras: int = 32,
    mu_a0_float32: bool = True,
    verbose: bool = True,
):
    # El cargador oficial devuelve (modelo, config, matrices_base, info).
    modelo, _config, _matrices, _info = cargar_modelo_desde_checkpoint(
        ruta_checkpoint, device="cpu")
    H, W = int(modelo.H), int(modelo.W)
    n_frames = int(modelo.n_frames)
    N_total = int(modelo.N)

    if verbose:
        print(f"checkpoint: N={N_total}  {W}x{H}  n_frames={n_frames}")
        print(f"grados: {modelo.grados}")

    imp = calcular_importancia(modelo, n_muestras)
    orden = orden_por_importancia(imp)
    K = N_total if top_k in (None, 0) else min(int(top_k), N_total)
    sel = orden[:K].contiguous()
    if verbose:
        print(f"importancia: max={imp.max():.5f}  mediana={imp.median():.5f}  "
              f"min={imp.min():.6f}")
        print(f"se conservan {K} de {N_total} gaussianas")

    grados = dict(modelo.grados)
    if grados_override:
        for a, g in grados_override.items():
            if a not in grados:
                raise SystemExit(f"atributo desconocido en --degradar_grados: {a}")
            if g > grados[a]:
                raise SystemExit(f"no se puede subir el grado de {a}: "
                                 f"{grados[a]} -> {g}")
            grados[a] = int(g)
        if verbose:
            print(f"grados tras truncar: {grados}")

    bloques, payload, offset = {}, [], 0
    orden_bloques = list(ATRIBUTOS)
    for nombre in ATRIBUTOS:
        c = _coefs_por_atributo(modelo, nombre)[sel]          # (K, dim, Korig+1)
        c = c[:, :, : grados[nombre] + 1].contiguous()        # truncado de grados
        arr = c.numpy().astype("<f2", copy=False)             # float16 little-endian
        raw = arr.tobytes(order="C")
        bloques[nombre] = {
            "dtype": "float16",
            "layout": "[N][dim][K+1], C-order",
            "shape": [K, DIMS[nombre], grados[nombre] + 1],
            "offset": offset,
            "length_bytes": len(raw),
        }
        payload.append(raw)
        offset += len(raw)

    # mu guarda coordenadas absolutas en pixeles: su coeficiente k=0 llega a
    # ~1080, y float16 solo tiene 11 bits de mantisa, asi que ahi el ULP vale
    # 1.0 px y el error alcanza 0.5 px. Medido sobre musical_20s_1600ep esto
    # cuesta 9 dB (47.11 vs 55.90 dB contra CUDA). Reponer SOLO ese coeficiente
    # en float32 recupera practicamente todo por 4 bytes por gaussiana.
    if mu_a0_float32:
        dc = _coefs_por_atributo(modelo, "mu")[sel][:, :, 0].contiguous()
        raw = dc.numpy().astype("<f4", copy=False).tobytes(order="C")
        bloques["mu_a0_f32"] = {
            "dtype": "float32",
            "layout": "[N][2], C-order",
            "shape": [K, 2],
            "offset": offset,
            "length_bytes": len(raw),
            "reemplaza": "coeficiente k=0 del bloque mu",
        }
        payload.append(raw)
        offset += len(raw)
        orden_bloques.append("mu_a0_f32")

    cabecera = {
        "magic": MAGIC.decode("ascii"),
        "version": VERSION,
        "n_gaussianas": K,
        "n_gaussianas_checkpoint": N_total,
        "ancho": W,
        "alto": H,
        "n_frames": n_frames,
        "fps": float(fps),
        "duracion_s": float(n_frames) / float(fps),
        "base_temporal": "chebyshev",
        "orden_gaussianas": "importancia_desc",
        "orden_bloques": orden_bloques,
        "grados": {a: int(grados[a]) for a in ATRIBUTOS},
        "dims": {a: DIMS[a] for a in ATRIBUTOS},
        "bloques": bloques,
        "tex_width": TEX_WIDTH,
        "activaciones": {
            "mu": "identidad",
            "theta": "identidad",
            "depth": "identidad",
            "opacity": "sigmoid",
            "color": "sigmoid",
            "scale": "exp(clamp(raw, log_scale_min, log_scale_max))",
        },
        "log_scale_min": math.log(0.5),
        "log_scale_max": math.log(float(max(H, W))),
        "render": {
            "k_sigma": 3.5,
            "tile_size": 16,
            "alpha_max": 0.99,
            "transmitancia_min": 1.0e-4,
            "fondo_rgb": [0.0, 0.0, 0.0],
            "orden_profundidad": "depth ascendente compuesto primero",
            "satura_color": False,
        },
    }

    cab_json = json.dumps(cabecera, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True).encode("utf-8")
    pad = (-(len(MAGIC) + 4 + len(cab_json))) % 4
    cab_json += b" " * pad

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_salida, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(cab_json)))
        f.write(cab_json)
        for raw in payload:
            f.write(raw)

    tam = ruta_salida.stat().st_size
    sha = hashlib.sha256(ruta_salida.read_bytes()).hexdigest()[:16]
    if verbose:
        print(f"escrito {ruta_salida}")
        print(f"  tamano: {tam:,} B = {tam / 1e6:.2f} MB")
        print(f"  sha256[:16]: {sha}")
    return {"ruta": ruta_salida, "bytes": tam, "sha256_16": sha,
            "N": K, "W": W, "H": H, "n_frames": n_frames,
            "fps": float(fps), "duracion_s": float(n_frames) / float(fps),
            "grados": grados}


def _parse_grados(txt: str | None) -> dict[str, int] | None:
    """--degradar_grados "mu=60,color=20"  ->  {"mu": 60, "color": 20}"""
    if not txt:
        return None
    out = {}
    for parte in txt.split(","):
        parte = parte.strip()
        if not parte:
            continue
        a, _, g = parte.partition("=")
        out[a.strip()] = int(g)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--top_k", type=int, default=0,
                    help="0 = todas las gaussianas (por defecto)")
    ap.add_argument("--fps", type=float, default=24.0,
                    help="NO esta en el checkpoint; ver FORMATO_MODELO_WEB.md")
    ap.add_argument("--degradar_grados", default=None,
                    help='truncar grados, p.ej. "mu=60,opacity=50"')
    ap.add_argument("--n_muestras_importancia", type=int, default=32)
    ap.add_argument("--sin_mu_a0_float32", action="store_true",
                    help="guarda TODO en float16; cuesta ~9 dB, ver el documento")
    args = ap.parse_args()

    exportar(
        args.checkpoint,
        Path(args.salida),
        args.top_k,
        args.fps,
        _parse_grados(args.degradar_grados),
        args.n_muestras_importancia,
        mu_a0_float32=not args.sin_mu_a0_float32,
    )


if __name__ == "__main__":
    main()
