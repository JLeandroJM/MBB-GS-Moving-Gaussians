"""
Figura 3 - Una gaussiana 2D aislada, fondo transparente, para usar de icono.

Evalua analiticamente exp(-1/2 (x-mu)^T Sigma^-1 (x-mu)) con los parametros
REALES de una gaussiana del checkpoint (escala, rotacion, color, opacidad).
No pasa por el rasterizador porque no hace falta: una sola primitiva sin
composicion alpha es exactamente esta formula.
"""
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import comun


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=240)
    ap.add_argument("--px", type=int, default=2048, help="lado del PNG")
    ap.add_argument("--anisotropia_min", type=float, default=1.8,
                    help="relacion entre ejes, para que se vea elipse y no circulo")
    ap.add_argument("--saturacion_min", type=float, default=0.0,
                    help="saturacion HSV minima; sube a ~0.5 para un icono vivo")
    ap.add_argument("--brillo_min", type=float, default=0.0,
                    help="valor HSV minimo; sin esto se eligen gaussianas casi negras")
    ap.add_argument("--indice", type=int, default=None,
                    help="usar esta gaussiana concreta en vez de elegir")
    ap.add_argument("--salida", default="03_gaussiana_aislada.png")
    args = ap.parse_args()

    coefs, meta = comun.cargar()
    H, W, T = meta["H"], meta["W"], meta["n_frames"]
    tau = np.array([2.0 * args.frame / (T - 1) - 1.0])

    sc = comun.activar("scale", comun.evaluar(coefs, meta, "scale", tau), H, W)[0]  # (N,2)
    op = torch.sigmoid(comun.evaluar(coefs, meta, "opacity", tau))[0, :, 0]
    col = torch.sigmoid(comun.evaluar(coefs, meta, "color", tau))[0]
    th = comun.evaluar(coefs, meta, "theta", tau)[0, :, 0]

    # Una gaussiana bien visible y claramente anisotropa: mejor icono.
    ratio = sc.max(1).values / sc.min(1).values.clamp(min=1e-6)
    mx = col.max(1).values
    sat = (mx - col.min(1).values) / mx.clamp(min=1e-6)   # saturacion HSV
    cand = torch.nonzero((op > 0.85) & (ratio > args.anisotropia_min) &
                         (sc.min(1).values > 1.0) &
                         (sat >= args.saturacion_min) &
                         (mx >= args.brillo_min)).squeeze(1)
    if cand.numel() == 0:
        raise SystemExit("ninguna gaussiana cumple los filtros; baja --saturacion_min")
    if args.indice is not None:
        i = args.indice
    else:
        # entre las que pasan, la de mayor saturacion*area: vistosa y grande
        puntaje = sat[cand] * mx[cand] * sc[cand, 0] * sc[cand, 1]
        i = cand[torch.argsort(puntaje, descending=True)[0]].item()
    print(f"  candidatas que pasan los filtros: {cand.numel()}")
    s1, s2 = sc[i].tolist(); theta = float(th[i]); c = col[i].tolist()
    print(f"  gaussiana #{i}: escala=({s1:.2f}, {s2:.2f}) px, theta={theta:.3f} rad, "
          f"opacidad={op[i]:.3f}, color=({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})")

    # Malla en unidades de sigma, con margen de 3.5 sigma (el k_sigma del render).
    k = 3.5
    n = args.px
    lim = k * max(s1, s2)
    g = np.linspace(-lim, lim, n)
    X, Y = np.meshgrid(g, g)
    ct, st = np.cos(theta), np.sin(theta)
    Xr = ct * X + st * Y
    Yr = -st * X + ct * Y
    G = np.exp(-0.5 * ((Xr / s1) ** 2 + (Yr / s2) ** 2))
    G[(Xr / s1) ** 2 + (Yr / s2) ** 2 > k ** 2] = 0.0      # mismo corte que el kernel

    rgba = np.zeros((n, n, 4), dtype=np.float32)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = c
    rgba[..., 3] = G * float(op[i])

    comun.OUT.mkdir(parents=True, exist_ok=True)
    ruta = comun.OUT / args.salida
    Image.fromarray((rgba * 255).astype(np.uint8), mode="RGBA").save(ruta)
    print(f"  guardado {ruta}  ({n}x{n}, RGBA con fondo transparente)")


if __name__ == "__main__":
    main()
