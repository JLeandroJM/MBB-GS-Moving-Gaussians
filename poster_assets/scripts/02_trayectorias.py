"""
Figura 2 - Trayectorias mu(t) de gaussianas representativas.

Medicion previa sobre el checkpoint: el desplazamiento NETO tipico es de ~7 px
(p50) y como mucho ~122 px, mientras que el recorrido total llega a cientos de
px. Es decir, las gaussianas oscilan alrededor de su zona en vez de cruzar el
cuadro. Dibujarlas sobre el frame completo (1080x720) produce garabatos
ilegibles, asi que por defecto se genera un recorte ampliado sobre la region
con mas movimiento, que es lo que se lee de lejos en un poster.

  --modo zoom    recorte ampliado (por defecto, recomendado para el poster)
  --modo full    frame completo, para contexto

Seleccion: se ordena por el diametro del bounding box de la trayectoria
(desplazamiento neto), no por longitud de arco, para no premiar el temblor en
sitio. No necesita CUDA.
"""
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image
import comun


def region_con_mas_movimiento(mu, span, mask, H, W, alto, ancho):
    """
    Ventana que maximiza el span acumulado, y luego re-centrada en el
    centroide de ese cluster para que las trayectorias no queden pegadas
    a un borde del recorte.
    """
    idx = torch.nonzero(mask).squeeze(1)
    y0 = mu[0, idx, 0].numpy(); x0 = mu[0, idx, 1].numpy(); w = span[idx].numpy()
    mejor, centro = -1.0, (H / 2.0, W / 2.0)
    for cy in np.arange(alto / 2, H - alto / 2 + 1, 30):
        for cx in np.arange(ancho / 2, W - ancho / 2 + 1, 30):
            sel = (np.abs(y0 - cy) < alto / 2) & (np.abs(x0 - cx) < ancho / 2)
            tot = w[sel].sum()
            if tot > mejor:
                mejor, centro = tot, (cy, cx)
    cy, cx = centro
    sel = (np.abs(y0 - cy) < alto / 2) & (np.abs(x0 - cx) < ancho / 2)
    if sel.sum() > 0:
        cy = float(np.average(y0[sel], weights=w[sel]))
        cx = float(np.average(x0[sel], weights=w[sel]))
    cy = float(np.clip(cy, alto / 2, H - alto / 2))
    cx = float(np.clip(cx, ancho / 2, W - ancho / 2))
    return cy, cx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", choices=["zoom", "full"], default="zoom")
    ap.add_argument("--frame_fondo", type=int, default=0)
    ap.add_argument("--n", type=int, default=14)
    ap.add_argument("--sep", type=float, default=None,
                    help="separacion minima entre trayectorias, px")
    ap.add_argument("--alto", type=int, default=260, help="alto del recorte, px")
    ap.add_argument("--ancho", type=int, default=390, help="ancho del recorte, px")
    ap.add_argument("--salida", default=None)
    ap.add_argument("--cmap", default="plasma")
    ap.add_argument("--dpi", type=int, default=None)
    args = ap.parse_args()

    coefs, meta = comun.cargar()
    H, W, T = meta["H"], meta["W"], meta["n_frames"]

    print("evaluando mu(t) y opacity(t) ...")
    mu = comun.evaluar(coefs, meta, "mu")
    op = torch.sigmoid(comun.evaluar(coefs, meta, "opacity"))[:, :, 0]

    span = torch.linalg.norm(mu.max(0).values - mu.min(0).values, dim=1)
    dentro = ((mu[:, :, 0] >= 0) & (mu[:, :, 0] < H) &
              (mu[:, :, 1] >= 0) & (mu[:, :, 1] < W)).all(0)
    mask = (op.mean(0) > 0.3) & dentro

    if args.modo == "zoom":
        cy, cx = region_con_mas_movimiento(mu, span, mask, H, W, args.alto, args.ancho)
        y0, y1 = int(cy - args.alto / 2), int(cy + args.alto / 2)
        x0, x1 = int(cx - args.ancho / 2), int(cx + args.ancho / 2)
        print(f"  recorte y=[{y0},{y1}] x=[{x0},{x1}]")
        en_caja = ((mu[0, :, 0] >= y0) & (mu[0, :, 0] < y1) &
                   (mu[0, :, 1] >= x0) & (mu[0, :, 1] < x1))
        cand = torch.nonzero(mask & en_caja & (span > 8)).squeeze(1)
    else:
        y0, y1, x0, x1 = 0, H, 0, W
        cand = torch.nonzero(mask & (span > 40)).squeeze(1)

    # Separacion minima entre trayectorias: sin esto se superponen en un
    # borron, porque el cluster de movimiento es compacto.
    orden = cand[torch.argsort(span[cand], descending=True)]
    sep = args.sep if args.sep is not None else (0.13 * min(args.alto, args.ancho))
    elegidas, centros = [], []
    for i in orden.tolist():
        c = mu[:, i, :].mean(0)
        if all(torch.linalg.norm(c - o).item() >= sep for o in centros):
            elegidas.append(i); centros.append(c)
        if len(elegidas) >= args.n:
            break
    print(f"  {len(elegidas)} trayectorias, separacion minima {sep:.0f} px, "
          f"span {span[elegidas[0]]:.0f} .. {span[elegidas[-1]]:.0f} px")

    fondo = np.asarray(Image.open(
        comun.CLIP_DIR / f"frame_{args.frame_fondo:04d}.png").convert("RGB"))[y0:y1, x0:x1]
    ah, aw = y1 - y0, x1 - x0

    fig, ax = plt.subplots(figsize=(aw / 100, ah / 100), dpi=100)
    ax.imshow(fondo, extent=[x0, x1, y1, y0])
    ax.imshow(np.ones((ah, aw, 4)) * np.array([1, 1, 1, 0.22]), extent=[x0, x1, y1, y0])

    cmap = plt.get_cmap(args.cmap)
    norm = plt.Normalize(0, T - 1)
    lw = 2.6 if args.modo == "zoom" else 2.0
    for i in elegidas:
        p = mu[:, i, :].numpy()
        pts = np.stack([p[:, 1], p[:, 0]], axis=1).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        # halo blanco para que la curva se despegue del fondo a distancia
        ax.add_collection(LineCollection(segs, colors="white",
                                         linewidths=lw + 1.8, alpha=0.8,
                                         capstyle="round"))
        lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=lw,
                            capstyle="round")
        lc.set_array(np.arange(T - 1))
        ax.add_collection(lc)
        ax.scatter(p[0, 1], p[0, 0], s=46, c=[cmap(0.0)], zorder=5,
                   edgecolors="white", linewidths=1.4)

    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.axis("off")
    salida = args.salida or f"02_trayectorias_{args.modo}.png"
    comun.guardar(fig, salida, dpi=args.dpi or (700 if args.modo == "zoom" else 340))
    print("  punto = instante inicial; el color avanza con el tiempo")


if __name__ == "__main__":
    main()
