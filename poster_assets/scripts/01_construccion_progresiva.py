"""
Figura 1 - Construccion progresiva: el mismo frame con solo las top-K gaussianas.

REQUIERE GPU NVIDIA con la extension raster_cuda compilada. Usa el MISMO
rasterizador del proyecto (gs2d_video.render.renderer.render_frame), no una
reimplementacion, para que la figura sea fiel al metodo.

Criterio de contribucion
------------------------
Se ordena por  alpha_i(t) * s1_i(t) * s2_i(t)  evaluado en el frame objetivo:

  - alpha_i(t) es la opacidad activada (sigmoide): cuanto aporta por pixel.
  - s1*s2 es proporcional al area efectiva de la elipse, asi que el producto
    es proporcional a la "masa de alpha" que la gaussiana deposita en la imagen
    (la integral de una gaussiana 2D es 2*pi*alpha*s1*s2, y el 2*pi es constante,
    no cambia el orden).

Es decir: se prioriza lo que mas cubre la imagen en ese instante. Alternativas
razonables serian ordenar por opacidad sola (ignora el tamano) o por el
gradiente de la perdida (mas fiel a "importancia" pero mucho mas caro y
dependiente del target). Se eligio la cobertura porque es la que produce una
construccion progresiva legible: primero las manchas grandes de color, luego
el detalle fino.

Uso
---
    python 01_construccion_progresiva.py \
        --checkpoint <ruta>/checkpoint_final.pt \
        --frame 240 --escala 2 \
        --ks 1 10 100 1000 0          # 0 = todas

--escala re-renderiza a mayor resolucion aprovechando que las gaussianas son
continuas: multiplica mu y scale por el factor y pide un lienzo mas grande.
No es interpolacion, es evaluar la misma representacion en una grilla mas fina.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--frame", type=int, default=240)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 10, 100, 1000, 0],
                    help="0 significa todas las gaussianas")
    ap.add_argument("--escala", type=float, default=2.0,
                    help="factor de supersampling del lienzo")
    ap.add_argument("--salida", default=None)
    ap.add_argument("--tile", type=int, default=16)
    ap.add_argument("--k_sigma", type=float, default=3.5)
    args = ap.parse_args()

    from gs2d_video.io.checkpoints import cargar_modelo_desde_checkpoint
    from gs2d_video.render.renderer import render_frame

    salida = Path(args.salida) if args.salida else Path(__file__).resolve().parents[1] / "img"
    salida.mkdir(parents=True, exist_ok=True)

    modelo, matrices, meta = cargar_modelo_desde_checkpoint(args.checkpoint, device="cuda")
    H, W = int(meta["H"]), int(meta["W"])
    params = modelo.evaluar_en_frame(args.frame, matrices)   # atributos ya activados

    alpha = params["opacity"].reshape(-1)
    escalas = params["scale"]
    contrib = alpha * escalas[:, 0] * escalas[:, 1]
    orden = torch.argsort(contrib, descending=True)
    print(f"contribucion: max={contrib.max():.4f} min={contrib.min():.6f} "
          f"mediana={contrib.median():.6f}")

    f = float(args.escala)
    Hs, Ws = int(round(H * f)), int(round(W * f))
    cfg = {"usar_cuda_tiled": True, "cuda_tile_size": args.tile, "cuda_k_sigma": args.k_sigma}

    for K in args.ks:
        idx = orden if K == 0 else orden[:K]
        sub = {k: v[idx].clone() for k, v in params.items()}
        if f != 1.0:
            sub["mu"] = sub["mu"] * f
            sub["scale"] = sub["scale"] * f
        img = render_frame(sub, Hs, Ws, cfg)                  # (3,H,W) o (H,W,3)
        arr = img.detach().float().clamp(0, 1).cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        nombre = f"01_progresiva_K{'todas' if K == 0 else K}.png"
        Image.fromarray((arr * 255).astype(np.uint8)).save(salida / nombre)
        print(f"  {nombre}  {Ws}x{Hs}  ({len(idx)} gaussianas)")


if __name__ == "__main__":
    main()
