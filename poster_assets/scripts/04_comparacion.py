"""
Figura 4 - Mismo frame: original vs reconstruido, en dos archivos separados.

Copia sin reescalar los PNG que ya existen (el clip original y los
frames_renderizados del experimento). No reinterpola: si el frame nativo mide
menos que --min_lado, avisa en vez de agrandar en silencio.
"""
import argparse
import shutil
import numpy as np
from PIL import Image
import comun


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=240)
    ap.add_argument("--min_lado", type=int, default=2000)
    args = ap.parse_args()

    orig = comun.CLIP_DIR / f"frame_{args.frame:04d}.png"
    recon = comun.EXP_DIR / "frames_renderizados" / f"frame_{args.frame:04d}.png"
    for p in (orig, recon):
        if not p.exists():
            raise SystemExit(f"no existe {p}")

    comun.OUT.mkdir(parents=True, exist_ok=True)
    destinos = []
    for p, nombre in ((orig, "04a_original.png"), (recon, "04b_reconstruido.png")):
        d = comun.OUT / nombre
        shutil.copyfile(p, d)
        destinos.append((d, Image.open(d).size))
        print(f"  guardado {d}  {Image.open(d).size[0]}x{Image.open(d).size[1]}")

    w, h = destinos[0][1]
    if max(w, h) < args.min_lado:
        print(f"\n  AVISO: el lado mayor es {max(w, h)} px, por debajo de los "
              f"{args.min_lado} px pedidos.")
        print("  No se interpola. Para obtener mas resolucion hay que re-renderizar")
        print("  el checkpoint a mayor tamano con el rasterizador CUDA, que esta")
        print("  disponible solo en una maquina con GPU NVIDIA. El original del clip")
        print("  tampoco existe a mas resolucion: fue extraido a esta.")

    # PSNR del par concreto, para poder citarlo en el pie de figura.
    a = np.asarray(Image.open(orig).convert("RGB"), dtype=np.float64) / 255.0
    b = np.asarray(Image.open(recon).convert("RGB"), dtype=np.float64) / 255.0
    if a.shape == b.shape:
        mse = float(np.mean((a - b) ** 2))
        print(f"\n  PSNR de este par (frame {args.frame}): "
              f"{10 * np.log10(1.0 / mse):.2f} dB   [calculado aqui sobre los dos PNG]")


if __name__ == "__main__":
    main()
