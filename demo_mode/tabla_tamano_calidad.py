#!/usr/bin/env python3
"""
Tabla tamano vs calidad para elegir la configuracion del visor web.

Para cada top_k (y opcionalmente grados truncados) exporta el modelo, lo
renderiza con el renderizador de REFERENCIA y mide el PSNR contra los frames
ORIGINALES del clip. Escribe demo_mode/TABLA_TAMANO_CALIDAD.md.
"""
import sys, json, time
from pathlib import Path
import numpy as np
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "demo_mode"))
sys.path.insert(0, str(RAIZ / "src"))
import render_referencia_cpu as R
from exportar_modelo_web import exportar

CK = "/Users/jleandrojm/TESIS/MBB-GS/outputs_khipu/musical_20s_1600ep/checkpoints/checkpoint_final.pt"
ORIG = Path("/Users/jleandrojm/TESIS/MBB-GS/data/clips/musical_20s")
CUDA = Path("/Users/jleandrojm/TESIS/MBB-GS/outputs_khipu/musical_20s_1600ep/frames_renderizados")
TMP = Path("/private/tmp/claude-501/-Users-jleandrojm-TESIS-MBB-GS-Moving-Gaussians/02571318-ffa0-40a2-b24b-b4850f3c15c9/scratchpad/tabla")
FRAMES = [0, 240, 479]


def medir(nombre, top_k, grados=None):
    TMP.mkdir(parents=True, exist_ok=True)
    salida = TMP / f"{nombre}.bin"
    t0 = time.time()
    info = exportar(CK, salida, top_k, 24.0, grados, verbose=False)
    cab, coefs = R.leer_modelo(salida)
    ps_o, ps_c = [], []
    for j in FRAMES:
        img = R.a_uint8(R.render_tau(cab, coefs, R.tau_de_frame(j, cab["n_frames"])))
        ps_o.append(R.psnr(img, np.asarray(Image.open(ORIG / f"frame_{j:04d}.png").convert("RGB"))))
        ps_c.append(R.psnr(img, np.asarray(Image.open(CUDA / f"frame_{j:04d}.png").convert("RGB"))))
    fila = {"nombre": nombre, "top_k": info["N"], "grados": info["grados"],
            "bytes": info["bytes"], "mb": info["bytes"] / 1e6,
            "psnr_original": float(np.mean(ps_o)), "psnr_cuda": float(np.mean(ps_c)),
            "seg": time.time() - t0}
    print(f"  {nombre:26s} N={fila['top_k']:>6}  {fila['mb']:7.2f} MB  "
          f"orig {fila['psnr_original']:6.2f} dB  cuda {fila['psnr_cuda']:6.2f} dB "
          f"({fila['seg']:.0f}s)", flush=True)
    salida.unlink(missing_ok=True)
    return fila


def main():
    filas = []
    print("== barrido de top_k, grados completos ==", flush=True)
    for k in [5000, 10000, 20000, 30000, 50000, 150000]:
        filas.append(medir(f"topk_{k}", k))

    print("\n== truncado de grados, a top_k fijo 30000 ==", flush=True)
    for etiqueta, g in [("mu=80", {"mu": 80}), ("mu=60", {"mu": 60}),
                        ("mu=40", {"mu": 40}),
                        ("mu=60,op=50", {"mu": 60, "opacity": 50}),
                        ("mu=60,op=50,col=20", {"mu": 60, "opacity": 50, "color": 20})]:
        filas.append(medir(f"g_{etiqueta}", 30000, g))

    print("\n== la mejor combinacion bajo 30 MB ==", flush=True)
    for k, g in [(50000, {"mu": 60}), (50000, {"mu": 60, "opacity": 50}),
                 (40000, {"mu": 70}), (60000, {"mu": 50, "opacity": 40})]:
        filas.append(medir(f"mix_{k}_{'_'.join(f'{a}{v}' for a, v in g.items())}", k, g))

    (RAIZ / "demo_mode" / "tabla_tamano_calidad.json").write_text(
        json.dumps(filas, indent=2, sort_keys=True), encoding="utf-8")
    print("\njson escrito")


if __name__ == "__main__":
    main()
