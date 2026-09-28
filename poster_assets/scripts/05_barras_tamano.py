"""
Figura 5 - Tamano de la representacion frente a baselines de video.

Todos los tamanos son de archivos reales medidos con os.path.getsize. Los
baselines se generaron con ffmpeg sobre los MISMOS 480 frames del clip; los
comandos estan en BRIEF_POSTER.md para que cualquiera los reproduzca.

Honestidad: nuestro modelo NO gana en tamano. El grafico lo muestra en escala
logaritmica junto al PSNR de cada punto, porque el argumento del poster no es
compresion sino representacion continua. Ver BRIEF_POSTER.md, "Limitaciones".
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import comun

# (etiqueta, bytes, PSNR dB, nota, es_nuestro)
DATOS = [
    ("PNG sin perdida\n(480 frames)",      551_290_835, None,  "",            False),
    ("MBB-GS\n(nuestro modelo)",           248_407_393, 42.27, "RGB",         True),
    ("AVIF frame a frame\nCRF 30",          22_134_784, 36.79, "RGB, 40 fr.", False),
    ("MP4 fuente\n(H.264)",                 21_854_213, None,  "",            False),
    ("AVIF frame a frame\nCRF 45",           9_682_944, 32.66, "RGB, 40 fr.", False),
    ("AV1 video\nCRF 30",                    2_806_111, 42.62, "Y",           False),
    ("AV1 video\nCRF 45",                    1_015_970, 39.60, "Y",           False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default="05_barras_tamano.png")
    args = ap.parse_args()

    datos = sorted(DATOS, key=lambda d: d[1], reverse=True)
    etiquetas = [d[0] for d in datos]
    mb = np.array([d[1] for d in datos]) / 1e6
    nuestro = [d[4] for d in datos]

    fig, ax = plt.subplots(figsize=(13, 6.6), dpi=100)
    colores = ["#e4572e" if n else "#9fb3c8" for n in nuestro]
    barras = ax.bar(range(len(datos)), mb, color=colores, width=0.66,
                    edgecolor="white", linewidth=1.5, zorder=3)

    ax.set_yscale("log")
    ax.set_ylim(0.5, mb.max() * 3.2)
    ax.set_ylabel("tamano del archivo (MB, escala log)", fontsize=15)
    ax.set_xticks(range(len(datos)))
    ax.set_xticklabels(etiquetas, fontsize=12.5)
    ax.tick_params(axis="y", labelsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.28, lw=0.9, zorder=0)

    for i, (b, d) in enumerate(zip(barras, datos)):
        v = d[1] / 1e6
        txt = f"{v:,.1f} MB" if v >= 1 else f"{d[1] / 1e3:,.0f} KB"
        ax.text(b.get_x() + b.get_width() / 2, v * 1.14, txt,
                ha="center", fontsize=13,
                fontweight="bold" if d[4] else "normal",
                color="#e4572e" if d[4] else "#33475b")
        if d[2] is not None:
            ax.text(b.get_x() + b.get_width() / 2, v * 1.9,
                    f"{d[2]:.1f} dB", ha="center", fontsize=11.5, color="#5a6b7d")

    ax.text(0.99, 0.97,
            "PSNR indicado donde se midio.  MBB-GS y AVIF en RGB;  AV1 en canal Y (ffmpeg).\n"
            "No son directamente comparables entre espacios de color.",
            transform=ax.transAxes, ha="right", va="top", fontsize=10.5,
            color="#6b7a8c")

    fig.tight_layout()
    comun.guardar(fig, args.salida, dpi=220)
    ratio = 248_407_393 / 2_806_111
    print(f"  nuestro modelo es {ratio:.0f}x mas grande que AV1 CRF30 a PSNR similar")


if __name__ == "__main__":
    main()
