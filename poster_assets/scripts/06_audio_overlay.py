"""
Figura 6 - Waveform original vs reconstruida por atomos de Gabor, superpuestas.

Lee los WAV que el entrenamiento ya dejo (target.wav / recon.wav). Elige
automaticamente un fragmento con energia, donde el ajuste se aprecia, en vez de
un tramo de silencio.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import comun

EXP_AUDIO = Path("/Users/jleandrojm/TESIS/MBB-GS/outputs/gabor/gabor_rock_8s_mono_ganador_N48k")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=str(EXP_AUDIO))
    ap.add_argument("--ms", type=float, default=25.0, help="duracion del fragmento")
    ap.add_argument("--leyenda", choices=["arriba", "ninguna"], default="arriba",
                    help="'arriba' la coloca fuera del area de trazado, sobre el "
                         "eje, para que nunca pise la senal; 'ninguna' la omite "
                         "y el pie de figura del poster explica los colores")
    ap.add_argument("--salida", default="06_audio_overlay.png")
    args = ap.parse_args()

    exp = Path(args.exp)
    tgt, sr = sf.read(exp / "target.wav")
    rec, _ = sf.read(exp / "recon.wav")
    if tgt.ndim > 1:
        tgt, rec = tgt[:, 0], rec[:, 0]
    n = int(args.ms * 1e-3 * sr)

    # Ventana de maxima energia: donde hay senal que ajustar.
    e = np.convolve(tgt ** 2, np.ones(n), mode="valid")
    i0 = int(np.argmax(e))
    t = np.arange(n) / sr * 1000.0
    a, b = tgt[i0:i0 + n], rec[i0:i0 + n]

    err = float(np.sqrt(np.mean((a - b) ** 2)))
    snr = 10 * np.log10(np.sum(a ** 2) / np.sum((a - b) ** 2))
    print(f"  {exp.name}: sr={sr} Hz, fragmento {args.ms} ms desde "
          f"{i0 / sr:.3f} s, SNR local {snr:.2f} dB, RMS error {err:.5f}")

    fig, ax = plt.subplots(figsize=(11, 3.8), dpi=100)
    ax.plot(t, a, lw=2.6, color="#1b2a4a", label="Original", zorder=2)
    ax.plot(t, b, lw=2.0, color="#e4572e", label="Reconstruido (Gabor)",
            zorder=3, alpha=0.9)
    ax.set_xlim(0, args.ms)
    ax.set_xlabel("tiempo (ms)", fontsize=13)
    ax.set_ylabel("amplitud", fontsize=13)
    if args.leyenda == "arriba":
        # Fuera del area de trazado: con una waveform que llena el cuadro no hay
        # ningun hueco interior donde la leyenda no tape picos.
        ax.legend(frameon=False, fontsize=13, ncol=2,
                  loc="lower left", bbox_to_anchor=(0.0, 1.01),
                  borderaxespad=0.0, handlelength=2.4, columnspacing=2.2)
    ax.tick_params(labelsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.8)
    fig.tight_layout()
    comun.guardar(fig, args.salida, dpi=260)

    m = exp / "metricas.json"
    if m.exists():
        d = json.load(open(m))
        print("  metricas globales del run:",
              {k: round(v, 4) for k, v in d.items() if isinstance(v, (int, float))})


if __name__ == "__main__":
    main()
