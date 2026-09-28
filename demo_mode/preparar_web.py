#!/usr/bin/env python3
"""
Prepara un modelo para el repositorio del visor: exporta, genera la miniatura
del frame central y actualiza demo_mode/export_web/manifest.json.

Nombres de archivo: minusculas, sin espacios, sin empezar por "_" ni "." para
que Jekyll (GitHub Pages) no los ignore. El --id se valida contra esa regla.

Determinista: mismo checkpoint y mismos argumentos -> .bin identico byte a byte.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "demo_mode"))
sys.path.insert(0, str(RAIZ / "src"))

import render_referencia_cpu as R           # noqa: E402
from exportar_modelo_web import exportar     # noqa: E402

SALIDA = RAIZ / "demo_mode" / "export_web"
RE_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def miniatura(cab, coefs, destino: Path, ancho: int = 640):
    """Frame central renderizado por la referencia, escalado a `ancho` px."""
    from PIL import Image
    j = (cab["n_frames"] - 1) / 2.0
    img = R.a_uint8(R.render_tau(cab, coefs, R.tau_de_frame(j, cab["n_frames"])))
    im = Image.fromarray(img)
    alto = max(1, round(im.height * ancho / im.width))
    im = im.resize((ancho, alto), Image.LANCZOS)
    im.save(destino, optimize=True)
    return im.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--id", required=True,
                    help="identificador en minusculas y guiones, p.ej. musical-20s")
    ap.add_argument("--titulo", required=True)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--degradar_grados", default=None)
    ap.add_argument("--ancho_miniatura", type=int, default=640)
    args = ap.parse_args()

    if not RE_ID.match(args.id):
        raise SystemExit(f"--id invalido: {args.id!r}. Solo minusculas, digitos y "
                         f"guiones, sin empezar por '-', '_' ni '.'")

    from exportar_modelo_web import _parse_grados
    SALIDA.mkdir(parents=True, exist_ok=True)
    bin_path = SALIDA / f"{args.id}.bin"
    png_path = SALIDA / f"{args.id}.png"

    info = exportar(args.checkpoint, bin_path, args.top_k, args.fps,
                    _parse_grados(args.degradar_grados))

    cab, coefs = R.leer_modelo(bin_path)
    w, h = miniatura(cab, coefs, png_path, args.ancho_miniatura)
    print(f"  miniatura {png_path.name}  {w}x{h}  "
          f"{png_path.stat().st_size / 1e3:.0f} KB")

    entrada = {
        "id": args.id,
        "titulo": args.titulo,
        "archivo": bin_path.name,
        "miniatura": png_path.name,
        "bytes": info["bytes"],
        "n_gaussianas": info["N"],
        "ancho": info["W"],
        "alto": info["H"],
        "n_frames": info["n_frames"],
        "fps": info["fps"],
        "duracion_s": info["duracion_s"],
        "grados": info["grados"],
        "sha256_16": info["sha256_16"],
    }

    man_path = SALIDA / "manifest.json"
    modelos = []
    if man_path.exists():
        modelos = json.loads(man_path.read_text(encoding="utf-8")).get("modelos", [])
    modelos = [m for m in modelos if m["id"] != args.id] + [entrada]
    modelos.sort(key=lambda m: m["id"])

    man_path.write_text(json.dumps(
        {"version": 1, "formato": "MBBGSW01", "modelos": modelos},
        indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    print(f"  manifest.json: {len(modelos)} modelo(s)")


if __name__ == "__main__":
    main()
