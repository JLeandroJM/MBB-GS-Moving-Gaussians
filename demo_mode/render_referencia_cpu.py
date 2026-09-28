#!/usr/bin/env python3
"""
Renderizador de referencia en CPU, implementacion directa de
demo_mode/FORMATO_MODELO_WEB.md.

Lee el ARCHIVO EXPORTADO (.bin), no el checkpoint, para validar el formato de
punta a punta: si este render coincide con el de CUDA, la especificacion que
recibe el visor web es correcta.

Prioriza claridad sobre velocidad. Replica el rasterizador CUDA paso a paso,
incluido el teselado, porque el recorte por k_sigma decide que gaussianas
afectan a que pixeles y por tanto cambia el resultado.

Subcomandos
-----------
  render     renderiza un frame o un tau arbitrario a PNG
  validar    compara contra los frames renderizados por CUDA y reporta PSNR
  vectores   genera los vectores de prueba de demo_mode/referencia/
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"MBBGSW01"


# --------------------------------------------------------------------------
# lectura del archivo exportado
# --------------------------------------------------------------------------
def leer_modelo(ruta: Path):
    datos = Path(ruta).read_bytes()
    if datos[:8] != MAGIC:
        raise ValueError(f"magic inesperado: {datos[:8]!r}")
    (n_cab,) = struct.unpack_from("<I", datos, 8)
    cab = json.loads(datos[12:12 + n_cab].decode("utf-8"))
    base = 12 + n_cab

    coefs = {}
    for nombre in cab["orden_bloques"]:
        b = cab["bloques"][nombre]
        dt = np.dtype("<f2" if b["dtype"] == "float16" else "<f4")
        arr = np.frombuffer(datos, dtype=dt,
                            count=b["length_bytes"] // dt.itemsize,
                            offset=base + b["offset"])
        coefs[nombre] = arr.reshape(b["shape"]).astype(np.float32)

    # Bloque opcional de alta precision: repone el coeficiente k=0 de mu.
    dc = coefs.pop("mu_a0_f32", None)
    if dc is not None:
        coefs["mu"] = coefs["mu"].copy()
        coefs["mu"][:, :, 0] = dc
    return cab, coefs


# --------------------------------------------------------------------------
# base temporal y activaciones
# --------------------------------------------------------------------------
def tau_de_frame(j: float, n_frames: int) -> float:
    """tau(j) = 2j/(n_frames-1) - 1.  j puede ser fraccionario."""
    if n_frames < 2:
        return 0.0
    return 2.0 * (float(j) / (n_frames - 1)) - 1.0


def fila_chebyshev(tau: float, grado: int) -> np.ndarray:
    """[T_0(tau), ..., T_grado(tau)] por la recurrencia estandar, en float64."""
    B = np.empty(grado + 1, dtype=np.float64)
    B[0] = 1.0
    if grado >= 1:
        B[1] = tau
    for k in range(2, grado + 1):
        B[k] = 2.0 * tau * B[k - 1] - B[k - 2]
    return B


def evaluar_atributos(cab, coefs, tau: float) -> dict[str, np.ndarray]:
    """Evalua los polinomios en tau y aplica las activaciones del proyecto."""
    crudo = {}
    for nombre in cab["grados"]:
        g = cab["grados"][nombre]
        B = fila_chebyshev(tau, g).astype(np.float32)      # (K+1,)
        crudo[nombre] = coefs[nombre] @ B                   # (N, dim)

    lo = np.float32(cab["log_scale_min"])
    hi = np.float32(cab["log_scale_max"])
    return {
        "mu": crudo["mu"],                                        # identidad
        "theta": crudo["theta"][:, 0],                            # identidad
        "depth": crudo["depth"][:, 0],                            # identidad
        "opacity": 1.0 / (1.0 + np.exp(-crudo["opacity"][:, 0])),  # sigmoid
        "color": 1.0 / (1.0 + np.exp(-crudo["color"])),            # sigmoid
        "scale": np.exp(np.clip(crudo["scale"], lo, hi)),          # exp con clamp
    }


def construir_conic(scale: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Inversa de la covarianza, tal como build_conic_kernel:
        inv_sx2 = 1/(sx^2 + 1e-8),  inv_sy2 = 1/(sy^2 + 1e-8)
        m00 = c^2 inv_sx2 + s^2 inv_sy2
        m01 = c s (inv_sx2 - inv_sy2)
        m11 = s^2 inv_sx2 + c^2 inv_sy2
    """
    sx, sy = scale[:, 0], scale[:, 1]
    s, c = np.sin(theta), np.cos(theta)
    inv_sx2 = 1.0 / (sx * sx + 1e-8)
    inv_sy2 = 1.0 / (sy * sy + 1e-8)
    return np.stack([
        c * c * inv_sx2 + s * s * inv_sy2,
        c * s * (inv_sx2 - inv_sy2),
        s * s * inv_sx2 + c * c * inv_sy2,
    ], axis=1).astype(np.float32)


# --------------------------------------------------------------------------
# rasterizacion
# --------------------------------------------------------------------------
def rasterizar(p, H, W, tile_size=16, k_sigma=3.5,
               alpha_max=0.99, t_min=1.0e-4, verbose=False) -> np.ndarray:
    """
    Replica raster_forward_tiled_kernel. Devuelve (H, W, 3) float32 sin
    recortar a [0,1]: el recorte ocurre solo al guardar el PNG.
    """
    mu, scale, theta = p["mu"], p["scale"], p["theta"]
    opacity, color, depth = p["opacity"], p["color"], p["depth"]
    conic = construir_conic(scale, theta)
    N = mu.shape[0]

    # --- caja envolvente por k_sigma (igual que el kernel de duplicacion) ---
    s, c = np.sin(theta), np.cos(theta)
    ext_f = k_sigma * np.sqrt((c * scale[:, 0]) ** 2 + (s * scale[:, 1]) ** 2)
    ext_c = k_sigma * np.sqrt((s * scale[:, 0]) ** 2 + (c * scale[:, 1]) ** 2)

    min_f = np.floor(mu[:, 0] - ext_f).astype(np.int64)
    max_f = np.ceil(mu[:, 0] + ext_f).astype(np.int64)
    min_c = np.floor(mu[:, 1] - ext_c).astype(np.int64)
    max_c = np.ceil(mu[:, 1] + ext_c).astype(np.int64)

    visible = ~((max_f < 0) | (min_f >= H) | (max_c < 0) | (min_c >= W))

    min_f = np.clip(min_f, 0, H - 1); max_f = np.clip(max_f, 0, H - 1)
    min_c = np.clip(min_c, 0, W - 1); max_c = np.clip(max_c, 0, W - 1)

    tiles_y = (H + tile_size - 1) // tile_size
    tiles_x = (W + tile_size - 1) // tile_size
    ty0 = np.clip(min_f // tile_size, 0, tiles_y - 1)
    ty1 = np.clip(max_f // tile_size, 0, tiles_y - 1)
    tx0 = np.clip(min_c // tile_size, 0, tiles_x - 1)
    tx1 = np.clip(max_c // tile_size, 0, tiles_x - 1)

    # --- profundidad cuantizada a 16 bits, igual que la clave de ordenamiento ---
    DEPTH_SCALE = 65536
    dmin, dmax = float(depth.min()), float(depth.max())
    dn = np.clip((depth - dmin) / (dmax - dmin + 1e-8), 0.0, 1.0)
    depth_q = np.clip((dn * (DEPTH_SCALE - 1)).astype(np.int64), 0, DEPTH_SCALE - 1)

    # --- duplicacion gaussiana -> tile, en el mismo orden que el kernel ---
    gids, keys = [], []
    for i in np.nonzero(visible)[0]:
        for ty in range(ty0[i], ty1[i] + 1):
            for tx in range(tx0[i], tx1[i] + 1):
                keys.append((ty * tiles_x + tx) * DEPTH_SCALE + depth_q[i])
                gids.append(i)
    if not gids:
        return np.zeros((H, W, 3), np.float32)
    keys = np.asarray(keys, dtype=np.int64)
    gids = np.asarray(gids, dtype=np.int64)

    # El radix sort de CUB es estable: los empates conservan el orden de
    # insercion, que es indice de gaussiana ascendente.
    orden = np.argsort(keys, kind="stable")
    keys, gids = keys[orden], gids[orden]
    tile_de_instancia = keys // DEPTH_SCALE
    if verbose:
        print(f"  {len(gids)} instancias gaussiana-tile "
              f"({len(gids) / max(1, int(visible.sum())):.1f} por gaussiana)")

    out = np.zeros((H, W, 3), np.float32)
    bordes = np.searchsorted(tile_de_instancia,
                             np.arange(tiles_y * tiles_x + 1), side="left")

    for tile_id in range(tiles_y * tiles_x):
        ini, fin = bordes[tile_id], bordes[tile_id + 1]
        if fin <= ini:
            continue
        idx = gids[ini:fin]

        ty, tx = divmod(tile_id, tiles_x)
        f0, f1 = ty * tile_size, min((ty + 1) * tile_size, H)
        c0, c1 = tx * tile_size, min((tx + 1) * tile_size, W)

        # centro de pixel: fila + 0.5, col + 0.5
        pr = np.arange(f0, f1, dtype=np.float32) + 0.5
        pc = np.arange(c0, c1, dtype=np.float32) + 0.5
        PR, PC = np.meshgrid(pr, pc, indexing="ij")
        PR = PR.reshape(-1); PC = PC.reshape(-1)              # (P,)

        acc = np.zeros((PR.size, 3), np.float32)
        T = np.ones(PR.size, np.float32)

        for j in idx:                       # de menor a mayor depth: front-to-back
            dx = PR - mu[j, 0]
            dy = PC - mu[j, 1]
            m00, m01, m11 = conic[j]
            quad = dx * (m00 * dx + m01 * dy) + dy * (m01 * dx + m11 * dy)
            alpha = opacity[j] * np.exp(-0.5 * quad)
            np.minimum(alpha, alpha_max, out=alpha)
            peso = alpha * T
            acc += peso[:, None] * color[j][None, :]
            T *= (1.0 - alpha)
            if not np.any(T >= t_min):      # equivalente al corte por T_MIN
                break

        out[f0:f1, c0:c1, :] = acc.reshape(f1 - f0, c1 - c0, 3)

    return out


def render_tau(cab, coefs, tau, **kw) -> np.ndarray:
    p = evaluar_atributos(cab, coefs, tau)
    r = cab["render"]
    return rasterizar(p, cab["alto"], cab["ancho"],
                      tile_size=kw.pop("tile_size", r["tile_size"]),
                      k_sigma=kw.pop("k_sigma", r["k_sigma"]),
                      alpha_max=r["alpha_max"],
                      t_min=r["transmitancia_min"], **kw)


def a_uint8(img: np.ndarray) -> np.ndarray:
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


def psnr(a_u8: np.ndarray, b_u8: np.ndarray) -> float:
    a = a_u8.astype(np.float64) / 255.0
    b = b_u8.astype(np.float64) / 255.0
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)


# --------------------------------------------------------------------------
# subcomandos
# --------------------------------------------------------------------------
def cmd_render(args):
    from PIL import Image
    cab, coefs = leer_modelo(args.modelo)
    tau = args.tau if args.tau is not None else tau_de_frame(args.frame, cab["n_frames"])
    print(f"N={cab['n_gaussianas']}  {cab['ancho']}x{cab['alto']}  tau={tau:.6f}")
    img = render_tau(cab, coefs, tau, verbose=True)
    Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a_uint8(img)).save(args.salida)
    print(f"  guardado {args.salida}")


def cmd_validar(args):
    from PIL import Image
    cab, coefs = leer_modelo(args.modelo)
    frames = [int(x) for x in args.frames.split(",")]
    print(f"modelo: N={cab['n_gaussianas']}  {cab['ancho']}x{cab['alto']}  "
          f"n_frames={cab['n_frames']}")
    print(f"referencia CUDA: {args.frames_cuda}\n")
    res = []
    for j in frames:
        ref_path = Path(args.frames_cuda) / f"frame_{j:04d}.png"
        if not ref_path.exists():
            print(f"  frame {j}: no existe {ref_path}"); continue
        ref = np.asarray(Image.open(ref_path).convert("RGB"))
        img = a_uint8(render_tau(cab, coefs, tau_de_frame(j, cab["n_frames"]),
                                 verbose=args.verbose))
        v = psnr(img, ref)
        res.append(v)
        print(f"  frame {j:4d}: PSNR referencia-CPU vs CUDA = {v:7.2f} dB")
        if args.guardar:
            Path(args.guardar).mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).save(Path(args.guardar) / f"cpu_{j:04d}.png")
    if res:
        m = float(np.mean(res))
        print(f"\n  media: {m:.2f} dB")
        print("  OK: la especificacion reproduce el rasterizador CUDA."
              if m >= 40 else
              "  ATENCION: por debajo de 40 dB, hay discrepancia en la especificacion.")


def cmd_vectores(args):
    from PIL import Image
    cab, coefs = leer_modelo(args.modelo)
    out = Path(args.salida); out.mkdir(parents=True, exist_ok=True)
    n_frames = cab["n_frames"]

    # Un tau por frame entero y uno estrictamente entre dos frames.
    casos = [
        ("frame0000", tau_de_frame(0, n_frames)),
        ("frame0240", tau_de_frame(240, n_frames)),
        ("entre_frames_240_241", tau_de_frame(240.5, n_frames)),
    ]
    indice = []
    for nombre, tau in casos:
        p = evaluar_atributos(cab, coefs, tau)
        n = min(20, cab["n_gaussianas"])
        datos = {
            "nombre": nombre,
            "tau": tau,
            "n_frames": n_frames,
            "nota": ("tau se obtiene con tau = 2*j/(n_frames-1) - 1; "
                     "j puede ser fraccionario"),
            "primeras_n_gaussianas": n,
            "atributos_activados": {
                "mu": p["mu"][:n].tolist(),
                "opacity": p["opacity"][:n].tolist(),
                "color": p["color"][:n].tolist(),
                "scale": p["scale"][:n].tolist(),
                "theta": p["theta"][:n].tolist(),
                "depth": p["depth"][:n].tolist(),
            },
        }
        (out / f"{nombre}.json").write_text(
            json.dumps(datos, indent=2, sort_keys=True), encoding="utf-8")
        img = a_uint8(render_tau(cab, coefs, tau, verbose=True))
        Image.fromarray(img).save(out / f"{nombre}.png")
        indice.append({"nombre": nombre, "tau": tau,
                       "json": f"{nombre}.json", "png": f"{nombre}.png"})
        print(f"  {nombre}: tau={tau:.6f}  -> {nombre}.json + {nombre}.png")

    (out / "index.json").write_text(
        json.dumps({"modelo": Path(args.modelo).name,
                    "n_gaussianas": cab["n_gaussianas"],
                    "ancho": cab["ancho"], "alto": cab["alto"],
                    "casos": indice}, indent=2, sort_keys=True),
        encoding="utf-8")
    print(f"  index.json escrito en {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render"); r.set_defaults(fn=cmd_render)
    r.add_argument("--modelo", required=True)
    r.add_argument("--frame", type=float, default=0)
    r.add_argument("--tau", type=float, default=None)
    r.add_argument("--salida", required=True)

    v = sub.add_parser("validar"); v.set_defaults(fn=cmd_validar)
    v.add_argument("--modelo", required=True)
    v.add_argument("--frames_cuda", required=True)
    v.add_argument("--frames", default="0,240,479")
    v.add_argument("--guardar", default=None)
    v.add_argument("--verbose", action="store_true")

    t = sub.add_parser("vectores"); t.set_defaults(fn=cmd_vectores)
    t.add_argument("--modelo", required=True)
    t.add_argument("--salida", default="demo_mode/referencia")

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
