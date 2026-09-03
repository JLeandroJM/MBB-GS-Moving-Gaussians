# Figures for the README

The main README has two figure slots, both currently commented out. Drop the
file in this folder, delete the surrounding comment markers in `README.md`, and
it appears.

Keep files small: GitHub renders them inline on every page load, so a teaser
above about 10 MB is a bad trade. Every command below is run from the repository
root.

---

## 1. `teaser.gif` — what the method does

The slot at the top of the README. It should answer "does this actually work"
in three seconds, before anyone reads a word.

**What to show.** A short loop of `original | reconstruction | amplified
difference`, side by side. The reconstruction script already emits exactly this
triptych:

```bash
python scripts/reconstruction/regenerar_clip_desde_checkpoint_streaming.py \
    --checkpoint outputs/<exp>/checkpoints/checkpoint_final.pt \
    --salida outputs/<exp>/teaser_frames \
    --inicio 0 --fin 90 \
    --comparaciones --factor_diff 5.0 \
    --device cuda
```

The triptychs are written next to the output folder, as
`comparaciones_streaming/comparacion_0000.png`. Turn them into a GIF with two
passes, using a palette to keep it sharp and small:

```bash
COMP=outputs/<exp>/comparaciones_streaming

ffmpeg -framerate 15 -i $COMP/comparacion_%04d.png \
    -vf "scale=1200:-1:flags=lanczos,palettegen" -y /tmp/pal.png

ffmpeg -framerate 15 -i $COMP/comparacion_%04d.png -i /tmp/pal.png \
    -lavfi "scale=1200:-1:flags=lanczos[x];[x][1:v]paletteuse" \
    -loop 0 -y assets/teaser.gif
```

Aim for 60–90 frames at 12–15 fps and about 1200 px wide. If the result is over
10 MB, drop the width to 900 or the frame count to 45.

**A stronger alternative.** Since temporal interpolation is the headline claim,
a slow-motion loop makes it better than a plain reconstruction: render the same
model at 4× the source frame rate with
`scripts/reconstruction/regenerar_fps_interpolado.py`, and caption it *"the
same model, evaluated between the recorded frames"*.

---

## 2. `architecture.png` — how the method works

The slot in the Method section. This is a diagram, not a screenshot: it has to
be drawn. Use Inkscape, draw.io, Figma or TikZ, export at 2× for retina, and
keep it under about 1 MB.

**What it has to convey.** The whole point of MBB-GS is that time enters *before*
rendering — the model is a set of coefficients, and a frame is what you get when
you evaluate them. A reader who understands only that has understood the paper.
So the diagram should read left to right as one row:

```
   coefficients          evaluate at t          Gaussians at t         frame
  ┌──────────────┐      ┌─────────────┐       ┌──────────────┐     ┌─────────┐
  │  a[i,0..K]   │─────▶│   τ(t) →    │──────▶│  μ, α, c,    │────▶│  Î(t)   │
  │  per Gaussian│      │   T_k(τ)    │       │  s, θ, d     │     │         │
  │  per attribute│     │  basis matrix│      │  (N of them) │     │         │
  └──────────────┘      └─────────────┘       └──────────────┘     └─────────┘
     stored              evaluated             activated            CUDA
     N × Σ(K_p+1)        per frame             sigmoid / exp        rasteriser
```

Four things worth getting right, because they are what distinguishes this from
generic Gaussian Splatting diagrams:

1. **The box on the left is the entire model.** Label it with the real size —
   414 coefficients per Gaussian for the production degrees, about 1.6 KB. That
   number is what makes "stores coefficients, not frames" concrete.
2. **Per-attribute degrees.** Draw the six attributes as rows of different
   lengths: `mu` long (100), `opacity` slightly shorter (80), down to `depth`
   very short (4). The unequal lengths are the design decision, visible at a
   glance.
3. **A gradient arrow returning underneath**, from the frame back to the
   coefficients, dashed. It shows that the rasteriser is differentiable and that
   training updates the polynomial coefficients directly — there is no network
   in the loop.
4. **A second, dotted arrow out of the τ box** pointing at a non-integer time,
   labelled *"any t, including t ∉ training frames"*. That is temporal
   interpolation, and it costs one arrow to explain.

If you want a fifth element, a small inset plot of one attribute's Chebyshev
curve `p_i(t)` against the frame index, with the training frames marked as dots
on the curve, makes the "continuous function sampled at frames" idea land
immediately. `scripts/visualization/viz_atributos_gaussiana_tiempo.py` produces
exactly that curve from a real checkpoint, so the inset can be real data rather
than a drawing:

```bash
python scripts/visualization/viz_atributos_gaussiana_tiempo.py \
    --checkpoint outputs/<exp>/checkpoints/checkpoint_final.pt \
    --salida assets/attr_curve --id 1234 \
    --inicio 0 --fin 600 --device cpu
```

---

## Other figures worth having

Not wired into the README, but strong material for the paper. All come from
`scripts/visualization/`:

| Figure | Command | Why it is good |
| --- | --- | --- |
| Gaussian trajectories over the frame | `viz_trayectorias_marcadores.py --checkpoint <ckpt> --modo top --n 12 --gif` | shows that individual primitives track content, not just that the sum looks right |
| Ellipses and velocities | `viz_elipses_velocidades.py --checkpoint <ckpt> --frame 120 --k_sigma 2.0` | makes the anisotropic 2D Gaussians and their motion visible |
| Training filmstrip | `viz_tira_evolucion.py --exp outputs/<exp> --gif` | how the reconstruction sharpens over epochs |
| Pruning | `viz_prune_checkpoint_by_stats.py` (in `scripts/compression/`) | which Gaussians the ranking removes and what survives |

For audio, `scripts/audio/visualizar.py --exp outputs/gabor/<exp>` produces
waveform, spectrogram and difference-spectrogram comparisons; it detects mono
versus stereo automatically.
