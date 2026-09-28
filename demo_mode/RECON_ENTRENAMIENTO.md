# Reconocimiento del entrenamiento

Documento de solo lectura: describe cómo funciona el entrenamiento en este
repositorio, sin haber ejecutado nada. Escrito para poder planificar el
entrenamiento de los 5 o 6 clips cortos de la demo.

Todas las rutas son relativas a la raíz del repositorio.

---

## 1. Cómo se lanza un entrenamiento de principio a fin

### 1.1 Instalar el entorno

```bash
conda create -n mbb-gs python=3.10 -y && conda activate mbb-gs
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install -e .                       # obligatorio: los scripts importan gs2d_video
cd cuda/raster_cuda && python setup.py build_ext --inplace && cd ../..
```

El rasterizador CUDA es obligatorio para entrenar: `render_frame` lanza
`RuntimeError` si `usar_cuda_tiled` es falso, y no hay camino CPU.

### 1.2 Preparar el clip

El entrenador espera una carpeta de PNG numerados en `data/clips/<nombre>/`:

```
data/clips/mi_clip/frame_0000.png, frame_0001.png, ...
```

Dos formas de generarla:

**a) Antes de entrenar**, con el script dedicado, que es el que conserva la
proporción original:

```bash
python scripts/data/extraer_clips_720p.py \
    --video data/videos/mi_video.mp4 \
    --nombre_clip mi_clip \
    --inicio_seg 12 --duracion_seg 5 \
    --H 360 --W 540 \
    --fps 24
```

Flags reales de ese script: `--video`, `--nombre_clip`, `--inicio_seg`,
`--duracion_seg`, `--max_frames`, `--fps`, `--H`, `--W`, `--forzar`.

**b) Durante el entrenamiento**, poniendo en el config `video_mp4`,
`fps_extraccion`, `n_frames_extraer` y `resolucion_extraccion`.
`src/gs2d_video/io/video.py` extrae con FFmpeg y cachea. **Cuidado:** ese camino
aplica un recorte cuadrado centrado al lado corto antes de escalar. Para
conservar la proporción del video conviene usar (a).

### 1.3 Entrenar

```bash
python scripts/train.py --config configs/video/final/ganador_motion_200k_1200ep.json
python scripts/train.py --config mi_config.json --nombre-experimento demo_clip1
```

`scripts/train.py` acepta exactamente dos flags: `--config` (obligatorio) y
`--nombre-experimento` (opcional, sobrescribe el del JSON).

### 1.4 Qué genera

```
outputs/<nombre_experimento>/
├── frames_renderizados/     PNG reconstruidos, post-pruning
├── checkpoints/             checkpoint_final.pt, modelo_pruneado.pt
├── logs/                    log_entrenamiento.csv, curvas, visualizaciones
├── metricas.json            agregados y por frame, separados pre/post pruning
├── metricas_por_frame.csv   una fila por frame
├── config_usada.json        copia literal del config que corrió
└── info_clip.json           clip, n_frames, H, W, fps, seed
```

`checkpoint_final.pt` se escribe **inmediatamente** después de entrenar, antes de
calcular ninguna métrica, así que un fallo en la evaluación no cuesta el modelo.
Por defecto el run se niega a arrancar si la carpeta ya existe; `sobreescribir_salida`
lo permite.

El checkpoint que consume el exportador de la demo es
`checkpoints/checkpoint_final.pt`.

---

## 2. Parámetros de configuración

Un experimento es un único JSON. Los que importan para la demo:

### Clip y ejecución

| Clave | Qué hace |
|---|---|
| `nombre_experimento` | nombre de la carpeta de salida |
| `clip` | carpeta bajo `data/clips/` |
| `max_frames` | trunca el clip |
| `device` | `cuda`; entrenar exige CUDA |
| `seed` | semilla de torch y del generador del modelo |
| `sobreescribir_salida` | reutilizar carpeta existente |

### Modelo

| Clave | Qué hace |
|---|---|
| `n_gaussianas_inicial` | N |
| `base_temporal` | `chebyshev` (principal) o `monomial` (ablación) |
| `grados` | grado polinómico por atributo |
| `escala_inicial_px` | escala inicial; su log es el `a_0` de `scale` |
| `inicializar_color_desde_frame0` | muestrea colores del frame 0; acelera mucho |

### Optimización

`n_epochs`, `lrs` (doce grupos: seis atributos × {`a0`, `high`}),
`sub_batch_frames`, `frames_por_epoch`, `beta_smoothness`, `pesos_smoothness`,
y la familia `scheduler_lento_*`.

### Pérdida

`tipo_loss` entre `baseline`, `l1_mse`, `motion`, `hard`, `edge`, `temporal`,
`motion_temporal`, `combo`. La receta validada es `motion` con
`lambda_motion: 2.0`, `lambda_dssim: 0.25` y `exponente_frame: 1`.

### Memoria

```json
"frames_en_cpu": true,
"frames_en_gpu_uint8": false,
"evitar_render_completo_en_train": true,
"usar_metricas_streaming": true
```

### 2.1 Qué acepta un clip de 3 a 8 segundos sin tocar código

**Nada del pipeline impide un clip corto.** `max_frames`, `clip`, la resolución
y `n_gaussianas_inicial` son datos de configuración. Lo único a tener en cuenta:

- **`n_frames` debe ser ≥ 2**, porque `τ(j) = 2j/(n_frames-1) - 1` divide entre
  `n_frames - 1`. Un clip de 3 s a 24 fps son 72 frames: de sobra.
- **Los grados deben ser bastante menores que `n_frames`.** Con 72 frames, un
  grado 100 en `mu` tiene más coeficientes que observaciones y el ajuste queda
  indeterminado. Regla práctica sensata: grado ≤ n_frames/2. Para 72 frames,
  `mu` entre 24 y 36.

  > Esto es razonamiento sobre la parametrización, no un resultado medido: no
  > hay en el repositorio ningún experimento que barra grado contra número de
  > frames.

- **`escala_inicial_px` es absoluta, en píxeles.** A 540×360 en lugar de
  1080×720 conviene reducirla proporcionalmente; el valor usado es 2.5 px.

Config de partida sugerido para un clip de 3 s a 24 fps y 540×360:

```json
{
  "nombre_experimento": "demo_clip1",
  "clip": "demo_clip1",
  "max_frames": 72,
  "device": "cuda",
  "seed": 42,
  "n_gaussianas_inicial": 40000,
  "inicializar_color_desde_frame0": true,
  "escala_inicial_px": 1.5,
  "grados": {"mu": 30, "opacity": 24, "color": 12, "scale": 8, "theta": 4, "depth": 3},
  "n_epochs": 600,
  "sub_batch_frames": 1,
  "tipo_loss": "motion",
  "lambda_motion": 2.0,
  "lambda_dssim": 0.25,
  "exponente_frame": 1.0,
  "usar_cuda_tiled": true,
  "cuda_tile_size": 16,
  "cuda_k_sigma": 3.5,
  "frames_en_cpu": true,
  "evitar_render_completo_en_train": true,
  "usar_metricas_streaming": true,
  "guardar_frames_rasterizados": true
}
```

Los `lrs` se pueden omitir: cada clave ausente cae a un valor por defecto. Copiar
el bloque completo de `configs/video/final/ganador_motion_200k_1200ep.json` es
lo más seguro, porque son los que produjeron los resultados reportados.

---

## 3. Cómo se lanzaban los trabajos en Khipu

`jobs/video/*.sbatch` y `jobs/audio/*.sbatch`, uno por experimento reportado.
Estructura, abreviada de `jobs/video/fase1_motion.sbatch`:

```bash
#SBATCH --job-name=fase1_motion
#SBATCH --partition=gpu
#SBATCH --gres=shard:6
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs_slurm/%x_%j.out

cd $HOME/MBB-GS
module load cuda/12.8
module load gnu12/12.4.0
module load miniconda/3.0
eval "$(conda shell.bash hook)"
conda activate mbb-gs

python scripts/train.py \
    --config configs/video/exp3_loss/fase1_motion.json \
    --nombre-experimento fase1_motion
```

Los recursos reales, leídos de los 24 archivos: todos los jobs de video piden
`shard:6` con 32–64 GB de RAM y entre 10 y 24 horas; el de 200k gaussianas pide
`shard:8` y 36 h. Los de audio van con 24–32 GB y 2 a 12 h. Todos en la
partición `gpu` salvo `motion_150k_0800ep`, que usó `data-science`.

`jobs/audio/gabor_mono.sbatch` y `gabor_stereo.sbatch` son genéricos: reciben el
config y el nombre como argumentos.

Antes de reutilizarlos hay que ajustar `cd $HOME/MBB-GS`, los `module load`, el
nombre del entorno conda y el `--mail-user`, y crear `logs_slurm/`.

**El repositorio no registra memoria de GPU.** Ni el log de entrenamiento ni las
métricas ni los `.out` de Slurm guardan `max_memory_allocated`. Las peticiones de
`--mem` son RAM de sistema, no VRAM. Para estimar VRAM habría que instrumentar,
o mirar el hecho de que `shard:6` bastó para 150k gaussianas a 720p.

---

## 4. Tiempos medidos y estimación para clips cortos

### 4.1 Lo medido

De `outputs_khipu/musical_20s_1600ep/logs/log_entrenamiento.csv`, 1600 filas:

| Dato | Valor |
|---|---|
| Épocas | 1600 |
| Tiempo por época | media **6.683 s**, mediana 6.635 s, rango 6.535–7.164 s |
| Tiempo total | 10 693 s = **2.97 h** |
| Configuración | 150 000 gaussianas, 480 frames, 720×1080, `motion`, `sub_batch_frames: 1` |
| VRAM | **no registrada** |

El tiempo por época es notablemente estable, lo que tiene sentido: cada época
recorre los 480 frames y hace el mismo trabajo.

### 4.2 Modelo de costo

Cada época renderiza los `n_frames` del clip y retropropaga. El costo del
rasterizador teselado crece con:

- el **número de gaussianas** N, aproximadamente lineal, porque cada una se
  duplica a los tiles que toca y se evalúa ahí;
- el **número de píxeles** H·W, aproximadamente lineal, porque el trabajo por
  tile es proporcional a sus píxeles por el número de gaussianas de su lista;
- el **número de frames**, lineal.

Con lo medido:

```
t_epoca ≈ 6.683 s  para  N=150k, píxeles=720·1080=777 600, frames=480

costo unitario = 6.683 / (150 000 · 777 600 · 480)
               = 1.193e-13 s por (gaussiana · píxel · frame)
```

### 4.3 Estimaciones

**Clip de 3 s a 24 fps = 72 frames, 40 000 gaussianas, 600 épocas.**

A 360×240 (86 400 px):

```
t_epoca = 1.193e-13 · 40 000 · 86 400 · 72 = 0.0297 s
total   = 0.0297 · 600 = 17.8 s
```

A 540×360 (194 400 px):

```
t_epoca = 1.193e-13 · 40 000 · 194 400 · 72 = 0.0668 s
total   = 0.0668 · 600 = 40.1 s
```

| Resolución | Píxeles | t/época | 600 épocas | 1500 épocas |
|---|---:|---:|---:|---:|
| 360 × 240 | 86 400 | 0.030 s | **18 s** | 45 s |
| 540 × 360 | 194 400 | 0.067 s | **40 s** | 100 s |
| 720 × 480 | 345 600 | 0.119 s | 71 s | 178 s |

**Estas cifras son optimistas y hay que tomarlas como cota inferior.** El modelo
lineal ignora:

- el **costo fijo por época** — lanzamiento de kernels, paso del optimizador
  sobre los doce grupos, sincronizaciones. A 0.03 s por época ese costo fijo
  probablemente domina, así que lo realista es del orden de **1 a 3 minutos**
  para 600 épocas, no 18 segundos;
- la **carga y transferencia de frames**, que con `frames_en_cpu: true` mueve un
  frame a GPU por iteración;
- las **métricas finales** (SSIM y sobre todo LPIPS) y el guardado de los PNG,
  que en clips cortos pueden pesar más que el entrenamiento.

La conclusión operativa aguanta igual: **un clip de 3 s a 540×360 con 40k
gaussianas se entrena en minutos, no en horas**. Entrenar los 5 o 6 clips de la
demo es cuestión de una tarde con una GPU, no de una reserva de cluster.

Conviene medir una corrida real antes de comprometer un calendario.

---

## 5. Invocar el entrenamiento desde Python, sin tocar el código existente

### 5.1 Lo que hay hoy

`scripts/train.py` tiene toda su lógica dentro de `main()`, que empieza por
`argparse.ArgumentParser()` y termina escribiendo a disco. No existe una función
del tipo `entrenar(config: dict) -> resultados`. La lógica reutilizable sí está
separada en el paquete:

| Pieza | Dónde |
|---|---|
| Carga de frames | `gs2d_video.io.frames` |
| Extracción desde MP4 | `gs2d_video.io.video.extraer_frames_de_video` |
| Matrices de base | `gs2d_video.core.bases.construir_matriz_chebyshev` |
| Modelo | `gs2d_video.core.modelo.GaussianasPolinomial2D` |
| Optimizador | `gs2d_video.core.optimizador.construir_optimizador` |
| **Bucle de entrenamiento** | `gs2d_video.training.trainer.entrenar_batch_full(modelo, frames, matrices_base, optimizer, config, carpeta_salida=None)` |
| Pruning | `gs2d_video.core.pruning_post.prunear_post` |
| Métricas | `gs2d_video.metrics.calidad.reporte_completo` |

`entrenar_batch_full` ya es una función normal que recibe un `config` como dict.
**El bucle de entrenamiento es invocable hoy mismo**; lo que no es reutilizable
es el orquestado de `main()`: resolución de rutas, extracción, guardado y
métricas.

### 5.2 Tres opciones, de menos a más intrusiva

**Opción 1 — subprocess. Cero cambios.**

```python
import json, subprocess, tempfile, pathlib

def entrenar(config: dict, raiz="."):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        ruta = f.name
    subprocess.run(["python", "scripts/train.py", "--config", ruta],
                   cwd=raiz, check=True)
    return pathlib.Path(raiz) / "outputs" / config["nombre_experimento"]
```

Es lo que ya hace `scripts/pipeline/run_pipeline_video_audio.py`, así que hay
precedente en el propio repositorio. Aísla fallos y libera la VRAM al terminar.
Lo malo: no devuelve objetos Python y no se puede intervenir a mitad.

**Opción 2 — reconstruir el orquestado en un script nuevo.**

Un archivo nuevo en `demo_mode/` que importe las piezas del paquete y repita la
secuencia de `main()`: cargar frames, construir matrices, crear modelo y
optimizador, llamar a `entrenar_batch_full`, guardar checkpoint. No modifica
nada, pero **duplica** la lógica de orquestado, con el riesgo de divergir de
`train.py` si este cambia.

**Opción 3 — refactor mínimo de `scripts/train.py`.**

Extraer el cuerpo de `main()` a `def entrenar_desde_config(config: dict, raiz:
str) -> dict` y dejar `main()` como un envoltorio que parsea argumentos y llama a
esa función. Son unas pocas líneas y no cambia ningún comportamiento: mismo
config, mismas salidas, mismo CLI. Es la solución correcta a medio plazo, pero
**modifica un archivo existente**, que esta tarea excluye.

**Recomendación para la demo:** opción 1. Entrenar 5 o 6 clips es un bucle de
subprocess y no justifica tocar el código. Si el visor llega a necesitar
entrenar bajo demanda, entonces sí vale la opción 3.

### 5.3 Cabos sueltos

- `scripts/train.py` resuelve rutas relativas a la raíz del repositorio,
  calculada desde `__file__`. Invocarlo desde otro directorio funciona, pero
  `clip` y `video_mp4` se interpretan respecto a esa raíz, no al directorio de
  trabajo.
- No hay forma documentada de cancelar o reanudar a mitad de una época.
  `checkpoint_inicial` y `cargar_optimizer_state` permiten reanudar desde un
  checkpoint guardado, y `checkpoint_cada_n_epochs` controla cada cuánto se
  guarda.
