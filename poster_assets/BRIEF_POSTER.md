# Brief para el póster — MBB-GS: Moving Gaussians

Demo Mode UTEC · 96 × 70 cm horizontal · público general + jurado técnico

**Regla de este documento:** cada cifra lleva entre corchetes el archivo del que
sale. Lo que no existe dice **PENDIENTE**. Nada está estimado ni redondeado a
ojo.

Generado el 2026-09-27. Todo lo de `poster_assets/` se creó nuevo; no se
reentrenó ni se modificó ningún experimento.

---

## 1. El método en tres niveles

**Una frase (público general)**
> En vez de guardar un video como miles de fotos, lo guardamos como un conjunto
> de manchas de color que se mueven y cambian siguiendo fórmulas — así el video
> deja de tener "fotogramas" y pasa a existir en cualquier instante.

**Un párrafo (visitante curioso)**
> Un video normal es una pila de imágenes independientes: la número 100 no sabe
> nada de la 101. Nosotros representamos todo el clip con una población fija de
> 150 000 manchas gaussianas 2D. Cada mancha tiene posición, color, tamaño,
> giro y transparencia, y cada una de esas propiedades no es un número fijo sino
> una curva suave en el tiempo. El archivo guarda los coeficientes de esas
> curvas, no las imágenes. Para ver el instante que sea, se evalúan las curvas
> en ese instante y se dibujan las manchas resultantes. Como las curvas son
> continuas, se puede pedir un instante que nunca se grabó y el modelo responde.

**Un párrafo (jurado técnico)**
> Representamos un clip de T frames con N gaussianas 2D anisótropas de población
> constante. Cada atributo p de cada gaussiana i — posición μ ∈ R², opacidad,
> color RGB, escala ∈ R², rotación θ y profundidad — se parametriza como un
> polinomio de Chebyshev de primer tipo en tiempo normalizado τ(t) ∈ [−1, 1]:
> p_i(t) = a_{i,0} + Σ_{k=1..K_p} a_{i,k} T_k(τ(t)). Cada atributo lleva su
> propio grado K_p. El modelo almacena únicamente los coeficientes a_{i,k}. El
> render de un frame evalúa los polinomios en τ(t) y pasa las gaussianas
> resultantes por un rasterizador CUDA diferenciable con teselado, ordenamiento
> por profundidad y composición alpha. El entrenamiento retropropaga el error de
> reconstrucción a través del rasterizador hasta los coeficientes; no hay red
> neuronal en el lazo. Se usa Chebyshev y no base monomial porque a grado ~100
> la matriz de Vandermonde sobre nodos equiespaciados está mal condicionada.

---

## 2. Experimento usado

**`musical_20s_1600ep`** — es el de mayor calidad entre los registros
disponibles: 42.27 dB frente a 35.36 dB del siguiente (`motion_150k_1600ep`)
[`results/video/*/metricas.json` del repo oficial]. Además es el único de esa
calidad que conserva `checkpoint_final.pt`, los 479 frames renderizados y el
clip original, que es lo que hace falta para generar figuras.

> No existe ningún experimento llamado `exp09` en el repositorio; se revisaron
> los 34 directorios de `outputs_khipu/` y los 37 registros de `results/`.

| Dato | Valor | Fuente |
|---|---|---|
| Clip | `musical_20s` | `config_usada.json` |
| Resolución | 720 × 1080 (alto × ancho) | `info_clip.json` |
| Frames | 480 (479 renderizados en disco) | `info_clip.json`, `ls frames_renderizados` |
| N gaussianas inicial | 150 000 | `config_usada.json: n_gaussianas_inicial` |
| N tras pruning | **150 000 — el pruning no eliminó ninguna** | `modelo_pruneado.pt: state_dict_coefs.N` |
| Umbral de pruning | 0.05 | `config_usada.json: umbral_pruning_post` |
| Grados | μ 100 · opacidad 80 · color 30 · escala 12 · θ 6 · profundidad 4 | `config_usada.json: grados` |
| Base temporal | Chebyshev (el config trae `base_temporal: null`, que el cargador interpreta como Chebyshev) | `config_usada.json`, `src/gs2d_video/io/checkpoints.py` |
| Pérdida | `motion`, λ_motion 2.0, λ_dssim 0.25 | `config_usada.json` |
| Épocas | 1600 | `config_usada.json: n_epochs` |
| Semilla | 42 | `info_clip.json` |

---

## 3. Resultados de video

| Métrica | Valor | Fuente |
|---|---|---|
| PSNR medio | **42.27 dB** | `metricas.json: post_pruning.agregados.psnr_promedio` = 42.2650 |
| PSNR mínimo | 38.72 dB | ídem `psnr_min` = 38.7171 |
| PSNR percentil 5 | 40.91 dB | ídem `psnr_p5` = 40.9061 |
| SSIM | **0.9838** | ídem `ssim_promedio` |
| LPIPS | **0.0175** | ídem `lpips_promedio` |
| PSNR temporal | 48.14 dB | ídem `psnr_temporal_promedio` = 48.1438 |
| Tiempo de entrenamiento | **2.97 h** (10 692.9 s) | suma de `tiempo_seg` sobre las 1600 filas de `logs/log_entrenamiento.csv` |
| Tiempo de render por frame | **PENDIENTE** | no se registra en ningún log del experimento |
| Tamaño del modelo | **248 407 393 B = 248.4 MB** | `os.path.getsize(checkpoints/checkpoint_final.pt)` |

El bloque `pre_pruning` de `metricas.json` viene en `null`: el run usó métricas
en streaming, que deja ese reporte vacío a propósito. Los números válidos son
los de `post_pruning`.

### Tamaño frente a baselines

Los baselines se generaron el 2026-09-27 sobre **los mismos 480 frames PNG** del
clip. Comandos exactos, reproducibles:

```bash
# AV1 como video
ffmpeg -framerate 24 -i data/clips/musical_20s/frame_%04d.png \
       -c:v libsvtav1 -crf {30,45} -preset 8 -pix_fmt yuv420p av1_crf{30,45}.mp4

# AVIF imagen por imagen (480 archivos independientes)
ffmpeg -i frame_XXXX.png -c:v libsvtav1 -crf {30,45} -preset 8 \
       -still-picture 1 -pix_fmt yuv420p frame_XXXX.avif
```

| Representación | Bytes | MB | PSNR | Espacio |
|---|---:|---:|---:|---|
| PNG sin pérdida, 480 frames | 551 290 835 | 551.3 | — (sin pérdida) | — |
| **MBB-GS (nuestro modelo)** | **248 407 393** | **248.4** | **42.27 dB** | RGB |
| AVIF frame a frame CRF 30 | 22 134 784 | 22.1 | 36.79 dB | RGB, muestra de 40 frames |
| MP4 fuente (H.264) | 21 854 213 | 21.9 | — | — |
| AVIF frame a frame CRF 45 | 9 682 944 | 9.7 | 32.66 dB | RGB, muestra de 40 frames |
| AV1 video CRF 30 | 2 806 111 | 2.8 | 42.62 dB | canal Y (ffmpeg) |
| AV1 video CRF 45 | 1 015 970 | 1.0 | 39.60 dB | canal Y (ffmpeg) |

> **Salvedad metodológica que hay que decir si preguntan:** el PSNR de ffmpeg es
> sobre canales YUV y el nuestro sobre RGB. No son directamente comparables. Aun
> tomando el margen más favorable, AV1 CRF 30 alcanza calidad equivalente en
> **2.8 MB contra nuestros 248.4 MB: unas 89 veces menos**.

---

## 4. Resultados de audio (Gabor splatting)

Experimento **`gabor_rock_8s_mono_ganador_N48k`** — el mejor mono con WAV
reconstruido disponible en disco.

| Dato | Valor | Fuente |
|---|---|---|
| Audio | 8.0 s, mono, 44 100 Hz, 352 800 muestras | `metricas.json`, `info_audio.json` |
| Átomos de Gabor | 48 000 | `metricas.json: n_atomos` |
| SNR | **20.72 dB** | `metricas.json: snr_db` = 20.7171 |
| SI-SDR | 20.68 dB | ídem `si_sdr_db` = 20.6802 |
| PSNR | 37.30 dB | ídem `psnr_db` |
| LSD | 5.28 dB | ídem `lsd_db` |
| Mel-L1 | 0.2976 | ídem `mel_l1` |
| MR-STFT | 1.7345 | ídem `mr_stft` |
| Tamaño del modelo | 960 004 B (0.96 MB) | ídem `bytes_modelo` |
| WAV int16 de referencia | 705 600 B (0.71 MB) | ídem `bytes_wav_int16` |
| Razón vs WAV | **0.735 — el modelo pesa MÁS que el WAV crudo** | ídem `ratio_compresion_vs_wav` |
| Tiempo de entrenamiento | 114.4 s | ídem `tiempo_total_s` |

> El mejor resultado de audio de la tesis (estéreo, 9 s, 160k átomos, 27.78 dB
> SNR) **no tiene registro en este repositorio ni WAV en disco**: sus salidas
> están solo en el Drive. Si el póster necesita esa cifra, hay que sacarla del
> informe y marcarla como tal. Aquí se usó el mejor experimento verificable.

---

## 5. Los beneficios más defendibles

Ordenados por cuán sólido es el respaldo.

**1. El tamaño del modelo no depende de la duración del clip.** *(El más fuerte,
y verificable en el sitio.)*
El modelo son N × Σ_p dim(p)·(K_p+1) coeficientes. Con los grados usados son 414
valores float32 por gaussiana = 1656 B; por 150 000 gaussianas, 248.4 MB —
exactamente el tamaño medido del checkpoint. Ese número **no contiene T**.
Duplicar los frames del clip no cambia el tamaño del archivo, mientras que en
PNG, AVIF o cualquier códec de video el tamaño crece con la duración.
*Cómo mostrarlo:* una línea plana (nosotros) contra una recta creciente (los
demás) al variar la duración.
*Límite honesto:* a duración fija de grados, un clip más largo tiene que
repartir la misma capacidad temporal entre más contenido, así que la calidad
baja. No está medido en qué punto deja de ser aceptable.

**2. Frames intermedios sin entrenar nada más.**
Los atributos son funciones continuas de t, así que se evalúan en instantes que
nunca se grabaron: no hay segunda red, ni flujo óptico, ni pasada extra de
entrenamiento. Es la misma operación del render normal con otro τ.
*Respaldo numérico:* 40.97 dB PSNR / 0.985 SSIM en el experimento de
interpolación de la tesis (90k gaussianas, 180 frames, 2200 épocas). **Ese run
no tiene registro en este repositorio** — la cifra viene del informe y hay que
citarla como tal.

**3. La representación es explícita e inspeccionable.**
Cada gaussiana es su propio objeto con sus coeficientes, así que después de
entrenar se puede preguntar cuál aporta, cuál se mueve y cuál es transitoria —
y podarlas o cuantizarlas sin reentrenar.
*Respaldo:* poda adaptativa al 20 % + cuantización UINT16 llevan un modelo de
~236 MB a ~85 MB con ~63 dB contra la reconstrucción completa (experimento de
150k gaussianas / 750 frames del informe; **sin registro en este repositorio**).
*Cuidado:* en el experimento que se usa para el póster la poda **no eliminó
ninguna gaussiana** con umbral 0.05. No presentar la poda como algo que ocurre
siempre.

**4. Un mismo marco para video y audio.**
La misma idea — primitivas explícitas en vez de muestras en una grilla — cubre
las dos modalidades, cambiando la primitiva: gaussianas con polinomios
temporales para imagen, átomos de Gabor sobre la forma de onda para sonido.
*Respaldo:* las métricas de la sección 4.
*Matiz técnico si preguntan:* el audio **no** usa polinomios temporales. Cada
átomo es estático y la señal es la suma; la oscilación ya está dentro de la
primitiva.

---

## 6. Limitaciones honestas — lo que puede preguntar el jurado

**"¿Comprime mejor que un códec?"** No. 248.4 MB contra 2.8 MB de AV1 a calidad
comparable: 89× peor. Esto es una **representación**, no un compresor. La
compresión de los coeficientes polinómicos es trabajo futuro; lo que hay hoy es
cuantización básica.

**"El modelo de audio pesa más que el WAV."** Correcto: 960 004 B contra 705 600 B,
razón 0.735. Con 48 000 átomos × 5 parámetros float32 no se baja del audio
crudo a esta duración.

**"¿Las gaussianas siguen los objetos?"** No, y conviene adelantarlo. Medido
sobre este checkpoint: el desplazamiento neto típico es de **7.0 px** (mediana),
39.4 px en el percentil 99 y 122.1 px como máximo, sobre un cuadro de
720 × 1080. En cambio el recorrido total acumulado llega a 692 px (p99). Es
decir, **oscilan dentro de su vecindad en vez de recorrer la escena**: son una
base local que codifica cómo cambia la apariencia de su zona, no un rastreador
de objetos. Se verificó el mismo patrón en otros tres experimentos.
*(Medición hecha hoy con `poster_assets/scripts/02_trayectorias.py`.)*

**"¿La interpolación es mejor que un método dedicado?"** No está comparada contra
ningún método de interpolación de frames. Y por construcción un polinomio es
suave entre los frames observados, lo cual es correcto para movimiento continuo
y **falso para movimiento rápido o discontinuo**: ahí los frames intermedios
muestran trayectorias que no ocurrieron.

**"¿Cuánto cuesta entrenar?"** 2.97 h de GPU para 20 s de video. El modelo es
**por clip**: no generaliza, no hay pesos reutilizables. Cada video nuevo se
entrena desde cero.

**"¿La poda siempre ayuda?"** En este experimento eliminó 0 gaussianas.

**"¿Por qué PSNR de espacios de color distintos?"** Está dicho en la tabla; es una
limitación de la medición hecha con ffmpeg, no del método.

---

## 7. Imágenes generadas

Todas en `poster_assets/img/`, PNG, sin títulos ni texto incrustado.

| Archivo | Resolución | Qué muestra |
|---|---|---|
| `02_trayectorias_zoom.png` | 2425 × 1617 | **Recomendada.** Recorte ampliado (y 181–481, x 630–1080) con 14 trayectorias μ(t) sobre el frame 0. Color = avance del tiempo, punto = instante inicial. Se ve que cada gaussiana describe un bucle cerrado en su vecindad. |
| `02_trayectorias_full.png` | 2827 × 1884 | El cuadro completo con 40 trayectorias, para contexto. A tamaño de póster las curvas quedan pequeñas: usar solo como apoyo. |
| `03_gaussiana_aislada.png` | 2048 × 2048 | Una gaussiana real del modelo (#18859: escala 21.6 × 4.8 px, θ = −0.079 rad, color RGB 0.87/0.33/0.29, opacidad 1.0), fondo transparente. Para ícono. |
| `04a_original.png` | 1080 × 720 | Frame 240 del clip original. |
| `04b_reconstruido.png` | 1080 × 720 | El mismo frame reconstruido por el modelo. PSNR de este par concreto: **39.87 dB** (calculado sobre los dos PNG). |
| `05_barras_tamano.png` | 2792 × 1386 | Barras de tamaño en escala log con el PSNR de cada punto y la nota sobre espacios de color. |
| `06_audio_overlay.png` | 2752 × 908 | Forma de onda original vs reconstruida, fragmento de 25 ms desde el segundo 5.909 (la ventana de más energía). SNR local del fragmento: 27.97 dB. Leyenda fuera del área de trazado, sobre el eje, para no tapar la señal. |
| `06b_audio_overlay_sin_leyenda.png` | 2777 × 909 | La misma figura sin leyenda, por si el pie de figura del póster ya explica los colores. |

### Resolución: aviso explícito

**`04a` y `04b` miden 1080 × 720, por debajo de los 2000 px pedidos, y NO se
interpolaron.** El clip fue extraído a esa resolución y no existe una versión
mayor. Para obtener más resolución hay que **re-renderizar el checkpoint a mayor
tamaño**, lo cual es posible porque las gaussianas son continuas: basta escalar
μ y `scale` por un factor y pedir un lienzo más grande. Eso lo hace
`01_construccion_progresiva.py --escala 2`, pero requiere GPU NVIDIA.

A 96 × 70 cm, 1080 px de ancho dan unos 150 ppp si la imagen ocupa ~18 cm de
ancho. Aceptable para mirar de cerca, justo para lectura a distancia. Si hay
acceso a una máquina con GPU antes de imprimir, conviene regenerar a `--escala 2`.

### Figura 1 (construcción progresiva): PENDIENTE

`poster_assets/scripts/01_construccion_progresiva.py` está **escrito, comentado
y listo**, pero **no se pudo ejecutar aquí**: usa el rasterizador real del
proyecto, que exige la extensión CUDA `raster_cuda`, y esta máquina es un Mac
sin GPU NVIDIA. Se intentó y falla con
`ModuleNotFoundError: No module named 'raster_cuda'`.

En una máquina con GPU y el entorno del proyecto:

```bash
python poster_assets/scripts/01_construccion_progresiva.py \
    --checkpoint <ruta>/musical_20s_1600ep/checkpoints/checkpoint_final.pt \
    --frame 240 --escala 2 --ks 1 10 100 1000 0
```

Produce `01_progresiva_K1.png`, `K10`, `K100`, `K1000` y `Ktodas`, a 2160 × 1440
con `--escala 2`.

**Criterio de contribución usado**, que es lo que pediste que explicara: se
ordena por **α_i(t) · s1_i(t) · s2_i(t)** evaluado en el frame objetivo. α es la
opacidad ya activada (cuánto aporta por píxel) y s1·s2 es proporcional al área
de la elipse, así que el producto es proporcional a la masa total de alpha que
la gaussiana deposita en la imagen — la integral de una gaussiana 2D es
2π·α·s1·s2 y la constante no altera el orden. Se prefirió sobre ordenar por
opacidad sola (ignora el tamaño: una mancha diminuta y opaca pesaría igual que
una grande) y sobre ordenar por el gradiente de la pérdida (más fiel a
"importancia", pero mucho más caro y dependiente del target). La cobertura da
además la construcción progresiva más legible: primero las manchas grandes de
color, luego el detalle.

---

## 8. Figuras ya existentes que sirven para el póster

Del repositorio oficial (`MBB-GS-Moving-Gaussians/assets/`):

| Ruta | Qué muestra |
|---|---|
| `assets/pipeline.png` | Diagrama del lazo de entrenamiento: video → modelo → render → frame → pérdida → backprop. Limpio y ya en el README. |
| `assets/top_gaussianas.png` | Clip original, modelo completo de 160k, y renders con solo las top-20k y top-80k por importancia visual. **Muy cercana a la figura 1 pedida**; si no se consigue GPU, puede sustituirla. |
| `assets/reconstruccion_n_gaussianas.png` | El mismo frame con 5k / 10k / 20k / 40k / 60k / 150k gaussianas. Ojo: son **modelos entrenados por separado**, no subconjuntos de uno solo — argumenta capacidad, no construcción progresiva. |

Del repositorio de trabajo (`/Users/jleandrojm/TESIS/MBB-GS/outputs_khipu/musical_20s_1600ep/`):

| Ruta | Qué muestra |
|---|---|
| `viz_marcadores/marcadores_top_animado.gif` | Gaussianas marcadas moviéndose. Si el póster lleva QR a algo animado, esta es la candidata. |
| `viz_marcadores/marcadores_top_estatico.png` | Versión fija de lo mismo. |
| `viz_elipses_vel/elipses_frame0240.png` | Elipses de las gaussianas dibujadas sobre el frame. |
| `viz_elipses_vel/velocidades_frame0240.png` | Campo de velocidades. |
| `viz_evolucion/convergencia_frame0240.gif` | Cómo converge la reconstrucción durante el entrenamiento. |
| `logs/loss_curve.png` | Curva de pérdida de las 1600 épocas. |
| `logs/heatmap_opacity_temporal.png` | Opacidad a lo largo del tiempo. |
| `logs/coeficientes_magnitudes.png` | Magnitud de los coeficientes por orden — apoya el argumento de Chebyshev. |
| `POLAR_BEAR/comparativo_importancia_5k_10k_20k.mp4` | Comparativo en video por importancia. |

---

## 9. Página web del proyecto

**PENDIENTE.** Existe un sitio estático completo en
`/Users/jleandrojm/TESIS/tesis-web/` (index.html, comparador de video con
slider, reproductor A/B de audio, figuras, BibTeX), pero **no está publicado**:
no tiene remoto git configurado y no aparece ninguna URL propia en el
repositorio. La única URL externa en el proyecto es la plantilla de la que
partió el diseño (`nerfies.github.io`).

Si se publica en GitHub Pages antes de imprimir, la URL sería del tipo
`https://jleandrojm.github.io/<nombre-del-repo>/`, pero **hay que confirmarla
tras publicarla** — no ponerla en el póster sin verificar que carga.

Alternativa ya disponible para el QR: el repositorio,
`https://github.com/JLeandroJM/MBB-GS-Moving-Gaussians` [`README.md`,
`CITATION.cff`].

---

## 10. Cómo se reproduce todo esto

```bash
cd poster_assets/scripts
python 02_trayectorias.py --modo zoom --alto 300 --ancho 450
python 02_trayectorias.py --modo full --n 40
python 03_gaussiana_aislada.py --saturacion_min 0.45 --brillo_min 0.6
python 04_comparacion.py --frame 240
python 05_barras_tamano.py
python 06_audio_overlay.py                      # leyenda arriba, fuera del trazado
python 06_audio_overlay.py --leyenda ninguna --salida 06b_audio_overlay_sin_leyenda.png
# requiere GPU NVIDIA:
python 01_construccion_progresiva.py --checkpoint <ckpt> --frame 240 --escala 2
```

Los scripts 02–06 solo necesitan `torch` (CPU), `numpy`, `matplotlib`, `pillow`
y `soundfile`; leen el checkpoint directamente y evalúan los polinomios de
Chebyshev en CPU, sin importar `gs2d_video`. Las rutas al experimento están en
`scripts/comun.py` (`EXP_DIR`, `CLIP_DIR`).

---

## 11. Resumen de lo PENDIENTE

1. **Figura 1, construcción progresiva** — script listo, falta una GPU NVIDIA.
2. **Tiempo de render por frame** — no está registrado en ningún log.
3. **URL de la página web** — el sitio existe pero no está publicado.
4. **Regenerar `04a`/`04b` a mayor resolución** — requiere GPU; hoy quedan a
   1080 × 720 sin interpolar.
5. **Métricas del mejor audio (estéreo 160k, 27.78 dB)** — no hay registro ni
   WAV en disco; solo están en el informe y en el Drive.
