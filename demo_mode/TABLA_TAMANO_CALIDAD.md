# Tamaño contra calidad — qué modelo subir a la web

Todas las medidas se hicieron con `demo_mode/render_referencia_cpu.py` leyendo
el archivo exportado, a resolución completa 1080 × 720, promediando los frames
**0, 240 y 479** de `musical_20s_1600ep`. El PSNR es contra los **frames
originales del clip**, no contra el render CUDA.

Reproducible con `demo_mode/tabla_tamano_calidad.py`.

---

## 1. Reducir el número de gaussianas

Prefijo del array ya ordenado por importancia, grados completos.

| top_k | Tamaño | PSNR vs original |
|---:|---:|---:|
| 5 000 | 4.18 MB | 12.57 dB |
| 10 000 | 8.36 MB | 13.71 dB |
| 20 000 | 16.72 MB | 15.75 dB |
| 30 000 | 25.08 MB | 17.46 dB |
| 50 000 | 41.80 MB | 20.19 dB |
| 75 000 | 62.70 MB | 23.27 dB |
| 100 000 | 83.60 MB | 26.55 dB |
| **150 000 (todas)** | **125.40 MB** | **39.52 dB** |

**Este eje no degrada con gracia.** Quitar un tercio de las gaussianas cuesta
13 dB. El salto de 100k a 150k vale 13 dB por sí solo.

La razón es que el modelo se entrenó con las 150 000 contribuyendo a la vez: la
imagen es la composición alpha de muchas primitivas solapadas y cada una aporta
poco. Al quitar una fracción grande, las que quedan no compensan — no fueron
entrenadas para hacerlo. Esto es distinto de la poda adaptativa del proyecto,
que elimina solo las que nunca son visibles.

El orden por importancia sigue siendo útil **como control de la demo** — ver la
imagen construirse de 1 a N gaussianas —, pero no como estrategia para que el
archivo pese menos.

---

## 2. Truncar los grados de los polinomios

Las 150 000 gaussianas, recortando el grado de los atributos.

| Grados | Tamaño | PSNR |
|---|---:|---:|
| completos (100/80/30/12/6/4) | 125.40 MB | **39.52 dB** |
| mu=60 | 101.40 MB | 29.46 dB |
| mu=40 | 89.40 MB | 27.75 dB |
| mu=30 | 83.40 MB | 27.57 dB |
| mu=40, op=40 | 77.40 MB | 22.84 dB |
| mu=30, op=30, col=15 | 54.90 MB | 23.11 dB |
| mu=25, op=20, col=10, sc=8 | 42.00 MB | 24.26 dB |
| mu=20, op=15, col=8, sc=6, th=4, d=3 | 33.60 MB | 25.13 dB |
| mu=16, op=12, col=6, sc=5, th=3, d=2 | 27.30 MB | 25.67 dB |
| **mu=12, op=10, col=5, sc=4, th=3, d=2** | **22.80 MB** | **25.97 dB** |

### Dos cosas que no se esperaban

**a) El primer recorte es carísimo.** Bajar `mu` de 100 a 60 cuesta 10 dB. Los
coeficientes de orden alto de la posición no son ruido: llevan información real.

**b) El truncado no es monótono.** Recortar *más* da mejor PSNR que recortar
poco: 22.80 MB con grados mínimos da 25.97 dB, mientras que 77.40 MB con
`mu=40, op=40` da 22.84 dB. Truncar la serie de un atributo pero no la de otro
rompe cancelaciones entre atributos que el entrenamiento aprendió conjuntamente;
un recorte uniforme y agresivo se comporta como un filtro paso bajo coherente.

**Consecuencia práctica: no se puede afinar el truncado con una búsqueda golosa
atributo por atributo.** Hay que medir cada combinación.

### Qué se pierde exactamente al truncar

La miniatura de la configuración recomendada
(`demo_mode/export_web/musical-20s.png`) lo muestra sin ambigüedad: **el fondo
estático queda nítido y las personas y la tela en movimiento salen
emborronadas.**

Tiene sentido: los grados altos codifican la variación temporal, así que
truncarlos borra precisamente el movimiento y conserva lo que no cambia. El
PSNR promediado sobre tres frames solo captura una parte de ese efecto.

**Y esto choca de frente con el propósito de la demo**, donde el visitante mueve
el tiempo y mira cómo cambia la escena. La configuración de 22.80 MB se ve bien
en una imagen fija y floja en movimiento.

---

## 3. Precisión de los coeficientes

Medido contra el render CUDA, frame 240, 150k gaussianas, grados completos:

| Precisión | PSNR vs CUDA | Tamaño extra |
|---|---:|---:|
| todo float16 | 47.11 dB | — |
| **`mu` k=0 en float32, resto float16** | **55.90 dB** | **+0.60 MB** |
| `mu` completo en float32 | 55.90 dB | +60.60 MB |
| todo float32 | 56.18 dB | +124.20 MB |

`mu` guarda píxeles absolutos, hasta ~1080, y ahí el ULP de float16 vale 1.0, o
sea medio píxel de error. Reponer **solo el término constante** en float32
recupera casi todo por un 0.5 % de tamaño. Está activado por defecto en el
exportador (bloque `mu_a0_f32`).

Es la única optimización de esta página que sale gratis.

---

## 4. Recomendación

### El problema de fondo

**El modelo completo pesa 125.40 MB y no cabe en GitHub sin LFS**, que tiene un
límite de 100 MB por archivo. Y toda reducción por debajo de eso cuesta al menos
10 dB.

Para este checkpoint concreto no hay ninguna configuración buena: o se sube algo
que GitHub rechaza, o se acepta una caída grande.

### Para `musical_20s`, si hay que subirlo hoy

```bash
python demo_mode/preparar_web.py \
    --checkpoint .../musical_20s_1600ep/checkpoints/checkpoint_final.pt \
    --id musical-20s --titulo "Musical, 20 s" --fps 24 \
    --degradar_grados "mu=12,opacity=10,color=5,scale=4,theta=3,depth=2"
```

| | |
|---|---|
| Gaussianas | 150 000, todas |
| Grados | mu 12, opacity 10, color 5, scale 4, theta 3, depth 2 |
| Tamaño | **22.80 MB**, bajo el objetivo de 30 MB |
| PSNR | 25.97 dB contra el original |
| Se ve | escena nítida, movimiento emborronado |

Es la mejor calidad medida por debajo de 30 MB. Sirve para que el visor tenga
algo que cargar mientras se preparan los modelos definitivos.

### Lo que de verdad conviene hacer

**No usar `musical_20s` en la demo. Entrenar clips cortos a propósito.**

`musical_20s` es un modelo grande porque se entrenó para el paper: 20 segundos,
720p, 150k gaussianas y grado 100 en `mu`. Un clip de la demo no necesita nada
de eso.

Para un clip de 3 s a 24 fps (72 frames) a 540 × 360, entrenado desde el
principio con 40 000 gaussianas y grados `mu=30, op=24, col=12, sc=8, th=4,
d=3`, el tamaño sale de la aritmética del formato:

```
coeficientes por gaussiana = 2·31 + 25 + 3·13 + 2·9 + 5 + 4 = 153
bytes = 40 000 · 153 · 2 = 12.2 MB, más 0.3 MB del bloque mu_a0_f32
```

**Unos 12.5 MB, y sin truncar nada**: esos grados son los que el modelo entrenó,
así que conserva toda su fidelidad temporal, que es justo lo que la demo
necesita y lo que el truncado destruye. Con 72 frames, un grado 30 en `mu` ya es
generoso.

Cinco o seis clips así pesan unos 75 MB en total, se versionan sin problema y
cargan rápido en la feria. En `demo_mode/RECON_ENTRENAMIENTO.md`, sección 2.1,
está el config de partida; la estimación de tiempo son minutos por clip.

---

## 5. Lo que no se midió

- **PSNR en instantes intermedios**, los que nunca se grabaron. No hay verdad de
  referencia contra la cual medirlos, así que el efecto del truncado sobre la
  interpolación temporal — la función estrella de la demo — está sin cuantificar.
- **Métricas perceptuales.** Todo es PSNR. SSIM y LPIPS capturarían mejor el
  emborronado del movimiento.
- **Solo tres frames por configuración**, de 480. Elegidos en los extremos y el
  centro; un barrido completo daría intervalos de confianza.
- **Un solo checkpoint.** Todas las conclusiones vienen de `musical_20s_1600ep`.
  Si el no-monótono del truncado se repite en otros modelos, está sin comprobar.
