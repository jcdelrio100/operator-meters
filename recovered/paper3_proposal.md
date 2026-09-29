# Propuesta — Tercer artículo

## Título (working)
**"Compress Toward the Label: A Rate–Relevance View of Matched and Learned Transforms"**

Alternativos:
- *Rate–Relevance, not Rate–Distortion: When Compression Bases Are the Right Front-End for Classification (and When to Learn One)*
- *The Gate as an Information-Bottleneck Meter: Reading, and Writing, the Equation of a Learned Transform*

## Una línea
El segundo artículo mostró *que* acoplar un diccionario fijo y una transformada aprendida funciona y que un gate auto-calibrante decide cuál hace falta. Este tercero explica **por qué**, con una tesis: la representación óptima para clasificar no comprime la *señal* (rate–distortion) sino la *etiqueta* (rate–relevance), el gate mide exactamente esa brecha, y la transformada resultante puede leerse —a veces como ecuación cerrada, a veces como el operador que diagonaliza.

---

## 1. Motivación y pregunta

Las bases de compresión clásicas (Fourier/DCT, wavelets) son las que usan JPEG, MP3, JPEG2000. Funcionan como front-end de clasificación porque decorrelan gratis. Pero *comprimir bien la señal* (rate–distortion) y *ser óptimo para la tarea* (rate–relevance) son objetivos distintos: el Information Bottleneck busca un **estadístico suficiente mínimo para la etiqueta `y`**, no para la señal `x`.

**Pregunta central:** ¿cuándo bastan las bases de compresión como representación, cuándo hay que aprender una, y qué mide exactamente el gate que decide entre ambas?

## 2. Tesis y hipótesis

- **H1 — El gate es un medidor de rate–relevance.** El gate protector `g_s2` cuantifica la relevancia que las bases de compresión dejan sobre la mesa. **[WP1 HECHO]**
  *Evidencia:* con una brecha *informacional* controlada `G = I(T_learned;Y) − I(T_dict;Y)` (bits, medida con cabezas separadas) sobre una familia `rotated(θ)` que rota la base discriminante de Fourier a aleatoria, el gate sigue a G con **Pearson r = 0.96** y ley lineal (rectificada) `g_s2 ≈ 0.04 + 0.26·G`; las 4 tareas 1-D reales caen sobre la misma recta.
- **H2 — La compresión es un buen *prior*, mal *objetivo* final.** La compresión *lossy* descarta detalle que la tarea necesita; conservar los coeficientes pre-cuantización y dejar que la tarea comprima (selección + gate + cabeza) lo recupera.
  *Evidencia piloto:* seleccionar features por energía (rate–distortion) da **0.274** en 1-D localizado frente a **0.999** por F-score (rate–relevance).
- **H3 — La expresabilidad simbólica vive en el lado de compresión.** Una U regularizada hacia compresión conserva estructura simbólica; una U dirigida por clasificación deriva fuera de alcance simbólico.
  *Evidencia piloto:* la DFT se recupera exacta por ajuste bilineal (fit 1.00, `a = 2π/N`); la sparsificante conserva estructura visible; la de clasificación se emborrona.
- **H4 — Identificación del operador.** En señales clásicas/comprimibles la transformada descubierta diagonaliza un operador *local/diferencial* simple; en las no clásicas, no.

## 3. Paquetes de trabajo

**WP1 — Formalizar y medir la brecha rate–relevance. ✅ HECHO (`wp1_gap_law.png`, `RESULTS_wp1_gap_law.md`).**
Brecha `G = I(T_learned;Y) − I(T_dict;Y)` a tasa fija K, estimada con MI vía entropía condicional (H(Y) − CE, en bits). Familia sintética `rotated(θ)` con brecha *controlada por diseño* (geodésica Fourier→aleatoria) + tareas reales. **Resultado:** `g_s2 ≈ 0.04 + 0.26·G`, **r = 0.96**, relación rectificada (gate = 0 mientras G ≤ 0, lineal después, con saturación leve). *Pendiente de refuerzo:* estimadores MINE/kNN como validación cruzada; curvas rate–distortion vs rate–relevance por representación; ampliar semillas.

**WP2 — Compresión como etapa de entrada, hecho bien.**
Barrido de códecs reales (JPEG/JPEG2000/MP3) a varios niveles de cuantización como front-end, vs. los mismos coeficientes *pre-cuantización* + selección por tarea. Mostrar dónde la cuantización perceptual destruye la señal discriminante (curva accuracy vs bitrate) y que el objetivo correcto es rate–relevance, no rate–distortion. Incluir la nota "gzip-como-distancia" (NCD/kNN) como uso *distinto* y válido (compresión como kernel, no como feature).

**WP3 — Leer/escribir la ecuación de U (PySR).**
Pipeline de canonicalización del gauge (ordenar filas por frecuencia; fijar fase) + PySR sobre `Re(U[j,k]) = f(j,k)`. Validación: recuperar `cos(2π·jk/N)` en la DFT. Objetivo real: la **sparsificante**, que muestra estructura no-Fourier por nombrar (¿chirp?, ¿DCT?). Negativo informativo esperado en la de clasificación. (Script `symbolic_pysr.py` ya listo; requiere máquina con Julia.)

**WP4 — Identificación del operador (SINDy-PDE).**
Contrastar si la transformada descubierta es la base propia de un operador simple `L` (p. ej. `L = a·∂²ₓ + V(x)`), identificando `L` con PDE-FIND/SINDy sobre la acción de U. Resultado buscado del tipo *"la transformada del ECG diagonaliza el operador L = …"* — leer U como una **ley**, no solo como una tabla. Reportar honestamente dónde no existe tal operador (rotaciones arbitrarias).

## 4. Resultados preliminares (ya en mano)

| Pieza | Resultado | Estado |
|---|---|---|
| **Ley del gate (H1/WP1)** | **`g_s2 ≈ 0.04+0.26·G`, r = 0.96** (brecha informacional controlada) | **hecho (`wp1_gap_law.png`)** |
| Compresión ≠ relevancia (H2) | localizado: energy-sel 0.27 vs task-sel 0.999 | hecho (`rate_relevance.png`) |
| Recuperación simbólica (H3) | DFT exacta, `a = 2π/N`; sparsificante estructurada | hecho (`symbolic_U.png`) |
| PySR real (WP3) | script listo | pendiente (Julia) |
| SINDy-PDE (WP4) | — | por hacer |

Es decir, el artículo ya nace con dos figuras de resultado y una tesis contrastada, no desde cero.

## 5. Contribuciones esperadas

1. Una **teoría rate–relevance del gate**: el gate como medidor operativo de Information Bottleneck, con la ley `g_s2 ∝ G`.
2. **Guía práctica** sobre compresión-como-front-end: cuándo sí (prior barato) y cuándo destruye la señal (lossy perceptual), con curvas accuracy-vs-bitrate.
3. Un **pipeline de lectura simbólica/operacional** de transformadas aprendidas (canonicalización + PySR + SINDy-PDE), convirtiendo "descubrir una transformada" en "escribir su ecuación o su operador".

## 6. Riesgos y mitigaciones

- *Estimar MI en alta dimensión es difícil* → trabajar sobre las K=16 features (baja dimensión); estimadores kNN/MINE; validar con tareas de brecha conocida.
- *PySR puede no hallar forma para U adaptadas* → es un negativo *informativo* (confirma H3); el objetivo principal es la sparsificante, donde hay estructura visible.
- *El operador de WP4 puede no existir* → reportar el mapa de dónde sí/no (clásicas sí, rotaciones no), que es en sí un resultado.
- *r=0.90 con 7 puntos* → ampliar a un banco grande con brecha controlada para robustez estadística.

## 7. Roadmap (orientativo)

1. **Mes 1–2:** WP1 (banco ampliado + estimadores MI + ley `g_s2 ∝ G`).
2. **Mes 2–3:** WP2 (códecs reales, curvas bitrate).
3. **Mes 3–4:** WP3 (PySR en máquina con Julia) — validación DFT + sparsificante.
4. **Mes 4–5:** WP4 (SINDy-PDE, exploratorio).
5. **Mes 5–6:** redacción, figuras, release Zenodo/GitHub (mismo formato que el segundo artículo).

## 8. Reproducibilidad

Mismo estándar que el segundo artículo: scripts deterministas por figura, 5 semillas con barras de error, repo standalone con README/LICENSE/CITATION/.zenodo.json, resultados negativos declarados. Reutiliza `src/` y `experiments/` de `fixed-learned-both`.

## 9. Referencias base

- Tishby, Pereira, Bialek — *The Information Bottleneck Method* (1999); Tishby & Zaslavsky — *Deep Learning and the IB Principle* (2015).
- Shannon — *rate–distortion*; Cover & Thomas — *Elements of Information Theory*.
- Rissanen — *MDL* (aprender = comprimir).
- Cranmer et al. — *PySR / Interpretable ML with symbolic regression* (2023).
- Brunton, Proctor, Kutz — *SINDy* (2016); Rudy et al. — *PDE-FIND* (2017).
- Jiang et al. — *"Low-Resource Text Classification: gzip + kNN"* (2023) — compresión como distancia (NCD).
- del Río Romero — artículos 1 (*Matched Transforms…*) y 2 (*Fixed, Learned, or Both?*).
