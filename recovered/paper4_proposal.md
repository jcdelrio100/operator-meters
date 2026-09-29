# Propuesta — Cuarto artículo

## Título (working)
**"When Does Fourier Diagonalize the Physics? A Gated Coupling of Spectral and Learned Bases for Compact Operator Learning"**

Alternativos:
- *Matched, Learned, or Both — for Dynamics: A Rate–Distortion / Koopman View of Reduced-Order Modeling*
- *The Gate as a Nonlinearity Meter: Reading the Operator a Learned Transform Diagonalizes*

## Una línea
Re-instanciar la arquitectura fijo / aprendido / acoplado para **modelizar dinámica** en vez de clasificar. El objetivo se **invierte** (predicción/reconstrucción, no cross-entropy), la transformada aprendida pasa a ser la **base de Koopman / modos propios** del sistema (bien planteada, a diferencia de clasificación), el gate se vuelve un **medidor de no-linealidad/heterogeneidad física**, y "leer U" pasa a ser **descubrir el operador que rige el fenómeno**.

## La inversión (clasificación → modelización)

| Pieza | Clasificación (art. 1–2) | Modelización física (este) |
|---|---|---|
| Objetivo | comprimir hacia la etiqueta (rate–relevance) | preservar/predecir el estado (rate–distortion / información predictiva) |
| S1 diccionario | Fourier/wavelet como features | métodos espectrales: Fourier **diagonaliza** operadores lineales homogéneos |
| S2 aprendida | base discriminante (infra-determinada) | **base de Koopman / modos propios** (bien determinada por la dinámica) |
| selección | F-score (separabilidad de clase) | energía / POD / amplitud de modo Koopman |
| cabeza | clasificador | **propagador** `u(t)→u(t+Δ)` (Koopman lineal + bloque no lineal) |
| gate g_s2 | ¿base discriminante fuera del banco de compresión? | **¿la física la diagonaliza una transformada clásica?** (linealidad/homogeneidad) |
| leer U | mal planteado (deriva) | **descubrir el operador** (Koopman/DMD/SINDy) — bien planteado |

## Hipótesis

- **H1 — El gate es un medidor de heterogeneidad/no-linealidad.** En física lineal homogénea (Fourier = base propia) el gate está cerrado y S1 es óptima; al romper la homogeneidad o la linealidad, el gate se abre en proporción.
- **H2 — La transformada aprendida redescubre los modos propios.** Warm-started desde Fourier, S2 converge a la base que diagonaliza la dinámica (eigenbase / Koopman), y "leer U" recupera esos modos.
- **H3 — El break-even se traslada a rate–distortion de la dinámica.** A tasa fija K, la eigenbase comprime la dinámica mejor que Fourier cuanto más heterogéneo/no lineal es el sistema.
- **H4 — Estructura y estabilidad.** Imponer conservación/simplecticidad y estabilidad de rollout mejora el horizonte de predicción; el acoplamiento con puerta hereda la robustez del art. 2.

## Paquetes de trabajo

**WP1 — Gate como medidor de heterogeneidad (PDE lineal). [PILOTO ✅ `phys_wp1_pilot.png`]**
Difusión con coeficiente espacial `a(x)`, knob de heterogeneidad β. **Resultado piloto (nivel operador):** en β=0 Fourier diagonaliza el propagador P (fracción off-diagonal = 0, método espectral exacto); al crecer β, **la no-diagonalización de Fourier sube monótona 0 → 0.27** mientras la eigenbase Φ lo diagonaliza siempre (≡0), y align(Φ,Fourier) cae 1.0 → 0.44. Es decir, la no-diagonalización de Fourier es un **medidor de heterogeneidad** limpio (análogo físico del gate). *Siguiente:* sustituir la Φ-oráculo por una V aprendida (warm-start Fourier) + propagador con puerta, y mostrar `g_s2 ∝ no-diagonalización` (estilo WP1 clasificación, r=0.96) y que V→Φ (redescubre los modos propios).

**WP2 — Gate como medidor de no-linealidad (Burgers / Kuramoto–Sivashinsky).**
Knob de no-linealidad ε: ε=0 (calor lineal) → Fourier exacta; ε↑ → acoplamiento de modos → hace falta bloque no lineal + base aprendida. Gate ∝ ε.

**WP3 — Descubrimiento del operador.**
SINDy / PDE-FIND / DMD sobre las coordenadas aprendidas: leer U = escribir la ecuación de gobierno. Conexión explícita con **Fourier Neural Operator** (= S1 en versión operador) y **ROM/DMD** (= S2); la contribución nueva es el **acoplamiento con puerta** entre base espectral fija y base Koopman aprendida.

**WP4 — Preservación de estructura y rollout.**
Transformadas que respetan conservación (energía/momento) / simplécticas; estabilidad a largo horizonte (radio espectral); error de forecast multi-paso.

## Contribuciones esperadas
1. Un **ROM con puerta** que combina base espectral fija y base Koopman aprendida, con el gate como diagnóstico interpretable de **no-linealidad/heterogeneidad**.
2. Trasladar el break-even y la ley del gate al dominio físico (rate–distortion de la dinámica).
3. **Descubrimiento de operador** vía lectura de U (Koopman/SINDy), bien planteado en física.

## Riesgos y mitigaciones
- *La cabeza no lineal sola resuelve Burgers* → usar heterogeneidad (WP1, lineal pero eigenbase≠Fourier) como caso limpio donde la base sí importa a tasa fija K.
- *Koopman con transformada de misma dimensión no linealiza del todo* → aceptar linealización parcial; comparar con lifting/observables; reportar honesto.
- *Inestabilidad de rollout* → empezar por predicción a un paso; WP4 aborda estabilidad.

## Estado
Reutiliza `src/` y `experiments/`. Pilotos del art. 2/3 aplicables (warm-start, gated head, gate). Primer experimento (WP1 difusión heterogénea) en marcha en esta sesión.
