# Propuesta — Quinto artículo (REENCUADRADA tras el chequeo de literatura del Paso 0)

*(La versión anterior, con la tesis geométrica, queda archivada en `paper5_proposal_v1_geometria.md`.)*

## Título (working)
**"The Geometry Gap: Measuring How Much of a Learned Representation Is Just a Change of Coordinates"**

Alternativos:
- *Warped, Not New: A Calibrated Decomposition of Learned Transforms into Reference, Warp and Residual*
- *Don't Learn the Basis, Learn the Geometry — and Measure the Difference*

## Una línea
Un **instrumento**: dado un front-end o transformada aprendida `T`, descomponerlo en (transformada de referencia `R` ∘ deformación de coordenadas `w`) + residuo, y reportar **qué fracción de `T` es sólo un cambio de coordenadas** — con la deformación recuperada como objeto *interpretable* y con los controles que hacen fiable el número.

## Por qué el reencuadre (resultado del Paso 0, `LITERATURE_CHECK.md`)
La versión anterior de esta ficha era un artículo de *teoría* ("transformada = geometría × trayectoria"). El chequeo de literatura la desmontó como contribución: las clases de Möbius, el diccionario grupo→transformada (coorbitas) y los polinomios ↔ Sturm–Liouville son clásicos, y **Marchetti et al. (PMLR 2024)** ya *demuestra* que los pesos de una red invariante recuperan la Fourier del grupo y que el grupo se lee de los pesos. Nada de eso es nuestro. Lo que **no** encontramos con nombre: una descomposición calibrada de UNA representación aprendida en (referencia ∘ warp) + residuo — CKA/SVCCA comparan *dos* representaciones por similitud, no descomponen una. **Toda la geometría se queda — como fundamento y validación, no como tesis.**

## El instrumento, operativamente

`GG(T; R, 𝒲) = fracción de T no explicada por R∘w, minimizada sobre w ∈ 𝒲`

con `𝒲` una familia de deformaciones declarada *a priori*: cambio monótono de coordenada (warp) + gauge por canal (ganancia/fase; el factor de Liouville `a^{1/4}` en el caso 1-D). Se reporta: **(1)** el valor del gap, **(2)** el warp óptimo — que es interpretable: en el caso físico *es la métrica*, en audio *es la escala frecuencial efectiva* —, **(3)** el residuo y su estructura.

**Controles obligatorios** (todos aprendidos a golpes en esta sesión y ya codificados como errores-a-no-repetir):
1. **Caso nulo**: calibrar contra el control trivial/homogéneo; una métrica que satura en el control se descarta (lección del residuo por modo).
2. **Deflación contra leyes puntuales**: una correlación alta puede ser una ley local sin geometría (lección del 0.993 de `1/a`).
3. **Test de ensanchamiento**: si el gap se estanca lejos de cero, ampliar `𝒲`; si baja a cero, era mala especificación; si no baja, el residuo es genuino — la mala especificación se anuncia sola (lección de WP1b).
4. **No promediar** lo que vive en átomos individuales (lección del artefacto de banda).

## Evidencia ya en mano (validación sintética del instrumento)
- **1-D (Liouville, `topo_transforms.py`)**: la base "nueva" de WP1b tenía energía monomodal 0.374 en la coordenada original y **0.996** tras el warp geodésico + gauge — el instrumento detecta que era **Fourier al 99.6 %**, y el warp recuperado *es* la métrica física.
- **2-D no separable (`geo2d.py`)**: descubrimiento de la deformación exacto (off-diag 0.4380 → 0.000000, `a(x,y)` con corr 1.0000) en el caso que una referencia separable no puede expresar.
- **Frontera (`sl_limits.py`)**: dónde la descomposición NO puede funcionar — sin simetría no hay descripción corta (1.1 % del hueco cerrado); no-normal: ninguna base ortogonal diagonaliza. **La simetría es la fuente de compresibilidad**: el mapa de aplicabilidad ya está medido.
- *Background* (no resultados): Möbius/órbitas (`topo_transforms` panel A), redundancia (`redundancy_probe.py`), granularidad (`card_probe.py`).

## Paquetes de trabajo

**WP0 — Cite-chase profundo (una tarde; go/no-go).** Cadenas de citas de CKA/SVCCA y de los análisis de front-ends (LEAF/SincNet). Pregunta única: ¿alguien ha publicado ya una descomposición (referencia ∘ warp) + residuo para representaciones aprendidas? Si sí → pivotar la contribución al protocolo de controles + la aplicación; si no → seguir.

**WP1 — Extender el gap a transformadas no ortogonales y sobrecompletas.** La definición actual asume bases (orto)normales. Los front-ends reales son *filterbanks*: sobrecompletos, no ortogonales, con compresión logarítmica y pooling. Extensión: ajustar warp frecuencial monótono + ganancia por canal entre respuestas en frecuencia; validar **plantando warps conocidos** (¿se recuperan?) y con los 4 controles. **Decisión de gauge ANTES de mirar datos reales** — qué cuenta como "cambio de coordenadas" y qué como estructura nueva se fija a priori, para no ajustar la definición al resultado.

**WP2 — El experimento ML sintético (heredado).** Mismo dato, dos front-ends: métrica parametrizada (pocos parámetros) vs base libre (~N²). Comparar identificabilidad, estabilidad entre semillas y extrapolación OOD con el protocolo de barras de error de WP4b del artículo 4.

**WP3 — La aplicación con respuesta desconocida: front-ends de audio.** El hueco confirmado en el Paso 0, ahora afinado con la lectura del paper de Sci Rep: éste **sí cuantifica** la deriva de LEAF (frecuencias centrales y anchos, deriva mínima) — **pero sólo puede hacerlo porque LEAF es paramétrico**: cada filtro declara su frecuencia central. Para front-ends de **forma libre** (conv1d sin parametrizar, TD-filterbanks) no hay parámetros que leer y la pregunta "¿cuánto es un mel deformado?" está abierta; *Should Audio Front-ends be Adaptive?* (2025) compara ocho datasets **sólo por accuracy**. Plan: mel fijo / LEAF / SincNet / conv libre en tareas públicas pequeñas (CPU); **predicciones registradas de antemano**: gap bajo para LEAF (consistente con Sci Rep — y sería una *validación cruzada* del instrumento contra su lectura paramétrica), desconocido para conv libre (el resultado nuevo). Reportar gap + warp recuperado (la "escala mel efectiva" aprendida) + residuo.

**WP4 — Frontera honesta.** No-normalidad, isoespectralidad (Gordon–Webb–Wolpert), ausencia de simetría; cuándo el gap es interpretable y cuándo no. Semilla hecha (`sl_limits.py`).

## Contribuciones esperadas
1. **El instrumento**: la descomposición calibrada (referencia ∘ warp) + residuo, con su protocolo de controles.
2. **El warp como lectura**: la deformación recuperada es un objeto con significado (métrica física / escala frecuencial efectiva), no un número opaco.
3. **Una respuesta cuantitativa a una pregunta abierta real**: cuánto de un front-end de audio de forma libre es un mel deformado — en datos públicos, reproducible en CPU.

## Riesgos y mitigaciones
- *WP0 encuentra un equivalente publicado* → pivote declarado arriba; el coste es una tarde.
- *La extensión no ortogonal es delicada* → gauge fijado a priori + validación con warps plantados antes de tocar datos reales.
- *Acceso a datasets de audio desde el entorno* → riesgo real (UCR directo estuvo bloqueado; pyts funcionó). Probar acceso (torchaudio/FSDD/librosa-bundled) **antes** de comprometer WP3; fallback: datasets pequeños empaquetados en PyPI.
- *Gap bajo en todo* (nada aprende geometría nueva) → resultado igualmente publicable: "los front-ends aprendidos son mel deformado al X %", con el warp como caracterización — coincide con la dirección de Sci Rep y la generaliza a forma libre.

## Estado
Ficha reencuadrada 2026-08-16 (Paso 2 del plan). Validación sintética hecha (arriba); WP0 y WP1 son lo siguiente; WP3 es el objetivo. Depende del artículo 4 sólo como cita (los medidores y sus controles).

## Referencias base
- Kornblith et al. — *Similarity of Neural Network Representations Revisited* (CKA), ICML 2019; Raghu et al. — SVCCA, NeurIPS 2017.
- Marchetti, Hillar, Kragic, Sanborn — *Harmonics of Learning*, PMLR 2024.
- Zeghidour et al. — *LEAF*, ICLR 2021; Ravanelli & Bengio — *SincNet*, 2018.
- *Should Audio Front-ends be Adaptive?* (2025); *A frequency analysis of filterbank initialisation for LEAF*, Sci. Rep. (2026); *What is Learnt by LEAF?* (2024).
- Feichtinger & Gröchenig — coorbitas; Grohs et al. — α-molecules (background).
- Gordon, Webb & Wolpert (1992) — isoespectralidad (frontera).
- del Río Romero — artículos 1, 2 y 4 (los medidores y sus controles).
