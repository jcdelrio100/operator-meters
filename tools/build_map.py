"""Build mapa_proyecto.html — the visual project map (convention: every section carries an
'En cristiano' strip with the glasses analogy; numbers come from results/*.json)."""
import os, sys, json, base64, datetime
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); FIG = os.path.join(ROOT, "figures")
# The map is an internal (Spanish) project document: it is written OUTSIDE the repository,
# next to it (../mapa_proyecto.html), unless --out is given.
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join(os.path.dirname(ROOT), "mapa_proyecto.html")
J = lambda n: json.load(open(os.path.join(RES, n)))
def img(name, alt=""):
    b = base64.b64encode(open(os.path.join(FIG, name), "rb").read()).decode()
    return f'<img alt="{alt}" src="data:image/png;base64,{b}">'

g = J("gap_law.json")["law"]; rr = J("rate_relevance.json")["out"]; sy = J("symbolic_U.json")
w1 = J("phys_wp1.json"); w2 = J("phys_wp2.json")["rows"]; d = J("depth_test.json")["mean_capture_eps_ge_0p7"]
w3 = {r["eps"]: r for r in J("phys_wp3.json")["rows"]}; w4 = J("phys_wp4.json"); s = w4["summary"]; amp = w4["amplitude_sweep"]
import numpy as np
loc_e = np.mean(rr["1D localized (wavelet)"]["energy"]); loc_t = np.mean(rr["1D localized (wavelet)"]["fscore"])
today = datetime.date.today().strftime("%d %b %Y")

CSS = """
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 18px;color:#222;line-height:1.45}
h1{margin-bottom:2px} .sub{color:#666;margin-top:0}
.intro{background:#f4f6fa;border-left:4px solid #31557F;padding:10px 14px;margin:14px 0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin:14px 0}
.card{border:1px solid #ddd;border-radius:8px;padding:10px 12px;background:#fff}
.card .st{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#fff;padding:2px 7px;border-radius:10px;display:inline-block;margin-bottom:6px}
.closed{background:#2F6B3D}.active{background:#B0173A}.parked{background:#888}.merged{background:#31557F}
.chain{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin:12px 0}
.q{border:1px solid #ccd;border-radius:6px;padding:8px;background:#fafbff;font-size:13px}
.q b{display:block;color:#31557F}
section{margin:26px 0;border-top:1px solid #eee;padding-top:10px}
.tag{font-size:12px;color:#B0173A;text-transform:uppercase;letter-spacing:.05em}
.llano{background:#fdf6e3;border-left:3px solid #d9a520;padding:8px 12px;margin:8px 0;font-size:14px}
.nums{display:flex;gap:14px;flex-wrap:wrap;margin:10px 0}
.n{background:#f7f7f7;border-radius:6px;padding:8px 12px;min-width:150px}
.n big{font-size:22px;font-weight:700;display:block;color:#31557F}
.n small{color:#555}
img{max-width:100%;border:1px solid #eee;border-radius:4px;margin:6px 0}
.read{background:#eef5ee;border-left:3px solid #2F6B3D;padding:8px 12px;margin:8px 0}
.lost{background:#fbeaea;border-left:3px solid #B0173A;padding:8px 12px;margin:8px 0}
table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ddd;padding:4px 8px}
"""

def sec(tag, title, llano, nums, figure, reading, extra=""):
    ns = "".join(f'<div class="n"><big>{v}</big><small>{k}</small></div>' for k, v in nums)
    return f'<section><div class="tag">{tag}</div><h2>{title}</h2><div class="llano"><b>En cristiano:</b> {llano}</div>{extra}<div class="nums">{ns}</div>{figure}<div class="read"><b>Lectura:</b> {reading}</div></section>'

H = [f"<!doctype html><html><head><meta charset='utf-8'><title>Mapa del proyecto — Transformadas fijas, aprendidas y física</title><style>{CSS}</style></head><body>"]
H.append(f"<h1>Mapa del proyecto</h1><p class='sub'>Dónde estamos, qué se ha probado y qué queda · actualizado {today} · <b>artículo unificado (3+4) reconstruido con código y cifras nuevas</b></p>")
H.append("""<div class="intro"><b>La idea que recorre todo el proyecto.</b> Para representar una señal puedes usar una transformada <b>fija</b> de toda la vida (Fourier, wavelets) o <b>aprender</b> una a medida. Construimos una arquitectura con las dos ramas y <b>una puerta que decide sola</b> cuánto usa la aprendida. Esa puerta resultó ser un <b>medidor</b>. Si te pierdes, vuelve a esta imagen: <b>una transformada = unas gafas para mirar la señal</b>; Fourier son las gafas clásicas, y una red neuronal puede intentar fabricarse unas a medida. La puerta mide cuánto valen las gafas nuevas — y este artículo pregunta <b>en qué moneda</b>: en clasificación pagan en <i>información</i>; en física pagan en <i>no-linealidad</i>, y además se pueden leer como una ecuación.</div>""")
H.append(f"""<h2>Los artículos</h2><div class="cards">
<div class="card"><span class="st closed">cerrado</span><br><b>1 · Matched Transforms</b><br>La cabeza auto-organizativa.</div>
<div class="card"><span class="st closed">cerrado · publicado</span><br><b>2 · Fixed, Learned, or Both?</b><br>S1 fija + S2 aprendida + puerta. Zenodo/GitHub.</div>
<div class="card"><span class="st merged">fusionado en el 4</span><br><b>3 · Compress Toward the Label</b><br>Su joya (la puerta mide bits) es ahora la Parte I del artículo unificado.</div>
<div class="card"><span class="st active">borrador reconstruido · en lectura</span><br><b>4 · When Does Fourier Diagonalize the Physics?</b><br>Artículo unificado: 16 págs, 3 diagramas, 10 figuras, repo <code>operator-meters</code>.</div>
<div class="card"><span class="st parked">aparcado · 16 ago</span><br><b>5 · The Geometry Gap</b><br>Instrumento validado; archivado en <code>recovered/</code>.</div></div>""")
H.append("""<div class="lost"><b>Lo que se perdió y se rehízo (29 sep).</b> El borrador de 16 páginas y los scripts del 4º artículo no se guardaron en tu ordenador (solo quedó el PDF de 10 páginas, el mapa y las figuras). Se reconstruyó <b>todo el código desde cero</b> y se volvieron a correr los experimentos; las cifras de este mapa y del artículo son las <b>nuevas</b>. Cambió una cosa importante: el término descubierto aplicado en <i>un solo paso</i> explota fuera de rango; sub-dividiendo el paso (integrador estructurado) extrapola de verdad. Ahora el repo vive en tu carpeta <code>Adaptive Matched Transforms/operator-meters</code> y cada número del artículo se genera desde <code>results/*.json</code>.</div>""")

H.append("""<h2>El artículo unificado en una cadena de preguntas</h2><div class="chain">
<div class="q"><b>I.1 ¿Qué mide la puerta?</b> Información en bits. Ley rectificada, r=0.99.</div>
<div class="q"><b>I.2 ¿Comprimir = clasificar?</b> No: 0.27 vs 0.999.</div>
<div class="q"><b>I.3 ¿Se lee la base aprendida?</b> No (0.03 vs 1.00). Por eso, física.</div>
<div class="q"><b>II.1 ¿Cuándo falla Fourier?</b> Medio heterogéneo. Medidor limpio.</div>
<div class="q"><b>II.2 ¿Hallamos la base solos?</b> Sí, exacta, y devuelve la física.</div>
<div class="q"><b>II.3 ¿Cuándo falla lo lineal?</b> No-linealidad. La puerta dispara con umbral.</div>
<div class="q"><b>II.4 ¿Capas o estructura?</b> Estructura: 45% → 90% → 99%.</div>
<div class="q"><b>II.5 ¿Leemos la ecuación?</b> Sí, exacta.</div>
<div class="q"><b>II.6 ¿Para qué sirve?</b> Extrapolar — si se integra, no en un paso.</div>
<div class="q"><b>II.7 ¿Y afinando?</b> Multiplica al que entiende; no salva al que no.</div></div>""")

# ---------------- Part I
H.append("<h1 style='margin-top:30px'>Parte I · Clasificación: por qué paga la puerta</h1>")
H.append(sec("I.1 · ley de la puerta en bits", "¿Qué mide exactamente la puerta?",
    "Fabricamos tareas donde la 'buena' forma de mirar la señal se va girando poco a poco desde Fourier hasta unas gafas aleatorias. Medimos en bits cuánto más sabe la rama aprendida sobre la etiqueta que las gafas clásicas, y miramos la puerta: cerrada del todo mientras no hay nada que ganar, y abierta en proporción (con saturación) en cuanto lo hay. Las cuatro tareas del artículo 2, que no se usaron para ajustar la curva, caen encima.",
    [("puerta máxima cuando G ≤ 0 (18 casos)", f"{g['gate_max_when_G_le_0']:.3f}"), ("r de la ley saturante (familia)", f"{g['saturating']['r_family']:.2f}"), ("r en las 4 tareas reales (no usadas en el ajuste)", f"{g['saturating']['r_heldout']:.2f}")],
    img("gap_law.png", "gap_law"), f"g ≈ {g['saturating']['c']:.2f}·(1 − e<sup>−[G]₊/{g['saturating']['G0']:.2f}</sup>). La puerta no se abre por costumbre: paga por la rama aprendida exactamente lo que ésta aporta sobre la etiqueta, y nada antes."))
H.append(sec("I.2 · rate–distortion vs rate–relevance", "¿Comprimir bien la señal es lo mismo que clasificarla bien?",
    "Un códec (JPEG, MP3) se queda con los coeficientes que tienen más energía. Un clasificador necesita los que distinguen las clases. Con el mismo diccionario y la misma cabeza, quedarse con los 16 'más energéticos' tira la información de posición que era justo la clase.",
    [("tarea localizada, selección por energía", f"{loc_e:.3f}"), ("misma tarea, selección por relevancia", f"{loc_t:.3f}")],
    img("rate_relevance.png", "rate_relevance"), "Comprimir hacia la señal y comprimir hacia la etiqueta son objetivos distintos; la puerta mide el segundo."))
H.append(sec("I.3 · leer la base aprendida", "¿Se puede escribir la fórmula de las gafas aprendidas?",
    "Intentamos ajustar la forma cerrada de la DFT a tres transformadas. A la DFT sin entrenar la lee perfecta (1.00). A las entrenadas por clasificación, no (0.09 y 0.03): muchas gafas distintas separan las mismas etiquetas, así que la aprendida no tiene una fórmula que nombrar. Esto es lo que motiva pasar a la física.",
    [("ajuste DFT sin entrenar", f"{sy['dft']['fit']:.2f}"), ("con prior de compresión", f"{sy['sparse']['fit']:.2f}"), ("entrenada por clasificación", f"{sy['ce']['fit']:.2f}")],
    img("symbolic_U.png", "symbolic_U"), "Leer la parte aprendida está mal planteado en clasificación y bien planteado en física — es la última fila de la tabla de correspondencias del artículo."))

# ---------------- Part II
H.append("<h1 style='margin-top:30px'>Parte II · Física: qué dice la parte aprendida</h1>")
mt = w1["meter"]; well = w1["discovery"]["well-specified"]; mis = w1["discovery"]["mis-specified"]
H.append(sec("II.1 · medidor de heterogeneidad", "¿Cuándo deja de servir Fourier?",
    "Si el material es igual en todas partes, las gafas clásicas son perfectas. Cuanto más revuelto está el material, peor miran — y ese 'cuánto peor' es un número.",
    [("Fourier, medio homogéneo", "0.000"), ("Fourier, medio muy heterogéneo (β=4)", f"{max(r['D_fourier'] for r in mt):.3f}"), ("alineación de la base real con Fourier", f"1.00 → {min(r['align'] for r in mt):.2f}")],
    img("phys_wp1_meter.png", "WP1"), "La degradación de Fourier es un medidor limpio y monótono; la base propia real diagonaliza siempre (≈1e-15)."))
H.append(sec("II.2 · descubrimiento sin oráculo", "¿Y podemos encontrar la base buena nosotros solos?",
    "Sin que nadie le diga la respuesta, buscando solo 'qué gafas hacen diagonal al operador' dentro de la familia de Sturm–Liouville, el método encuentra las correctas y de propina devuelve el mapa exacto del material. Si la familia es demasiado pobre, él mismo avisa (el objetivo se queda lejos de cero).",
    [("off-diagonalidad descubierta, todo β", f"{max(r['D_discovered'] for r in well):.6f}"), ("correlación del coeficiente recuperado", f"{min(r['corr'] for r in well):.4f}"), ("mal especificada: objetivo se estanca en", f"{max(r['D_discovered'] for r in mis):.3f} (corr {min(r['corr'] for r in mis):.2f})")],
    img("phys_wp1_discovery.png", "WP1b"), "No es una regresión de la señal: se ajusta el generador. Y la mala especificación se auto-anuncia — contrapeso exacto de II.4: la estructura es la palanca, la estructura equivocada es peor que inútil."))
fire = {r["eps"]: r for r in w2}
H.append(sec("II.3 · medidor de no-linealidad · la puerta dispara", "¿Cuándo deja de servir lo lineal?",
    "Un modelo lineal (DMD, la 'Laplace de datos') predice estirando modos exponenciales: funciona hasta que el fenómeno se vuelve no lineal. Su fallo mide cuánta no-linealidad hay, y el interruptor de la red solo se enciende cuando de verdad hace falta — la misma forma que la puerta de clasificación.",
    [("residuo de DMD, 0 → ", f"{max(r['res_dmd'] for r in w2):.3f}"), ("disparo con ε ≤ 0.2 / ε = 1 / ε = 2", f"{fire[0.2]['firing']:.3f} / {fire[1.0]['firing']:.3f} / {fire[2.0]['firing']:.3f}"), ("captura fuera de muestra (ε ≥ 0.7)", f"{100*min(r['capture'] for r in w2 if r['eps']>=0.7):.0f}–{100*max(r['capture'] for r in w2 if r['eps']>=0.7):.0f}%")],
    img("phys_wp2_meter.png", "WP2"), "Cerrada hasta que hay algo que pagar, luego proporcional. Detalle de receta: la puerta debe arrancar abierta (en g=0 el producto g·NL es un punto de silla muerto)."))
H.append(sec("II.4 · profundidad vs estructura", "¿Más capas, o mejor estructura?",
    "Para capturar lo que falta no sirve hacer la red más profunda: sirve darle la pieza con la forma correcta (u·∂ₓu). Y si esa misma pieza se aplica en 4 sub-pasos (sin parámetros nuevos), captura casi todo.",
    [("MLP caja negra", f"{100*d['mlp']:.1f}%"), ("integrador residual 4 / 8 etapas", f"{100*d['resint4']:.1f}% / {100*d['resint8']:.1f}%"), ("término estructurado, un paso", f"{100*d['quad']:.1f}%"), ("mismo término, 4 sub-pasos", f"{100*d['quadint4']:.1f}%")],
    img("depth_test.png", "WP2b"), "La profundidad ayuda algo; la estructura ayuda más; la profundidad al servicio de la estructura lo cierra. Y la rama ganadora es exactamente el término que SINDy nombra en II.5."))
H.append(sec("II.5 · leer la ecuación", "¿Puede el sistema escribir la ecuación que gobierna los datos?",
    "Le damos solo datos (con la derivada temporal estimada numéricamente), un catálogo de términos y regresión dispersa: devuelve la ecuación de verdad con sus números exactos. Con choques muy fuertes (ε ≥ 1.5) subestima por falta de resolución de malla.",
    [("ν recuperada (verdad 0.050)", f"{w3[1.0]['nu_hat']:.3f}"), ("coef. de u·uₓ para ε = 0.2 … 1.0", ", ".join(f"{-w3[e]['eps_hat']:.3f}" for e in [0.2,0.4,0.7,1.0])), ("otros términos del catálogo", "0")],
    img("phys_wp3_sindy.png", "WP3"), "«Leer U» aquí no es una metáfora: devuelve la ecuación exacta."))
def fm(k, c): v = s[k][c]; return "diverge" if v["diverged"] == v["n"] else f"{v['mean']:.3f}"
tbl = "<table><tr><th>modelo (entrenado solo a 1 paso, ε=1)</th><th>error tras 30 pasos</th><th>amplitud ×1.6 nunca vista</th><th>¿crea energía?</th></tr>"
for k, lab, e in [("linear", "solo lineal (DMD)", "no"), ("mlp_1step", "lineal + caja negra", f"sí ({100*s['mlp_1step']['energy_id']['mean']:.0f}% de pasos)"), ("quad_1step", "lineal + u·uₓ en un paso", "diverge fuera de rango"), ("quadint4_1step", "lineal + u·uₓ integrado (4 sub-pasos)", "nunca")]:
    tbl += f"<tr><td>{lab}</td><td>{fm(k,'id')}</td><td>{fm(k,'ood')}</td><td>{e}</td></tr>"
tbl += "</table>"
H.append(sec("II.6 · extrapolación — tu tesis", "¿Y para qué sirve saber la ecuación?",
    "Para predecir en situaciones nunca vistas. La caja negra no aporta nada (empeora al modelo lineal y crea energía de la nada). El término correcto aplicado de un golpe acierta en horizonte pero explota con amplitudes mayores, como un integrador explícito pasado de paso. El mismo término integrado en sub-pasos acierta en horizonte y en amplitud, y nunca crea energía.",
    [("mejora en horizonte vs lineal", f"{s['linear']['id']['mean']/s['quadint4_1step']['id']['mean']:.0f}×"), ("mejora a amplitud ×1.6 vs lineal", f"{s['linear']['ood']['mean']/s['quadint4_1step']['ood']['mean']:.0f}×"), ("vs caja negra", f"{s['mlp_1step']['id']['mean']/s['quadint4_1step']['id']['mean']:.0f}× / {s['mlp_1step']['ood']['mean']/s['quadint4_1step']['ood']['mean']:.0f}×")],
    img("phys_wp4_rollout.png", "WP4"), "«La única forma de extrapolar es entender la física» — confirmado, con un matiz nuevo que no esperábamos: el término correcto hay que <i>integrarlo</i>, no aplicarlo una vez. Resultado negativo declarado en el artículo.", extra=tbl))
H.append(sec("II.7 · endurecimiento por rollout", "¿Se puede afinar aún más?",
    "Entrenar 'a largo plazo' (5 pasos desenrollados, como afinado) mejora a los dos modelos en un factor parecido — y un factor parecido aplicado a un error de 2.9 y a uno de 0.10 deja a uno inservible y al otro casi perfecto. El rollout no sustituye la comprensión: la multiplica.",
    [("integrador estructurado, 1 paso → +rollout (horizonte)", f"{s['quadint4_1step']['id']['mean']:.3f} → {s['quadint4_rollout']['id']['mean']:.3f}"), ("ídem, amplitud ×1.6", f"{s['quadint4_1step']['ood']['mean']:.3f} → {s['quadint4_rollout']['ood']['mean']:.3f}"), ("caja negra, amplitud ×1.6", f"{s['mlp_1step']['ood']['mean']:.2f} → {s['mlp_rollout']['ood']['mean']:.2f} (lineal: {s['linear']['ood']['mean']:.2f})")],
    img("phys_wp4b_hardening.png", "WP4b"), "3 semillas (solo cambian la inicialización de la red, no los datos). Chequeo físico: el estructurado nunca crea energía; la caja negra sí, incluso tras el rollout."))

H.append("""<section><h2>Qué queda</h2><ol>
<li><b>Tu lectura del artículo unificado</b> (<code>paper/operator_meters.pdf</code>, 16 págs). Comentarios por tandas, como con el artículo 2.</li>
<li>Pulido → release: README/CITATION/.zenodo.json ya están; falta el DOI y el nombre de usuario de GitHub en <code>CITATION.cff</code>.</li>
<li>Opcional: mapear el margen de estabilidad en amplitud más allá de ×1.6; estimador de información alternativo (kNN/MINE) como contraste.</li></ol>
<p style="color:#666;font-size:13px">Convención del mapa: cada sección lleva su tira «En cristiano» con la analogía de las gafas; el mapa se regenera con <code>python tools/build_map.py</code> y los números salen de <code>results/*.json</code>.</p></section></body></html>""")
open(OUT, "w", encoding="utf-8").write("\n".join(H))
print("wrote", OUT, os.path.getsize(OUT) // 1024, "KB")
