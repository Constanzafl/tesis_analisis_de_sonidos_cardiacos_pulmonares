# Paso a paso

Notebooks para entender y decidir cada parte del trabajo, de a una. Lo que se
decida acá se pasa después a los `.py` finales de la tesis. El código de
`notebooks/` y `src/` queda como referencia y no se toca.

Cada notebook termina con **preguntas de control**: si las podés responder con
tus palabras, se pasa a la siguiente.

| # | Notebook | Pregunta que responde |
|---|---|---|
| 01 | El audio | ¿Qué es una grabación digital y dónde está la energía de un sonido cardíaco y uno pulmonar? |
| 02 | Preprocesamiento | ¿Qué hace un filtro (pasa bajo, pasa banda), la normalización y el remuestreo? |
| 03 | Features | ¿Qué mide cada característica y cómo 105 números resumen un audio? |
| 04 | Features de ritmo | Detectar latidos y medir la regularidad de los intervalos: ¿ayuda con las arritmias? |
| 05 | Exploración no supervisada | ¿Normal y patológico se separan solos? (PCA, k-means) |
| 06 | Métricas | Matriz de confusión, sensibilidad, especificidad, F1, curva ROC y AUC: qué son y por qué se usan |
| 07 | Validación | Por grabación (principal) y por caso (comparación) |
| 08 | Modelo a fondo: regresión logística | Cómo funciona, qué es `C`, cómo leer los coeficientes |
| 09 | Modelo a fondo: random forest (de consulta) | Del árbol al bosque; por qué tiene buen AUC y mala especificidad en cardíacos |
| 10 | Todos los modelos: la línea de base | Los 9 modelos en pocas palabras, `class_weight` vs SMOTE, las mismas métricas para todos, por tipo de patología y control de pseudo-casos |
| 11 | Correcciones de la extracción | 4000 Hz, tramas del borde y planitud: ¿qué cambia en cada modelo? Es la nueva línea de base |
| 12 | Filtros y normalización | Los filtros candidatos, con y sin normalización de volumen: ¿ayudan? ¿El resultado depende del volumen? |
| 13 | Features de ritmo | ¿Sumar el ritmo (notebook 04) a las features espectrales mejora las arritmias? |
| 14 | Umbral y ajustes del modelo | Umbral elegido dentro de la validación, `min_samples_leaf` del bosque y `C` de la regresión logística |
| 15 | Mezclas | ¿El preprocesamiento ayuda cuando suenan corazón y pulmón juntos? |
| 16 | ICBHI 2017 | Validación externa en pacientes reales (pulmonar) |

**Resultados para el documento:** `resultados_tesis.ipynb` reproduce las tablas
y figuras nuevas de la versión 5 de la tesis (capítulos 6 y 9: los nueve
modelos con la extracción corregida, la regresión logística en detalle, el AUC
por tipo y los pseudo-casos). Las figuras quedan en `outputs/figuras_tesis/` y
las tablas en `outputs/tesis_*.csv`. Las tablas y figuras del capítulo 7 salen
de las notebooks 11 y 12, y las del umbral, de la 14.

Las notebooks 12, 13 y 14 se comparan **cada una contra la línea de base de la
11**, para medir cada cambio por separado (una cosa por vez). Desde la 11, los
modelos son cuatro, elegidos por su tipo: regresión logística, SVM-RBF, random
forest y XGBoost, con `class_weight`.

**Criterio transversal:** además del resultado global normal/patológico, se
reporta **por tipo de patología** (qué fracción de cada tipo se detecta). La
clase "patológico" mezcla fenómenos de timbre (soplos, ruidos agregados), de
baja frecuencia (R3, R4) y de ritmo (arritmias), y un método puede servir para
unos y no para otros.

## El dataset viene prefiltrado (Torabi et al., 2025, Tabla I)

| Subconjunto | Modo del estetoscopio (3M Littmann CORE) | Respuesta |
|---|---|---|
| HS | Campana (*Bell*) | 20–200 Hz; caída de 20 dB/octava por encima de 300 Hz |
| LS | Diafragma | 100–500 Hz; caída de 10 dB/octava por debajo de 200 Hz |
| Mix | *Midrange* | 50–500 Hz; refuerzo de +20 dB en 550 Hz |

Además: cancelación activa de ruido (85 %, 20–500 Hz) y un pasa banda por
software posterior (cortes no informados). El equipo graba a 22.050 Hz, pero
los archivos publicados están a 4000 Hz (verificado en los 535).

Consecuencias:
- La "banda de energía" de cada subconjunto (notebook 01) refleja sonido +
  filtro del equipo, no solo fisiología.
- HS y LS no se grabaron con el mismo filtro: parte de la diferencia de
  desempeño entre subconjuntos puede deberse al modo (la campana atenúa > 300 Hz,
  donde está parte de algunos soplos).
- Las mezclas se grabaron en otro modo: el desplazamiento de dominio limpio →
  mezcla (+20 dB, otro espectro) se explica en buena parte por eso, no solo por
  enmascaramiento.
- El notch de 146 Hz y la normalización de volumen no están cubiertos por el
  prefiltrado del paper.
- Para ICBHI (validación externa) hay que imitar el modo diafragma
  (~100–500 Hz) y llevar a 4000 Hz.

## Hallazgo sobre el "optimismo" (notebooks 07, 09 y 10)

La diferencia "AUC por grabación − AUC por caso" no es solo fuga: también
refleja el esquema de validación (dejar afuera grupos enteros de una clase).
Un control con pseudo-casos al azar (mismo tamaño y clase, sin ser "hermanos")
separa las dos cosas:

| Modelo | | Por grabación | Pseudo-caso (control) | Caso real |
|---|---|---|---|---|
| Regresión logística | Pulmonares | 0,998 | 0,991 (0,97–1,00) | 0,967 → fuga real, chica (~0,02) |
| Regresión logística | Cardíacos | 0,711 | 0,621 (0,39–0,76) | 0,610 → sin fuga medible; la caída es del esquema |
| Random forest | Pulmonares | 0,949 | 0,842 (0,77–0,88) | 0,757 → fuga real grande (~0,09) |
| Random forest | Cardíacos | 0,785 | 0,569 (0,43–0,74) | 0,645 → sin fuga medible; caída grande por el esquema |

- El bosque con árboles completos se aprovecha más de la grabación "hermana"
  (hojas puras ≈ copiar a la grabación más parecida). En pulmonares, la idea de
  que "memoriza el caso" se sostiene.
- En cardíacos hay solo dos casos normales (F_N: 6 grabaciones, M_N: 3). Al
  dejar uno afuera se entrena con 3 o 6 normales y la escala de probabilidades
  de los normales sube: el AUC cae sin que haya memorización.

En la notebook 10 se aplicó el mismo control a los 9 modelos (versión
corregida, `class_weight`, 5 sorteos):
- **Pulmonares:** en casi todos, el caso real queda por debajo de todos los
  pseudo-casos. La fuga es chica en la regresión logística (~0,02), intermedia
  en el SVM-RBF, el Voting y el MLP (~0,04) y en el SVM lineal (~0,07), y
  grande en el random forest (~0,10) y XGBoost (~0,11).
- **Cardíacos:** ningún modelo muestra fuga medible; las caídas al validar por
  caso vienen del esquema.

## Hallazgo: tres configuraciones de la tesis que no hacían lo que parecían (notebook 10)

| Modelo | Problema | Corrección | Efecto |
|---|---|---|---|
| XGBoost | No acepta `class_weight`; en las filas "class_weight" entrenaba sin balanceo | `scale_pos_weight = normales / patológicos` | Especificidad de 0,58 a 0,75 (LS) y de 0,33 a 0,59 (HS), AUC parecido |
| LightGBM | `min_child_samples=20` (por defecto): con ~40 grabaciones, árboles de una sola pregunta | `min_child_samples=5` | AUC de 0,87 a 0,90 (LS) y de 0,69 a 0,77 (HS) |
| MLP | `early_stopping=True` aparta ~6 grabaciones y para a las 12–14 vueltas: casi no entrena | sin `early_stopping` | AUC de 0,44 a 0,94 (LS) y de 0,65 a 0,77 (HS) |

Además, para los SVM se usa su puntaje (`decision_function`) y no las
probabilidades de Platt, que con tan pocos datos se distorsionan.

## Hallazgo: las arritmias quedan cerca del azar con todos los modelos (notebook 10)

AUC de cada tipo de patología contra los normales: los soplos sistólicos
precoces y tardíos y el diastólico tardío se distinguen bien (0,7–1,0 con la
mayoría de los modelos), pero la fibrilación auricular (0,38–0,70) y el
bloqueo AV (0,48–0,59) quedan cerca del azar con casi todos. Las features
actuales no ven el ritmo, y ningún modelo lo compensa: hay que sumar las
features de ritmo (notebook 04) en la notebook 13.

## Hallazgo: AUC alto con especificidad casi nula (notebook 09)

En cardíacos, el random forest de la tesis **ordena bien** (AUC ~0,8), pero les
da a **todos** los normales una probabilidad mayor que 0,5: con el umbral 0,5,
la especificidad es 0,07–0,11. Con árboles completos (`min_samples_leaf=1`) las
hojas son puras y `class_weight` casi no actúa. Comparar solo por AUC escondía
este problema. Para la notebook 14, fijadas de antemano: `min_samples_leaf`
mayor que 1, SMOTE, o un umbral elegido dentro de la validación cruzada.

## Hallazgo: la extracción corregida cambia el panorama cardíaco (notebook 11)

Las tres correcciones de la extracción (cargar a 4000 Hz con tramas de la
misma duración, sin relleno en los bordes, y la planitud que se arregla sola)
se midieron con un diseño 2 × 2 y los cuatro modelos. AUC por grabación:

| Modelo | Pulmonares: tesis → corregida | Cardíacos: tesis → corregida |
|---|---|---|
| Regresión logística | 1,00 → 1,00 | 0,68 → **0,83** (mejora real: +0,15, IC pareado +0,03 a +0,27) |
| SVM-RBF | 0,98 → 0,98 | 0,73 → 0,83 (al borde: IC −0,01 a +0,24) |
| Random forest | 0,95 → 0,93 | 0,79 → 0,82 |
| XGBoost | 0,93 → 0,87 | 0,82 → 0,75 (al borde, y frágil: con tramas de 64 ms vuelve a 0,84) |

- **Pulmonares:** el resultado de la tesis no dependía de estos errores.
- **Cardíacos:** los modelos lineales pasan de los peores a estar arriba; la
  regresión logística queda con la mejor exactitud balanceada (0,78,
  especificidad 0,70). Sigue sin haber un ganador claro (0,75–0,83, intervalos
  superpuestos). La mejora viene sobre todo de cargar a 4000 Hz; el R3 y el
  soplo mesosistólico son los tipos que más mejoran.
- **Los máximos y mínimos de MFCC "importantes" (notebooks 08 y 09) no eran el
  relleno:** `mfcc_5_max`, `mfcc_4_max` y `mfcc_5_min` salen del borde en el
  2 % de las grabaciones y separan igual sin relleno. Los contaminados eran
  otros (`mfcc_3_min` en el 94 %, `mfcc_4_min` en el 78 %, `mfcc_1_max` en el
  70 %), y eran ruido: sacarlos ayuda a la regresión logística, que usa todas
  las features, y casi nada a los árboles, que eligen.
- **Fuga por caso (pseudo-casos):** en pulmonares sigue siendo chica en la
  regresión logística (~0,03); en los modelos de árboles se achica y ya no es
  medible. En cardíacos, la regresión logística validada por caso pasa de 0,61
  a 0,73.
- La línea de base corregida quedó en `outputs/hs_features_corregidas.csv` y
  `outputs/ls_features_corregidas.csv` (101 features).
- Para comparar versiones se usa desde ahora el **bootstrap pareado** (las
  mismas grabaciones, la diferencia de AUC con su intervalo).

## Hallazgo: filtros y normalización (notebook 12)

Los tres candidatos fijados de antemano, con y sin normalización RMS, sobre la
extracción corregida (40 comparaciones con bootstrap pareado; se leen los
patrones que se repiten en los cuatro modelos):
- **Cardíacos: la banda clínica de 25–400 Hz empeora a los cuatro modelos**
  (el random forest, de 0,82 a 0,66). Los patológicos tienen más energía entre
  400 y 1000 Hz (la fracción de energía entre 600 y 1000 Hz es la que más
  separa sola, 0,80) y el filtro la borra; pierden sobre todo los soplos. Es una
  banda pensada para **segmentar** (encontrar R1 y R2), no para **clasificar**.
- **Barrido del pasa bajo cardíaco** (curva de sensibilidad, no para elegir):
  con cortes de 100–200 Hz, todos los modelos quedan cerca del azar; el AUC sube
  hasta ~600 Hz y desde ahí queda plano, al nivel de no filtrar.
- **Pulmonares: la banda clínica (100–1000 Hz) con normalización** no cambia la
  regresión logística (~1,00) y mejora los modelos de árboles (random forest de
  0,93 a 0,99, XGBoost de 0,87 a 0,96).
- **El resultado pulmonar no depende del volumen:** el volumen solo separa
  0,78; con banda clínica + RMS ninguna feature queda asociada al volumen
  original (|r| > 0,8) y el AUC no cae. Además, la fuga por caso deja de ser
  medible y la regresión logística validada por caso sube de 0,96 a 0,99.
- Por encima de ~1100 Hz solo hay piso de ruido del equipo, y en pulmonares
  está más alto en los patológicos (≈ −79 contra −85 dB): una pista de
  "condición de grabación" (notebook 05) que la banda clínica saca.
- Recomendación: pulmonares con banda clínica + RMS (por robustez, no por
  AUC); cardíacos sin filtro (normalizar es indiferente).

## Hallazgo: las features de ritmo simples no ayudan (notebook 13)

Sumar las 5 features de ritmo de la notebook 04 a las 101 espectrales
corregidas no cambia el AUC de ningún modelo (diferencias de −0,002 a +0,009,
todos los intervalos pareados incluyen el 0). Por qué:
- **Direcciones opuestas:** el modelo aprende la dirección de la mayoría de
  los patológicos (más eventos por segundo, intervalos más parejos); el
  bloqueo AV va para el otro lado, y con el ritmo solo la regresión logística
  lo ordena al revés (AUC 0,16). La taquicardia, en cambio, queda perfecta
  (1,00).
- **El detector simple no encuentra R1 y R2:** un mismo caso normal del
  maniquí da frecuencias estimadas de 32 a 72 lpm (`F_N`) o de 41 a 115 lpm
  (`M_N`) según el foco. Una feature motivada por la semiología (variabilidad
  de la intensidad de R1, probada una sola vez) tampoco separa: los normales
  salen más variables (0,53) que las fibrilaciones (0,36).
- **Muy pocas arritmias:** 4 fibrilaciones auriculares, 3 bloqueos AV y 3
  taquicardias.

Para la tesis: las arritmias son la principal limitación del clasificador
cardíaco. Trabajo futuro: segmentar R1 y R2 (Springer et al., 2016) para medir
los intervalos R1–R1 y la intensidad de cada R1.

## Hallazgo: el umbral elegido dentro de la validación (notebook 14)

El umbral no cambia el AUC, pero sí la sensibilidad y la especificidad. Se
eligió **dentro** de la validación (validación cruzada interna sobre el
entrenamiento de cada partición) con dos criterios fijados de antemano:
Youden (equilibrio) y tamizaje (sensibilidad de al menos 0,90).

| Cardíacos | Umbral 0,5 (sens. / esp.) | Youden | Tamizaje |
|---|---|---|---|
| Regresión logística | 0,86 / 0,70 | 0,79 / 0,70 | 0,89 / 0,56 |
| SVM-RBF | 0,93 / 0,41 | **0,79 / 0,89** | 0,88 / 0,70 |
| Random forest | 0,98 / **0,00** | 0,74 / 0,63 | 0,93 / 0,41 |
| XGBoost | 0,83 / 0,41 | 0,69 / 0,59 | 0,92 / 0,22 |

- El random forest cardíaco pasa de no reconocer ningún normal a reconocer
  0,63 con Youden, con el mismo AUC. De los tres remedios fijados en las
  notebooks 09 y 10, el umbral es el mejor: las hojas de 5 (especificidad 0,37
  con 0,5) bajan un poco el AUC (0,82 → 0,79, baja chica pero real) y SMOTE
  (0,33) ayuda menos.
- En pulmonares, la regresión logística con el criterio de tamizaje da
  sensibilidad 0,96 y especificidad 1,00.
- `C = 0,1` (notebook 08) no cambia el AUC ni la fuga por caso.
- Con 9 normales cardíacos, cada normal vale 0,11 de especificidad: las
  diferencias entre modelos son de una o dos grabaciones.
- Para la tesis: reportar, además del AUC, la sensibilidad y la especificidad
  con un umbral elegido dentro de la validación, aclarando el criterio.

## Correcciones pendientes para los archivos finales

- **Extracción** (implementada y medida en la notebook 11, función `extraer`):
  cargar a 4000 Hz (la frecuencia nativa), tramas de 372 muestras con saltos de
  93 (93 y 23 ms, como la tesis), `center=False` en todas las features (sin
  relleno en los bordes) y `n_bands=4` en el contraste espectral (5 bandas
  hasta 2000 Hz). La planitud se corrige sola. Quedan 101 features.
- Modelos: las tres correcciones de la notebook 10 (XGBoost con
  `scale_pos_weight`, LightGBM con `min_child_samples=5`, MLP sin
  `early_stopping`) y el puntaje de los SVM en lugar de sus probabilidades.

## Preprocesamientos candidatos (se comparan en la notebook 12)

Fijados de antemano, con argumentos de la señal y clínicos (notebook 02), para
no elegir "el que mejor dé" entre muchos.

| Candidato | Cardíacos (HS) | Pulmonares (LS) | Por qué |
|---|---|---|---|
| Sin filtro | — | — | Referencia (la línea de base de la notebook 11) |
| Limpieza mínima | pasa banda 20–1000 Hz + notch 146 Hz | pasa banda 20–1000 Hz + notch 146 Hz | Saca solo lo no fisiológico: retumbo profundo, siseo > 1100 Hz y el tono del equipo |
| Banda clínica | pasa banda 25–400 Hz + notch 146 Hz | pasa banda 100–1000 Hz + notch 146 Hz | HS: la banda del método de segmentación de Springer, Tarassenko y Clifford (IEEE TBME, 2016), la referencia del desafío PhysioNet/CinC 2016. LS: atenúa sonido cardíaco y retumbo; el corte de 100 Hz es nuestro (las guías CORSA usan cortes del orden de 60 Hz en el registro) |

Cada candidato se prueba con y sin normalización de amplitud (RMS), para
medir cuánto del resultado depende del volumen.

## Decisiones propuestas para la versión final (a confirmar con la directora)

Salen de las notebooks 11 a 14; ninguna se eligió por ser "la que mejor dio".
- **Extracción corregida** (notebook 11) para todo.
- **Pulmonares:** pasa banda 100–1000 Hz + notch 146 Hz + normalización RMS
  (notebook 12), por robustez: el resultado deja de depender del volumen y de
  la firma de la grabación, sin perder desempeño.
- **Cardíacos:** sin filtro (la banda de 25–400 Hz empeora, notebook 12) y sin
  las features de ritmo simples (no aportan, notebook 13).
- **Modelos:** los cuatro elegidos por tipo, con `class_weight`; la regresión
  logística como modelo principal para explicar.
- **Umbral:** elegido dentro de la validación, con un criterio clínico fijado
  de antemano (tamizaje o Youden: falta decidir cuál, notebook 14).
