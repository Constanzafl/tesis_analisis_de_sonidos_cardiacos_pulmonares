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
| 11 | Preprocesamiento | Filtros candidatos y correcciones pendientes (4000 Hz, tramas del borde, ritmo): ¿qué cambia en cada modelo? |
| 12 | Mezclas | ¿El preprocesamiento ayuda cuando suenan corazón y pulmón juntos? |
| 13 | ICBHI 2017 | Validación externa en pacientes reales (pulmonar) |

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
features de ritmo (notebook 04) en la notebook 11.

## Hallazgo: AUC alto con especificidad casi nula (notebook 09)

En cardíacos, el random forest de la tesis **ordena bien** (AUC ~0,8), pero les
da a **todos** los normales una probabilidad mayor que 0,5: con el umbral 0,5,
la especificidad es 0,07–0,11. Con árboles completos (`min_samples_leaf=1`) las
hojas son puras y `class_weight` casi no actúa. Comparar solo por AUC escondía
este problema. Para la notebook 11, fijadas de antemano: `min_samples_leaf`
mayor que 1, SMOTE, o un umbral elegido dentro de la validación cruzada.

## Correcciones pendientes para los archivos finales

- Cargar a la frecuencia nativa (4000 Hz), ajustando el tamaño de trama en
  tiempo (p. ej. 256 muestras = 64 ms). A 22.050 Hz la mitad de las bandas mel
  y del espectro miran una zona sin sonido (notebooks 01 y 03).
- Descartar las tramas del borde antes de calcular máximos y mínimos
  (notebook 03, sección 9.1).
- La planitud espectral es ~0 a 22.050 Hz y no aporta información (sección 9.2).
- Modelos: las tres correcciones de la notebook 10 (XGBoost con
  `scale_pos_weight`, LightGBM con `min_child_samples=5`, MLP sin
  `early_stopping`) y el puntaje de los SVM en lugar de sus probabilidades.

## Preprocesamientos candidatos (se comparan en la notebook 11)

Fijados de antemano, con argumentos de la señal y clínicos (notebook 02), para
no elegir "el que mejor dé" entre muchos.

| Candidato | Cardíacos (HS) | Pulmonares (LS) | Por qué |
|---|---|---|---|
| Sin filtro | — | — | Referencia (lo que usa la tesis hoy) |
| Limpieza mínima | pasa banda 20–1000 Hz + notch 146 Hz | pasa banda 20–1000 Hz + notch 146 Hz | Saca solo lo no fisiológico: retumbo profundo, siseo > 1100 Hz y el tono del equipo |
| Banda clínica | pasa banda 25–400 Hz + notch 146 Hz | pasa banda 100–1000 Hz + notch 146 Hz | HS: rango habitual en fonocardiograma (verificar cita). LS: atenúa sonido cardíaco y retumbo |

Cada candidato se prueba con y sin normalización de amplitud (RMS), para
medir cuánto del resultado depende del volumen.
