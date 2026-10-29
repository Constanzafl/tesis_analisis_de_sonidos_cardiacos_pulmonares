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
| 09 | Modelo a fondo: random forest | Del árbol al bosque, `max_depth`, `n_estimators`, importancia |
| 10 | El resto de los modelos y el desbalance | SVM, boosting, voting, MLP por encima; `class_weight` vs SMOTE |
| 11 | Comparación | Preprocesamientos × features × modelos: ¿qué importa más? |
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

## Correcciones pendientes para los archivos finales

- Cargar a la frecuencia nativa (4000 Hz), ajustando el tamaño de trama en
  tiempo (p. ej. 256 muestras = 64 ms). A 22.050 Hz la mitad de las bandas mel
  y del espectro miran una zona sin sonido (notebooks 01 y 03).
- Descartar las tramas del borde antes de calcular máximos y mínimos
  (notebook 03, sección 9.1).
- La planitud espectral es ~0 a 22.050 Hz y no aporta información (sección 9.2).

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
