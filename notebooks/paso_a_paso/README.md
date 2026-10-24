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
| 04 | Exploración no supervisada | ¿Normal y patológico se separan solos? (PCA, k-means) |
| 05 | Métricas | Matriz de confusión, sensibilidad, especificidad, F1, curva ROC y AUC: qué son y por qué se usan |
| 06 | Validación | Por grabación (principal) y por caso (comparación) |
| 07 | Modelo a fondo: regresión logística | Cómo funciona, qué es `C`, cómo leer los coeficientes |
| 08 | Modelo a fondo: random forest | Del árbol al bosque, `max_depth`, `n_estimators`, importancia |
| 09 | El resto de los modelos y el desbalance | SVM, boosting, voting, MLP por encima; `class_weight` vs SMOTE |
| 10 | Comparación | Preprocesamientos × modelos: ¿qué importa más? |
| 11 | Mezclas | ¿El preprocesamiento ayuda cuando suenan corazón y pulmón juntos? |
| 12 | ICBHI 2017 | Validación externa en pacientes reales (pulmonar) |
