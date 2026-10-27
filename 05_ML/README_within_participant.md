# Analisis exploratorio por participante

`02_within_participant.py` lee exclusivamente `data_features`. No modifica
preprocesado, epochs, features ni resultados anteriores.

Protocolo: 1_BACK=0, 3_BACK=1; una LR por persona/metodo, cinco folds externos
por grabaciones completas y GridSearch de tres folds internos tambien por
grabaciones. Misma asignacion externa de sesiones para los cinco metodos,
definida con all_mean. No es una evaluacion prospectiva (entrenar pasado y
predecir futuro), ni una CV entre personas. Se mantienen sesiones completas
para evitar compartir muestras EEG solapadas entre train y test.

Pipeline: mediana -> ANOVA k=50 -> StandardScaler -> LR balanceada.
C=[0.01, 0.1, 1, 10, 100], penalty=[l1,l2], solver=liblinear.
Cada transformacion se ajusta en el train del fold correspondiente.
Las columnas completamente vacias en train se conservan como constantes
con keep_empty_features=True para mantener la correspondencia de nombres.
No hay normalizacion por grabacion ni transformaciones log/logit adicionales.

Ejecutar desde D:\TFM_clean:

```powershell
.venv/Scripts/python.exe 05_ML/02_within_participant.py
```

`--audit-only --output results/ml_within_participant_audit` valida sesiones,
clases y separacion de grupos sin entrenar. `--methods all_mean` o
`--participants participant_01` permiten limitar la ejecucion.

Salidas en results/ml_within_participant_lr:

- config.json y split_manifest.csv: configuracion y sesiones de cada fold.
- results_per_fold.csv y summary.csv: metricas y media/DE entre folds.
- predictions.csv: predicciones externas con identificadores y performance.
- selected_features.csv: 50 variables seleccionadas y coeficientes por fold.
- feature_stability.csv: frecuencia de seleccion; una variable seleccionada
  puede tener coeficiente cero con L1. Coeficiente positivo apunta a 3_BACK.
- performance_descriptive.csv: fraccion de predicciones correctas por persona,
  condicion y categoria de performance. only_correct exige al menos un evento
  y todos Correct; contains_error incluye cualquier error; no_event es aparte.

Los epochs aceptados pueden diferir entre rereferencias. Comparar sesiones
comunes no garantiza comparar exactamente los mismos epochs. Los resultados
describen cada pipeline con su rechazo propio, como en el analisis existente.
La DE entre folds es descriptiva, no un intervalo de confianza. Los folds
comparten datos de entrenamiento. Las dos personas no permiten generalizar
a una poblacion ni concluir causalmente que una mejora se debe al sujeto.
Los identificadores participant deben representar a las mismas personas
fisicas entre sesiones para interpretar este analisis como intraindividual.

Performance se analiza despues de predecir, sin seleccionar solo 3_BACK
correctos para entrenar: eso cambiaria la poblacion objetivo y seleccionaria
una clase de forma distinta. En 3_BACK la fraccion correcta es sensibilidad,
no balanced accuracy. Un evento puede aparecer en dos epochs: no se realizan
tests de significacion ni se cuentan estos epochs como eventos independientes.

La version 01_claude.py escribe en la misma carpeta que 01_1back_vs_3back.py
y guarda modelos en los mismos nombres. Los CSV existentes tienen formato
de CV anidada, no el test unico del script original. No se modifican aqui.

NORMALIZE_PER_GROUP=True en 01_claude.py calcula media/DE de toda la grabacion
incluida la de test y todas sus condiciones. Es adaptacion offline al test,
no escalado aprendido solo con train. No se incluye en esta prueba principal.
Un futuro contraste True/False debe identificarse como analisis adicional
offline y mantener iguales las demas decisiones para aislar su efecto.

## Comparacion con PSD fino (1-45 Hz)

```powershell
.venv/Scripts/python.exe 05_ML/02_within_participant.py --input-dir data_features_psd --psd-max 45 --output results/ml_within_participant_psd_1_45_lr
.venv/Scripts/python.exe 05_ML/03_compare_feature_sets.py
```

Se excluyen las columnas con limite superior mayor de 45 Hz antes del ML.
Quedan 44 bins por canal (1-2 hasta 44-45 Hz): 704, 2816 o 2464 variables
candidatas segun el numero de derivaciones; se seleccionan 50 dentro de train.
Las particiones externas siguen definidas con los datos originales all_mean,
y las internas usan el mismo algoritmo y semilla. El cargador comprueba que
los epochs, etiquetas y limites de muestras coincidan con las features
originales. El comparador verifica ademas las asignaciones externas y cada
epoch de test, y guarda diferencias pareadas por fold sin tests inferenciales.

Es una comparacion exploratoria sobre los mismos folds ya examinados, no una
validacion confirmatoria nueva. Los CSV de extraccion no se modifican y los
resultados anteriores se conservan en su carpeta.
