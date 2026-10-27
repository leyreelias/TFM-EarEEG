# Datos conductuales: revision inicial

Se revisaron los 28 psychopymarkers_data.csv originales en data/. Columnas:
timestamp, marker, notes. No se encontraron markers explicitamente identificados
como estimulo, tecla, trial o tiempo de reaccion; notes no tiene valores.
Hay Correct, Wrong_Pressed y Wrong_Not_Pressed, ademas de inicio/final de
bloques y otros markers. Esto no permite identificar de forma fiable la pareja
presentacion de estimulo/pulsacion para calcular RT. No asumir que el timestamp
de Correct equivale exactamente al de la pulsacion; verificar codigo de tarea.
Los registros originales de PsychoPy, si existen fuera de esta carpeta, podrian
contener RT o timestamps de cada estimulo y respuesta.

Los markers originales permiten contar eventos registrados por bloque/tarea,
sin duplicarlos debido al solapamiento de epochs. Antes de calcular accuracy
por ensayo, hay que verificar si se registran todos los ensayos y si Correct
incluye rechazos correctos (no pulsar ante un no-target).
La ausencia de pulsacion no equivale necesariamente a error: depende del ensayo.
Un ensayo sin respuesta no tiene RT observado; no asignarle 0 segundos.

Incluir conducta en un modelo no es falsear si se declara y evalua correctamente.
Para este TFM, mantener EEG solo como comparacion principal de rereferencias.
Una extension podria comparar conducta sola, EEG solo y ambos, con las mismas
particiones. Una mejora del modelo combinado no puede atribuirse solo al EEG.
No usar rendimiento del bloque completo en una prediccion online de un epoch
anterior al final de ese bloque. No introducir el nombre de la condicion como
feature; no derivar la etiqueta de una medida que tambien se usa como entrada.

Ejemplo de estudio que analiza accuracy y RT junto con EEG en N-back:
https://pmc.ncbi.nlm.nih.gov/articles/PMC5742112/
