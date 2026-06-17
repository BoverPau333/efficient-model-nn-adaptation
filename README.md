# Efficient Model NN Adaptation

Repositorio asociado a un Trabajo Fin de Grado centrado en el estudio de métodos eficientes para adaptar redes neuronales tras modificar el conjunto de clases de un problema de clasificación de imágenes.

El objetivo principal del proyecto es comparar distintas estrategias de adaptación cuando se elimina o se añade una clase, intentando reducir el coste de reentrenamiento sin perder demasiado rendimiento. Para ello se analizan diferentes métodos en términos de precisión, tiempo de adaptación, olvido, número de ejemplos utilizados y número de parámetros entrenados.

## Descripción del proyecto

En muchos escenarios reales, los modelos de clasificación deben adaptarse cuando cambia el conjunto de clases. Por ejemplo, puede ser necesario eliminar una categoría, incorporar una nueva clase o reajustar el modelo sin volver a entrenarlo desde cero.

Este proyecto estudia diferentes formas de realizar esa adaptación de manera eficiente. En lugar de utilizar siempre un reentrenamiento completo, se comparan métodos que reutilizan el conocimiento ya aprendido por el modelo base.

Los escenarios principales estudiados son:

- Eliminación de clases.
- Adición de nuevas clases.
- Reducción del conjunto de entrenamiento.
- Selección de ejemplos guiada por similitud entre clases.
- Adaptación mediante aprendizaje few-shot con prototipos.

## Métodos evaluados

Los métodos principales incluidos en el repositorio son:

- **Baseline**: reentrenamiento completo del modelo tras modificar el conjunto de clases.
- **Fine-tuning con backbone congelado**: solo se entrena la cabecera de clasificación.
- **Fine-tuning en dos fases**: primero se entrena la cabecera y después se permite actualizar también el backbone.
- **Fine-tuning selectivo guiado por distancia**: se seleccionan subconjuntos de datos usando similitud entre clases calculada mediante embeddings.
- **Few-shot con prototipos**: se clasifican las muestras usando prototipos de clase en el espacio de embeddings, sin reentrenar el modelo.

## Datasets y modelos

El proyecto trabaja con distintos conjuntos de datos de clasificación de imágenes:

- CIFAR-10.
- Fashion-MNIST.
- Fruits-360.

También se evalúan varias arquitecturas convolucionales preentrenadas, como:

- ResNet18.
- EfficientNet-B0.
- MobileNetV3-Small.

## Métricas de evaluación

Los experimentos comparan los métodos utilizando varias métricas:

- Accuracy global.
- Accuracy por clase.
- Accuracy de la clase añadida, en el escenario de adición.
- Olvido sobre las clases previas o restantes.
- Tiempo de adaptación.
- Número de ejemplos utilizados.
- Número de parámetros entrenados.
- Número de épocas necesarias.

## Estudio de similitud entre clases

Una parte importante del proyecto consiste en analizar cómo medir la similitud entre clases. Para ello se comparan distintas representaciones internas del modelo:

- Embeddings extraídos del backbone.
- Logits de la cabecera de clasificación.

También se comparan distintas métricas de distancia, especialmente:

- Distancia coseno.
- Distancia euclídea.
- Variantes normalizadas.

El objetivo de este análisis es identificar clases cercanas y utilizar esa información para seleccionar ejemplos más relevantes durante la adaptación.

## Estructura del repositorio

```text
efficient-model-nn-adaptation/
│
├── experiments/
│   ├── baseline_addition_after_class_introduction.py
│   ├── baseline_retrain_after_class_removal.py
│   ├── finetuning_retrain_after_class_removal.py
│   ├── frozen_backbone_head_retrain_after_class_removal.py
│   ├── retrain_prototypical_fewshot.py
│   ├── embedding_distance_comparison.py
│   ├── logits_vs_embeddings_distance_comparison.py
│   └── ...
│
├── scripts/
│   ├── baseline_addition.sh
│   ├── baseline_elimination.sh
│   ├── similarity_visualization.sh
│   └── ...
│
├── src/
│   ├── adaptation/
│   │   ├── class_addition_experiment_utils.py
│   │   ├── class_removal_experiment_utils.py
│   │   ├── dynamic_dataset_selection.py
│   │   ├── dynamic_finetuning_utils.py
│   │   ├── finetuning_schedule_utils.py
│   │   └── prototypical_utils.py
│   │
│   ├── analysis/
│   │   ├── class_addition_method_comparison.py
│   │   ├── class_addition_percentage_analysis.py
│   │   ├── class_removal_analysis.py
│   │   ├── class_removal_method_comparison.py
│   │   └── ...
│   │
│   ├── core/
│   │   ├── class_distance.py
│   │   ├── distancias.py
│   │   ├── embedding_utils.py
│   │   ├── embeddings.py
│   │   ├── experiment_utils.py
│   │   ├── reduction.py
│   │   ├── results_utils.py
│   │   ├── training.py
│   │   └── visualization.py
│   │
│   ├── dataset/
│   │   └── ...
│   │
│   ├── experiments_config/  
│   │   └── ...
│   │
│   ├── config.py
│   ├── models.py
│   ├── metricas_embeddings.py
│   ├── metrics_addition.py
│   └── metrics_elimination.py
│
├── environment.yml
├── LICENSE
└── README.md
````

## Organización general

La carpeta `src/` contiene el código reutilizable del proyecto. Incluye la carga de datasets, definición de modelos, entrenamiento, extracción de embeddings, cálculo de distancias, métricas y utilidades de visualización.

La carpeta `experiments/` contiene los experimentos principales. Cada archivo ejecuta una parte concreta del estudio: baseline, fine-tuning, selección por distancia, few-shot, comparación entre embeddings y logits, generación de gráficas o resumen de resultados.

La carpeta `scripts/` contiene lanzadores `.sh` para ejecutar los experimentos de forma más sencilla y reproducible en el servidor.

## Instalación

El entorno del proyecto se define mediante `environment.yml`.

```bash
git clone https://github.com/BoverPau333/efficient-model-nn-adaptation.git
cd efficient-model-nn-adaptation

conda env create -f environment.yml
conda activate tfg
```

Las dependencias principales incluyen:

* Python 3.10.
* PyTorch.
* Torchvision.
* NumPy.
* Pandas.
* Matplotlib.
* Seaborn.
* Scikit-learn.

## Ejecución de experimentos

Los experimentos pueden ejecutarse directamente desde Python o mediante los scripts incluidos en la carpeta `scripts/`.

Ejemplos:

```bash
# Entrenamiento de modelos base
bash scripts/full_training_reference_imagenet.sh

# Eliminación de clases
bash scripts/baseline_elimination.sh

# Adición de clases
bash scripts/baseline_addition.sh

# Fine-tuning selectivo guiado por distancia
bash scripts/dynamic_embedding_finetuning.sh
bash scripts/dynamic_embedding_finetuning_addition.sh

# Few-shot con prototipos
bash scripts/prototypical_fewshot.sh
bash scripts/prototypical_fewshot_addition.sh

# Estudio de similitud entre clases
bash scripts/embedding_distance_comparison.sh
```

## Flujo experimental

El flujo general del proyecto es el siguiente:

1. Entrenar modelos base sobre los datasets seleccionados.
2. Analizar el comportamiento inicial del modelo y detectar clases conflictivas.
3. Extraer embeddings y logits para estudiar la similitud entre clases.
4. Comparar métricas de distancia para seleccionar clases cercanas.
5. Ejecutar experimentos de eliminación de clases.
6. Ejecutar experimentos de adición de clases.
7. Comparar métodos de adaptación en términos de accuracy, tiempo, olvido, ejemplos y parámetros.
8. Generar tablas y gráficas para analizar los resultados.

## Resultados generales

Los resultados obtenidos muestran que el reentrenamiento completo no siempre es la opción más eficiente. Aunque sirve como referencia, suele requerir mucho tiempo y no siempre mejora a los métodos que reutilizan el conocimiento previo.

El fine-tuning permite reducir de forma importante el coste de adaptación. El entrenamiento solo de la cabecera es rápido y conserva bien el conocimiento previo, mientras que el fine-tuning en dos fases ofrece mejores resultados cuando se permite actualizar también el backbone.

La selección guiada por distancia entre clases permite usar menos datos de forma más informativa. El uso de embeddings junto con distancia coseno resulta especialmente útil para detectar clases relacionadas y construir subconjuntos de entrenamiento más relevantes.

El método few-shot con prototipos es la alternativa más rápida, ya que no requiere reentrenamiento. Sin embargo, en adición de clases puede tener dificultades para aprender correctamente la nueva categoría cuando se dispone de pocos ejemplos.

## Conclusión

Este repositorio proporciona una base experimental para estudiar cómo adaptar redes neuronales de clasificación cuando cambia el conjunto de clases. La comparación entre métodos muestra que es posible reducir considerablemente el coste de adaptación reutilizando modelos previamente entrenados, seleccionando mejor los datos y aplicando estrategias de aprendizaje con pocos ejemplos.

