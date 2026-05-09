# Proyecto TPI - Procesamiento de Lenguaje Natural (PLN)
**Sistema de Recomendación y Evaluación de Búsqueda de Películas**

Este proyecto es un Trabajo Práctico Integrador (TPI) enfocado en la evaluación rigurosa de algoritmos de recuperación de información (Information Retrieval). El objetivo principal es comparar el rendimiento de una búsqueda tradicional basada en palabras clave (BM25) frente a una búsqueda semántica basada en embeddings densos, y finalmente, evaluar un modelo híbrido que combina ambos enfoques.

El sistema utiliza un corpus de películas combinando catálogos de Netflix, Amazon Prime, Disney+ y Hulu. En lugar de utilizar metadatos aislados, se generan "narrativas de contenido" utilizando f-strings en Python para que los modelos semánticos capturen mejor el contexto gramatical de cada película.

## 📁 Arquitectura del Proyecto

El proyecto está diseñado de forma modular para separar el procesamiento de datos, la lógica del motor de búsqueda y el pipeline de evaluación académica (generación de *qrels* y métricas).

- `01_data_processing.ipynb`: (Opcional) Se encarga de cargar los datasets originales, eliminar películas duplicadas, generar las narrativas y calcular los embeddings vectoriales. Para evitar que tengas que recalcular esto en tu máquina, los resultados ya se encuentran exportados en el repositorio dentro del archivo `movies_with_embeddings.pkl`.
- `search_engine.py`: Módulo central de Python que contiene la lógica de los tres motores de búsqueda:
  - `BM25Searcher`: Búsqueda *sparse* utilizando la librería `rank_bm25` y tokenización con `nltk`.
  - `DenseSearcher`: Búsqueda *dense* mediante similitud de coseno utilizando los embeddings generados.
  - `HybridSearcher`: Búsqueda híbrida que fusiona los resultados de BM25 y Dense utilizando el algoritmo **Reciprocal Rank Fusion (RRF)**.
- `02_pooling_generation.ipynb`: Implementa la técnica de *Pooling* (común en conferencias como TREC). Toma una lista de consultas de prueba, extrae el Top 5 de resultados únicos de los tres motores y exporta un archivo `qrels_to_grade.csv` para que un humano pueda realizar el etiquetado manual de relevancia.
- `03_evaluation.ipynb`: Es el motor de evaluación final. Lee el archivo CSV calificado manualmente y calcula las métricas matemáticas para probar qué algoritmo es superior. Las métricas utilizadas son:
  - **NDCG@5 (Normalized Discounted Cumulative Gain):** Evalúa la relevancia en una escala gradual (0 a 5).
  - **Precision@5 y MRR (Mean Reciprocal Rank):** Evalúa la relevancia de forma binaria, utilizando un umbral de corte configurable (ej. puntajes >= 3 se consideran relevantes).

## 🚀 Cómo Ejecutar el Proyecto

1. **Instalar Dependencias:**
   Asegúrate de tener instaladas las librerías necesarias listadas en `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

2. **Procesar Datos y Embeddings (OPCIONAL):**
   El archivo `movies_with_embeddings.pkl` (que contiene los datos limpios y los vectores precalculados) ya se encuentra incluido en el repositorio. Por lo tanto, **no es necesario** que ejecutes el notebook `01_data_processing.ipynb` a menos que modifiques los CSVs originales o desees reconstruir los embeddings desde cero. ¡Puedes saltar directamente al paso 3!

3. **Generar el Pool de Evaluación:**
   Abre `02_pooling_generation.ipynb`. Puedes modificar la lista `TEST_QUERIES` con las consultas que desees probar. Ejecuta el notebook para generar el archivo `qrels_to_grade.csv`.

4. **Etiquetado Manual (Ground Truth):**
   Abre el archivo `qrels_to_grade.csv` generado y completa la columna `relevance` con un valor del `0` al `5` (donde 0 es completamente irrelevante y 5 es una sugerencia perfecta). Guarda el archivo.

5. **Evaluar Resultados:**
   Abre y ejecuta `03_evaluation.ipynb`. El notebook calculará automáticamente las métricas NDCG, MRR y Precisión, mostrando una tabla comparativa final con el rendimiento de BM25, Dense Search y Hybrid Search.