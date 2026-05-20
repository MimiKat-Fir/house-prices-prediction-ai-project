# House Prices Prediction AI Project

Proyecto colaborativo de Machine Learning para analizar datos de viviendas y construir un modelo de prediccion de precios.

## Estructura del repositorio

```text
.
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   └── data_description.txt
├── notebooks/
│   └── house-prices-prediction-using-tfdf_our_notebook.ipynb
├── .gitignore
└── README.md
```

## Archivos principales

- `data/train.csv`: datos de entrenamiento.
- `data/test.csv`: datos para generar predicciones.
- `data/sample_submission.csv`: formato esperado para la entrega.
- `data/data_description.txt`: descripcion de las variables del dataset.
- `notebooks/house-prices-prediction-using-tfdf_our_notebook.ipynb`: notebook principal del proyecto.

## Como empezar

Clona el repositorio:

```bash
git clone https://github.com/MimiKat-Fir/house-prices-prediction-ai-project.git
cd house-prices-prediction-ai-project
```

Abre el notebook desde Jupyter, VS Code o Google Colab:

```text
notebooks/house-prices-prediction-using-tfdf_our_notebook.ipynb
```

## Flujo recomendado para trabajar en equipo

Antes de empezar a cambiar archivos:

```bash
git pull
```

Despues de hacer cambios:

```bash
git status
git add .
git commit -m "Describe los cambios"
git push
```

## Recomendaciones

- Evitad editar el mismo notebook al mismo tiempo para reducir conflictos.
- Si alguien va a hacer cambios grandes, avisad al grupo antes.
- Guardad nuevos notebooks dentro de `notebooks/`.
- Guardad nuevos datasets o archivos de entrada dentro de `data/`.
