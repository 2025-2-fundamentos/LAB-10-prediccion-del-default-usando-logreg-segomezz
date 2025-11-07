  # %% [markdown]
#  ## Paso 1
#  
#  flake8: noqa: E501
# 
#  En este dataset se desea pronosticar el default (pago) del cliente el próximo
#  mes a partir de 23 variables explicativas.
# 
#    LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el credito familiar (suplementario).
# 
#    SEX: Genero (1=male; 2=female).
# 
#    EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
# 
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#     
#    AGE: Edad (years).
# 
# 
#        PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#        PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#        PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#        PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#        PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#        PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
# 
#     BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#     BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#     BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#     BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#     BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#     BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
# 
#    
#     PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#     PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#     PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#     PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#     PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#     PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
# 
#  La variable "default payment next month" corresponde a la variable objetivo.
# 
#  El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
#  en la carpeta "files/input/".
# 
#  Los pasos que debe seguir para la construcción de un modelo de
#  clasificación están descritos a continuación.
# 
# 
#  Paso 1.
#  Realice la limpieza de los datasets:
#  - Renombre la columna "default payment next month" a "default".
#  - Remueva la columna "ID".
#  - Elimine los registros con informacion no disponible.
#  - Para la columna EDUCATION, valores > 4 indican niveles superiores
#    de educación, agrupe estos valores en la categoría "others".

# %%
from sklearn.metrics import accuracy_score, classification_report,precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score,ParameterGrid, GridSearchCV
from itertools import product
import gzip
import joblib
import json
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler




def carga_limpieza():
    # Leer los datasets descomprimidos
    train_raw = pd.read_csv("files/input/train_default_of_credit_card_clients.csv")
    test_raw = pd.read_csv("files/input/test_default_of_credit_card_clients.csv")

    # Crear copias para evitar modificar los originales
    train_dataset = train_raw.copy()
    test_dataset = test_raw.copy()

    # --Renombrar columna objetivo
    train_dataset.rename(columns={"default payment next month": "default"}, inplace=True)
    test_dataset.rename(columns={"default payment next month": "default"}, inplace=True)

    # --Remover columna "ID"
    train_dataset.drop(columns="ID", inplace=True)
    test_dataset.drop(columns="ID", inplace=True)

    # --Eliminar registros con información no disponible
    train_dataset.dropna(inplace=True)
    train_dataset.drop_duplicates(inplace=True)
    test_dataset.dropna(inplace=True)
    test_dataset.drop_duplicates(inplace=True)

    # --Agrupar valores de EDUCATION > 4 en categoría "others" (5)
    train_dataset.loc[train_dataset["EDUCATION"] > 4, "EDUCATION"] = 5
    test_dataset.loc[test_dataset["EDUCATION"] > 4, "EDUCATION"] = 5

    

    # ⚙️ Transformación logarítmica en variables de montos
    #monto_cols = [
     #   "LIMIT_BAL",
      #  "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
       # "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
    #]

    #for col in monto_cols:
        # Reemplazar valores negativos por 0 antes del log
        #train_dataset[col] = np.where(train_dataset[col] < 0, 0, train_dataset[col])
        #test_dataset[col] = np.where(test_dataset[col] < 0, 0, test_dataset[col])

        # Aplicar log(1 + x)
        #train_dataset[col] = np.log1p(train_dataset[col])
        #test_dataset[col] = np.log1p(test_dataset[col])

    return train_dataset, test_dataset

# %% [markdown]
# ## Paso 2
# 
#  Divida los datasets en x_train, y_train, x_test, y_test.
# 
# 
# 
# 

# %%
# Separar características (X) y variable objetivo (y)

def Division_Datasets(train_dataset,test_dataset):
    x_train = train_dataset.drop(columns=["default"])
    y_train = train_dataset["default"]

# Características y variable objetivo para prueba
    x_test = test_dataset.drop(columns=["default"])
    y_test = test_dataset["default"]
    

    return x_train,y_train,x_test,y_test


# %% [markdown]
# 
# 
# # Paso 3.
#  Cree un pipeline para el modelo de clasificación. Este pipeline debe
#  contener las siguientes capas:
#  - Transforma las variables categoricas usando el método
#    one-hot-encoding.
#  - Ajusta un modelo de bosques aleatorios (rando forest).
# 

# %%
# Asegúrate de que las columnas categóricas sean cadenas


def build_pipeline(x_train,x_test,estimator):

    categorical_columns = ["SEX", "EDUCATION", "MARRIAGE"]
    for col in categorical_columns:
        x_train[col] = x_train[col].astype(int)
        x_test[col] = x_test[col].astype(int)

     #Definir las categorías esperadas basadas en los datos únicos
    sex_categories = sorted(x_train["SEX"].unique().tolist())
    education_categories = sorted(x_train["EDUCATION"].unique().tolist()) 
    marriage_categories = sorted(x_train["MARRIAGE"].unique().tolist())

    numeric_columns = [col for col in x_train.columns if col not in categorical_columns]
    # Crear el preprocesador para las variables categóricas
    # OPCIÓN 1: Sin especificar categorías (más flexible)
    preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(
            categories=[sex_categories, education_categories, marriage_categories],
            handle_unknown="ignore",
            sparse_output=False
        ), categorical_columns),
        ("num", MinMaxScaler(), numeric_columns),
    ],
    remainder="drop"
)
    selectkbest = SelectKBest(score_func=f_classif)

    pipeline = Pipeline(
        steps=[
            ("tranformer", preprocessor),
            ("selectkbest", selectkbest),
            ("estimator", estimator),
        ],
        verbose=False,
    )

    return pipeline  

# %% [markdown]
# ## Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
#  Use 10 splits para la validación cruzada. Use la función de precision
#  balanceada para medir la precisión del modelo.

# %%
def Grid_Search(pipeline,x_train,y_train):

    # Grid de hiperparámetros más amplio
    param_grid = {
    "selectkbest__k": [5],
    "estimator__C": [5,6],
    "estimator__penalty": ["l1", "l2",'elasticnet', None],
    "estimator__class_weight": [
        None,
        {0:1, 1:1.3},
        
    ]
}



    grid_search = GridSearchCV(
    pipeline,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=10,
        n_jobs=-1,
        verbose=2  #
    )
    grid_search.fit(x_train, y_train)

        # Resultados
    print("Mejores hiperparámetros encontrados:")
    print(grid_search.best_params_)
    print(f"Mejor score (balanced_accuracy): {grid_search.best_score_:.4f}")

    return grid_search

# %% [markdown]
# ## Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
#  Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
# 
# 
# 

# %%

def save_model_as_gzip(model, filepath):
    """
    Guarda un modelo como un archivo comprimido con gzip.
    
    Args:
        model: El modelo a guardar (por ejemplo, un objeto GridSearchCV).
        filepath: Ruta del archivo donde se guardará el modelo.
    """
    # Crear el directorio si no existe
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Guardar el modelo comprimido
    with gzip.open(filepath, "wb") as f:
        pickle.dump(model, f)

def load_model(filepath = 'files/models', name = 'model.pkl.gz'):
    import os
    import gzip
    import pickle

    model_path = os.path.join(filepath, name)

    if not os.path.exists(model_path):
        return None
    with gzip.open(model_path, 'rb') as f:
        model = pickle.load(f)

    return model


# %% [markdown]
# ## Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
#  y f1-score para los conjuntos de entrenamiento y prueba.
#  Guardelas en el archivo files/output/metrics.json. Cada fila
#  del archivo es un diccionario con las metricas de un modelo.
#  Este diccionario tiene un campo para indicar si es el conjunto
#  de entrenamiento o prueba. Por ejemplo:
# 
#  {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
#  {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
# 

# %%
def calculate_and_save_metrics(model, x_train, y_train, x_test, y_test, output_path):
    """
    Calcula las métricas para los conjuntos de entrenamiento y prueba y las guarda en un archivo JSONL.

    Args:
        model: Modelo entrenado.
        x_train: Datos de entrenamiento (features).
        y_train: Etiquetas de entrenamiento.
        x_test: Datos de prueba (features).
        y_test: Etiquetas de prueba.
        output_path: Ruta del archivo JSON donde se guardarán las métricas.
    """


    y_train_pred = model.best_estimator_.predict(x_train)
    y_test_pred = model.best_estimator_.predict(x_test)

    # Diccionarios con métricas
    train_metrics = {
        "type": "metrics",
        "dataset": "train",
        "precision": float(precision_score(y_train, y_train_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_train, y_train_pred)),
        "recall": float(recall_score(y_train, y_train_pred,zero_division=0 )),
        "f1_score":float( f1_score(y_train, y_train_pred, zero_division=0))
    }
    test_metrics = {
        "type": "metrics",
        "dataset": "test",
        "precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_test_pred)),
        "recall": float(recall_score(y_test, y_test_pred,zero_division=0 )),
        "f1_score":float( f1_score(y_test, y_test_pred, zero_division=0))
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Guardar en formato JSONL (una línea por dict)
    with open(output_path, "w") as f:
        f.write(json.dumps(train_metrics) + "\n")
        f.write(json.dumps(test_metrics) + "\n")

    print(f"✅ Métricas guardadas en {output_path} (JSONL)")


# %% [markdown]
# 
# ## Paso 7.
#  Calcule las matrices de confusion para los conjuntos de entrenamiento y
#  prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
#  del archivo es un diccionario con las metricas de un modelo.
#  de entrenamiento o prueba. Por ejemplo:
# 
#  {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
#  {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}

# %%
def calculate_and_save_confusion_matrices(model, x_train, y_train, x_test, y_test, output_path):
    """
    Calcula matrices de confusión para train/test y las guarda en formato JSONL.
    """

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    train_cm_dict = {
        "type": "cm_matrix",
        "dataset": "train",
        "true_0": {"predicted_0": int(cm_train[0, 0]), "predicted_1": int(cm_train[0, 1])},
        "true_1": {"predicted_0": int(cm_train[1, 0]), "predicted_1": int(cm_train[1, 1])}
    }
    test_cm_dict = {
        "type": "cm_matrix",
        "dataset": "test",
        "true_0": {"predicted_0": int(cm_test[0, 0]), "predicted_1": int(cm_test[0, 1])},
        "true_1": {"predicted_0": int(cm_test[1, 0]), "predicted_1": int(cm_test[1, 1])}
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "a") as f:  # 👈 append para no sobrescribir las métricas
        f.write(json.dumps(train_cm_dict) + "\n")
        f.write(json.dumps(test_cm_dict) + "\n")

    print(f"✅ Matrices de confusión guardadas en {output_path} (JSONL)")


# %% [markdown]
# ## Flujo Final
# 
# 

# %%
def Flujo_Final():
    train_dataset,test_dataset=carga_limpieza()
    x_train,y_train,x_test,y_test=Division_Datasets(train_dataset,test_dataset)

    pipeline=build_pipeline(x_train,x_test,estimator=LogisticRegression(max_iter=1000,solver='lbfgs',random_state=123))

    model=Grid_Search(pipeline,x_train,y_train)
    
    save_model_as_gzip(model, "files/models/model.pkl.gz")

    calculate_and_save_metrics(model, x_train, y_train, x_test, y_test, "files/output/metrics.json")

    calculate_and_save_confusion_matrices(model, x_train, y_train, x_test, y_test, "files/output/metrics.json")

Flujo_Final()
    

