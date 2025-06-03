import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor 
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
from sklearn.model_selection import cross_val_score


def crear_y_entrenar_bagging_regressor(datacorr, columnas_features, columna_target, path_guardado_modelo, path_guardado_preprocesador):
    print(f"\n--- Entrenando Modelo ---")
    df = pd.read_csv("./dataset/yield_proc.csv")
    X = df[columnas_features]
    y = df[columna_target]

    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    random_states = 456
    scores = []


    X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state= random_states,
        )

    # Identificar tipos de columnas 
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(include='object').columns.tolist()
    
    print(f"Features numericas para preprocesar: {numeric_features}")
    print(f"Features categoricas para preprocesar: {categorical_features}")

    # crear transformadores de preprocesamiento
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocesador = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' 
    )

    model = BaggingRegressor(
        estimator=DecisionTreeRegressor(
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=4
        ),

        n_estimators= 120,
        max_samples= 0.8,
        max_features= 0.8,
        random_state=42,   
        n_jobs=-1,          
        oob_score=True      
    )

    cv_scores = cross_val_score(
        Pipeline([('preprocessor', preprocesador), ('regressor', model)]),
        X, y, 
        cv=5,
        scoring='r2'
    )
    print(f"\nCross-validation R² scores: {cv_scores}")
    print(f"Mean CV R²: Varianza de {cv_scores.mean():.4f} STD: (+/- {cv_scores.std() * 2:.4f})") # vamos a esperar una estimacion general del rendimiento del modelo

    pipeline_completo = Pipeline(steps=[('preprocessor', preprocesador),
                                        ('regressor', model)])

    print("Entrenando pipeline completo...")
    pipeline_completo.fit(X_train, y_train)
    score = pipeline_completo.score(X_val, y_val)
    scores.append(score)

    print(f"Promedio: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

    score_val = pipeline_completo.score(X_val, y_val) 
    print(f"R^2 en conjunto de validación: {score_val:.4f}")
    if hasattr(pipeline_completo.named_steps['regressor'], 'oob_score_') and pipeline_completo.named_steps['regressor'].oob_score_:
         print(f"Puntuación Out-of-Bag del BaggingRegressor: {pipeline_completo.named_steps['regressor'].oob_score_:.4f}")
    try:
        joblib.dump(pipeline_completo, path_guardado_modelo)
        print(f"Pipeline completo guardado en: {path_guardado_modelo}")
    except Exception as e:
        print(f"Error al guardar el pipeline: {e}")

    return pipeline_completo

def predecir_rendimiento_usuario(datos_usuario_df, path_modelo_cargado):
    print(f"\n--- Prediccion ---")
    if not isinstance(datos_usuario_df, pd.DataFrame):
        print("Error: los datos del usuario deben ser un DataFrame.")
        return None

    try:
        pipeline_cargado = joblib.load(path_modelo_cargado)
        print(f"Pipeline cargado desde: {path_modelo_cargado}")
    except FileNotFoundError:
        print(f"Error: Archivo de modelo no encontrado en {path_modelo_cargado}")
        return None
    except Exception as e:
        print(f"Error al cargar el pipeline: {e}")
        return None
    try:
        predicciones = pipeline_cargado.predict(datos_usuario_df)
        print(f"Predicciones: {predicciones}")
        return predicciones
    except Exception as e:
        print(f"Error durante la predicción: {e}")
        print("Verifica que los datos de entrada tengan las columnas correctas y tipos de datos esperados por el preprocesador.")
        return None

if __name__ == "__main__":
    df = pd.read_csv("./dataset/yield_proc.csv")

    df_principal = df
    df_principal.dropna(subset=['hg/ha_yield'], inplace=True) 

    columnas_features_seleccionadas = ['Area', 'Year', 'Item_x', 'rain_per_year', 'avg_temp']
    columna_target_seleccionada = 'hg/ha_yield'
    
    ruta_modelo_guardado = "./bagging_yield_pipeline.joblib"
    ruta_preprocesador_guardado = "./yield_preprocessor.joblib" 
    



    # Entrenar y guardar el modelo
    pipeline_entrenado = crear_y_entrenar_bagging_regressor(
        df_principal,
        columnas_features_seleccionadas,
        columna_target_seleccionada,
        ruta_modelo_guardado,
        ruta_preprocesador_guardado
    )

    if pipeline_entrenado: 
        datos_nuevos_usuario = pd.DataFrame({
            'Area': ['Argentina', 'Uruguay'],
            'Item_x': ['Maize', 'Soybeans'],
            'Year': [2007, 2013],
            'rain_per_year': [657, 1100],
            'avg_temp': [23.5, 25]
        })
        
        datos_nuevos_usuario = datos_nuevos_usuario[columnas_features_seleccionadas]


        predicciones_usuario = predecir_rendimiento_usuario(datos_nuevos_usuario, ruta_modelo_guardado)

        if predicciones_usuario is not None:
            print("\nResultados para el usuario:")
            for i, pred in enumerate(predicciones_usuario):
                print(f"  Predicción para entrada {i+1}: {pred:.2f} hg/ha")