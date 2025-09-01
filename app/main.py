
from fastapi import FastAPI
import pandas as pd
import numpy as np
from my_preprocessor import Preprocessor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from sqlalchemy import create_engine, text

# Connexion SQL Server
server = '192.168.10.241'
database = 'Orca BI'
# username = 'adm.nav2'
# password = 'ORCA2022'
driver = 'ODBC Driver 17 for SQL Server'

# engine = create_engine(
#     f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver}'
# )

# Authentification Windows
engine = create_engine(
    f'mssql+pyodbc://@{server}/{database}?driver={driver}&trusted_connection=yes'
)
try:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        print("✅ Connexion OK ! Résultat test :", result.scalar())
except Exception as e:
    print("❌ Erreur de connexion :", e)

app = FastAPI()
preprocessor = Preprocessor()

@app.get("/preprocess")

def preprocess_only():
    df = pd.read_sql("SELECT * FROM dbo.Data_Ai_Train;", engine)
    df_processed = preprocessor.fit(df).transform(df)
    return {"message": "Prétraitement terminé, vérifier console pour shape", "shape": df_processed.shape}

@app.get("/train")
def train_model():
    try:
        # 1️⃣ Charger les données
        df = pd.read_sql("SELECT * FROM dbo.Data_Ai_Train;", engine)

        # 2️⃣ Prétraitement
        df_processed = preprocessor.fit(df).transform(df)
        print("✅ Prétraitement terminé, shape :", df_processed.shape)

        # 3️⃣ Séparer X et y
        # Supposons que la cible est "Quantite"
        if "Quantite" not in df.columns:
            return {"error": "La colonne 'Quantite' n'existe pas dans les données"}
        
        y = df["Quantite"]
        X = df_processed.drop(columns=["Quantite"], errors="ignore")

        # 4️⃣ Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 5️⃣ Entraîner XGBoost
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)

        # 6️⃣ Prédictions
        y_pred = model.predict(X_test)

        # 7️⃣ Calcul des métriques
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # 8️⃣ Sauvegarde du modèle
        joblib.dump(model, "xgb_model.pkl")
        joblib.dump(preprocessor, "preprocessor.pkl")

        # 9️⃣ Retourner les métriques
        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }

    except Exception as e:
        print("❌ Erreur durant l'entraînement :", e)
        return {"error": str(e)}