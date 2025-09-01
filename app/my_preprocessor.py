# main.py
# ===============================
# IMPORTS
# ===============================
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb
import joblib
from sklearn.preprocessing import StandardScaler

# ===============================
# 1. Classe Preprocessor
# ===============================
class Preprocessor:
    def __init__(self):
        self.freq_maps = {}
        self.oe = None

    def fit(self, df):
        df = df.copy()
        # Encodage catégoriel
        df["Sector"] = df["Sector"].fillna("Unknown")
        df["Article_Family"] = df["Article_Family"].fillna("Unknown")
        self.oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[["Sector", "Article_Family"]] = self.oe.fit_transform(df[["Sector", "Article_Family"]])

        # Frequency encoding
        for col in ["Article_ID", "Client_ID", "Customer_Name"]:
            if col in df.columns:
                self.freq_maps[col] = df[col].value_counts().to_dict()

        

        return self

    def transform(self, df):
        df = df.copy()
        # Encodage
        df["Sector"] = df["Sector"].fillna("Unknown")
        df["Article_Family"] = df["Article_Family"].fillna("Unknown")
        df[["Sector", "Article_Family"]] = self.oe.transform(df[["Sector", "Article_Family"]])

        # Frequency encoding
        for col in ["Article_ID", "Client_ID", "Customer_Name"]:
            if col in df.columns and col in self.freq_maps:
                df[col + "_freq"] = df[col].map(self.freq_maps[col]).fillna(0)

        # Features temporelles
        if "Month" in df.columns:
            df["Month_sin"] = np.sin(2 * np.pi * df["Month"]/12)
            df["Month_cos"] = np.cos(2 * np.pi * df["Month"]/12)
            def month_to_season(m):
                if m in [12,1,2]: return 0
                elif m in [3,4,5]: return 1
                elif m in [6,7,8]: return 2
                else: return 3
            df["Season"] = df["Month"].apply(month_to_season)

        # Rolling features
        if "Article_ID" in df.columns and "Quantite" in df.columns:
            df = df.sort_values(["Article_ID", "Year", "Month"])
            df["Quantite_roll3"] = df.groupby("Article_ID")["Quantite"].transform(lambda x: x.rolling(3, min_periods=1).mean())
            df["Quantite_roll6"] = df.groupby("Article_ID")["Quantite"].transform(lambda x: x.rolling(6, min_periods=1).mean())
            df["Quantite_diff"] = df.groupby("Article_ID")["Quantite"].diff().fillna(0)
            df["Quantite_pct_change"] = df.groupby("Article_ID")["Quantite"].pct_change().fillna(0)
        else:
            df["Quantite_roll3"] = 0
            df["Quantite_roll6"] = 0
            df["Quantite_diff"] = 0
            df["Quantite_pct_change"] = 0

        # Nettoyage NaN / inf
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Suppression colonnes inutiles
        drop_cols = ['Article_ID', 'Client_ID', 'Date_ID', 'Customer_Name',
                     'Article_Description', 'Day', 'Quarter', 'Month',
                     'Chiffre_Affaires', 'Marge_V', 'Prix_Vente_Unitaire']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        

        return df

preprocessor = Preprocessor()

# Sauvegarder le préprocesseur pour réutilisation
joblib.dump(preprocessor, "preprocessor.pkl")
