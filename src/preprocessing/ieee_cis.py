"""
Preprocessing spécialisé pour IEEE-CIS Fraud Detection

Ce module contient toutes les étapes de préprocessing optimisées pour le dataset IEEE-CIS,
incluant la gestion des valeurs manquantes, l'encodage des variables et l'optimisation mémoire.

Author: [Votre Nom]
Date: Juin 2025
"""

import pandas as pd
import numpy as np
import gc
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class IEEEPreprocessor:
    """
    Preprocesseur spécialisé pour les données IEEE-CIS Fraud Detection.
    
    Ce preprocesseur gère les spécificités du dataset IEEE-CIS :
    - Données de transaction (train_transaction.csv)
    - Données d'identité (train_identity.csv) 
    - Variables V1-V339 avec patterns spécifiques
    - Optimisation mémoire pour gros datasets
    """
    
    def __init__(self, reduce_memory: bool = True, verbose: bool = True):
        """
        Initialise le preprocesseur IEEE-CIS.
        
        Args:
            reduce_memory: Optimiser l'usage mémoire
            verbose: Afficher les logs de progression
        """
        self.reduce_memory = reduce_memory
        self.verbose = verbose
        self.categorical_encoders = {}
        self.numerical_scalers = {}
        self.feature_names = {}
        
        # Colonnes connues du dataset IEEE-CIS
        self.transaction_cols = [
            'TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt',
            'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
            'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain'
        ]
        
        # Colonnes C1-C14 (categorical)
        self.c_cols = [f'C{i}' for i in range(1, 15)]
        
        # Colonnes D1-D15 (days)
        self.d_cols = [f'D{i}' for i in range(1, 16)]
        
        # Colonnes M1-M9 (match)
        self.m_cols = [f'M{i}' for i in range(1, 10)]
        
        # Colonnes V1-V339 (Vesta features)
        self.v_cols = [f'V{i}' for i in range(1, 340)]
        
        if self.verbose:
            print("IEEEPreprocessor initialisé")
    
    def load_data(self, 
                  transaction_path: str, 
                  identity_path: Optional[str] = None,
                  sample_frac: Optional[float] = None) -> pd.DataFrame:
        """
        Charge les données IEEE-CIS avec optimisation mémoire.
        
        Args:
            transaction_path: Chemin vers train_transaction.csv
            identity_path: Chemin vers train_identity.csv (optionnel)
            sample_frac: Fraction des données à charger (pour tests)
            
        Returns:
            DataFrame fusionné et optimisé
        """
        if self.verbose:
            print("📊 Chargement des données IEEE-CIS...")
        
        # Chargement des transactions
        if self.verbose:
            print("  ➤ Chargement train_transaction.csv...")
        
        df_trans = pd.read_csv(transaction_path)
        
        if sample_frac:
            df_trans = df_trans.sample(frac=sample_frac, random_state=42)
            if self.verbose:
                print(f"  ➤ Échantillonnage: {len(df_trans)} lignes")
        
        # Chargement des identités si disponible
        if identity_path:
            if self.verbose:
                print("  ➤ Chargement train_identity.csv...")
            
            df_identity = pd.read_csv(identity_path)
            
            if sample_frac:
                # Garder seulement les IDs présents dans l'échantillon
                df_identity = df_identity[
                    df_identity['TransactionID'].isin(df_trans['TransactionID'])
                ]
            
            # Fusion des datasets
            df = df_trans.merge(df_identity, on='TransactionID', how='left')
            
            if self.verbose:
                print(f"  ➤ Fusion réussie: {len(df)} transactions")
        else:
            df = df_trans
        
        # Optimisation mémoire
        if self.reduce_memory:
            df = self._reduce_memory_usage(df)
        
        if self.verbose:
            print(f"✅ Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
            print(f"   Fraude: {df['isFraud'].sum()} transactions ({df['isFraud'].mean():.3%})")
        
        return df
    
    def _reduce_memory_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimise l'usage mémoire en ajustant les types de données.
        
        Args:
            df: DataFrame à optimiser
            
        Returns:
            DataFrame optimisé
        """
        start_mem = df.memory_usage(deep=True).sum() / 1024**2
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float32)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        
        end_mem = df.memory_usage(deep=True).sum() / 1024**2
        
        if self.verbose:
            print(f'💾 Optimisation mémoire: {start_mem:.1f}MB -> {end_mem:.1f}MB '
                  f'({100 * (start_mem - end_mem) / start_mem:.1f}% réduction)')
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gestion intelligente des valeurs manquantes pour IEEE-CIS.
        
        Args:
            df: DataFrame avec valeurs manquantes
            
        Returns:
            DataFrame avec valeurs manquantes traitées
        """
        if self.verbose:
            print("🔧 Traitement des valeurs manquantes...")
        
        df_clean = df.copy()
        
        # Compter les NaN par colonne
        nan_counts = df_clean.isnull().sum()
        high_nan_cols = nan_counts[nan_counts > len(df_clean) * 0.9].index.tolist()
        
        if high_nan_cols and self.verbose:
            print(f"  ➤ Suppression de {len(high_nan_cols)} colonnes avec >90% NaN")
            df_clean = df_clean.drop(columns=high_nan_cols)
        
        # Traitement spécifique par type de colonne
        
        # 1. Colonnes D (days) - remplacer par -1 (souvent utilisé dans IEEE-CIS)
        for col in self.d_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(-1)
        
        # 2. Colonnes M (match) - mode ou -1
        for col in self.m_cols:
            if col in df_clean.columns:
                mode_val = df_clean[col].mode()
                fill_val = mode_val[0] if len(mode_val) > 0 else -1
                df_clean[col] = df_clean[col].fillna(fill_val)
        
        # 3. Colonnes catégorielles (card, addr, email domains)
        categorical_cols = ['card4', 'card6', 'P_emaildomain', 'R_emaildomain']
        for col in categorical_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna('unknown')
        
        # 4. Colonnes numériques - médiane
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
        
        # 5. Colonnes V - traitement spécial (souvent binaires ou catégorielles)
        for col in self.v_cols:
            if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
                # Si la colonne semble binaire
                unique_vals = df_clean[col].dropna().unique()
                if len(unique_vals) <= 2:
                    # Mode pour binaires
                    mode_val = df_clean[col].mode()
                    fill_val = mode_val[0] if len(mode_val) > 0 else 0
                    df_clean[col] = df_clean[col].fillna(fill_val)
                else:
                    # Médiane pour continues
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        if self.verbose:
            remaining_nulls = df_clean.isnull().sum().sum()
            print(f"✅ Valeurs manquantes traitées. Restant: {remaining_nulls}")
        
        return df_clean
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée des features temporelles à partir de TransactionDT.
        
        Args:
            df: DataFrame avec colonne TransactionDT
            
        Returns:
            DataFrame avec nouvelles features temporelles
        """
        if 'TransactionDT' not in df.columns:
            return df
        
        if self.verbose:
            print("⏰ Création des features temporelles...")
        
        df_time = df.copy()
        
        # TransactionDT est en secondes depuis un point de référence
        # Convertir en datetime relatif
        df_time['DT_hour'] = (df_time['TransactionDT'] / 3600) % 24
        df_time['DT_day'] = (df_time['TransactionDT'] / (3600 * 24)) % 7
        df_time['DT_week'] = (df_time['TransactionDT'] / (3600 * 24 * 7)) % 52
        df_time['DT_month'] = (df_time['TransactionDT'] / (3600 * 24 * 30)) % 12
        
        # Features cycliques (important pour la fraude)
        df_time['DT_hour_sin'] = np.sin(2 * np.pi * df_time['DT_hour'] / 24)
        df_time['DT_hour_cos'] = np.cos(2 * np.pi * df_time['DT_hour'] / 24)
        df_time['DT_day_sin'] = np.sin(2 * np.pi * df_time['DT_day'] / 7)
        df_time['DT_day_cos'] = np.cos(2 * np.pi * df_time['DT_day'] / 7)
        
        # Patterns de fraude temporels
        df_time['is_night'] = ((df_time['DT_hour'] >= 23) | (df_time['DT_hour'] <= 6)).astype(int)
        df_time['is_weekend'] = (df_time['DT_day'] >= 5).astype(int)
        df_time['is_business_hours'] = (
            (df_time['DT_hour'] >= 9) & (df_time['DT_hour'] <= 17) & (df_time['DT_day'] < 5)
        ).astype(int)
        
        if self.verbose:
            print(f"  ➤ {12} nouvelles features temporelles créées")
        
        return df_time
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode les variables catégorielles avec techniques optimisées pour la fraude.
        
        Args:
            df: DataFrame à encoder
            fit: Si True, fit les encoders. Si False, utilise les encoders existants.
            
        Returns:
            DataFrame avec variables encodées
        """
        if self.verbose:
            print("🔤 Encodage des variables catégorielles...")
        
        df_encoded = df.copy()
        
        # Variables catégorielles principales
        cat_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain']
        
        # Variables M (souvent catégorielles)
        cat_cols.extend([col for col in self.m_cols if col in df.columns])
        
        for col in cat_cols:
            if col in df.columns:
                if fit:
                    # Frequency encoding (très efficace pour la fraude)
                    freq_map = df[col].value_counts().to_dict()
                    self.categorical_encoders[f'{col}_freq'] = freq_map
                    df_encoded[f'{col}_freq'] = df[col].map(freq_map).fillna(0)
                    
                    # Label encoding simple
                    unique_vals = df[col].unique()
                    label_map = {val: idx for idx, val in enumerate(unique_vals)}
                    self.categorical_encoders[f'{col}_label'] = label_map
                    df_encoded[f'{col}_label'] = df[col].map(label_map).fillna(-1)
                    
                else:
                    # Utiliser les encoders existants
                    if f'{col}_freq' in self.categorical_encoders:
                        df_encoded[f'{col}_freq'] = df[col].map(
                            self.categorical_encoders[f'{col}_freq']
                        ).fillna(0)
                    
                    if f'{col}_label' in self.categorical_encoders:
                        df_encoded[f'{col}_label'] = df[col].map(
                            self.categorical_encoders[f'{col}_label']
                        ).fillna(-1)
        
        if self.verbose:
            print(f"  ➤ {len(cat_cols)} variables catégorielles encodées")
        
        return df_encoded
    
    def create_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée des features d'agrégation par card, email, etc.
        
        Args:
            df: DataFrame source
            
        Returns:
            DataFrame avec features d'agrégation
        """
        if self.verbose:
            print("📊 Création des features d'agrégation...")
        
        df_agg = df.copy()
        
        # Agrégations par card1 (très important pour la fraude)
        if 'card1' in df.columns:
            card1_agg = df.groupby('card1')['TransactionAmt'].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).add_prefix('card1_')
            df_agg = df_agg.merge(card1_agg, left_on='card1', right_index=True, how='left')
            
            # Ratio par rapport à la moyenne de la carte
            df_agg['amt_vs_card1_mean'] = df_agg['TransactionAmt'] / df_agg['card1_mean']
        
        # Agrégations par email domain
        for email_col in ['P_emaildomain', 'R_emaildomain']:
            if email_col in df.columns:
                email_agg = df.groupby(email_col)['TransactionAmt'].agg([
                    'count', 'mean'
                ]).add_prefix(f'{email_col}_')
                df_agg = df_agg.merge(email_agg, left_on=email_col, right_index=True, how='left')
        
        if self.verbose:
            n_new_features = len([col for col in df_agg.columns if col not in df.columns])
            print(f"  ➤ {n_new_features} features d'agrégation créées")
        
        return df_agg
    
    def preprocess_full_pipeline(self, 
                                 df: pd.DataFrame, 
                                 fit: bool = True,
                                 create_features: bool = True) -> pd.DataFrame:
        """
        Pipeline complet de preprocessing pour IEEE-CIS.
        
        Args:
            df: DataFrame à preprocesser
            fit: Si True, fit les transformateurs
            create_features: Si True, crée les nouvelles features
            
        Returns:
            DataFrame fully preprocessed
        """
        if self.verbose:
            print("🚀 Pipeline complet de preprocessing IEEE-CIS")
            print(f"   Input: {df.shape[0]} lignes, {df.shape[1]} colonnes")
        
        # 1. Gestion valeurs manquantes
        df_processed = self.handle_missing_values(df)
        
        # 2. Features temporelles
        if create_features:
            df_processed = self.create_time_features(df_processed)
        
        # 3. Encodage catégoriel
        df_processed = self.encode_categorical_features(df_processed, fit=fit)
        
        # 4. Features d'agrégation
        if create_features and 'TransactionAmt' in df.columns:
            df_processed = self.create_aggregation_features(df_processed)
        
        # 5. Nettoyage final
        # Supprimer les colonnes originales encodées
        cols_to_drop = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain']
        cols_to_drop = [col for col in cols_to_drop if col in df_processed.columns]
        if cols_to_drop:
            df_processed = df_processed.drop(columns=cols_to_drop)
        
        if self.verbose:
            print(f"✅ Pipeline terminé: {df_processed.shape[0]} lignes, {df_processed.shape[1]} colonnes")
            if 'isFraud' in df_processed.columns:
                print(f"   Distribution fraude: {df_processed['isFraud'].mean():.3%}")
        
        # Nettoyage mémoire
        gc.collect()
        
        return df_processed


# Fonctions utilitaires pour l'exploration des données
def explore_ieee_dataset(df: pd.DataFrame) -> Dict:
    """
    Exploration rapide du dataset IEEE-CIS.
    
    Args:
        df: DataFrame IEEE-CIS
        
    Returns:
        Dictionnaire avec statistiques d'exploration
    """
    stats = {
        'shape': df.shape,
        'fraud_rate': df['isFraud'].mean() if 'isFraud' in df.columns else None,
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
    }
    
    # Analyse des colonnes V
    v_cols = [col for col in df.columns if col.startswith('V')]
    stats['v_columns_count'] = len(v_cols)
    
    # Distribution des montants
    if 'TransactionAmt' in df.columns:
        stats['transaction_stats'] = {
            'mean': df['TransactionAmt'].mean(),
            'median': df['TransactionAmt'].median(),
            'max': df['TransactionAmt'].max(),
            'std': df['TransactionAmt'].std()
        }
    
    return stats


# Exemple d'utilisation
if __name__ == "__main__":
    # Test du preprocesseur
    preprocessor = IEEEPreprocessor(verbose=True)
    
    print("✅ IEEEPreprocessor prêt à l'usage!")
    print("\nUtilisation:")
    print("df = preprocessor.load_data('train_transaction.csv', 'train_identity.csv')")
    print("df_processed = preprocessor.preprocess_full_pipeline(df)")
