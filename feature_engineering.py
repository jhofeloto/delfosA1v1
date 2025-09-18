#!/usr/bin/env python3
"""
Feature Engineering Avanzado para el pipeline DM2
==================================================

Incluye:
- ✅ Creación de features derivadas
- ✅ Binning inteligente
- ✅ Interacciones entre variables
- ✅ Encoding avanzado
- ✅ Selección de características
- ✅ Transformaciones no lineales
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings("ignore")

class MedicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature Engineer especializado para datos médicos de DM2
    """
    
    def __init__(self, 
                 create_interactions=True,
                 create_ratios=True,
                 create_bins=True,
                 n_interactions=10,
                 random_state=42):
        
        self.create_interactions = create_interactions
        self.create_ratios = create_ratios
        self.create_bins = create_bins
        self.n_interactions = n_interactions
        self.random_state = random_state
        
        # Variables importantes para DM2 (conocimiento médico)
        self.medical_vars = {
            'anthropometric': ['edad', 'peso', 'talla', 'imc', 'perimetro_abdominal'],
            'vital_signs': ['tas', 'tad'],
            'lifestyle': ['realiza_ejercicio', 'frecuencia_frutas', 'Consumo_Cigarrillo'],
            'clinical': ['medicamentos_hta', 'puntaje_total', 'riesgo_dm'],
            'family_history': ['Dx_Diabetes_Tipo2_Familia']
        }
        
        self.feature_names_ = []
        self.binners_ = {}
        
    def fit(self, X, y=None):
        """Aprende los parámetros para feature engineering"""
        self.feature_names_ = list(X.columns)
        
        # Identificar columnas numéricas disponibles
        self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Preparar binning para variables numéricas importantes
        if self.create_bins:
            for col in self.numeric_cols_:
                if col in ['imc', 'edad', 'tas', 'tad', 'perimetro_abdominal']:
                    # Usar binning personalizado basado en conocimiento médico
                    if col == 'imc':
                        # Categorías estándar de IMC
                        self.binners_[col] = 'custom_imc'
                    elif col == 'edad':
                        # Grupos etarios relevantes para DM2
                        self.binners_[col] = 'custom_age'
                    elif col in ['tas', 'tad']:
                        # Categorías de presión arterial
                        self.binners_[col] = 'custom_bp'
                    else:
                        # Binning automático
                        try:
                            binner = KBinsDiscretizer(
                                n_bins=min(5, len(X[col].dropna().unique())),
                                encode='ordinal',
                                strategy='quantile'
                            )
                            if len(X[col].dropna()) > 0:
                                binner.fit(X[[col]].dropna())
                                self.binners_[col] = binner
                        except:
                            pass
        
        return self
    
    def transform(self, X):
        """Aplica feature engineering"""
        X_new = X.copy()
        
        # 1. Features médicas derivadas
        X_new = self._create_medical_features(X_new)
        
        # 2. Binning inteligente
        if self.create_bins:
            X_new = self._create_binned_features(X_new)
        
        # 3. Ratios e interacciones
        if self.create_ratios:
            X_new = self._create_ratio_features(X_new)
        
        if self.create_interactions:
            X_new = self._create_interaction_features(X_new)
        
        # 4. Features categóricas derivadas
        X_new = self._create_categorical_features(X_new)
        
        return X_new
    
    def _create_medical_features(self, X):
        """Crea features basadas en conocimiento médico"""
        
        # BMI categories (si tenemos IMC)
        if 'imc' in X.columns:
            X['imc_category'] = pd.cut(
                X['imc'],
                bins=[-np.inf, 18.5, 24.9, 29.9, 34.9, np.inf],
                labels=['bajo_peso', 'normal', 'sobrepeso', 'obesidad_1', 'obesidad_2+']
            )
            
            # IMC risk score
            X['imc_risk_score'] = np.where(
                X['imc'] >= 30, 3,  # Obesidad
                np.where(X['imc'] >= 25, 2,  # Sobrepeso
                        np.where(X['imc'] >= 18.5, 1, 0))  # Normal o bajo peso
            )
        
        # Age groups relevant for diabetes
        if 'edad' in X.columns:
            X['edad_grupo'] = pd.cut(
                X['edad'],
                bins=[-np.inf, 30, 45, 60, np.inf],
                labels=['joven', 'adulto_joven', 'adulto_mayor', 'senior']
            )
            
            # Age risk (diabetes risk increases with age)
            X['edad_risk_score'] = np.where(
                X['edad'] >= 60, 3,
                np.where(X['edad'] >= 45, 2,
                        np.where(X['edad'] >= 30, 1, 0))
            )
        
        # Blood pressure categories
        if 'tas' in X.columns and 'tad' in X.columns:
            X['hipertension'] = (
                (X['tas'] >= 140) | (X['tad'] >= 90)
            ).astype(int)
            
            # Presión de pulso
            X['pulse_pressure'] = X['tas'] - X['tad']
        
        return X
    
    def _create_binned_features(self, X):
        """Crea features binneadas"""
        
        for col, binner in self.binners_.items():
            if col not in X.columns:
                continue
                
            if isinstance(binner, str):
                # Custom binning
                if binner == 'custom_imc' and 'imc' in X.columns:
                    X[f'{col}_bin'] = pd.cut(
                        X[col],
                        bins=[-np.inf, 18.5, 24.9, 29.9, np.inf],
                        labels=['bajo', 'normal', 'sobrepeso', 'obesidad']
                    )
                elif binner == 'custom_age' and 'edad' in X.columns:
                    X[f'{col}_bin'] = pd.cut(
                        X[col],
                        bins=[-np.inf, 35, 50, 65, np.inf],
                        labels=['joven', 'adulto', 'maduro', 'mayor']
                    )
            else:
                # Sklearn binner
                try:
                    X[f'{col}_bin'] = binner.transform(X[[col]].fillna(X[col].median()))
                except:
                    pass
        
        return X
    
    def _create_ratio_features(self, X):
        """Crea features de ratios médicamente relevantes"""
        
        # Ratio cintura-talla (si disponible)
        if 'perimetro_abdominal' in X.columns and 'talla' in X.columns:
            X['waist_height_ratio'] = X['perimetro_abdominal'] / X['talla']
        
        # Presión arterial media
        if 'tas' in X.columns and 'tad' in X.columns:
            X['mean_arterial_pressure'] = X['tad'] + ((X['tas'] - X['tad']) / 3)
        
        return X
    
    def _create_interaction_features(self, X):
        """Crea interacciones entre variables importantes"""
        
        important_interactions = [
            ('edad', 'imc'),
            ('edad', 'tas'),
            ('tas', 'tad')
        ]
        
        for var1, var2 in important_interactions:
            if var1 in X.columns and var2 in X.columns:
                # Solo para numéricas
                if (X[var1].dtype in ['int64', 'float64'] and 
                    X[var2].dtype in ['int64', 'float64']):
                    X[f'{var1}_x_{var2}'] = X[var1] * X[var2]
        
        return X
    
    def _create_categorical_features(self, X):
        """Crea features categóricas derivadas"""
        
        # Perfil de riesgo combinado
        risk_score = 0
        
        if 'edad' in X.columns:
            risk_score += (X['edad'] >= 60).astype(int)
        
        if 'imc' in X.columns:
            risk_score += (X['imc'] >= 30).astype(int)
        
        if 'tas' in X.columns:
            risk_score += (X['tas'] >= 140).astype(int)
        
        X['risk_profile'] = np.where(
            risk_score >= 2, 'alto_riesgo',
            np.where(risk_score >= 1, 'riesgo_medio', 'bajo_riesgo')
        )
        
        return X


class FeatureSelector:
    """Selector de características con múltiples métodos"""
    
    def __init__(self, n_features=50, random_state=42):
        self.n_features = n_features
        self.random_state = random_state
        self.selected_features_ = []
        self.feature_scores_ = {}
    
    def fit_select(self, X, y):
        """Selecciona las mejores características usando múltiples métodos"""
        
        # Solo considerar características numéricas para algunos métodos
        X_numeric = X.select_dtypes(include=[np.number])
        
        feature_importance = {}
        
        # Método 1: F-statistics (ANOVA)
        if len(X_numeric.columns) > 0:
            try:
                f_selector = SelectKBest(f_classif, k='all')
                f_selector.fit(X_numeric, y)
                
                f_scores = dict(zip(X_numeric.columns, f_selector.scores_))
                for feat, score in f_scores.items():
                    feature_importance[feat] = feature_importance.get(feat, 0) + score * 0.5
                    
                self.feature_scores_['f_classif'] = f_scores
            except:
                pass
        
        # Método 2: Random Forest feature importance
        try:
            # Preparar datos para Random Forest (manejar categóricas)
            X_rf = pd.get_dummies(X, drop_first=True)
            
            rf = RandomForestClassifier(
                n_estimators=50, 
                random_state=self.random_state,
                n_jobs=-1
            )
            rf.fit(X_rf, y)
            
            rf_scores = dict(zip(X_rf.columns, rf.feature_importances_))
            for feat, score in rf_scores.items():
                feature_importance[feat] = feature_importance.get(feat, 0) + score * 0.5
                
            self.feature_scores_['random_forest'] = rf_scores
        except:
            pass
        
        # Seleccionar top features
        if feature_importance:
            sorted_features = sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            self.selected_features_ = [feat for feat, _ in sorted_features[:self.n_features]]
        else:
            # Fallback: usar todas las features
            self.selected_features_ = list(X.columns)
        
        return self.selected_features_
    
    def transform(self, X):
        """Aplica selección de características"""
        available_features = [f for f in self.selected_features_ if f in X.columns]
        return X[available_features]

if __name__ == "__main__":
    print("🔧 Feature Engineering para DM2 - Módulo cargado")
    print("💡 Incluye: Features médicas, binning, interacciones y selección")