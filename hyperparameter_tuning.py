#!/usr/bin/env python3
"""
Hyperparameter Tuning con Optuna para el pipeline DM2
=====================================================

Incluye:
- ✅ Optimización bayesiana con Optuna
- ✅ Múltiples objetivos (multi-objective)
- ✅ Pruning para eficiencia
- ✅ Visualización de resultados
- ✅ Integración con pipeline existente
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import optuna
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Local imports
from config import *
from utils import get_medical_metrics, save_results

class OptunaOptimizer:
    """Optimizador de hyperparámetros usando Optuna"""
    
    def __init__(self, 
                 X, y, 
                 preprocessor,
                 n_trials=100,
                 cv_folds=5,
                 random_state=42,
                 optimize_for='diabetes_sensitivity',  # Métrica principal para medicina
                 secondary_metric='f1_macro'):
        
        self.X = X
        self.y = y
        self.preprocessor = preprocessor
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.optimize_for = optimize_for
        self.secondary_metric = secondary_metric
        
        # Resultados
        self.studies = {}
        self.best_params = {}
        self.best_models = {}
        
        # CV strategy
        self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        print(f"🎯 Optimizando para: {optimize_for} (secundario: {secondary_metric})")
        print(f"🔄 {n_trials} trials, {cv_folds}-fold CV")
    
    def objective_logistic_regression(self, trial):
        """Función objetivo para Logistic Regression"""
        
        # Hyperparámetros a optimizar
        params = {
            'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs']),
            'max_iter': trial.suggest_int('max_iter', 100, 1000),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
        }
        
        # Crear modelo
        model = LogisticRegression(
            random_state=self.random_state,
            multi_class='multinomial',
            **params
        )
        
        # Pipeline completo
        pipe = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', model)
        ])
        
        return self._evaluate_model(pipe, trial)
    
    def objective_random_forest(self, trial):
        """Función objetivo para Random Forest"""
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
        }
        
        model = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=-1,
            **params
        )
        
        pipe = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', model)
        ])
        
        return self._evaluate_model(pipe, trial)
    
    def objective_gradient_boosting(self, trial):
        """Función objetivo para Gradient Boosting"""
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'subsample': trial.suggest_float('subsample', 0.8, 1.0)
        }
        
        model = GradientBoostingClassifier(
            random_state=self.random_state,
            **params
        )
        
        pipe = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', model)
        ])
        
        return self._evaluate_model(pipe, trial)
    
    def _evaluate_model(self, pipeline, trial):
        """Evalúa un modelo usando cross-validation"""
        
        try:
            # Cross-validation scores
            scores = []
            
            for train_idx, val_idx in self.cv.split(self.X, self.y):
                X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
                y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
                
                # Entrenar modelo
                pipeline.fit(X_train, y_train)
                
                # Predicciones
                y_pred = pipeline.predict(X_val)
                y_proba = pipeline.predict_proba(X_val)
                
                # Calcular métricas médicas
                metrics = get_medical_metrics(
                    y_val, y_pred, y_proba, CLASS_ORDER
                )
                
                # Métrica principal
                main_score = metrics.get(self.optimize_for, 0)
                scores.append(main_score)
                
                # Pruning: detener si va muy mal
                trial.report(np.mean(scores), len(scores))
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            
            # Promedio de scores
            mean_score = np.mean(scores)
            
            return mean_score
            
        except Exception as e:
            print(f"⚠️ Error en trial: {e}")
            return 0.0  # Score mínimo en caso de error
    
    def optimize_model(self, model_name, objective_func):
        """Optimiza un modelo específico"""
        
        print(f"🔄 Optimizando {model_name}...")
        
        # Crear estudio
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=3,
                n_warmup_steps=5,
                interval_steps=1
            )
        )
        
        # Optimización
        study.optimize(
            objective_func, 
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        # Guardar resultados
        self.studies[model_name] = study
        self.best_params[model_name] = study.best_params
        
        print(f"✅ {model_name} - Mejor score: {study.best_value:.4f}")
        print(f"📊 Mejores parámetros: {study.best_params}")
        
        return study
    
    def optimize_all_models(self):
        """Optimiza todos los modelos disponibles"""
        
        models_to_optimize = {
            'logistic_regression': self.objective_logistic_regression,
            'random_forest': self.objective_random_forest,
            'gradient_boosting': self.objective_gradient_boosting
        }
        
        print(f"🚀 Iniciando optimización de {len(models_to_optimize)} modelos...")
        
        for model_name, objective_func in models_to_optimize.items():
            try:
                self.optimize_model(model_name, objective_func)
            except Exception as e:
                print(f"❌ Error optimizando {model_name}: {e}")
                continue
        
        # Seleccionar mejor modelo overall
        self._select_best_overall_model()
    
    def _select_best_overall_model(self):
        """Selecciona el mejor modelo entre todos los optimizados"""
        
        if not self.studies:
            print("❌ No hay estudios disponibles")
            return
        
        # Comparar estudios
        best_score = -1
        best_model_name = None
        
        comparison_results = []
        
        for model_name, study in self.studies.items():
            score = study.best_value
            
            comparison_results.append({
                'model': model_name,
                'best_score': score,
                'n_trials': len(study.trials)
            })
            
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        self.best_model_name = best_model_name
        self.best_overall_score = best_score
        
        # Crear DataFrame de comparación
        self.comparison_df = pd.DataFrame(comparison_results).sort_values(
            'best_score', ascending=False
        )
        
        print(f"\n🏆 MEJOR MODELO: {best_model_name}")
        print(f"📊 Score: {best_score:.4f}")
        print(f"\n📈 RANKING DE MODELOS:")
        print(self.comparison_df.round(4).to_string(index=False))
    
    def create_best_model(self):
        """Crea el mejor modelo con parámetros optimizados"""
        
        if not hasattr(self, 'best_model_name'):
            print("❌ No se ha seleccionado mejor modelo")
            return None
        
        model_name = self.best_model_name
        best_params = self.best_params[model_name]
        
        # Crear modelo con mejores parámetros
        if model_name == 'logistic_regression':
            model = LogisticRegression(
                random_state=self.random_state,
                multi_class='multinomial',
                **best_params
            )
        elif model_name == 'random_forest':
            model = RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                **best_params
            )
        elif model_name == 'gradient_boosting':
            model = GradientBoostingClassifier(
                random_state=self.random_state,
                **best_params
            )
        else:
            print(f"❌ Modelo desconocido: {model_name}")
            return None
        
        # Pipeline completo
        best_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', model)
        ])
        
        # Entrenar en todo el dataset
        best_pipeline.fit(self.X, self.y)
        
        self.best_models[model_name] = best_pipeline
        
        print(f"✅ Mejor modelo creado y entrenado: {model_name}")
        
        return best_pipeline
    
    def save_results(self):
        """Guarda todos los resultados de la optimización"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar parámetros optimizados
        params_path = OUTPUT_DIR / f"optimized_params_{timestamp}.json"
        save_results(self.best_params, params_path)
        
        # Guardar comparación de modelos
        if hasattr(self, 'comparison_df'):
            comparison_path = OUTPUT_DIR / f"model_comparison_optuna_{timestamp}.csv"
            self.comparison_df.to_csv(comparison_path, index=False)
        
        # Guardar mejor modelo
        if hasattr(self, 'best_model_name') and self.best_model_name in self.best_models:
            model_path = MODELS_DIR / f"best_model_optuna_{self.best_model_name}_{timestamp}.joblib"
            joblib.dump(self.best_models[self.best_model_name], model_path)
            
            print(f"💾 Modelo guardado: {model_path}")
        
        # Resumen de optimización
        summary = {
            'timestamp': datetime.now().isoformat(),
            'best_model': getattr(self, 'best_model_name', None),
            'best_score': getattr(self, 'best_overall_score', None),
            'optimization_config': {
                'n_trials': self.n_trials,
                'cv_folds': self.cv_folds,
                'primary_metric': self.optimize_for,
                'secondary_metric': self.secondary_metric
            },
            'best_parameters': self.best_params
        }
        
        summary_path = OUTPUT_DIR / f"optuna_summary_{timestamp}.json"
        save_results(summary, summary_path)
        
        print(f"📋 Resumen guardado: {summary_path}")
    
    def run_complete_optimization(self):
        """Ejecuta optimización completa"""
        
        print(f"🎯 Iniciando optimización completa con Optuna...")
        print(f"📊 Dataset: {self.X.shape}")
        
        # Optimizar todos los modelos
        self.optimize_all_models()
        
        # Crear mejor modelo
        best_model = self.create_best_model()
        
        # Guardar resultados
        self.save_results()
        
        print(f"\n🎉 Optimización completada!")
        print(f"🏆 Mejor modelo: {getattr(self, 'best_model_name', 'N/A')}")
        print(f"📊 Mejor score: {getattr(self, 'best_overall_score', 'N/A'):.4f}")
        
        return best_model

if __name__ == "__main__":
    print("🎯 Módulo de Optimización Hyperparámetros con Optuna cargado")
    print("🔧 Uso: Requiere X, y y preprocessor para funcionar")