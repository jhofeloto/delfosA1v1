#!/usr/bin/env python3
"""
Pipeline Mejorado para Predicción de Estado Glucémico (DM2)
============================================================

Mejoras implementadas:
- ✅ Rutas relativas y configuración externa
- ✅ Técnicas de balancing de clases
- ✅ Métricas médicas específicas
- ✅ Validación cruzada robusta
- ✅ Manejo de errores y logging
- ✅ Reportes detallados
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

# ML imports
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

# Local imports
from config import *
from utils import get_medical_metrics, save_results, load_and_validate_data, generate_detailed_report

class DM2Pipeline:
    """Pipeline completo para predicción de DM2 con mejoras implementadas"""
    
    def __init__(self, config_override=None):
        self.config = config_override or {}
        self.results = {}
        self.best_model = None
        self.feature_names = []
        
        # Setup logging
        self.setup_logging()
        
        print("🚀 Iniciando Pipeline DM2 Mejorado...")
        print(f"📅 Timestamp: {datetime.now()}")
        
    def setup_logging(self):
        """Configura logging básico"""
        self.log_file = OUTPUT_DIR / f"dm2_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
    def log(self, message: str, level: str = "INFO"):
        """Log simple a archivo y consola"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        print(log_entry)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + "\n")
    
    def load_data(self) -> bool:
        """Carga y valida los datos"""
        self.log("Cargando dataset...")
        
        if not DATA_PATH.exists():
            self.log(f"❌ Dataset no encontrado: {DATA_PATH}", "ERROR")
            return False
            
        self.df, is_valid = load_and_validate_data(DATA_PATH, TARGET)
        
        if not is_valid:
            return False
            
        self.log(f"✅ Dataset cargado: {self.df.shape[0]} filas, {self.df.shape[1]} columnas")
        return True
    
    def prepare_features(self):
        """Prepara features y target con exclusiones apropiadas"""
        self.log("Preparando features...")
        
        # Filtrar columnas existentes
        blacklist = [col for col in BLACKLIST_COLS if col in self.df.columns]
        pii = [col for col in PII_COLS if col in self.df.columns]
        
        # Selección automática por tipo
        num_cols = self.df.select_dtypes(include=[np.number]).columns.difference(blacklist + pii).tolist()
        cat_cols = self.df.select_dtypes(include=["object"]).columns.difference(blacklist + pii).tolist()
        
        self.log(f"📊 Variables numéricas: {len(num_cols)}")
        self.log(f"📊 Variables categóricas: {len(cat_cols)}")
        
        # Remover filas con target faltante
        initial_size = len(self.df)
        self.df = self.df.dropna(subset=[TARGET]).copy()
        final_size = len(self.df)
        
        if initial_size != final_size:
            self.log(f"🧹 Removidas {initial_size - final_size} filas con target faltante")
        
        # Preparar matrices
        self.X = self.df[num_cols + cat_cols].copy()
        self.y = self.df[TARGET].astype("category")
        self.y = self.y.cat.set_categories(CLASS_ORDER)
        
        # Guardar nombres de columnas
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        
        self.log(f"✅ Features preparadas: X{self.X.shape}, y{self.y.shape}")
        
    def create_preprocessor(self):
        """Crea pipeline de preprocesamiento"""
        self.log("Creando preprocesador...")
        
        # Preprocessing steps
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop=None))
        ])
        
        self.preprocessor = ColumnTransformer([
            ('num', numeric_transformer, self.num_cols),
            ('cat', categorical_transformer, self.cat_cols)
        ], remainder='drop')
        
        self.log("✅ Preprocesador creado")
    
    def create_models(self):
        """Crea modelos con diferentes estrategias de balancing"""
        self.log("Creando modelos...")
        
        # Calcular class weights
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(self.y), 
            y=self.y
        )
        class_weight_dict = dict(zip(np.unique(self.y), class_weights))
        
        self.log(f"📊 Class weights: {class_weight_dict}")
        
        self.models = {}
        
        # Modelo 1: Logistic Regression con class weights
        logreg_weighted = LogisticRegression(
            max_iter=1000,
            multi_class='multinomial',
            class_weight='balanced',
            random_state=RANDOM_STATE
        )
        
        # Modelo 2: Gradient Boosting con class weights
        # Nota: GBC no soporta class_weight directamente para multiclass
        gbc = GradientBoostingClassifier(
            random_state=RANDOM_STATE,
            n_estimators=100
        )
        
        # Modelo 3: Random Forest con class weights
        rf_weighted = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=RANDOM_STATE
        )
        
        # Pipeline sin SMOTE (para modelos con class_weight)
        pipe_logreg = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', logreg_weighted)
        ])
        
        pipe_rf = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', rf_weighted)
        ])
        
        # Pipeline con SMOTE
        pipe_gbc_smote = ImbPipeline([
            ('preprocessor', self.preprocessor),
            ('smote', SMOTE(k_neighbors=min(3, len(self.y[self.y == 'Diabetes']) - 1), random_state=RANDOM_STATE)),
            ('classifier', gbc)
        ])
        
        # Modelos calibrados
        self.models = {
            'logreg_weighted': CalibratedClassifierCV(pipe_logreg, method='isotonic', cv=3),
            'rf_weighted': CalibratedClassifierCV(pipe_rf, method='isotonic', cv=3),
            'gbc_smote': CalibratedClassifierCV(pipe_gbc_smote, method='isotonic', cv=3)
        }
        
        self.log(f"✅ {len(self.models)} modelos creados")
    
    def evaluate_models(self):
        """Evalúa modelos con validación cruzada y métricas médicas"""
        self.log("Evaluando modelos con validación cruzada...")
        
        # Custom scorer para métricas médicas
        def medical_scorer(y_true, y_pred, **kwargs):
            metrics = get_medical_metrics(y_true, y_pred, labels=CLASS_ORDER)
            return metrics['f1_macro']
        
        medical_score = make_scorer(medical_scorer)
        
        # Validación cruzada estratificada
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        self.results = {}
        
        for name, model in self.models.items():
            self.log(f"🔄 Evaluando: {name}")
            
            try:
                # Cross validation con múltiples métricas
                cv_results = cross_validate(
                    model, self.X, self.y,
                    cv=cv,
                    scoring={
                        'f1_macro': 'f1_macro',
                        'balanced_accuracy': 'balanced_accuracy',
                        'medical': medical_score
                    },
                    return_train_score=False,
                    n_jobs=-1
                )
                
                # Entrenar modelo completo para métricas detalladas
                model.fit(self.X, self.y)
                y_pred = model.predict(self.X)
                y_proba = model.predict_proba(self.X)
                
                # Calcular métricas médicas detalladas
                detailed_metrics = get_medical_metrics(
                    self.y, y_pred, y_proba, CLASS_ORDER
                )
                
                # Combinar resultados
                self.results[name] = {
                    **detailed_metrics,
                    'cv_f1_macro_mean': np.mean(cv_results['test_f1_macro']),
                    'cv_f1_macro_std': np.std(cv_results['test_f1_macro']),
                    'cv_balanced_accuracy_mean': np.mean(cv_results['test_balanced_accuracy']),
                    'cv_balanced_accuracy_std': np.std(cv_results['test_balanced_accuracy']),
                }
                
                self.log(f"✅ {name}: F1={detailed_metrics['f1_macro']:.4f}, "
                        f"Diabetes Sens={detailed_metrics['diabetes_sensitivity']:.4f}")
                
            except Exception as e:
                self.log(f"❌ Error evaluando {name}: {e}", "ERROR")
                continue
        
        # Seleccionar mejor modelo
        if self.results:
            # Priorizar sensibilidad para diabetes, luego F1 macro
            best_name = max(self.results.keys(), key=lambda k: (
                self.results[k]['diabetes_sensitivity'],
                self.results[k]['f1_macro']
            ))
            
            self.best_model_name = best_name
            self.best_model = self.models[best_name]
            
            self.log(f"🏆 Mejor modelo: {best_name}")
        else:
            self.log("❌ No se pudo evaluar ningún modelo", "ERROR")
    
    def train_final_model(self):
        """Entrena el modelo final en todo el dataset"""
        if self.best_model is None:
            self.log("❌ No hay modelo seleccionado", "ERROR")
            return
        
        self.log("Entrenando modelo final...")
        
        # Entrenar en todo el dataset
        self.best_model.fit(self.X, self.y)
        
        # Guardar modelo
        model_path = MODELS_DIR / f"best_model_{self.best_model_name}.joblib"
        joblib.dump(self.best_model, model_path)
        
        self.log(f"✅ Modelo final guardado: {model_path}")
        
        # Guardar metadatos
        metadata = {
            'model_name': self.best_model_name,
            'training_date': datetime.now().isoformat(),
            'dataset_shape': self.X.shape,
            'target_distribution': self.y.value_counts().to_dict(),
            'feature_columns': {
                'numeric': self.num_cols,
                'categorical': self.cat_cols
            }
        }
        
        metadata_path = MODELS_DIR / f"metadata_{self.best_model_name}.json"
        save_results(metadata, metadata_path)
        
        self.log(f"✅ Metadatos guardados: {metadata_path}")
    
    def generate_reports(self):
        """Genera reportes de resultados"""
        self.log("Generando reportes...")
        
        # Resumen en JSON
        summary = {
            'timestamp': datetime.now().isoformat(),
            'best_model': self.best_model_name,
            'dataset_info': {
                'shape': self.X.shape,
                'target_distribution': self.y.value_counts().to_dict()
            },
            'results': self.results
        }
        
        summary_path = OUTPUT_DIR / "results_summary.json"
        save_results(summary, summary_path)
        
        # Reporte detallado en texto
        detailed_report = generate_detailed_report(self.results, CLASS_ORDER)
        
        report_path = OUTPUT_DIR / "detailed_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(detailed_report)
        
        # DataFrame de resultados para fácil visualización
        results_df = pd.DataFrame([
            {
                'model': name,
                'f1_macro': results['f1_macro'],
                'balanced_accuracy': results['balanced_accuracy'],
                'diabetes_sensitivity': results['diabetes_sensitivity'],
                'diabetes_specificity': results['diabetes_specificity'],
                'diabetes_ppv': results['diabetes_ppv']
            }
            for name, results in self.results.items()
        ]).sort_values('diabetes_sensitivity', ascending=False)
        
        results_csv_path = OUTPUT_DIR / "model_comparison.csv"
        results_df.to_csv(results_csv_path, index=False)
        
        self.log(f"✅ Reportes generados en {OUTPUT_DIR}")
        
        # Mostrar tabla resumen
        print("\n" + "="*80)
        print("📊 RESULTADOS COMPARACIÓN DE MODELOS")
        print("="*80)
        print(results_df.round(4).to_string(index=False))
        print("="*80)
    
    def run_complete_pipeline(self):
        """Ejecuta el pipeline completo"""
        try:
            # Paso 1: Cargar datos
            if not self.load_data():
                return False
            
            # Paso 2: Preparar features
            self.prepare_features()
            
            # Paso 3: Crear preprocesador
            self.create_preprocessor()
            
            # Paso 4: Crear modelos
            self.create_models()
            
            # Paso 5: Evaluar modelos
            self.evaluate_models()
            
            # Paso 6: Entrenar modelo final
            self.train_final_model()
            
            # Paso 7: Generar reportes
            self.generate_reports()
            
            self.log("🎉 Pipeline completado exitosamente!")
            return True
            
        except Exception as e:
            self.log(f"💥 Error crítico en pipeline: {e}", "ERROR")
            return False

if __name__ == "__main__":
    # Ejecutar pipeline
    pipeline = DM2Pipeline()
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n🎉 ¡Pipeline ejecutado exitosamente!")
        print(f"📁 Revisa los resultados en: {OUTPUT_DIR}")
        print(f"🤖 Mejor modelo: {pipeline.best_model_name}")
    else:
        print("\n💥 Pipeline falló. Revisa los logs para más detalles.")