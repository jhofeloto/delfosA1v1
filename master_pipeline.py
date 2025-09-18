#!/usr/bin/env python3
"""
Master Pipeline - Ejecutor Principal del Sistema DM2
====================================================

Pipeline completo que integra todas las mejoras:
- ✅ Análisis Exploratorio de Datos (EDA)
- ✅ Feature Engineering Avanzado
- ✅ Técnicas de Balancing
- ✅ Múltiples Modelos con Calibración
- ✅ Hyperparameter Tuning con Optuna
- ✅ Métricas Médicas Específicas
- ✅ Reportes Detallados
- ✅ Visualizaciones Completas
- ✅ Sistema de Logging
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
import argparse

# Sklearn imports
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Local imports
from config import *
from utils import load_and_validate_data, save_results
from eda_analysis import EDAAnalyzer
from feature_engineering import MedicalFeatureEngineer, FeatureSelector
from dm2_pipeline_improved import DM2Pipeline
from hyperparameter_tuning import OptunaOptimizer

class MasterPipeline:
    """Pipeline maestro que ejecuta todo el flujo de ML para DM2"""
    
    def __init__(self, 
                 run_eda=True,
                 run_feature_engineering=True,
                 run_hyperparameter_tuning=True,
                 optuna_trials=50,  # Reducido para demo
                 enable_advanced_models=True):
        
        self.config = {
            'run_eda': run_eda,
            'run_feature_engineering': run_feature_engineering,
            'run_hyperparameter_tuning': run_hyperparameter_tuning,
            'optuna_trials': optuna_trials,
            'enable_advanced_models': enable_advanced_models
        }
        
        self.results = {}
        self.models = {}
        self.best_model = None
        
        # Setup
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_logging()
        
        print("🚀 MASTER PIPELINE DM2 - INICIADO")
        print("=" * 50)
        print(f"📅 Timestamp: {self.timestamp}")
        print(f"⚙️  Configuración: {self.config}")
        print("=" * 50)
    
    def setup_logging(self):
        """Configura logging del pipeline maestro"""
        self.log_file = OUTPUT_DIR / f"master_pipeline_{self.timestamp}.log"
        
        # Crear directorios específicos para esta ejecución
        self.execution_dir = OUTPUT_DIR / f"execution_{self.timestamp}"
        self.execution_dir.mkdir(exist_ok=True)
        
        (self.execution_dir / "eda").mkdir(exist_ok=True)
        (self.execution_dir / "models").mkdir(exist_ok=True)
        (self.execution_dir / "reports").mkdir(exist_ok=True)
        (self.execution_dir / "visualizations").mkdir(exist_ok=True)
    
    def log(self, message: str, level: str = "INFO"):
        """Log con timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        print(log_entry)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + "\\n")
    
    def step_1_load_and_validate_data(self):
        """Paso 1: Carga y validación inicial de datos"""
        self.log("=== PASO 1: CARGA Y VALIDACIÓN DE DATOS ===")
        
        if not DATA_PATH.exists():
            self.log(f"❌ Dataset no encontrado: {DATA_PATH}", "ERROR")
            return False
        
        self.df, is_valid = load_and_validate_data(DATA_PATH, TARGET)
        
        if not is_valid:
            self.log("❌ Datos no válidos", "ERROR")
            return False
        
        # Estadísticas básicas
        self.log(f"✅ Dataset cargado exitosamente")
        self.log(f"📊 Shape: {self.df.shape}")
        self.log(f"🎯 Distribución target: {self.df[TARGET].value_counts().to_dict()}")
        
        # Guardar info básica
        basic_info = {
            'dataset_shape': self.df.shape,
            'target_distribution': self.df[TARGET].value_counts().to_dict(),
            'missing_values': self.df.isnull().sum().sum(),
            'duplicate_rows': self.df.duplicated().sum()
        }
        
        save_results(basic_info, self.execution_dir / "reports" / "basic_dataset_info.json")
        
        return True
    
    def step_2_exploratory_data_analysis(self):
        """Paso 2: Análisis Exploratorio de Datos"""
        self.log("=== PASO 2: ANÁLISIS EXPLORATORIO DE DATOS ===")
        
        if not self.config['run_eda']:
            self.log("⏭️ EDA deshabilitado, saltando...")
            return True
        
        try:
            # Ejecutar EDA completo
            eda_analyzer = EDAAnalyzer()
            success = eda_analyzer.run_complete_eda()
            
            if success:
                self.log("✅ EDA completado exitosamente")
                
                # Mover archivos de EDA a directorio de ejecución
                import shutil
                eda_files = list(OUTPUT_DIR.glob("*"))
                
                for file in eda_files:
                    if file.name.startswith(('eda_', 'correlation_', 'target_', 'numeric_')):
                        try:
                            shutil.move(str(file), str(self.execution_dir / "eda" / file.name))
                        except:
                            pass
                
                self.results['eda'] = {'status': 'completed', 'files_generated': True}
                return True
            else:
                self.log("⚠️ EDA falló, continuando sin EDA", "WARNING")
                self.results['eda'] = {'status': 'failed'}
                return True  # Continuar aunque falle EDA
                
        except Exception as e:
            self.log(f"❌ Error en EDA: {e}", "ERROR")
            self.results['eda'] = {'status': 'error', 'message': str(e)}
            return True  # Continuar aunque falle EDA
    
    def step_3_feature_engineering(self):
        """Paso 3: Feature Engineering Avanzado"""
        self.log("=== PASO 3: FEATURE ENGINEERING ===")
        
        # Preparar datos base
        blacklist = [col for col in BLACKLIST_COLS if col in self.df.columns]
        pii = [col for col in PII_COLS if col in self.df.columns]
        
        # Features iniciales
        num_cols = self.df.select_dtypes(include=[np.number]).columns.difference(blacklist + pii).tolist()
        cat_cols = self.df.select_dtypes(include=["object"]).columns.difference(blacklist + pii).tolist()
        
        self.X_base = self.df[num_cols + cat_cols].copy()
        self.y = self.df[TARGET].astype("category").cat.set_categories(CLASS_ORDER)
        
        self.log(f"📊 Features base: {self.X_base.shape[1]} ({len(num_cols)} num, {len(cat_cols)} cat)")
        
        if self.config['run_feature_engineering']:
            try:
                # Feature Engineering Médico
                self.log("🔧 Aplicando feature engineering médico...")
                
                medical_engineer = MedicalFeatureEngineer(
                    create_interactions=True,
                    create_ratios=True,
                    create_bins=True,
                    n_interactions=10
                )
                
                # Aplicar transformaciones
                medical_engineer.fit(self.X_base)
                X_engineered = medical_engineer.transform(self.X_base)
                
                self.log(f"✅ Features después de engineering: {X_engineered.shape[1]}")
                self.log(f"📈 Nuevas features: {X_engineered.shape[1] - self.X_base.shape[1]}")
                
                # Feature Selection
                self.log("🎯 Seleccionando mejores features...")
                
                selector = FeatureSelector(n_features=min(50, X_engineered.shape[1]))
                selected_features = selector.fit_select(X_engineered, self.y)
                
                self.X_final = selector.transform(X_engineered)
                
                self.log(f"✅ Features finales seleccionadas: {len(selected_features)}")
                
                # Guardar información de features
                feature_info = {
                    'original_features': self.X_base.shape[1],
                    'engineered_features': X_engineered.shape[1],
                    'selected_features': len(selected_features),
                    'selected_feature_names': selected_features,
                    'feature_engineering_applied': True
                }
                
                save_results(feature_info, self.execution_dir / "reports" / "feature_engineering_report.json")
                
                self.feature_engineer = medical_engineer
                self.feature_selector = selector
                
            except Exception as e:
                self.log(f"⚠️ Error en feature engineering: {e}", "WARNING")
                self.log("📊 Usando features originales")
                self.X_final = self.X_base
                
                feature_info = {
                    'original_features': self.X_base.shape[1],
                    'feature_engineering_applied': False,
                    'error': str(e)
                }
                
                save_results(feature_info, self.execution_dir / "reports" / "feature_engineering_report.json")
        else:
            self.log("⏭️ Feature engineering deshabilitado")
            self.X_final = self.X_base
        
        self.log(f"📊 Dataset final para modelado: X{self.X_final.shape}, y{self.y.shape}")
        return True
    
    def step_4_baseline_modeling(self):
        """Paso 4: Modelado Baseline Mejorado"""
        self.log("=== PASO 4: MODELADO BASELINE ===")
        
        try:
            # Crear preprocesador para features finales
            num_cols_final = self.X_final.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols_final = self.X_final.select_dtypes(include=["object", "category"]).columns.tolist()
            
            numeric_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', drop=None))
            ])
            
            self.preprocessor = ColumnTransformer([
                ('num', numeric_transformer, num_cols_final),
                ('cat', categorical_transformer, cat_cols_final)
            ], remainder='drop')
            
            # Ejecutar pipeline baseline mejorado
            baseline_pipeline = DM2Pipeline()
            baseline_pipeline.df = self.df
            baseline_pipeline.X = self.X_final
            baseline_pipeline.y = self.y
            baseline_pipeline.num_cols = num_cols_final
            baseline_pipeline.cat_cols = cat_cols_final
            baseline_pipeline.preprocessor = self.preprocessor
            
            # Saltar pasos ya ejecutados y ir directo a modelado
            baseline_pipeline.create_models()
            baseline_pipeline.evaluate_models()
            
            self.baseline_results = baseline_pipeline.results
            self.baseline_best_model = baseline_pipeline.best_model
            self.baseline_best_name = baseline_pipeline.best_model_name
            
            self.log(f"✅ Baseline completado - Mejor: {self.baseline_best_name}")
            self.log(f"📊 F1 Macro: {self.baseline_results[self.baseline_best_name]['f1_macro']:.4f}")
            
            return True
            
        except Exception as e:
            self.log(f"❌ Error en modelado baseline: {e}", "ERROR")
            return False
    
    def step_5_hyperparameter_optimization(self):
        """Paso 5: Optimización de Hyperparámetros"""
        self.log("=== PASO 5: OPTIMIZACIÓN HYPERPARÁMETROS ===")
        
        if not self.config['run_hyperparameter_tuning']:
            self.log("⏭️ Optimización deshabilitada")
            self.final_best_model = self.baseline_best_model
            self.final_best_name = self.baseline_best_name
            return True
        
        try:
            self.log(f"🎯 Iniciando optimización con {self.config['optuna_trials']} trials...")
            
            # Optimización con Optuna
            optimizer = OptunaOptimizer(
                X=self.X_final,
                y=self.y,
                preprocessor=self.preprocessor,
                n_trials=self.config['optuna_trials'],
                cv_folds=CV_FOLDS,
                optimize_for='diabetes_sensitivity'  # Priorizar sensibilidad médica
            )
            
            # Ejecutar optimización completa
            optimized_model = optimizer.run_complete_optimization()
            
            if optimized_model is not None:
                self.final_best_model = optimized_model
                self.final_best_name = optimizer.best_model_name
                self.optuna_results = {
                    'best_model': optimizer.best_model_name,
                    'best_score': optimizer.best_overall_score,
                    'best_params': optimizer.best_params
                }
                
                self.log(f"✅ Optimización completada - Mejor: {self.final_best_name}")
                self.log(f"📊 Score optimizado: {optimizer.best_overall_score:.4f}")
            else:
                self.log("⚠️ Optimización falló, usando baseline", "WARNING")
                self.final_best_model = self.baseline_best_model
                self.final_best_name = self.baseline_best_name
            
            return True
            
        except Exception as e:
            self.log(f"❌ Error en optimización: {e}", "ERROR")
            self.log("📊 Usando modelo baseline")
            self.final_best_model = self.baseline_best_model
            self.final_best_name = self.baseline_best_name
            return True
    
    def step_6_final_evaluation_and_reports(self):
        """Paso 6: Evaluación Final y Reportes"""
        self.log("=== PASO 6: EVALUACIÓN FINAL ===")
        
        try:
            # Entrenar modelo final en todo el dataset
            if self.final_best_model is not None:
                self.final_best_model.fit(self.X_final, self.y)
                
                # Predicciones finales
                y_pred_final = self.final_best_model.predict(self.X_final)
                y_proba_final = self.final_best_model.predict_proba(self.X_final)
                
                # Métricas finales
                from utils import get_medical_metrics
                final_metrics = get_medical_metrics(
                    self.y, y_pred_final, y_proba_final, CLASS_ORDER
                )
                
                self.log("📊 MÉTRICAS FINALES:")
                self.log(f"   F1 Macro: {final_metrics['f1_macro']:.4f}")
                self.log(f"   Balanced Acc: {final_metrics['balanced_accuracy']:.4f}")
                self.log(f"   Diabetes Sensitivity: {final_metrics['diabetes_sensitivity']:.4f}")
                self.log(f"   Diabetes Specificity: {final_metrics['diabetes_specificity']:.4f}")
                
                # Guardar modelo final
                import joblib
                model_path = self.execution_dir / "models" / f"final_best_model_{self.final_best_name}.joblib"
                joblib.dump(self.final_best_model, model_path)
                self.log(f"💾 Modelo final guardado: {model_path}")
                
                # Crear resumen ejecutivo completo
                executive_summary = {
                    'execution_timestamp': self.timestamp,
                    'dataset_info': {
                        'shape': self.df.shape,
                        'target_distribution': self.df[TARGET].value_counts().to_dict()
                    },
                    'pipeline_config': self.config,
                    'final_model': {
                        'name': self.final_best_name,
                        'metrics': final_metrics
                    },
                    'baseline_comparison': getattr(self, 'baseline_results', {}),
                    'optimization_results': getattr(self, 'optuna_results', {}),
                    'execution_directory': str(self.execution_dir)
                }
                
                save_results(executive_summary, self.execution_dir / "reports" / "executive_summary.json")
                
                # Generar reporte en texto plano
                self.generate_text_report(executive_summary, final_metrics)
                
                self.log("✅ Evaluación final completada")
                return True
                
            else:
                self.log("❌ No hay modelo final disponible", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"❌ Error en evaluación final: {e}", "ERROR")
            return False
    
    def generate_text_report(self, summary, metrics):
        """Genera reporte ejecutivo en texto plano"""
        
        report_lines = [
            "=" * 80,
            "🏥 REPORTE EJECUTIVO - PIPELINE DM2",
            "=" * 80,
            f"📅 Fecha de Ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"🔧 Pipeline Version: Master Pipeline v2.0",
            "",
            "📊 INFORMACIÓN DEL DATASET",
            "-" * 40,
            f"• Tamaño: {summary['dataset_info']['shape'][0]} pacientes, {summary['dataset_info']['shape'][1]} variables",
            f"• Distribución de clases:"
        ]
        
        for clase, count in summary['dataset_info']['target_distribution'].items():
            pct = (count / sum(summary['dataset_info']['target_distribution'].values())) * 100
            report_lines.append(f"  - {clase}: {count} ({pct:.1f}%)")
        
        report_lines.extend([
            "",
            "🤖 MODELO FINAL",
            "-" * 40,
            f"• Algoritmo: {summary['final_model']['name']}",
            f"• Configuración: {summary['pipeline_config']}",
            "",
            "📈 MÉTRICAS DE RENDIMIENTO",
            "-" * 40,
            f"• F1 Macro: {metrics['f1_macro']:.4f}",
            f"• Balanced Accuracy: {metrics['balanced_accuracy']:.4f}",
            f"• Diabetes Sensitivity: {metrics['diabetes_sensitivity']:.4f}",
            f"• Diabetes Specificity: {metrics['diabetes_specificity']:.4f}",
            f"• Diabetes PPV: {metrics['diabetes_ppv']:.4f}",
            "",
            "⚕️ INTERPRETACIÓN CLÍNICA",
            "-" * 40
        ])
        
        # Interpretación automática
        sens = metrics['diabetes_sensitivity']
        spec = metrics['diabetes_specificity']
        
        if sens >= 0.8:
            report_lines.append("• ✅ Excelente detección de diabetes (alta sensibilidad)")
        elif sens >= 0.6:
            report_lines.append("• ⚠️ Detección moderada de diabetes")
        else:
            report_lines.append("• ❌ Baja detección de diabetes - requiere mejoras")
        
        if spec >= 0.8:
            report_lines.append("• ✅ Baja tasa de falsos positivos (alta especificidad)")
        elif spec >= 0.6:
            report_lines.append("• ⚠️ Especificidad moderada")
        else:
            report_lines.append("• ❌ Alta tasa de falsos positivos")
        
        report_lines.extend([
            "",
            f"📁 Archivos generados en: {self.execution_dir}",
            "=" * 80
        ])
        
        # Guardar reporte
        report_text = "\\n".join(report_lines)
        report_path = self.execution_dir / "reports" / "executive_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        # Mostrar en consola
        print("\\n" + report_text)
        
        self.log(f"📋 Reporte ejecutivo guardado: {report_path}")
    
    def run_complete_pipeline(self):
        """Ejecuta el pipeline completo"""
        
        start_time = datetime.now()
        
        try:
            self.log("🚀 INICIANDO PIPELINE MAESTRO COMPLETO")
            
            # Paso 1: Carga de datos
            if not self.step_1_load_and_validate_data():
                return False
            
            # Paso 2: EDA
            if not self.step_2_exploratory_data_analysis():
                return False
            
            # Paso 3: Feature Engineering
            if not self.step_3_feature_engineering():
                return False
            
            # Paso 4: Modelado Baseline
            if not self.step_4_baseline_modeling():
                return False
            
            # Paso 5: Optimización
            if not self.step_5_hyperparameter_optimization():
                return False
            
            # Paso 6: Evaluación Final
            if not self.step_6_final_evaluation_and_reports():
                return False
            
            # Calcular tiempo total
            end_time = datetime.now()
            total_time = end_time - start_time
            
            self.log(f"🎉 PIPELINE COMPLETADO EXITOSAMENTE")
            self.log(f"⏱️ Tiempo total: {total_time}")
            self.log(f"📁 Resultados en: {self.execution_dir}")
            
            return True
            
        except Exception as e:
            self.log(f"💥 ERROR CRÍTICO EN PIPELINE: {e}", "ERROR")
            return False

def main():
    """Función principal con argumentos CLI"""
    
    parser = argparse.ArgumentParser(description='Master Pipeline DM2')
    parser.add_argument('--no-eda', action='store_true', help='Saltar análisis exploratorio')
    parser.add_argument('--no-feature-engineering', action='store_true', help='Saltar feature engineering')
    parser.add_argument('--no-optimization', action='store_true', help='Saltar optimización hyperparámetros')
    parser.add_argument('--optuna-trials', type=int, default=50, help='Número de trials para Optuna')
    parser.add_argument('--quick', action='store_true', help='Ejecución rápida (menos trials)')
    
    args = parser.parse_args()
    
    # Configuración basada en argumentos
    config = {
        'run_eda': not args.no_eda,
        'run_feature_engineering': not args.no_feature_engineering,
        'run_hyperparameter_tuning': not args.no_optimization,
        'optuna_trials': 20 if args.quick else args.optuna_trials,
        'enable_advanced_models': True
    }
    
    # Ejecutar pipeline
    pipeline = MasterPipeline(**config)
    success = pipeline.run_complete_pipeline()
    
    if success:
        print(f"\\n🎉 ¡ÉXITO! Revisa los resultados en: {pipeline.execution_dir}")
        return 0
    else:
        print(f"\\n💥 FALLÓ. Revisa los logs en: {pipeline.log_file}")
        return 1

if __name__ == "__main__":
    exit(main())