#!/usr/bin/env python3
"""
Ejecutor Rápido para Testing del Pipeline DM2
==============================================

Versión simplificada para testing rápido sin optimizaciones pesadas
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import sys

# Agregar directorio actual al path
sys.path.insert(0, str(Path(__file__).parent))

from master_pipeline import MasterPipeline
from config import DATA_PATH

def quick_test():
    """Test rápido del pipeline"""
    
    print("🧪 EJECUTOR DE TEST RÁPIDO")
    print("=" * 40)
    
    # Verificar que existe el dataset
    if not DATA_PATH.exists():
        print(f"❌ Dataset no encontrado: {DATA_PATH}")
        print("   Asegúrate de tener 'output-glucosa_labeled.csv' en el directorio")
        return False
    
    # Configuración de test rápido
    config = {
        'run_eda': True,
        'run_feature_engineering': True,
        'run_hyperparameter_tuning': True,
        'optuna_trials': 10,  # Solo 10 trials para test rápido
        'enable_advanced_models': False  # Solo modelos básicos
    }
    
    print("⚡ Configuración de test rápido:")
    print(f"   - EDA: {'✅' if config['run_eda'] else '❌'}")
    print(f"   - Feature Engineering: {'✅' if config['run_feature_engineering'] else '❌'}")
    print(f"   - Hyperparameter Tuning: {config['optuna_trials']} trials")
    print()
    
    try:
        # Ejecutar pipeline
        pipeline = MasterPipeline(**config)
        success = pipeline.run_complete_pipeline()
        
        if success:
            print(f"\\n🎉 ¡TEST EXITOSO!")
            print(f"📁 Resultados en: {pipeline.execution_dir}")
            
            # Mostrar métricas clave si están disponibles
            if hasattr(pipeline, 'final_best_model') and pipeline.final_best_model:
                print(f"🤖 Mejor modelo: {pipeline.final_best_name}")
            
            return True
        else:
            print(f"\\n❌ TEST FALLÓ")
            print(f"📋 Revisa logs en: {pipeline.log_file}")
            return False
            
    except KeyboardInterrupt:
        print("\\n⏹️ Test interrumpido por usuario")
        return False
    except Exception as e:
        print(f"\\n💥 Error inesperado: {e}")
        return False

if __name__ == "__main__":
    success = quick_test()
    exit(0 if success else 1)