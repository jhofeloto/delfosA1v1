#!/usr/bin/env python3
"""
Setup e Instalación de Dependencias para Pipeline DM2
=====================================================
"""

import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Instala las dependencias necesarias"""
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ Archivo requirements.txt no encontrado")
        return False
    
    try:
        print("📦 Instalando dependencias...")
        
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencias instaladas exitosamente")
            return True
        else:
            print(f"❌ Error instalando dependencias: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_data_file():
    """Verifica que el archivo de datos exista"""
    
    data_file = Path(__file__).parent / "output-glucosa_labeled.csv"
    
    if data_file.exists():
        print(f"✅ Dataset encontrado: {data_file}")
        return True
    else:
        print(f"⚠️ Dataset no encontrado: {data_file}")
        print("   Asegúrate de tener el archivo 'output-glucosa_labeled.csv' en el directorio del proyecto")
        return False

def main():
    """Función principal de setup"""
    
    print("🔧 SETUP PIPELINE DM2")
    print("=" * 30)
    
    # Instalar dependencias
    deps_ok = install_requirements()
    
    # Verificar dataset
    data_ok = check_data_file()
    
    print("\\n📋 RESUMEN:")
    print(f"   Dependencias: {'✅' if deps_ok else '❌'}")
    print(f"   Dataset: {'✅' if data_ok else '⚠️'}")
    
    if deps_ok:
        print("\\n🚀 ¡Listo para ejecutar!")
        print("   Ejecuta: python master_pipeline.py")
        print("   O para ejecución rápida: python master_pipeline.py --quick")
    else:
        print("\\n❌ Setup incompleto. Revisa los errores anteriores.")
    
    return 0 if (deps_ok and data_ok) else 1

if __name__ == "__main__":
    exit(main())