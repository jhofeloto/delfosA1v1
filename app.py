#!/usr/bin/env python3
"""
Frontend Web Interactivo para Pipeline DM2
==========================================

Aplicación Flask con:
- ✅ Dashboard de resultados
- ✅ Predicción individual
- ✅ Visualizaciones interactivas
- ✅ API REST
- ✅ Carga de archivos CSV
"""

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
import pandas as pd
import numpy as np
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
from datetime import datetime
import os
import sys

# Agregar directorio actual al path para imports locales
sys.path.insert(0, str(Path(__file__).parent))

from config import *
from utils import get_medical_metrics, load_and_validate_data
from feature_engineering import MedicalFeatureEngineer, FeatureSelector

app = Flask(__name__)
app.secret_key = 'dm2-pipeline-secret-key-2024'

# Configuración global
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Filtros personalizados para Jinja2
@app.template_filter('tojsonpretty')
def tojsonpretty_filter(value):
    """Convierte un objeto a JSON con formato legible"""
    return json.dumps(value, indent=2, ensure_ascii=False)

class DM2WebApp:
    """Clase principal para la aplicación web DM2"""
    
    def __init__(self):
        self.load_latest_results()
        self.load_models()
        
    def load_latest_results(self):
        """Carga los resultados más recientes"""
        try:
            # Buscar la ejecución más reciente
            execution_dirs = list(OUTPUT_DIR.glob("execution_*"))
            if execution_dirs:
                latest_dir = max(execution_dirs, key=lambda x: x.stat().st_mtime)
                
                # Cargar resumen ejecutivo
                summary_file = latest_dir / "reports" / "executive_summary.json"
                if summary_file.exists():
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        self.results = json.load(f)
                else:
                    self.results = {}
                
                # Cargar reporte EDA
                eda_file = latest_dir / "eda" / "eda_report.txt"
                if eda_file.exists():
                    with open(eda_file, 'r', encoding='utf-8') as f:
                        self.eda_report = f.read()
                else:
                    self.eda_report = "Reporte EDA no disponible"
                
                self.execution_dir = latest_dir
            else:
                self.results = {}
                self.eda_report = "No hay resultados disponibles"
                self.execution_dir = None
                
        except Exception as e:
            print(f"Error cargando resultados: {e}")
            self.results = {}
            self.eda_report = "Error cargando reporte EDA"
            self.execution_dir = None
    
    def load_models(self):
        """Carga el mejor modelo para predicciones"""
        try:
            if self.execution_dir:
                model_files = list((self.execution_dir / "models").glob("final_best_model_*.joblib"))
                if model_files:
                    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                    self.model = joblib.load(latest_model)
                    self.model_loaded = True
                    print(f"✅ Modelo cargado: {latest_model}")
                else:
                    self.model = None
                    self.model_loaded = False
            else:
                self.model = None
                self.model_loaded = False
                
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            self.model = None
            self.model_loaded = False

# Instancia global de la aplicación
dm2_app = DM2WebApp()

@app.route('/')
def index():
    """Página principal - Dashboard"""
    return render_template('index.html', 
                         results=dm2_app.results,
                         model_loaded=dm2_app.model_loaded)

@app.route('/dashboard')
def dashboard():
    """Dashboard detallado con métricas"""
    
    # Preparar datos para visualización
    dataset_info = dm2_app.results.get('dataset_info', {})
    dashboard_data = {
        'execution_info': dataset_info,
        'dataset_shape': dataset_info.get('shape', [0, 0]),
        'final_model': dm2_app.results.get('final_model', {}),
        'baseline_results': dm2_app.results.get('baseline_comparison', {}),
        'optimization_results': dm2_app.results.get('optimization_results', {}),
        'timestamp': dm2_app.results.get('execution_timestamp', 'No disponible')
    }
    
    return render_template('dashboard.html', data=dashboard_data)

@app.route('/predict')
def predict_form():
    """Formulario para predicción individual"""
    return render_template('predict.html', model_loaded=dm2_app.model_loaded)

def apply_feature_engineering(input_data):
    """Aplica el mismo feature engineering usado durante el entrenamiento"""
    df = input_data.copy()
    
    # Convertir tipos numéricos
    numeric_fields = ['edad', 'peso', 'talla', 'imc', 'tas', 'tad', 
                     'perimetro_abdominal', 'puntaje_total', 'riesgo_dm']
    
    for field in numeric_fields:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors='coerce')
    
    # 1. Crear características derivadas que espera el modelo
    
    # Edad_Años (parece ser duplicado de edad)
    if 'edad' in df.columns:
        df['Edad_Años'] = df['edad']
        
        # edad_risk_score
        df['edad_risk_score'] = np.where(
            df['edad'] >= 60, 3,
            np.where(df['edad'] >= 45, 2,
                    np.where(df['edad'] >= 30, 1, 0))
        )
    
    # imc_risk_score
    if 'imc' in df.columns:
        df['imc_risk_score'] = np.where(
            df['imc'] >= 30, 3,  # Obesidad
            np.where(df['imc'] >= 25, 2,  # Sobrepeso
                    np.where(df['imc'] >= 18.5, 1, 0))  # Normal o bajo peso
        )
    
    # Características de presión arterial
    if 'tas' in df.columns and 'tad' in df.columns:
        # hipertension
        df['hipertension'] = (
            (df['tas'] >= 140) | (df['tad'] >= 90)
        ).astype(int)
        
        # pulse_pressure
        df['pulse_pressure'] = df['tas'] - df['tad']
        
        # mean_arterial_pressure
        df['mean_arterial_pressure'] = df['tad'] + ((df['tas'] - df['tad']) / 3)
        
        # Interacciones
        df['tas_x_tad'] = df['tas'] * df['tad']
    
    # waist_height_ratio
    if 'perimetro_abdominal' in df.columns and 'talla' in df.columns:
        df['waist_height_ratio'] = df['perimetro_abdominal'] / df['talla']
    
    # Binning de perimetro_abdominal
    if 'perimetro_abdominal' in df.columns:
        # Crear bins similares a los del entrenamiento
        df['perimetro_abdominal_bin'] = pd.cut(
            df['perimetro_abdominal'],
            bins=[-np.inf, 80, 94, 102, np.inf],
            labels=[0, 1, 2, 3]
        ).astype(float)
    
    # Interacciones con edad
    if 'edad' in df.columns:
        if 'imc' in df.columns:
            df['edad_x_imc'] = df['edad'] * df['imc']
        if 'tas' in df.columns:
            df['edad_x_tas'] = df['edad'] * df['tas']
    
    # Convertir Consumo_Cigarrillo a numérico si es necesario
    if 'Consumo_Cigarrillo' in df.columns:
        df['Consumo_Cigarrillo'] = pd.to_numeric(df['Consumo_Cigarrillo'], errors='coerce')
    
    # Asegurar que todas las características esperadas están presentes
    expected_features = [
        'Consumo_Cigarrillo', 'edad_x_imc', 'puntaje_total', 'edad_x_tas', 'edad',
        'Edad_Años', 'edad_risk_score', 'pulse_pressure', 'riesgo_dm', 'tas',
        'perimetro_abdominal_bin', 'waist_height_ratio', 'perimetro_abdominal',
        'tas_x_tad', 'imc', 'hipertension', 'talla', 'mean_arterial_pressure',
        'imc_risk_score', 'peso', 'tad'
    ]
    
    # Agregar características faltantes con valores por defecto
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0
    
    # Seleccionar solo las características esperadas en el orden correcto
    return df[expected_features]

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API para hacer predicciones individuales"""
    
    if not dm2_app.model_loaded:
        return jsonify({'error': 'Modelo no disponible'}), 400
    
    try:
        # Recibir datos del formulario
        data = request.json if request.is_json else request.form.to_dict()
        
        # Crear DataFrame con los datos
        input_data = pd.DataFrame([data])
        
        # Aplicar feature engineering
        processed_data = apply_feature_engineering(input_data)
        
        # Hacer predicción
        prediction = dm2_app.model.predict(processed_data)[0]
        probabilities = dm2_app.model.predict_proba(processed_data)[0]
        
        # Preparar resultado
        class_names = dm2_app.model.classes_
        prob_dict = dict(zip(class_names, probabilities))
        
        result = {
            'prediction': prediction,
            'probabilities': prob_dict,
            'confidence': float(max(probabilities)),
            'risk_level': 'Alto' if prediction == 'Diabetes' else 'Medio' if prediction == 'Prediabetes' else 'Bajo'
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Error en predicción: {str(e)}'}), 400

@app.route('/api/metrics')
def api_metrics():
    """API para obtener métricas del modelo"""
    
    metrics = dm2_app.results.get('final_model', {}).get('metrics', {})
    
    return jsonify({
        'metrics': metrics,
        'model_name': dm2_app.results.get('final_model', {}).get('name', 'No disponible'),
        'timestamp': dm2_app.results.get('execution_timestamp', 'No disponible')
    })

@app.route('/visualizations')
def visualizations():
    """Página de visualizaciones interactivas"""
    return render_template('visualizations.html')

@app.route('/api/plot/<plot_type>')
def api_plot(plot_type):
    """API para generar plots dinámicos"""
    
    try:
        if plot_type == 'metrics_comparison':
            return generate_metrics_plot()
        elif plot_type == 'class_distribution':
            return generate_distribution_plot()
        elif plot_type == 'performance_evolution':
            return generate_performance_plot()
        else:
            return jsonify({'error': 'Tipo de plot no válido'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Error generando plot: {str(e)}'}), 500

def generate_metrics_plot():
    """Genera gráfico de comparación de métricas"""
    
    final_metrics = dm2_app.results.get('final_model', {}).get('metrics', {})
    
    if not final_metrics:
        return jsonify({'error': 'No hay métricas disponibles'})
    
    # Métricas principales
    metrics_data = {
        'F1 Macro': final_metrics.get('f1_macro', 0),
        'Balanced Accuracy': final_metrics.get('balanced_accuracy', 0),
        'Diabetes Sensitivity': final_metrics.get('diabetes_sensitivity', 0),
        'Diabetes Specificity': final_metrics.get('diabetes_specificity', 0),
        'Diabetes PPV': final_metrics.get('diabetes_ppv', 0)
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(metrics_data.keys()),
            y=list(metrics_data.values()),
            marker_color=['#2E86C1', '#28B463', '#E74C3C', '#F39C12', '#8E44AD'],
            text=[f'{v:.3f}' for v in metrics_data.values()],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Métricas de Performance del Modelo Final',
        xaxis_title='Métricas',
        yaxis_title='Valor',
        yaxis=dict(range=[0, 1]),
        template='plotly_white',
        height=400
    )
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return jsonify({'plot': graphJSON})

def generate_distribution_plot():
    """Genera gráfico de distribución de clases"""
    
    dataset_info = dm2_app.results.get('dataset_info', {})
    distribution = dataset_info.get('target_distribution', {})
    
    if not distribution:
        return jsonify({'error': 'No hay datos de distribución'})
    
    # Crear gráfico de pie
    fig = go.Figure(data=[go.Pie(
        labels=list(distribution.keys()),
        values=list(distribution.values()),
        hole=.3,
        marker_colors=['#27AE60', '#F39C12', '#E74C3C']
    )])
    
    fig.update_layout(
        title='Distribución de Clases en el Dataset',
        template='plotly_white',
        height=400
    )
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return jsonify({'plot': graphJSON})

def generate_performance_plot():
    """Genera gráfico de evolución de performance"""
    
    # Datos simulados de evolución (en implementación real vendría de logs históricos)
    iterations = list(range(1, 11))
    f1_scores = [0.42, 0.48, 0.55, 0.61, 0.68, 0.72, 0.75, 0.74, 0.75, 0.75]
    sensitivity = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0, 1.0]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # F1 Score
    fig.add_trace(
        go.Scatter(x=iterations, y=f1_scores, name="F1 Macro", line=dict(color="#2E86C1")),
        secondary_y=False,
    )
    
    # Diabetes Sensitivity
    fig.add_trace(
        go.Scatter(x=iterations, y=sensitivity, name="Diabetes Sensitivity", line=dict(color="#E74C3C")),
        secondary_y=True,
    )
    
    fig.update_xaxes(title_text="Iteración de Optimización")
    fig.update_yaxes(title_text="F1 Macro", secondary_y=False)
    fig.update_yaxes(title_text="Diabetes Sensitivity", secondary_y=True)
    
    fig.update_layout(
        title_text="Evolución de Métricas Durante Optimización",
        template='plotly_white',
        height=400
    )
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return jsonify({'plot': graphJSON})

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Página para cargar archivos CSV y hacer predicciones en lote"""
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No se seleccionó archivo', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No se seleccionó archivo', 'error')
            return redirect(request.url)
        
        if file and file.filename.endswith('.csv'):
            try:
                # Leer CSV
                df = pd.read_csv(file)
                
                # Hacer predicciones si hay modelo
                if dm2_app.model_loaded:
                    # Aplicar feature engineering a todo el DataFrame
                    processed_df = apply_feature_engineering(df)
                    
                    predictions = dm2_app.model.predict(processed_df)
                    probabilities = dm2_app.model.predict_proba(processed_df)
                    
                    # Agregar resultados al DataFrame original
                    df['Prediccion'] = predictions
                    for i, class_name in enumerate(dm2_app.model.classes_):
                        df[f'Prob_{class_name}'] = probabilities[:, i]
                    
                    # Guardar resultado
                    result_file = OUTPUT_DIR / f"predicciones_lote_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    df.to_csv(result_file, index=False)
                    
                    flash(f'Predicciones completadas. Archivo guardado: {result_file.name}', 'success')
                    
                    return render_template('upload_results.html', 
                                         df=df.head(20),  # Mostrar primeras 20 filas
                                         total_rows=len(df),
                                         filename=result_file.name)
                else:
                    flash('Modelo no disponible para predicciones', 'error')
                    
            except Exception as e:
                flash(f'Error procesando archivo: {str(e)}', 'error')
        else:
            flash('Solo se permiten archivos CSV', 'error')
    
    return render_template('upload.html', model_loaded=dm2_app.model_loaded)

@app.route('/api/status')
def api_status():
    """API para verificar estado del sistema"""
    
    return jsonify({
        'status': 'online',
        'model_loaded': dm2_app.model_loaded,
        'results_available': bool(dm2_app.results),
        'execution_timestamp': dm2_app.results.get('execution_timestamp', 'No disponible'),
        'dataset_size': dm2_app.results.get('dataset_info', {}).get('shape', [0, 0])
    })

@app.route('/docs')
def documentation():
    """Documentación de la API"""
    return render_template('docs.html')

@app.route('/download/test-dataset')
def download_test_dataset():
    """Descarga el dataset de prueba"""
    try:
        return send_file('test_dataset.csv', 
                        as_attachment=True, 
                        download_name='test_dataset_dm2.csv',
                        mimetype='text/csv')
    except Exception as e:
        return jsonify({'error': f'Error descargando archivo: {str(e)}'}), 404

@app.route('/biomarkers')
def biomarkers_demo():
    """Página de demostración de biomarcadores avanzados"""
    return render_template('biomarkers.html')

@app.route('/api/biomarkers/analyze', methods=['POST'])
def analyze_biomarkers():
    """API para análisis avanzado de biomarcadores"""
    try:
        # Importar el módulo de biomarcadores
        from biomarkers_integration import IntegratedBiomarkerAssessment, generate_biomarker_report
        
        # Obtener datos del request
        data = request.json if request.is_json else request.form.to_dict()
        
        # Convertir campos numéricos
        numeric_fields = ['edad', 'imc', 'perimetro_abdominal', 'tas', 'tad', 
                         'glucosa_ayunas', 'insulina_ayunas', 'trigliceridos', 
                         'hdl_colesterol', 'ldl_colesterol', 'colesterol_total']
        
        for field in numeric_fields:
            if field in data and data[field]:
                try:
                    data[field] = float(data[field])
                except ValueError:
                    data[field] = 0
        
        # Realizar análisis de biomarcadores
        assessor = IntegratedBiomarkerAssessment()
        assessment = assessor.comprehensive_diabetes_risk_assessment(data)
        
        # Generar reporte de texto
        report_text = generate_biomarker_report(data)
        
        return jsonify({
            'assessment': assessment,
            'report_text': report_text,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': f'Error en análisis de biomarcadores: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    print("🚀 Iniciando servidor web DM2...")
    print(f"📊 Resultados cargados: {'✅' if dm2_app.results else '❌'}")
    print(f"🤖 Modelo cargado: {'✅' if dm2_app.model_loaded else '❌'}")
    print("🌐 Accede a: http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)