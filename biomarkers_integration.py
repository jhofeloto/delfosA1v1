#!/usr/bin/env python3
"""
Integración de Biomarcadores - Ejemplo Práctico
===============================================

Este módulo demuestra cómo integrar biomarcadores avanzados
en el pipeline DM2 existente.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

class BiomarkerCalculator:
    """Calculadora de biomarcadores avanzados para DM2"""
    
    def __init__(self):
        self.biomarker_definitions = {
            'homa_ir': {
                'name': 'HOMA-IR (Resistencia Insulínica)',
                'formula': '(glucosa_ayunas * insulina_ayunas) / 22.5',
                'normal_range': (0.5, 2.5),
                'units': 'index',
                'category': 'metabolic'
            },
            'tg_hdl_ratio': {
                'name': 'Ratio Triglicéridos/HDL',
                'formula': 'trigliceridos / hdl_colesterol',
                'normal_range': (0.0, 3.5),
                'units': 'ratio',
                'category': 'lipidic'
            },
            'metabolic_syndrome_score': {
                'name': 'Puntaje Síndrome Metabólico',
                'formula': 'weighted_sum_of_criteria',
                'normal_range': (0, 2),
                'units': 'score',
                'category': 'composite'
            },
            'diabetes_risk_biomarker': {
                'name': 'Biomarcador Riesgo DM2',
                'formula': 'ml_prediction_based',
                'normal_range': (0.0, 1.0),
                'units': 'probability',
                'category': 'predictive'
            }
        }
    
    def calculate_homa_ir(self, glucose_mg_dl: float, insulin_uiu_ml: float) -> Dict:
        """Calcula HOMA-IR (Homeostatic Model Assessment)"""
        
        if glucose_mg_dl <= 0 or insulin_uiu_ml <= 0:
            return {'value': None, 'interpretation': 'Invalid input values'}
        
        homa_ir = (glucose_mg_dl * insulin_uiu_ml) / 22.5
        
        # Interpretación clínica
        if homa_ir < 1.0:
            interpretation = 'Sensibilidad insulínica óptima'
            risk_level = 'low'
        elif homa_ir <= 2.5:
            interpretation = 'Sensibilidad insulínica normal'
            risk_level = 'low'
        elif homa_ir <= 5.0:
            interpretation = 'Resistencia insulínica moderada'
            risk_level = 'moderate'
        else:
            interpretation = 'Resistencia insulínica severa'
            risk_level = 'high'
        
        return {
            'value': round(homa_ir, 2),
            'interpretation': interpretation,
            'risk_level': risk_level,
            'reference_range': self.biomarker_definitions['homa_ir']['normal_range']
        }
    
    def calculate_metabolic_syndrome_score(self, patient_data: Dict) -> Dict:
        """Calcula puntaje de síndrome metabólico según criterios ATP III"""
        
        score = 0
        criteria_met = []
        
        # Criterio 1: Circunferencia abdominal
        perimeter = patient_data.get('perimetro_abdominal', 0)
        gender = patient_data.get('sexo', '').upper()
        
        if gender == 'M' and perimeter >= 102:
            score += 1
            criteria_met.append('Circunferencia abdominal elevada (≥102 cm)')
        elif gender == 'F' and perimeter >= 88:
            score += 1
            criteria_met.append('Circunferencia abdominal elevada (≥88 cm)')
        
        # Criterio 2: Triglicéridos
        triglycerides = patient_data.get('trigliceridos', 0)
        if triglycerides >= 150:
            score += 1
            criteria_met.append('Triglicéridos elevados (≥150 mg/dL)')
        
        # Criterio 3: HDL Colesterol
        hdl = patient_data.get('hdl_colesterol', 100)
        if (gender == 'M' and hdl < 40) or (gender == 'F' and hdl < 50):
            score += 1
            criteria_met.append('HDL colesterol bajo')
        
        # Criterio 4: Presión arterial
        systolic = patient_data.get('tas', 0)
        diastolic = patient_data.get('tad', 0)
        if systolic >= 130 or diastolic >= 85:
            score += 1
            criteria_met.append('Presión arterial elevada')
        
        # Criterio 5: Glucosa en ayunas
        glucose = patient_data.get('glucosa_ayunas', 0)
        if glucose >= 100:
            score += 1
            criteria_met.append('Glucosa en ayunas elevada (≥100 mg/dL)')
        
        # Interpretación
        if score >= 3:
            interpretation = 'Síndrome metabólico presente'
            risk_level = 'high'
        elif score == 2:
            interpretation = 'Riesgo elevado de síndrome metabólico'
            risk_level = 'moderate'
        else:
            interpretation = 'Bajo riesgo de síndrome metabólico'
            risk_level = 'low'
        
        return {
            'value': score,
            'interpretation': interpretation,
            'risk_level': risk_level,
            'criteria_met': criteria_met,
            'total_criteria': 5
        }
    
    def calculate_advanced_lipid_ratios(self, lipid_panel: Dict) -> Dict:
        """Calcula ratios lipídicos avanzados"""
        
        results = {}
        
        # Ratio TG/HDL
        tg = lipid_panel.get('trigliceridos', 0)
        hdl = lipid_panel.get('hdl_colesterol', 1)  # Evitar división por cero
        
        if hdl > 0:
            tg_hdl_ratio = tg / hdl
            
            if tg_hdl_ratio < 2.0:
                tg_hdl_interp = 'Ratio óptimo'
                tg_hdl_risk = 'low'
            elif tg_hdl_ratio < 3.5:
                tg_hdl_interp = 'Ratio limítrofe'
                tg_hdl_risk = 'moderate'
            else:
                tg_hdl_interp = 'Ratio elevado - alto riesgo'
                tg_hdl_risk = 'high'
            
            results['tg_hdl_ratio'] = {
                'value': round(tg_hdl_ratio, 2),
                'interpretation': tg_hdl_interp,
                'risk_level': tg_hdl_risk
            }
        
        # Ratio LDL/HDL
        ldl = lipid_panel.get('ldl_colesterol', 0)
        if hdl > 0:
            ldl_hdl_ratio = ldl / hdl
            
            if ldl_hdl_ratio < 2.5:
                ldl_hdl_interp = 'Ratio óptimo'
                ldl_hdl_risk = 'low'
            elif ldl_hdl_ratio < 3.5:
                ldl_hdl_interp = 'Ratio limítrofe'
                ldl_hdl_risk = 'moderate'
            else:
                ldl_hdl_interp = 'Ratio elevado'
                ldl_hdl_risk = 'high'
            
            results['ldl_hdl_ratio'] = {
                'value': round(ldl_hdl_ratio, 2),
                'interpretation': ldl_hdl_interp,
                'risk_level': ldl_hdl_risk
            }
        
        # Colesterol no-HDL
        total_chol = lipid_panel.get('colesterol_total', 0)
        non_hdl = total_chol - hdl
        
        if non_hdl < 130:
            non_hdl_interp = 'Óptimo'
            non_hdl_risk = 'low'
        elif non_hdl < 160:
            non_hdl_interp = 'Cerca de óptimo'
            non_hdl_risk = 'low'
        elif non_hdl < 190:
            non_hdl_interp = 'Limítrofe alto'
            non_hdl_risk = 'moderate'
        else:
            non_hdl_interp = 'Alto'
            non_hdl_risk = 'high'
        
        results['non_hdl_cholesterol'] = {
            'value': round(non_hdl, 1),
            'interpretation': non_hdl_interp,
            'risk_level': non_hdl_risk
        }
        
        return results

class BiomarkerTrendAnalyzer:
    """Analizador de tendencias para biomarcadores temporales"""
    
    def __init__(self):
        self.min_data_points = 3
        
    def analyze_glucose_variability(self, glucose_readings: List[Tuple[datetime, float]]) -> Dict:
        """Analiza variabilidad glucémica"""
        
        if len(glucose_readings) < self.min_data_points:
            return {'error': 'Insufficient data points'}
        
        values = [reading[1] for reading in glucose_readings]
        
        # Estadísticas básicas
        mean_glucose = np.mean(values)
        std_glucose = np.std(values)
        cv_glucose = (std_glucose / mean_glucose) * 100  # Coeficiente de variación
        
        # Time in Range (TIR) - % tiempo en rango objetivo (70-180 mg/dL)
        in_range = sum(1 for v in values if 70 <= v <= 180)
        tir_percentage = (in_range / len(values)) * 100
        
        # Episodios de hipoglucemia e hiperglucemia
        hypo_episodes = sum(1 for v in values if v < 70)
        hyper_episodes = sum(1 for v in values if v > 180)
        
        # Interpretación clínica
        if tir_percentage >= 70:
            glucose_control = 'Excelente control glucémico'
            control_level = 'excellent'
        elif tir_percentage >= 50:
            glucose_control = 'Buen control glucémico'
            control_level = 'good'
        else:
            glucose_control = 'Control glucémico deficiente'
            control_level = 'poor'
        
        return {
            'mean_glucose': round(mean_glucose, 1),
            'glucose_variability': round(cv_glucose, 1),
            'time_in_range_percent': round(tir_percentage, 1),
            'hypoglycemia_episodes': hypo_episodes,
            'hyperglycemia_episodes': hyper_episodes,
            'interpretation': glucose_control,
            'control_level': control_level,
            'total_readings': len(values)
        }
    
    def detect_dawn_phenomenon(self, hourly_glucose: Dict[int, List[float]]) -> Dict:
        """Detecta fenómeno del amanecer en glucosa"""
        
        # Comparar glucosa 3-6 AM vs 6-9 AM
        pre_dawn = hourly_glucose.get(3, []) + hourly_glucose.get(4, []) + hourly_glucose.get(5, [])
        dawn = hourly_glucose.get(6, []) + hourly_glucose.get(7, []) + hourly_glucose.get(8, [])
        
        if not pre_dawn or not dawn:
            return {'detected': False, 'reason': 'Insufficient data'}
        
        pre_dawn_avg = np.mean(pre_dawn)
        dawn_avg = np.mean(dawn)
        
        # Incremento >30 mg/dL sugiere fenómeno del amanecer
        glucose_rise = dawn_avg - pre_dawn_avg
        
        if glucose_rise >= 30:
            return {
                'detected': True,
                'glucose_rise_mg_dl': round(glucose_rise, 1),
                'pre_dawn_avg': round(pre_dawn_avg, 1),
                'dawn_avg': round(dawn_avg, 1),
                'severity': 'moderate' if glucose_rise < 50 else 'severe'
            }
        else:
            return {
                'detected': False,
                'glucose_rise_mg_dl': round(glucose_rise, 1),
                'interpretation': 'Normal glucose pattern'
            }

class IntegratedBiomarkerAssessment:
    """Evaluación integrada de múltiples biomarcadores"""
    
    def __init__(self):
        self.calculator = BiomarkerCalculator()
        self.trend_analyzer = BiomarkerTrendAnalyzer()
        
    def comprehensive_diabetes_risk_assessment(self, patient_data: Dict) -> Dict:
        """Evaluación completa de riesgo de diabetes usando múltiples biomarcadores"""
        
        assessment = {
            'patient_id': patient_data.get('patient_id', 'unknown'),
            'assessment_date': datetime.now().isoformat(),
            'biomarkers': {},
            'risk_factors': {},
            'recommendations': [],
            'overall_risk': 'low'
        }
        
        # 1. Biomarcadores básicos
        if 'glucosa_ayunas' in patient_data and 'insulina_ayunas' in patient_data:
            homa_ir = self.calculator.calculate_homa_ir(
                patient_data['glucosa_ayunas'],
                patient_data['insulina_ayunas']
            )
            assessment['biomarkers']['homa_ir'] = homa_ir
        
        # 2. Síndrome metabólico
        metabolic_syndrome = self.calculator.calculate_metabolic_syndrome_score(patient_data)
        assessment['biomarkers']['metabolic_syndrome'] = metabolic_syndrome
        
        # 3. Panel lipídico avanzado
        if any(key in patient_data for key in ['trigliceridos', 'hdl_colesterol', 'ldl_colesterol']):
            lipid_ratios = self.calculator.calculate_advanced_lipid_ratios(patient_data)
            assessment['biomarkers']['lipid_profile'] = lipid_ratios
        
        # 4. Factores de riesgo adicionales
        risk_score = 0
        
        # Edad
        age = patient_data.get('edad', 0)
        if age >= 45:
            risk_score += 2
            assessment['risk_factors']['age'] = 'Risk factor (≥45 years)'
        
        # IMC
        bmi = patient_data.get('imc', 0)
        if bmi >= 30:
            risk_score += 3
            assessment['risk_factors']['obesity'] = 'High risk (BMI ≥30)'
        elif bmi >= 25:
            risk_score += 1
            assessment['risk_factors']['overweight'] = 'Moderate risk (BMI 25-30)'
        
        # Antecedentes familiares
        family_history = patient_data.get('Dx_Diabetes_Tipo2_Familia', 'No')
        if 'Padres o Hermanos' in family_history:
            risk_score += 3
            assessment['risk_factors']['family_history'] = 'High risk (first-degree relatives)'
        elif 'Abuelos' in family_history:
            risk_score += 1
            assessment['risk_factors']['family_history'] = 'Moderate risk (second-degree relatives)'
        
        # Hipertensión
        systolic = patient_data.get('tas', 0)
        if systolic >= 140 or patient_data.get('medicamentos_hta') == 'SI':
            risk_score += 2
            assessment['risk_factors']['hypertension'] = 'Present'
        
        # 5. Determinar riesgo general y recomendaciones
        if risk_score >= 8:
            assessment['overall_risk'] = 'high'
            assessment['recommendations'].extend([
                'Evaluación endocrinológica urgente',
                'Prueba de tolerancia oral a glucosa',
                'HbA1c cada 3 meses',
                'Intervención intensiva en estilo de vida'
            ])
        elif risk_score >= 5:
            assessment['overall_risk'] = 'moderate'
            assessment['recommendations'].extend([
                'Consulta con médico interno',
                'HbA1c cada 6 meses',
                'Programa de pérdida de peso',
                'Ejercicio regular supervisado'
            ])
        else:
            assessment['overall_risk'] = 'low'
            assessment['recommendations'].extend([
                'Seguimiento anual de rutina',
                'Mantener peso saludable',
                'Actividad física regular'
            ])
        
        # 6. Agregar recomendaciones específicas por biomarcador
        if 'homa_ir' in assessment['biomarkers']:
            homa_risk = assessment['biomarkers']['homa_ir']['risk_level']
            if homa_risk in ['moderate', 'high']:
                assessment['recommendations'].append('Considerar metformina')
        
        if 'metabolic_syndrome' in assessment['biomarkers']:
            ms_score = assessment['biomarkers']['metabolic_syndrome']['value']
            if ms_score >= 3:
                assessment['recommendations'].append('Manejo integral de síndrome metabólico')
        
        return assessment

def generate_biomarker_report(patient_data: Dict, output_path: Optional[str] = None) -> str:
    """Genera reporte completo de biomarcadores"""
    
    assessor = IntegratedBiomarkerAssessment()
    assessment = assessor.comprehensive_diabetes_risk_assessment(patient_data)
    
    # Generar reporte en formato texto
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("REPORTE DE BIOMARCADORES - RIESGO DIABETES TIPO 2")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    # Información del paciente
    report_lines.append(f"Paciente: {assessment['patient_id']}")
    report_lines.append(f"Fecha de evaluación: {assessment['assessment_date'][:10]}")
    report_lines.append(f"Riesgo general: {assessment['overall_risk'].upper()}")
    report_lines.append("")
    
    # Biomarcadores
    report_lines.append("BIOMARCADORES EVALUADOS:")
    report_lines.append("-" * 30)
    
    for biomarker_name, biomarker_data in assessment['biomarkers'].items():
        report_lines.append(f"\n{biomarker_name.upper()}:")
        
        if isinstance(biomarker_data, dict):
            if 'value' in biomarker_data:
                report_lines.append(f"  Valor: {biomarker_data['value']}")
                report_lines.append(f"  Interpretación: {biomarker_data['interpretation']}")
                report_lines.append(f"  Nivel de riesgo: {biomarker_data['risk_level']}")
        else:
            # Para biomarcadores complejos como panel lipídico
            for sub_marker, sub_data in biomarker_data.items():
                report_lines.append(f"  {sub_marker}: {sub_data['value']} - {sub_data['interpretation']}")
    
    # Factores de riesgo
    if assessment['risk_factors']:
        report_lines.append("\n\nFACTORES DE RIESGO IDENTIFICADOS:")
        report_lines.append("-" * 35)
        for factor, description in assessment['risk_factors'].items():
            report_lines.append(f"• {factor}: {description}")
    
    # Recomendaciones
    report_lines.append("\n\nRECOMENDACIONES:")
    report_lines.append("-" * 15)
    for i, recommendation in enumerate(assessment['recommendations'], 1):
        report_lines.append(f"{i}. {recommendation}")
    
    report_lines.append("\n" + "=" * 60)
    report_lines.append("Nota: Esta evaluación es orientativa. Consulte con su médico.")
    report_lines.append("=" * 60)
    
    report_text = "\n".join(report_lines)
    
    # Guardar archivo si se especifica ruta
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
    
    return report_text

# Ejemplo de uso
def demo_biomarker_analysis():
    """Demostración del análisis de biomarcadores"""
    
    # Datos de ejemplo de un paciente
    patient_example = {
        'patient_id': 'PAT001',
        'edad': 52,
        'sexo': 'M',
        'imc': 29.5,
        'perimetro_abdominal': 105,
        'tas': 145,
        'tad': 92,
        'glucosa_ayunas': 118,
        'insulina_ayunas': 15.2,
        'trigliceridos': 185,
        'hdl_colesterol': 38,
        'ldl_colesterol': 145,
        'colesterol_total': 215,
        'medicamentos_hta': 'SI',
        'Dx_Diabetes_Tipo2_Familia': 'Si: Padres o Hermanos'
    }
    
    # Generar reporte
    report = generate_biomarker_report(patient_example)
    print(report)
    
    return report

if __name__ == "__main__":
    # Ejecutar demostración
    demo_biomarker_analysis()