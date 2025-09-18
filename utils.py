"""
Utilidades para el pipeline de DM2
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    f1_score, balanced_accuracy_score, 
    precision_recall_fscore_support,
    roc_auc_score
)
from typing import Dict, List, Tuple, Any
import json
from pathlib import Path

def get_medical_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                       y_proba: np.ndarray = None, 
                       labels: List[str] = None,
                       focus_class: str = "Diabetes") -> Dict[str, float]:
    """
    Calcula métricas médicas específicas para el problema de DM2
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones del modelo
        y_proba: Probabilidades (opcional)
        labels: Lista de clases en orden
        focus_class: Clase de mayor importancia clínica
    
    Returns:
        Dict con métricas médicas
    """
    if labels is None:
        labels = ["Normal", "Prediabetes", "Diabetes"]
    
    # Métricas generales
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Métricas por clase
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    
    # Métricas específicas para la clase focal (Diabetes)
    if focus_class in labels:
        focus_idx = labels.index(focus_class)
        
        # Convertir a problema binario: Focus vs No-Focus
        y_true_bin = (y_true == focus_class).astype(int)
        y_pred_bin = (y_pred == focus_class).astype(int)
        
        # Matriz de confusión binaria
        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
        
        sensitivity = tp / (tp + fn + 1e-12)  # Recall, True Positive Rate
        specificity = tn / (tn + fp + 1e-12)  # True Negative Rate
        ppv = tp / (tp + fp + 1e-12)          # Positive Predictive Value
        npv = tn / (tn + fn + 1e-12)          # Negative Predictive Value
        
        # AUC si hay probabilidades
        auc_focus = None
        if y_proba is not None and len(np.unique(y_true_bin)) > 1:
            try:
                if y_proba.ndim > 1 and focus_idx < y_proba.shape[1]:
                    auc_focus = roc_auc_score(y_true_bin, y_proba[:, focus_idx])
                else:
                    auc_focus = roc_auc_score(y_true_bin, y_pred_bin)
            except:
                auc_focus = None
    else:
        sensitivity = specificity = ppv = npv = auc_focus = 0.0
        focus_idx = -1
    
    metrics = {
        # Métricas generales
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'balanced_accuracy': float(balanced_acc),
        
        # Métricas por clase
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'f1_per_class': f1.tolist(),
        'support_per_class': support.tolist(),
        
        # Métricas específicas para clase focal
        f'{focus_class.lower()}_sensitivity': float(sensitivity),
        f'{focus_class.lower()}_specificity': float(specificity),
        f'{focus_class.lower()}_ppv': float(ppv),
        f'{focus_class.lower()}_npv': float(npv),
        f'{focus_class.lower()}_precision': float(precision[focus_idx]) if focus_idx >= 0 else 0.0,
        f'{focus_class.lower()}_recall': float(recall[focus_idx]) if focus_idx >= 0 else 0.0,
        f'{focus_class.lower()}_f1': float(f1[focus_idx]) if focus_idx >= 0 else 0.0,
    }
    
    if auc_focus is not None:
        metrics[f'{focus_class.lower()}_auc'] = float(auc_focus)
    
    return metrics

def save_results(results: Dict[str, Any], filepath: Path) -> None:
    """Guarda resultados en JSON con formato legible"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

def load_and_validate_data(data_path: Path, target_col: str) -> Tuple[pd.DataFrame, bool]:
    """
    Carga y valida el dataset
    
    Returns:
        Tuple: (dataframe, is_valid)
    """
    try:
        df = pd.read_csv(data_path)
        
        # Validaciones básicas
        if df.empty:
            print("❌ Dataset vacío")
            return df, False
            
        if target_col not in df.columns:
            print(f"❌ Columna target '{target_col}' no encontrada")
            return df, False
            
        # Verificar casos por clase
        class_counts = df[target_col].value_counts()
        print(f"📊 Distribución de clases:\n{class_counts}")
        
        # Advertencias sobre desbalance
        min_class_count = class_counts.min()
        if min_class_count < 10:
            print(f"⚠️  Clase minoritaria tiene solo {min_class_count} casos")
            
        if class_counts.min() / class_counts.max() < 0.1:
            print(f"⚠️  Desbalance severo detectado (ratio: {class_counts.min() / class_counts.max():.3f})")
        
        return df, True
        
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return pd.DataFrame(), False

def generate_detailed_report(results: Dict[str, Any], labels: List[str]) -> str:
    """Genera reporte detallado en formato texto"""
    
    report = "=" * 80 + "\n"
    report += "📈 REPORTE DETALLADO - PIPELINE DM2\n"
    report += "=" * 80 + "\n\n"
    
    # Mejor modelo
    if 'best_model' in results:
        report += f"🏆 MEJOR MODELO: {results['best_model']}\n"
        report += f"📊 F1 Macro: {results.get('best_f1_macro', 'N/A'):.4f}\n\n"
    
    # Métricas por modelo
    for model_name, metrics in results.items():
        if isinstance(metrics, dict) and 'f1_macro' in metrics:
            report += f"🔸 MODELO: {model_name.upper()}\n"
            report += f"   F1 Macro: {metrics['f1_macro']:.4f}\n"
            report += f"   Balanced Acc: {metrics['balanced_accuracy']:.4f}\n"
            
            # Métricas de Diabetes
            if 'diabetes_sensitivity' in metrics:
                report += f"   Diabetes Sensitivity: {metrics['diabetes_sensitivity']:.4f}\n"
                report += f"   Diabetes Specificity: {metrics['diabetes_specificity']:.4f}\n"
                report += f"   Diabetes PPV: {metrics['diabetes_ppv']:.4f}\n"
            
            report += "\n"
    
    report += "=" * 80 + "\n"
    
    return report

print("✅ Utilidades cargadas correctamente")