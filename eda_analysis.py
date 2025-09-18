#!/usr/bin/env python3
"""
Análisis Exploratorio de Datos (EDA) para el dataset DM2
========================================================

Incluye:
- ✅ Estadísticas descriptivas
- ✅ Análisis de distribuciones
- ✅ Correlaciones y asociaciones
- ✅ Detección de outliers
- ✅ Visualizaciones informativas
- ✅ Análisis de calidad de datos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Scipy para estadísticas
from scipy import stats
from scipy.stats import chi2_contingency

# Local imports
from config import *
from utils import load_and_validate_data

class EDAAnalyzer:
    """Analizador completo para EDA del dataset DM2"""
    
    def __init__(self):
        self.df = None
        self.numeric_cols = []
        self.categorical_cols = []
        self.report = []
        
        # Configurar estilo de plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        print("🔍 Inicializando EDA Analyzer...")
    
    def load_data(self):
        """Carga el dataset"""
        self.df, is_valid = load_and_validate_data(DATA_PATH, TARGET)
        
        if not is_valid:
            return False
        
        # Identificar tipos de columnas
        blacklist = [col for col in BLACKLIST_COLS if col in self.df.columns]
        pii = [col for col in PII_COLS if col in self.df.columns]
        
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.difference(blacklist + pii).tolist()
        self.categorical_cols = self.df.select_dtypes(include=["object"]).columns.difference(blacklist + pii + [TARGET]).tolist()
        
        self.report.append(f"📊 Dataset shape: {self.df.shape}")
        self.report.append(f"🔢 Numeric columns: {len(self.numeric_cols)}")
        self.report.append(f"📝 Categorical columns: {len(self.categorical_cols)}")
        
        return True
    
    def basic_statistics(self):
        """Estadísticas descriptivas básicas"""
        print("📈 Generando estadísticas descriptivas...")
        
        # Target distribution
        target_dist = self.df[TARGET].value_counts()
        target_pct = self.df[TARGET].value_counts(normalize=True) * 100
        
        self.report.append("\n🎯 DISTRIBUCIÓN DEL TARGET")
        self.report.append("=" * 40)
        for class_name in CLASS_ORDER:
            count = target_dist.get(class_name, 0)
            pct = target_pct.get(class_name, 0)
            self.report.append(f"{class_name}: {count} ({pct:.1f}%)")
        
        # Missing values analysis
        missing_analysis = []
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            missing_pct = (missing_count / len(self.df)) * 100
            if missing_count > 0:
                missing_analysis.append((col, missing_count, missing_pct))
        
        if missing_analysis:
            self.report.append("\n🔍 VALORES FALTANTES")
            self.report.append("=" * 40)
            missing_analysis.sort(key=lambda x: x[2], reverse=True)
            for col, count, pct in missing_analysis[:10]:  # Top 10
                self.report.append(f"{col}: {count} ({pct:.1f}%)")
        
        # Numeric columns statistics
        if self.numeric_cols:
            numeric_stats = self.df[self.numeric_cols].describe()
            
            # Save to file
            stats_path = OUTPUT_DIR / "numeric_statistics.csv"
            numeric_stats.to_csv(stats_path)
            self.report.append(f"\n📊 Estadísticas numéricas guardadas en: {stats_path}")
    
    def analyze_correlations(self):
        """Análisis de correlaciones para variables numéricas"""
        print("🔗 Analizando correlaciones...")
        
        if len(self.numeric_cols) < 2:
            self.report.append("\n⚠️ Insuficientes variables numéricas para correlación")
            return
        
        # Correlation matrix
        corr_matrix = self.df[self.numeric_cols].corr()
        
        # Find high correlations (excluding diagonal)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    high_corr_pairs.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        corr_val
                    ))
        
        if high_corr_pairs:
            self.report.append("\n🔗 CORRELACIONES ALTAS (|r| > 0.7)")
            self.report.append("=" * 50)
            for var1, var2, corr_val in high_corr_pairs:
                self.report.append(f"{var1} ↔ {var2}: {corr_val:.3f}")
        
        # Generate correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Matriz de Correlaciones - Variables Numéricas')
        plt.tight_layout()
        
        corr_plot_path = OUTPUT_DIR / "correlation_heatmap.png"
        plt.savefig(corr_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.report.append(f"\n📈 Heatmap de correlaciones guardado en: {corr_plot_path}")
    
    def analyze_target_relationships(self):
        """Analiza relación de variables con el target"""
        print("🎯 Analizando relaciones con el target...")
        
        significant_vars = []
        
        # Para variables numéricas: ANOVA
        for col in self.numeric_cols:
            if self.df[col].isnull().all():
                continue
                
            groups = [self.df[self.df[TARGET] == class_name][col].dropna() 
                     for class_name in CLASS_ORDER 
                     if len(self.df[self.df[TARGET] == class_name]) > 0]
            
            if len(groups) >= 2 and all(len(g) > 0 for g in groups):
                try:
                    f_stat, p_value = stats.f_oneway(*groups)
                    if p_value < 0.05:
                        significant_vars.append((col, 'numeric', f_stat, p_value))
                except:
                    pass
        
        # Para variables categóricas: Chi-squared
        for col in self.categorical_cols:
            if self.df[col].isnull().all():
                continue
                
            # Create contingency table
            contingency = pd.crosstab(self.df[col], self.df[TARGET])
            
            # Ensure minimum cell count
            if contingency.min().min() >= 5:
                try:
                    chi2, p_value, _, _ = chi2_contingency(contingency)
                    if p_value < 0.05:
                        significant_vars.append((col, 'categorical', chi2, p_value))
                except:
                    pass
        
        if significant_vars:
            self.report.append("\n🎯 VARIABLES SIGNIFICATIVAMENTE ASOCIADAS AL TARGET (p < 0.05)")
            self.report.append("=" * 70)
            significant_vars.sort(key=lambda x: x[3])  # Sort by p-value
            
            for var, var_type, stat, p_val in significant_vars[:15]:  # Top 15
                stat_name = "F-stat" if var_type == 'numeric' else "Chi2"
                self.report.append(f"{var} ({var_type}): {stat_name}={stat:.3f}, p={p_val:.4f}")
        
        return significant_vars
    
    def detect_outliers(self):
        """Detección de outliers usando IQR"""
        print("🚨 Detectando outliers...")
        
        outlier_summary = []
        
        for col in self.numeric_cols:
            if self.df[col].isnull().all():
                continue
                
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_pct = (outlier_count / len(self.df)) * 100
            
            if outlier_count > 0:
                outlier_summary.append((col, outlier_count, outlier_pct))
        
        if outlier_summary:
            self.report.append("\n🚨 OUTLIERS DETECTADOS (IQR method)")
            self.report.append("=" * 40)
            outlier_summary.sort(key=lambda x: x[2], reverse=True)
            
            for col, count, pct in outlier_summary[:10]:  # Top 10
                self.report.append(f"{col}: {count} ({pct:.1f}%)")
    
    def generate_visualizations(self):
        """Genera visualizaciones clave"""
        print("📊 Generando visualizaciones...")
        
        # 1. Target distribution
        plt.figure(figsize=(10, 6))
        target_counts = self.df[TARGET].value_counts()
        colors = sns.color_palette("husl", len(target_counts))
        
        plt.subplot(1, 2, 1)
        target_counts.plot(kind='bar', color=colors)
        plt.title('Distribución de Clases')
        plt.xlabel('Clase')
        plt.ylabel('Frecuencia')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        target_counts.plot(kind='pie', autopct='%1.1f%%', colors=colors)
        plt.title('Proporción de Clases')
        plt.ylabel('')
        
        plt.tight_layout()
        target_plot_path = OUTPUT_DIR / "target_distribution.png"
        plt.savefig(target_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Numeric variables distribution by target
        if self.numeric_cols:
            n_cols = min(4, len(self.numeric_cols))
            n_rows = (len(self.numeric_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
            
            for i, col in enumerate(self.numeric_cols[:n_rows*n_cols]):
                ax = axes[i] if len(axes) > 1 else axes
                
                for class_name in CLASS_ORDER:
                    class_data = self.df[self.df[TARGET] == class_name][col].dropna()
                    if len(class_data) > 0:
                        ax.hist(class_data, alpha=0.7, label=class_name, bins=20)
                
                ax.set_title(f'Distribución: {col}')
                ax.legend()
                ax.set_xlabel(col)
                ax.set_ylabel('Frecuencia')
            
            # Hide empty subplots
            for i in range(len(self.numeric_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            numeric_dist_path = OUTPUT_DIR / "numeric_distributions.png"
            plt.savefig(numeric_dist_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        self.report.append(f"\n📊 Visualizaciones guardadas en: {OUTPUT_DIR}")
    
    def generate_summary_report(self):
        """Genera reporte resumen del EDA"""
        print("📝 Generando reporte resumen...")
        
        # Add data quality summary
        self.report.append("\n📋 RESUMEN DE CALIDAD DE DATOS")
        self.report.append("=" * 50)
        
        total_missing = self.df.isnull().sum().sum()
        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_pct = (total_missing / total_cells) * 100
        
        self.report.append(f"Total de valores faltantes: {total_missing} ({missing_pct:.2f}%)")
        self.report.append(f"Filas completas: {self.df.dropna().shape[0]} ({(self.df.dropna().shape[0]/len(self.df)*100):.1f}%)")
        
        # Duplicates
        duplicates = self.df.duplicated().sum()
        self.report.append(f"Filas duplicadas: {duplicates}")
        
        # Save report
        report_text = "\n".join(self.report)
        
        eda_report_path = OUTPUT_DIR / "eda_report.txt"
        with open(eda_report_path, 'w', encoding='utf-8') as f:
            f.write("🔍 ANÁLISIS EXPLORATORIO DE DATOS - DATASET DM2\n")
            f.write("=" * 60 + "\n\n")
            f.write(report_text)
        
        print(f"\n✅ Reporte EDA completo guardado en: {eda_report_path}")
        
        # Display key findings
        print("\n" + "="*60)
        print("📊 HALLAZGOS CLAVE DEL EDA")
        print("="*60)
        for line in self.report[:20]:  # Show first 20 lines
            print(line)
        print("="*60)
    
    def run_complete_eda(self):
        """Ejecuta el análisis EDA completo"""
        try:
            print("🚀 Iniciando Análisis Exploratorio de Datos...")
            
            if not self.load_data():
                return False
            
            self.basic_statistics()
            self.analyze_correlations()
            significant_vars = self.analyze_target_relationships()
            self.detect_outliers()
            self.generate_visualizations()
            self.generate_summary_report()
            
            print("\n🎉 ¡EDA completado exitosamente!")
            return True
            
        except Exception as e:
            print(f"💥 Error en EDA: {e}")
            return False

if __name__ == "__main__":
    analyzer = EDAAnalyzer()
    success = analyzer.run_complete_eda()
    
    if success:
        print(f"\n📁 Todos los archivos del EDA están en: {OUTPUT_DIR}")
    else:
        print("\n💥 EDA falló. Revisa los errores anteriores.")