# Dataset de Prueba DM2 - Documentación

## Descripción del Archivo de Prueba

**Archivo:** `test_dataset.csv`  
**Tamaño:** 25 pacientes simulados  
**Propósito:** Probar la funcionalidad de carga masiva y predicción por lotes

## Estructura de Datos

### Campos Demográficos
- **edad**: Edad del paciente (18-70 años)
- **sexo**: M (Masculino) / F (Femenino)
- **genero**: Masculino / Femenino / Otro
- **regimen**: Subsidiado / Contributivo / Especial

### Medidas Antropométricas
- **peso**: Peso en kilogramos (50-100 kg)
- **talla**: Altura en centímetros (155-180 cm)
- **imc**: Índice de Masa Corporal (calculado automáticamente)
- **perimetro_abdominal**: Circunferencia abdominal en cm (70-115 cm)

### Datos Clínicos
- **tas**: Tensión Arterial Sistólica (105-170 mmHg)
- **tad**: Tensión Arterial Diastólica (65-105 mmHg)
- **puntaje_total**: Puntaje de riesgo total (2.0-9.5)
- **riesgo_dm**: Valor numérico de riesgo DM2 (0.9-7.8)

### Estilo de Vida y Antecedentes
- **realiza_ejercicio**: SI / NO
- **frecuencia_frutas**: SI / NO  
- **Consumo_Cigarrillo**: 0 (No) / 1 (Sí)
- **medicamentos_hta**: SI / NO
- **Dx_Diabetes_Tipo2_Familia**: 
  - "No"
  - "Si: Padres o Hermanos"  
  - "Si: Abuelos, Tios o Primos Hermanos (pero no Padres, Hermanos o Hijos)"

## Casos de Uso Incluidos

### 1. **Bajo Riesgo** (8 casos)
- Pacientes jóvenes (26-35 años)
- IMC normal (21-25)
- Sin factores de riesgo significativos
- Expectativa: Clasificación "Normal"

### 2. **Riesgo Moderado** (9 casos)
- Pacientes mediana edad (37-50 años)
- Sobrepeso leve-moderado (IMC 25-30)
- Algunos factores de riesgo
- Expectativa: Clasificación "Normal" o "Prediabetes"

### 3. **Alto Riesgo** (8 casos)
- Pacientes mayores (52-67 años)
- Obesidad (IMC >30)
- Múltiples factores de riesgo
- Expectativa: Clasificación "Prediabetes" o "Diabetes"

## Instrucciones de Uso

### Carga del Dataset
1. Navega a la sección "Carga de Archivos" en la aplicación web
2. Selecciona el archivo `test_dataset.csv`
3. Haz clic en "Subir y Procesar"
4. El sistema procesará automáticamente todas las filas

### Resultados Esperados
- **Tiempo de procesamiento:** <5 segundos
- **Predicciones:** 25 resultados con probabilidades
- **Formato de salida:** CSV con columnas originales + predicciones
- **Métricas:** Distribución de clases predichas

## Validación de Resultados

### Distribución Esperada (aproximada)
- **Normal:** ~32% (8 casos)
- **Prediabetes:** ~36% (9 casos) 
- **Diabetes:** ~32% (8 casos)

### Campos de Salida Adicionales
- `Prediccion`: Clase predicha (Normal/Prediabetes/Diabetes)
- `Prob_Normal`: Probabilidad de clase Normal (0-1)
- `Prob_Prediabetes`: Probabilidad de clase Prediabetes (0-1)  
- `Prob_Diabetes`: Probabilidad de clase Diabetes (0-1)

## Notas Técnicas

- Todos los datos son **sintéticos** y generados para testing
- Los valores están dentro de rangos clínicos realistas
- El dataset está **balanceado** entre diferentes categorías de riesgo
- Compatible con el formato esperado por el modelo DM2

## Casos de Error para Testing

Para probar la robustez del sistema, puedes modificar el archivo:
- **Valores faltantes:** Dejar algunas celdas vacías
- **Formatos incorrectos:** Cambiar números por texto
- **Rangos inválidos:** Usar valores fuera de rangos médicos

---

*Dataset creado para testing del Pipeline DM2 v2.0*  
*Fecha: Septiembre 2024*