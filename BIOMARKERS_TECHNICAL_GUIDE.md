# Guía Técnica de Integración de Biomarcadores - Pipeline DM2

## 🏗️ Arquitectura Técnica para Biomarcadores

### Stack Tecnológico Recomendado

```
┌─────────────────────────────────────────────────────────┐
│                    CAPA DE PRESENTACIÓN                 │
│  React.js + TypeScript + Material-UI + Chart.js       │
└─────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────┐
│                    CAPA DE API REST                     │
│     FastAPI + Pydantic + Authentication + CORS        │
└─────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────┐
│                  CAPA DE LÓGICA DE NEGOCIO              │
│  Python + Pandas + NumPy + SciPy + Scikit-Learn      │
└─────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────┐
│                   CAPA DE DATOS                         │
│  PostgreSQL + Redis + InfluxDB + MinIO + MongoDB      │
└─────────────────────────────────────────────────────────┘
```

---

## 🗄️ Modelo de Datos para Biomarcadores

### Schema PostgreSQL

```sql
-- Tabla principal de pacientes
CREATE TABLE patients (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    birth_date DATE NOT NULL,
    gender VARCHAR(10) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Catálogo de biomarcadores
CREATE TABLE biomarker_types (
    id SERIAL PRIMARY KEY,
    code VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    category VARCHAR(50) NOT NULL, -- 'molecular', 'physiological', 'digital', etc.
    unit VARCHAR(20),
    reference_range_min DECIMAL(10,4),
    reference_range_max DECIMAL(10,4),
    description TEXT,
    is_active BOOLEAN DEFAULT true
);

-- Mediciones de biomarcadores
CREATE TABLE biomarker_measurements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID REFERENCES patients(id),
    biomarker_type_id INTEGER REFERENCES biomarker_types(id),
    value DECIMAL(15,6) NOT NULL,
    measurement_date TIMESTAMP NOT NULL,
    device_id VARCHAR(50),
    quality_score DECIMAL(3,2), -- 0.00-1.00
    flags JSONB, -- {"outlier": false, "validated": true}
    raw_data JSONB, -- Datos originales del sensor
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Índices para optimización
CREATE INDEX idx_biomarker_patient_date ON biomarker_measurements(patient_id, measurement_date);
CREATE INDEX idx_biomarker_type_date ON biomarker_measurements(biomarker_type_id, measurement_date);
CREATE INDEX idx_quality_score ON biomarker_measurements(quality_score) WHERE quality_score >= 0.8;
```

### Schema InfluxDB (Series Temporales)

```sql
-- Para datos de wearables y sensores continuos
CREATE MEASUREMENT continuous_biomarkers (
    time: timestamp,
    patient_id: tag,
    device_id: tag,
    biomarker_code: tag,
    value: field,
    quality: field,
    battery_level: field
)

-- Políticas de retención
CREATE RETENTION POLICY "raw_data" ON "biomarkers_db" DURATION 90d REPLICATION 1
CREATE RETENTION POLICY "downsampled" ON "biomarkers_db" DURATION 2y REPLICATION 1

-- Continuous queries para agregación
CREATE CONTINUOUS QUERY "hourly_averages" ON "biomarkers_db"
BEGIN
  SELECT mean("value") AS "mean_value", max("value") AS "max_value", min("value") AS "min_value"
  INTO "downsampled"."biomarker_hourly"
  FROM "continuous_biomarkers"
  GROUP BY time(1h), "patient_id", "biomarker_code"
END
```

---

## 🔄 API Design Patterns

### RESTful Endpoints

```python
# app/routers/biomarkers.py
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from datetime import datetime

router = APIRouter(prefix="/api/v1/biomarkers", tags=["biomarkers"])

@router.get("/types")
async def get_biomarker_types(
    category: Optional[str] = None,
    active_only: bool = True
) -> List[BiomarkerType]:
    """Obtiene catálogo de tipos de biomarcadores"""
    pass

@router.post("/measurements")
async def create_measurement(
    measurement: BiomarkerMeasurementCreate,
    current_user: User = Depends(get_current_user)
) -> BiomarkerMeasurement:
    """Registra nueva medición de biomarcador"""
    pass

@router.get("/patients/{patient_id}/measurements")
async def get_patient_measurements(
    patient_id: str,
    biomarker_codes: Optional[List[str]] = Query(None),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = Query(100, le=1000)
) -> List[BiomarkerMeasurement]:
    """Obtiene mediciones de un paciente"""
    pass

@router.get("/patients/{patient_id}/trends")
async def get_biomarker_trends(
    patient_id: str,
    biomarker_code: str,
    period: str = Query("30d", regex="^(7d|30d|90d|1y)$")
) -> BiomarkerTrend:
    """Calcula tendencias de biomarcadores"""
    pass

@router.post("/batch/upload")
async def batch_upload_measurements(
    file: UploadFile,
    device_id: Optional[str] = None
) -> BatchUploadResult:
    """Carga masiva de mediciones desde CSV/JSON"""
    pass
```

### Modelos Pydantic

```python
# app/schemas/biomarkers.py
from pydantic import BaseModel, validator, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class BiomarkerCategory(str, Enum):
    MOLECULAR = "molecular"
    PHYSIOLOGICAL = "physiological"
    DIGITAL = "digital"
    IMAGING = "imaging"

class BiomarkerType(BaseModel):
    id: int
    code: str = Field(..., regex="^[A-Z0-9_]+$")
    name: str
    category: BiomarkerCategory
    unit: Optional[str] = None
    reference_range_min: Optional[float] = None
    reference_range_max: Optional[float] = None
    description: Optional[str] = None

class BiomarkerMeasurementCreate(BaseModel):
    patient_id: str
    biomarker_code: str
    value: float
    measurement_date: datetime
    device_id: Optional[str] = None
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    flags: Optional[Dict[str, Any]] = None
    raw_data: Optional[Dict[str, Any]] = None

    @validator('value')
    def validate_value(cls, v):
        if not -999999.0 <= v <= 999999.0:
            raise ValueError('Value out of allowed range')
        return v

class BiomarkerMeasurement(BiomarkerMeasurementCreate):
    id: str
    created_at: datetime
    biomarker_type: BiomarkerType

    class Config:
        orm_mode = True

class BiomarkerTrend(BaseModel):
    biomarker_code: str
    patient_id: str
    period: str
    measurements_count: int
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_slope: float
    current_value: Optional[float]
    average_value: float
    min_value: float
    max_value: float
    coefficient_of_variation: float
    quality_metrics: Dict[str, float]
```

---

## 🔬 Procesamiento de Datos de Biomarcadores

### Pipeline de Calidad de Datos

```python
# app/services/biomarker_processing.py
import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, Dict, Any

class BiomarkerProcessor:
    
    def __init__(self, biomarker_type: BiomarkerType):
        self.biomarker_type = biomarker_type
        self.outlier_threshold = 3.0  # Z-score
        
    def validate_measurement(self, value: float) -> Tuple[bool, Dict[str, Any]]:
        """Valida una medición individual"""
        flags = {}
        is_valid = True
        
        # Validar rango de referencia
        if self.biomarker_type.reference_range_min is not None:
            if value < self.biomarker_type.reference_range_min:
                flags['below_reference'] = True
                
        if self.biomarker_type.reference_range_max is not None:
            if value > self.biomarker_type.reference_range_max:
                flags['above_reference'] = True
        
        # Validar valores extremos
        if abs(value) > 1e6:  # Valor absurdamente alto
            flags['extreme_value'] = True
            is_valid = False
            
        return is_valid, flags
    
    def detect_outliers(self, values: List[float]) -> List[bool]:
        """Detecta outliers usando Z-score modificado"""
        if len(values) < 3:
            return [False] * len(values)
            
        values_array = np.array(values)
        median = np.median(values_array)
        mad = stats.median_abs_deviation(values_array)
        
        # Z-score modificado (más robusto que Z-score estándar)
        modified_z_scores = 0.6745 * (values_array - median) / mad
        
        return np.abs(modified_z_scores) > self.outlier_threshold
    
    def calculate_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcula tendencias estadísticas"""
        if len(df) < 2:
            return {"trend_direction": "insufficient_data"}
        
        # Ordenar por fecha
        df_sorted = df.sort_values('measurement_date')
        
        # Regresión lineal simple
        x = np.arange(len(df_sorted))
        y = df_sorted['value'].values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Determinar dirección de tendencia
        if abs(slope) < std_err * 2:  # No significativo estadísticamente
            trend_direction = "stable"
        elif slope > 0:
            trend_direction = "increasing"
        else:
            trend_direction = "decreasing"
        
        return {
            "trend_direction": trend_direction,
            "trend_slope": slope,
            "r_squared": r_value ** 2,
            "p_value": p_value,
            "standard_error": std_err,
            "coefficient_of_variation": np.std(y) / np.mean(y) if np.mean(y) != 0 else 0
        }

class RealTimeBiomarkerAnalyzer:
    """Análisis en tiempo real para datos de wearables"""
    
    def __init__(self):
        self.window_size = 100  # Ventana deslizante
        self.alert_thresholds = {}
        
    def process_stream(self, patient_id: str, biomarker_code: str, 
                      value: float, timestamp: datetime) -> Dict[str, Any]:
        """Procesa datos en streaming"""
        
        # Obtener últimas mediciones
        recent_values = self.get_recent_values(patient_id, biomarker_code)
        
        # Análisis de ventana deslizante
        alerts = []
        
        # Detectar cambios súbitos
        if len(recent_values) >= 2:
            last_value = recent_values[-1]
            change_rate = abs(value - last_value) / last_value if last_value != 0 else 0
            
            if change_rate > 0.2:  # Cambio > 20%
                alerts.append({
                    "type": "sudden_change",
                    "severity": "medium" if change_rate < 0.5 else "high",
                    "change_rate": change_rate
                })
        
        # Detectar patrones anómalos
        if len(recent_values) >= self.window_size:
            z_score = self.calculate_z_score(value, recent_values)
            if abs(z_score) > 2.5:
                alerts.append({
                    "type": "statistical_anomaly",
                    "severity": "high",
                    "z_score": z_score
                })
        
        return {
            "alerts": alerts,
            "quality_score": self.calculate_quality_score(value, recent_values),
            "processed_at": datetime.utcnow()
        }
```

---

## 📊 Machine Learning para Biomarcadores

### Feature Engineering Especializado

```python
# app/ml/biomarker_features.py
import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.preprocessing import StandardScaler, RobustScaler

class BiomarkerFeatureEngineer:
    
    def __init__(self):
        self.scalers = {}
        self.feature_names = []
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características temporales"""
        df = df.copy()
        
        # Características básicas de tiempo
        df['hour'] = df['measurement_date'].dt.hour
        df['day_of_week'] = df['measurement_date'].dt.dayofweek
        df['month'] = df['measurement_date'].dt.month
        
        # Características circadianas
        df['circadian_phase'] = np.sin(2 * np.pi * df['hour'] / 24)
        
        # Tiempo desde última comida (asumiendo horarios típicos)
        meal_hours = [7, 12, 19]  # Desayuno, almuerzo, cena
        df['time_since_meal'] = df['hour'].apply(
            lambda h: min([abs(h - meal) for meal in meal_hours])
        )
        
        return df
    
    def create_statistical_features(self, df: pd.DataFrame, 
                                  window_sizes: List[int] = [7, 30, 90]) -> pd.DataFrame:
        """Crea características estadísticas con ventanas deslizantes"""
        df = df.copy()
        
        for window in window_sizes:
            # Media móvil
            df[f'rolling_mean_{window}d'] = df.groupby('patient_id')['value'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            # Desviación estándar móvil
            df[f'rolling_std_{window}d'] = df.groupby('patient_id')['value'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            
            # Percentiles móviles
            df[f'rolling_p25_{window}d'] = df.groupby('patient_id')['value'].transform(
                lambda x: x.rolling(window=window, min_periods=1).quantile(0.25)
            )
            
            df[f'rolling_p75_{window}d'] = df.groupby('patient_id')['value'].transform(
                lambda x: x.rolling(window=window, min_periods=1).quantile(0.75)
            )
            
            # Tendencia (slope de regresión lineal)
            df[f'trend_slope_{window}d'] = df.groupby('patient_id')['value'].transform(
                lambda x: self._calculate_rolling_slope(x, window)
            )
        
        return df
    
    def create_cross_biomarker_features(self, biomarker_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Crea características combinando múltiples biomarcadores"""
        
        # Ejemplo: Ratio de biomarcadores relacionados
        if 'glucose' in biomarker_dict and 'insulin' in biomarker_dict:
            glucose_df = biomarker_dict['glucose']
            insulin_df = biomarker_dict['insulin']
            
            # Merge por paciente y fecha
            combined = pd.merge(glucose_df, insulin_df, 
                              on=['patient_id', 'measurement_date'], 
                              suffixes=('_glucose', '_insulin'))
            
            # HOMA-IR calculado
            combined['homa_ir'] = (combined['value_glucose'] * combined['value_insulin']) / 22.5
            
            return combined
        
        return pd.DataFrame()
    
    def _calculate_rolling_slope(self, series: pd.Series, window: int) -> pd.Series:
        """Calcula la pendiente en ventana deslizante"""
        def slope_func(y):
            if len(y) < 2:
                return 0
            x = np.arange(len(y))
            return np.polyfit(x, y, 1)[0]
        
        return series.rolling(window=window, min_periods=2).apply(slope_func)

class BiomarkerMLPipeline:
    """Pipeline completo de ML para biomarcadores"""
    
    def __init__(self):
        self.feature_engineer = BiomarkerFeatureEngineer()
        self.model = None
        self.is_trained = False
    
    def prepare_data(self, biomarker_data: Dict[str, pd.DataFrame], 
                    target_column: str = 'diabetes_risk') -> Tuple[np.ndarray, np.ndarray]:
        """Prepara datos para entrenamiento"""
        
        # Feature engineering
        features_list = []
        for biomarker_code, df in biomarker_data.items():
            # Características temporales
            df_temporal = self.feature_engineer.create_temporal_features(df)
            
            # Características estadísticas
            df_stats = self.feature_engineer.create_statistical_features(df_temporal)
            
            features_list.append(df_stats)
        
        # Combinar todas las características
        combined_features = pd.concat(features_list, ignore_index=True)
        
        # Crear características cruzadas
        cross_features = self.feature_engineer.create_cross_biomarker_features(biomarker_data)
        
        if not cross_features.empty:
            combined_features = pd.merge(combined_features, cross_features, 
                                       on=['patient_id', 'measurement_date'], 
                                       how='left')
        
        # Preparar X, y
        feature_cols = [col for col in combined_features.columns 
                       if col not in ['patient_id', 'measurement_date', target_column]]
        
        X = combined_features[feature_cols].fillna(0)
        y = combined_features[target_column] if target_column in combined_features.columns else None
        
        return X.values, y.values if y is not None else None
```

---

## 🔐 Seguridad y Privacidad

### Implementación de Privacidad Diferencial

```python
# app/security/differential_privacy.py
import numpy as np
from typing import Dict, Any

class DifferentialPrivacyManager:
    
    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon  # Privacy budget
        
    def add_laplace_noise(self, value: float, sensitivity: float) -> float:
        """Añade ruido Laplaciano para privacidad diferencial"""
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return value + noise
    
    def private_mean(self, values: List[float], bounds: Tuple[float, float]) -> float:
        """Calcula media privada con clipping de valores"""
        clipped_values = [np.clip(v, bounds[0], bounds[1]) for v in values]
        true_mean = np.mean(clipped_values)
        sensitivity = (bounds[1] - bounds[0]) / len(values)
        return self.add_laplace_noise(true_mean, sensitivity)

class DataMinimizer:
    """Minimización de datos según principios GDPR"""
    
    @staticmethod
    def anonymize_patient_data(df: pd.DataFrame) -> pd.DataFrame:
        """Anonimiza datos sensibles del paciente"""
        df_anon = df.copy()
        
        # Remover identificadores directos
        if 'patient_name' in df_anon.columns:
            df_anon = df_anon.drop('patient_name', axis=1)
        
        # Generalizar fechas de nacimiento a rangos de edad
        if 'birth_date' in df_anon.columns:
            df_anon['age_group'] = pd.cut(df_anon['age'], 
                                        bins=[0, 30, 50, 65, 100], 
                                        labels=['<30', '30-50', '50-65', '>65'])
            df_anon = df_anon.drop('birth_date', axis=1)
        
        # K-anonymity para combinaciones de características
        quasi_identifiers = ['age_group', 'gender', 'location']
        df_anon = apply_k_anonymity(df_anon, quasi_identifiers, k=5)
        
        return df_anon
```

---

## 📱 Integración con Dispositivos IoT

### Cliente para Dispositivos Wearables

```python
# app/integrations/wearables.py
import asyncio
import aiohttp
from typing import Dict, List
import json
from datetime import datetime

class WearableDataCollector:
    
    def __init__(self):
        self.supported_devices = {
            'fitbit': FitbitAPI(),
            'apple_health': AppleHealthAPI(),
            'garmin': GarminAPI(),
            'google_fit': GoogleFitAPI()
        }
    
    async def collect_biomarker_data(self, user_id: str, 
                                   device_type: str, 
                                   start_date: datetime,
                                   end_date: datetime) -> List[Dict]:
        """Recolecta datos de biomarcadores desde wearables"""
        
        if device_type not in self.supported_devices:
            raise ValueError(f"Dispositivo {device_type} no soportado")
        
        api = self.supported_devices[device_type]
        
        # Obtener diferentes tipos de datos
        tasks = [
            api.get_heart_rate(user_id, start_date, end_date),
            api.get_activity_data(user_id, start_date, end_date),
            api.get_sleep_data(user_id, start_date, end_date),
            api.get_biometric_data(user_id, start_date, end_date)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Procesar y normalizar datos
        normalized_data = []
        for result in results:
            if not isinstance(result, Exception):
                normalized_data.extend(self._normalize_data(result, device_type))
        
        return normalized_data
    
    def _normalize_data(self, raw_data: Dict, device_type: str) -> List[Dict]:
        """Normaliza datos de diferentes dispositivos a formato estándar"""
        normalized = []
        
        for record in raw_data.get('data', []):
            biomarker_record = {
                'patient_id': record['user_id'],
                'biomarker_code': self._map_biomarker_code(record['type']),
                'value': record['value'],
                'measurement_date': record['timestamp'],
                'device_id': f"{device_type}_{record.get('device_id', 'unknown')}",
                'quality_score': record.get('confidence', 1.0),
                'raw_data': record
            }
            normalized.append(biomarker_record)
        
        return normalized

class RealTimeDataProcessor:
    """Procesador en tiempo real para streams de IoT"""
    
    def __init__(self):
        self.redis_client = redis.Redis()
        self.alert_manager = AlertManager()
    
    async def process_incoming_data(self, data: Dict):
        """Procesa datos entrantes en tiempo real"""
        
        # Validar datos
        if not self._validate_data(data):
            return {"status": "error", "message": "Invalid data format"}
        
        # Procesar con analyzer en tiempo real
        analyzer = RealTimeBiomarkerAnalyzer()
        analysis_result = analyzer.process_stream(
            data['patient_id'],
            data['biomarker_code'],
            data['value'],
            datetime.fromisoformat(data['timestamp'])
        )
        
        # Guardar en caché para acceso rápido
        cache_key = f"biomarker:{data['patient_id']}:{data['biomarker_code']}"
        self.redis_client.setex(cache_key, 3600, json.dumps(data))
        
        # Generar alertas si es necesario
        if analysis_result['alerts']:
            await self.alert_manager.send_alerts(
                data['patient_id'], 
                analysis_result['alerts']
            )
        
        return {"status": "processed", "analysis": analysis_result}
```

---

## 🚀 Despliegue y Escalabilidad

### Configuración Docker

```dockerfile
# docker/Dockerfile.biomarkers
FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de aplicación
COPY . .

# Variables de entorno
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production

# Exponer puerto
EXPOSE 8000

# Comando de inicio
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Docker Compose para Stack Completo

```yaml
# docker-compose.biomarkers.yml
version: '3.8'

services:
  biomarkers-api:
    build: 
      context: .
      dockerfile: docker/Dockerfile.biomarkers
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/biomarkers
      - REDIS_URL=redis://redis:6379
      - INFLUX_URL=http://influxdb:8086
    depends_on:
      - postgres
      - redis
      - influxdb
    restart: unless-stopped

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=biomarkers
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  influxdb:
    image: influxdb:2.7
    ports:
      - "8086:8086"
    environment:
      - INFLUXDB_DB=biomarkers
      - INFLUXDB_ADMIN_USER=admin
      - INFLUXDB_ADMIN_PASSWORD=password
    volumes:
      - influx_data:/var/lib/influxdb2

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources

volumes:
  postgres_data:
  redis_data:
  influx_data:
  grafana_data:
```

---

## 📈 Monitoreo y Observabilidad

### Métricas de Sistema

```python
# app/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Métricas de negocio
biomarker_measurements_total = Counter(
    'biomarker_measurements_total',
    'Total number of biomarker measurements processed',
    ['biomarker_type', 'device_type', 'status']
)

data_quality_score = Histogram(
    'biomarker_data_quality_score',
    'Quality score of biomarker measurements',
    ['biomarker_type'],
    buckets=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
)

active_patients = Gauge(
    'biomarkers_active_patients',
    'Number of patients with recent biomarker data',
    ['time_window']
)

# Métricas de performance
processing_time = Histogram(
    'biomarker_processing_duration_seconds',
    'Time spent processing biomarker data',
    ['operation_type']
)

class MetricsCollector:
    
    def record_measurement(self, biomarker_type: str, device_type: str, 
                          quality_score: float, processing_time: float):
        """Registra métricas de una medición"""
        
        # Incrementar contador
        biomarker_measurements_total.labels(
            biomarker_type=biomarker_type,
            device_type=device_type,
            status='success'
        ).inc()
        
        # Registrar calidad
        data_quality_score.labels(biomarker_type=biomarker_type).observe(quality_score)
        
        # Tiempo de procesamiento
        processing_time.labels(operation_type='measurement_processing').observe(processing_time)

# Decorator para métricas automáticas
def track_processing_time(operation_type: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            processing_time.labels(operation_type=operation_type).observe(duration)
            return result
        return wrapper
    return decorator
```

---

## 🔄 CI/CD Pipeline

### GitHub Actions

```yaml
# .github/workflows/biomarkers-ci.yml
name: Biomarkers CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test
          POSTGRES_DB: test_biomarkers
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run tests
      env:
        DATABASE_URL: postgresql://postgres:test@localhost:5432/test_biomarkers
      run: |
        pytest tests/ -v --cov=app --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Run security scan
      uses: securecodewarrior/github-action-add-sarif@v1
      with:
        sarif-file: security-scan-results.sarif

  deploy:
    needs: [test, security-scan]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build and push Docker image
      env:
        DOCKER_REGISTRY: ${{ secrets.DOCKER_REGISTRY }}
      run: |
        docker build -t $DOCKER_REGISTRY/biomarkers-api:$GITHUB_SHA .
        docker push $DOCKER_REGISTRY/biomarkers-api:$GITHUB_SHA
    
    - name: Deploy to production
      run: |
        # Deployment script here
        echo "Deploying to production..."
```

---

*Guía Técnica actualizada: Septiembre 2024*  
*Versión: 2.0 - Pipeline DM2 Avanzado*