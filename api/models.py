from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ModeloTipo(str, Enum):
    ORIGINAL = "original"
    ETAPAS = "etapas"


class SimulacionRequest(BaseModel):
    """Solicitud para ejecutar una simulación"""
    parametros: Dict[str, Any] = Field(..., description="Parámetros del modelo")
    estado_inicial: List[float] = Field(..., description="Estado inicial de las poblaciones")
    tiempo: List[float] = Field(..., description="Puntos de tiempo para la simulación (días)")
    modelo: ModeloTipo = Field(default=ModeloTipo.ETAPAS, description="Tipo de modelo a usar")


class SimulacionResponse(BaseModel):
    """Respuesta de una simulación"""
    success: bool
    tiempo: List[float]
    resultados: Dict[str, List[float]]
    mensaje: Optional[str] = None


class EscenarioBase(BaseModel):
    """Base para un escenario de liberación"""
    id: str
    nombre: str
    descripcion: str
    capacidad_pupas_dia: float
    parametros: Dict[str, Any]
    estado_inicial: List[float]
    dias_simulacion: int
    fecha_creacion: datetime


class EscenarioCreate(BaseModel):
    """Solicitud para crear un escenario"""
    nombre: str
    descripcion: str
    capacidad_pupas_dia: float
    parametros: Optional[Dict[str, Any]] = None
    estado_inicial: Optional[List[float]] = None
    dias_simulacion: int = 365


class EscenarioResponse(EscenarioBase):
    """Respuesta con resultados del escenario"""
    resultados: Optional[Dict[str, Any]] = None


class CalibracionRequest(BaseModel):
    """Solicitud para calibrar el modelo"""
    parametros_iniciales: Optional[Dict[str, float]] = None
    bounds: Optional[Dict[str, List[float]]] = None
    max_iteraciones: int = 200
    especies: List[str] = Field(default=["bovinos", "porcinos"])


class CalibracionResponse(BaseModel):
    """Respuesta de calibración"""
    success: bool
    parametros_optimos: Dict[str, float]
    intervalos_confianza: Dict[str, List[float]]
    metricas: Dict[str, Any]
    escalas: Dict[str, float]
    mensaje: Optional[str] = None


class CalibracionStatus(BaseModel):
    """Estado actual de la calibración"""
    parametros_optimos: Dict[str, float]
    intervalos_confianza: Dict[str, List[float]]
    escalas: Dict[str, float]
    metricas: Dict[str, Any]
    fecha_actualizacion: Optional[datetime] = None


class HealthResponse(BaseModel):
    """Respuesta de health check"""
    status: str
    version: str
    timestamp: datetime
    modelo_disponible: bool
    datos_disponibles: bool


class ErrorResponse(BaseModel):
    """Respuesta de error"""
    detail: str
    timestamp: datetime = Field(default_factory=datetime.now)