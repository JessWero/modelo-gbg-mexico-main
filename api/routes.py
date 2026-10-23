from fastapi import APIRouter, HTTPException, status
from datetime import datetime
from typing import Dict, Any, List, Optional

from api.models import (
    SimulacionRequest,
    SimulacionResponse,
    EscenarioCreate,
    EscenarioResponse,
    CalibracionRequest,
    CalibracionResponse,
    CalibracionStatus,
    HealthResponse,
    ErrorResponse,
    ModeloTipo,
)
from api.services import SimulacionService, EscenarioService, CalibracionService

router = APIRouter()


# ============================================================
# HEALTH CHECK
# ============================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Verificar estado de la API",
    description="Endpoint para verificar que la API está funcionando correctamente"
)
async def health_check():
    """Health check endpoint"""
    # Verificar disponibilidad del modelo y datos
    modelo_ok = True
    datos_ok = True

    try:
        from src.modelo.parametros import PARAMETROS
        _ = PARAMETROS
    except Exception:
        modelo_ok = False

    try:
        from pathlib import Path
        data_file = Path(__file__).resolve().parent.parent / "serie_mensual_analisis.csv"
        datos_ok = data_file.exists()
    except Exception:
        datos_ok = False

    return HealthResponse(
        status="healthy" if (modelo_ok and datos_ok) else "degraded",
        version="1.0.0",
        timestamp=datetime.now(),
        modelo_disponible=modelo_ok,
        datos_disponibles=datos_ok
    )


# ============================================================
# SIMULACIÓN
# ============================================================

@router.post(
    "/simular",
    response_model=SimulacionResponse,
    summary="Ejecutar simulación del modelo GBG",
    description="""
    Ejecuta una simulación del modelo matemático GBG con los parámetros proporcionados.
    
    - **modelo**: "original" (3 estados: Mw, Fw, Ms) o "etapas" (7 estados: H, L, P, Mw, Fw, Ps, Ms)
    - **parametros**: Diccionario con todos los parámetros del modelo
    - **estado_inicial**: Valores iniciales para cada estado
    - **tiempo**: Array de puntos de tiempo en días (mínimo 2 puntos)
    """
)
async def simular(request: SimulacionRequest):
    """Ejecuta una simulación del modelo GBG"""
    request_data = {
        "parametros": request.parametros,
        "estado_inicial": request.estado_inicial,
        "tiempo": request.tiempo,
        "modelo": request.modelo.value
    }

    resultado = SimulacionService.ejecutar(request_data)

    if not resultado["success"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=resultado["mensaje"]
        )

    return SimulacionResponse(**resultado)


@router.get(
    "/parametros/base",
    summary="Obtener parámetros base del modelo",
    description="Retorna los parámetros por defecto del modelo GBG"
)
async def get_parametros_base():
    """Obtiene los parámetros base del modelo"""
    return SimulacionService.get_parametros_base()


# ============================================================
# ESCENARIOS
# ============================================================

@router.get(
    "/escenarios",
    response_model=List[EscenarioResponse],
    summary="Listar todos los escenarios",
    description="Retorna la lista de escenarios de liberación predefinidos"
)
async def listar_escenarios():
    """Lista todos los escenarios disponibles"""
    return EscenarioService.listar_escenarios()


@router.get(
    "/escenarios/{escenario_id}",
    response_model=EscenarioResponse,
    summary="Obtener un escenario específico",
    description="Retorna los detalles y resultados de un escenario específico"
)
async def obtener_escenario(escenario_id: str):
    """Obtiene un escenario por su ID"""
    escenario = EscenarioService.obtener_escenario(escenario_id)
    if not escenario:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Escenario '{escenario_id}' no encontrado"
        )
    return EscenarioResponse(**escenario)


@router.post(
    "/escenarios",
    response_model=EscenarioResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Crear un nuevo escenario",
    description="Crea un nuevo escenario de liberación personalizado"
)
async def crear_escenario(request: EscenarioCreate):
    """Crea un nuevo escenario personalizado"""
    # Generar ID único
    import uuid
    escenario_id = f"custom_{uuid.uuid4().hex[:8]}"

    # Usar parámetros base si no se proporcionan
    parametros = request.parametros or SimulacionService.get_parametros_base()
    estado_inicial = request.estado_inicial or EscenarioService.ESTADO_INICIAL_DEFAULT

    # Simular el escenario
    try:
        resultados = EscenarioService._simular_escenario(request.capacidad_pupas_dia)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error al simular escenario: {str(e)}"
        )

    escenario = {
        "id": escenario_id,
        "nombre": request.nombre,
        "descripcion": request.descripcion,
        "capacidad_pupas_dia": request.capacidad_pupas_dia,
        "parametros": parametros,
        "estado_inicial": estado_inicial,
        "dias_simulacion": request.dias_simulacion,
        "fecha_creacion": datetime.now(),
        "resultados": resultados
    }

    return EscenarioResponse(**escenario)


# ============================================================
# CALIBRACIÓN
# ============================================================

@router.get(
    "/calibracion",
    response_model=CalibracionStatus,
    summary="Obtener estado actual de calibración",
    description="Retorna los resultados de la última calibración realizada"
)
async def get_calibracion():
    """Obtiene los resultados de la última calibración"""
    calibracion = CalibracionService.get_calibracion_actual()
    if not calibracion:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No hay calibración disponible. Ejecute POST /calibrar primero."
        )
    return CalibracionStatus(**calibracion)


@router.post(
    "/calibrar",
    response_model=CalibracionResponse,
    summary="Ejecutar calibración del modelo",
    description="""
    Ejecuta la calibración del modelo GBG usando datos observados de bovinos y porcinos.
    
    La calibración ajusta los parámetros:
    - **b**: Tasa de nacimiento
    - **mu_f**: Mortalidad de hembras
    - **c**: Competitividad de machos estériles
    
    Utiliza optimización least_squares con bounds biológicos y calcula IC 95% via aproximación Laplace.
    """
)
async def calibrar(request: CalibracionRequest):
    """Ejecuta la calibración del modelo"""
    request_data = {
        "parametros_iniciales": request.parametros_iniciales,
        "bounds": request.bounds,
        "max_iteraciones": request.max_iteraciones,
        "especies": request.especies
    }

    resultado = CalibracionService.calibrar(request_data)

    if not resultado["success"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=resultado["mensaje"]
        )

    return CalibracionResponse(**resultado)