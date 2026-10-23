import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp

from src.modelo.modelo_etapas import modelo_gbg_etapas
from src.modelo.parametros import PARAMETROS as PARAMETROS_BASE
from src.simulacion.simulador import ejecutar_simulacion


# Configuración de rutas
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR


class SimulacionService:
    """Servicio para ejecutar simulaciones del modelo GBG"""

    @staticmethod
    def ejecutar(request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta una simulación con los parámetros proporcionados
        """
        try:
            parametros = request_data["parametros"]
            
            # Convertir produccion_pupas a callable si es un número
            if "produccion_pupas" in parametros and not callable(parametros["produccion_pupas"]):
                valor = float(parametros["produccion_pupas"])
                parametros["produccion_pupas"] = lambda t, v=valor: v
            
            estado_inicial = np.array(request_data["estado_inicial"], dtype=float)
            tiempo = np.array(request_data["tiempo"], dtype=float)
            modelo_tipo = request_data.get("modelo", "etapas")

            # Validaciones básicas
            if len(tiempo) < 2:
                raise ValueError("El tiempo debe contener al menos dos puntos")
            if len(estado_inicial) not in [3, 7]:
                raise ValueError("El estado inicial debe tener 3 (original) o 7 (etapas) elementos")

            # Ejecutar simulación
            resultado = ejecutar_simulacion(
                parametros=parametros,
                estado_inicial=estado_inicial,
                tiempo=tiempo,
                modelo=modelo_tipo
            )

            # Preparar resultados
            if modelo_tipo == "original":
                nombres_estados = ["Mw", "Fw", "Ms"]
            else:
                nombres_estados = ["H", "L", "P", "Mw", "Fw", "Ps", "Ms"]

            resultados_dict = {
                nombre: resultado.y[i].tolist()
                for i, nombre in enumerate(nombres_estados)
            }

            return {
                "success": True,
                "tiempo": resultado.t.tolist(),
                "resultados": resultados_dict,
                "mensaje": "Simulación completada exitosamente"
            }

        except Exception as e:
            return {
                "success": False,
                "tiempo": [],
                "resultados": {},
                "mensaje": f"Error en simulación: {str(e)}"
            }

    @staticmethod
    def get_parametros_base() -> Dict[str, Any]:
        """Obtiene los parámetros base del modelo"""
        params = deepcopy(PARAMETROS_BASE)
        # Convertir función a representación serializable
        if callable(params.get("produccion_pupas")):
            params["produccion_pupas"] = "funcion: produccion_pupas(t)"
        return params


class EscenarioService:
    """Servicio para gestionar escenarios de liberación"""

    ARCHIVO_ESCENARIOS = DATA_DIR / "escenarios_liberacion.csv"
    ARCHIVO_RESULTADOS = DATA_DIR / "escenarios_liberacion_resultados.csv"
    ARCHIVO_TRAYECTORIAS = DATA_DIR / "escenarios_liberacion_trayectorias.csv"

    # Escenarios predefinidos
    ESCENARIOS_BASE = [
        {
            "id": "base_0",
            "nombre": "Sin liberación (control)",
            "descripcion": "Escenario base sin liberación de machos estériles",
            "capacidad_pupas_dia": 0,
        },
        {
            "id": "base_1000",
            "nombre": "Liberación baja (1,000 pupas/día)",
            "descripcion": "Escenario con liberación de 1,000 pupas por día",
            "capacidad_pupas_dia": 1_000,
        },
        {
            "id": "base_10000",
            "nombre": "Liberación media (10,000 pupas/día)",
            "descripcion": "Escenario con liberación de 10,000 pupas por día",
            "capacidad_pupas_dia": 10_000,
        },
        {
            "id": "base_100000",
            "nombre": "Liberación alta (100,000 pupas/día)",
            "descripcion": "Escenario con liberación de 100,000 pupas por día",
            "capacidad_pupas_dia": 100_000,
        },
    ]

    ESTADO_INICIAL_DEFAULT = [1000, 500, 200, 1000, 1000, 0, 0]
    DIAS_DEFAULT = 365

    @classmethod
    def listar_escenarios(cls) -> List[Dict[str, Any]]:
        """Lista todos los escenarios disponibles"""
        escenarios = []
        for esc in cls.ESCENARIOS_BASE:
            escenarios.append({
                "id": esc["id"],
                "nombre": esc["nombre"],
                "descripcion": esc["descripcion"],
                "capacidad_pupas_dia": esc["capacidad_pupas_dia"],
                "parametros": cls._get_parametros_escenario(esc["capacidad_pupas_dia"]),
                "estado_inicial": cls.ESTADO_INICIAL_DEFAULT,
                "dias_simulacion": cls.DIAS_DEFAULT,
                "fecha_creacion": datetime.now().isoformat(),
            })
        return escenarios

    @classmethod
    def obtener_escenario(cls, escenario_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene un escenario específico por ID"""
        for esc in cls.ESCENARIOS_BASE:
            if esc["id"] == escenario_id:
                # Ejecutar simulación para obtener resultados
                resultados = cls._simular_escenario(esc["capacidad_pupas_dia"])
                return {
                    "id": esc["id"],
                    "nombre": esc["nombre"],
                    "descripcion": esc["descripcion"],
                    "capacidad_pupas_dia": esc["capacidad_pupas_dia"],
                    "parametros": cls._get_parametros_escenario(esc["capacidad_pupas_dia"]),
                    "estado_inicial": cls.ESTADO_INICIAL_DEFAULT,
                    "dias_simulacion": cls.DIAS_DEFAULT,
                    "fecha_creacion": datetime.now().isoformat(),
                    "resultados": resultados
                }
        return None

    @classmethod
    def _get_parametros_escenario(cls, capacidad_pupas_dia: float) -> Dict[str, Any]:
        """Genera parámetros para un escenario específico"""
        params = deepcopy(PARAMETROS_BASE)
        params["produccion_pupas"] = lambda t: float(capacidad_pupas_dia)
        # Convertir para serialización
        params_serializable = {}
        for k, v in params.items():
            if callable(v):
                params_serializable[k] = f"funcion: {k}(t) = {capacidad_pupas_dia}"
            else:
                params_serializable[k] = v
        return params_serializable

    @classmethod
    def _simular_escenario(cls, capacidad_pupas_dia: float) -> Dict[str, Any]:
        """Ejecuta la simulación para un escenario"""
        parametros = deepcopy(PARAMETROS_BASE)
        parametros["produccion_pupas"] = lambda t: float(capacidad_pupas_dia)

        def sistema(t, estado):
            poblacion = estado[:7]
            derivadas = modelo_gbg_etapas(t, poblacion, parametros)

            machos_en_campo = max(0.0, poblacion[6])
            liberacion_dia = max(
                0.0,
                derivadas[6] + parametros["mu_s"] * machos_en_campo,
            )

            return [*derivadas, liberacion_dia]

        tiempos = np.arange(0, cls.DIAS_DEFAULT + 1, dtype=float)

        resultado = solve_ivp(
            sistema,
            (0, cls.DIAS_DEFAULT),
            [*cls.ESTADO_INICIAL_DEFAULT, 0.0],
            t_eval=tiempos,
            rtol=1e-7,
            atol=1e-9,
        )

        if not resultado.success:
            raise RuntimeError(resultado.message)

        datos = resultado.y
        nombres_estados = ["H", "L", "P", "Mw", "Fw", "Ps", "Ms"]

        return {
            "capacidad_pupas_dia": capacidad_pupas_dia,
            "dias_simulados": cls.DIAS_DEFAULT,
            "liberados_acumulados": float(datos[7, -1]),
            "machos_esteriles_vivos_dia_final": float(datos[6, -1]),
            "machos_silvestres_dia_final": float(datos[3, -1]),
            "hembras_silvestres_dia_final": float(datos[4, -1]),
            "minimo_poblacion_silvestre": float(np.min(datos[:5])),
            "trayectorias": {
                nombre: datos[i].tolist()
                for i, nombre in enumerate(nombres_estados)
            },
            "tiempo": tiempos.tolist(),
            "integracion_exitosa": resultado.success,
            "tipo_resultado": "exploratorio; parametros biologicos no calibrados",
        }


class CalibracionService:
    """Servicio para calibración del modelo"""

    ARCHIVO_PARAMS = DATA_DIR / "calibracion_parametros.csv"
    ARCHIVO_ESCALAS = DATA_DIR / "calibracion_escalas.csv"
    ARCHIVO_DATOS = DATA_DIR / "serie_mensual_analisis.csv"

    NOMBRES_PARAMS = ["b", "mu_f", "c"]
    BOUNDS = {
        "b": (0.05, 0.50),
        "mu_f": (0.02, 0.25),
        "c": (0.10, 1.00),
    }
    LIMITES_INF = [0.05, 0.02, 0.10]
    LIMITES_SUP = [0.50, 0.25, 1.00]
    X0_DEFAULT = [0.20, 0.08, 0.60]

    ESTADO_INICIAL = np.array([1000, 500, 200, 1000, 1000, 0, 0], dtype=float)

    @classmethod
    def get_calibracion_actual(cls) -> Optional[Dict[str, Any]]:
        """Obtiene los resultados de la última calibración"""
        if not cls.ARCHIVO_PARAMS.exists() or not cls.ARCHIVO_ESCALAS.exists():
            return None

        try:
            df_params = pd.read_csv(cls.ARCHIVO_PARAMS)
            df_escalas = pd.read_csv(cls.ARCHIVO_ESCALAS)

            parametros_optimos = {}
            intervalos_confianza = {}

            for _, row in df_params.iterrows():
                parametros_optimos[row["parametro"]] = row["valor_optimo"]
                intervalos_confianza[row["parametro"]] = [
                    row["ic_2.5"],
                    row["ic_97.5"]
                ]

            escalas = {}
            metricas = {}
            for _, row in df_escalas.iterrows():
                escalas[row["especie"]] = row["escala_empirica"]
                metricas[row["especie"]] = {
                    "rmse_log": row["rmse_log"],
                    "mae": row["mae"],
                    "n_meses": row["n_meses"],
                }

            return {
                "parametros_optimos": parametros_optimos,
                "intervalos_confianza": intervalos_confianza,
                "escalas": escalas,
                "metricas": metricas,
                "fecha_actualizacion": datetime.fromtimestamp(cls.ARCHIVO_PARAMS.stat().st_mtime).isoformat()
            }
        except Exception:
            return None

    @classmethod
    def calibrar(cls, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta la calibración del modelo"""
        try:
            # Cargar datos
            series = cls._cargar_datos(request_data.get("especies", ["bovinos", "porcinos"]))

            # Parámetros iniciales y bounds
            x0 = request_data.get("parametros_iniciales")
            if x0:
                x0 = [x0.get(p, cls.X0_DEFAULT[i]) for i, p in enumerate(cls.NOMBRES_PARAMS)]
            else:
                x0 = cls.X0_DEFAULT

            bounds_inf = cls.LIMITES_INF
            bounds_sup = cls.LIMITES_SUP
            if request_data.get("bounds"):
                bounds_inf = [request_data["bounds"].get(p, cls.LIMITES_INF[i])[0] for i, p in enumerate(cls.NOMBRES_PARAMS)]
                bounds_sup = [request_data["bounds"].get(p, cls.LIMITES_SUP[i])[1] for i, p in enumerate(cls.NOMBRES_PARAMS)]

            max_nfev = request_data.get("max_iteraciones", 200)

            # Optimización
            res = cls._ajustar_deterministico(series, x0, bounds_inf, bounds_sup, max_nfev)
            x_opt = res.x

            # IC 95% Laplace
            ic_low, ic_high, se, sigma2 = cls._calcular_ic_laplace(res, series, bounds_inf, bounds_sup)

            # Métricas y escalas
            metricas, escalas = cls._calcular_metricas_y_escalas(x_opt, series)

            # Guardar resultados
            cls._guardar_resultados(res, ic_low, ic_high, se, sigma2, metricas, escalas)

            # Preparar respuesta
            parametros_optimos = {cls.NOMBRES_PARAMS[i]: float(x_opt[i]) for i in range(len(cls.NOMBRES_PARAMS))}
            intervalos_confianza = {cls.NOMBRES_PARAMS[i]: [float(ic_low[i]), float(ic_high[i])] for i in range(len(cls.NOMBRES_PARAMS))}

            return {
                "success": True,
                "parametros_optimos": parametros_optimos,
                "intervalos_confianza": intervalos_confianza,
                "metricas": metricas,
                "escalas": escalas,
                "mensaje": "Calibración completada exitosamente"
            }

        except Exception as e:
            return {
                "success": False,
                "parametros_optimos": {},
                "intervalos_confianza": {},
                "metricas": {},
                "escalas": {},
                "mensaje": f"Error en calibración: {str(e)}"
            }

    @classmethod
    def _cargar_datos(cls, especies: List[str]) -> Dict[str, Any]:
        """Carga los datos de observación"""
        df = pd.read_csv(cls.ARCHIVO_DATOS)
        df["periodo"] = pd.to_datetime(df["periodo"].astype(str), format="%Y-%m", errors="coerce")
        df = df.dropna(subset=["periodo"]).sort_values("periodo").reset_index(drop=True)

        series = {}
        for hosp in especies:
            if hosp not in ["bovinos", "porcinos"]:
                continue
            col_comp = f"{hosp}_completo"
            comp = df[col_comp].astype(str).str.strip().str.lower().eq("si")
            casos = pd.to_numeric(df[hosp], errors="coerce")
            val = casos.notna() & comp
            dfv = df.loc[val].copy()
            dfv["casos"] = casos.loc[val].values
            if len(dfv) >= 5:
                series[hosp] = {
                    "df": dfv,
                    "casos": dfv["casos"].to_numpy(float),
                    "fechas": pd.DatetimeIndex(dfv["periodo"]),
                    "n_meses": len(dfv),
                }
        return series

    @classmethod
    def _simular_hembras_en_fechas(cls, parametros: Dict, fechas_objetivo: pd.DatetimeIndex) -> np.ndarray:
        """Simula e interpola hembras en fechas exactas"""
        if len(fechas_objetivo) == 0:
            return np.array([])

        fecha_inicio = fechas_objetivo.min()
        fecha_fin = fechas_objetivo.max()
        n_dias = (fecha_fin - fecha_inicio).days + 60

        t_eval = np.arange(0, n_dias + 1, 7, dtype=float)

        resultado = ejecutar_simulacion(
            parametros=parametros,
            estado_inicial=cls.ESTADO_INICIAL.copy(),
            tiempo=t_eval,
            modelo="etapas",
        )

        if not resultado.success:
            raise RuntimeError(f"Simulación falló: {resultado.message}")

        dias_objetivo = (fechas_objetivo - fecha_inicio).days.astype(float)
        hembras = np.interp(dias_objetivo, t_eval, resultado.y[4])
        return hembras

    @classmethod
    def _construir_parametros(cls, x: np.ndarray) -> Dict:
        """Construye diccionario de parámetros a partir de vector"""
        p = deepcopy(PARAMETROS_BASE)
        p["b"] = float(x[0])
        p["mu_f"] = float(x[1])
        p["c"] = float(x[2])
        return p

    @classmethod
    def _residuos_conjuntos(cls, x: np.ndarray, series: Dict) -> np.ndarray:
        """Residuos log para todas las series"""
        p = cls._construir_parametros(x)
        all_res = []

        for hosp, data in series.items():
            hembras = cls._simular_hembras_en_fechas(p, data["fechas"])
            escala = max(np.mean(data["casos"]) / max(np.mean(hembras), 1e-6), 1e-6)
            pred = escala * hembras
            res = np.log1p(pred) - np.log1p(data["casos"])
            all_res.append(res)

        return np.concatenate(all_res)

    @classmethod
    def _ajustar_deterministico(cls, series: Dict, x0: List, bounds_inf: List, bounds_sup: List, max_nfev: int):
        """Optimización determinista"""
        def fun_res(x):
            return cls._residuos_conjuntos(x, series)

        res = least_squares(
            fun=fun_res,
            x0=x0,
            bounds=(bounds_inf, bounds_sup),
            loss="soft_l1",
            f_scale=0.5,
            max_nfev=max_nfev,
            verbose=0,
        )
        return res

    @classmethod
    def _calcular_ic_laplace(cls, res, series: Dict, bounds_inf: List, bounds_sup: List) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """IC 95% via aproximación Laplace (Jacobiano)"""
        n_obs = sum(len(data["casos"]) for data in series.values())
        n_params = len(cls.NOMBRES_PARAMS)

        jac = res.jac
        if jac is None:
            from scipy.optimize import approx_derivative
            jac = approx_derivative(lambda x: cls._residuos_conjuntos(x, series), res.x)

        rss = 2 * res.cost
        sigma2 = rss / max(n_obs - n_params, 1)

        jtj = jac.T @ jac
        cond = np.linalg.cond(jtj)

        if cond > 1e12:
            jtj = jtj + 1e-6 * np.eye(n_params)

        try:
            cov = sigma2 * np.linalg.inv(jtj)
        except np.linalg.LinAlgError:
            cov = sigma2 * np.linalg.pinv(jtj)

        se = np.sqrt(np.diag(cov))
        z = 1.96
        ic_low = np.maximum(res.x - z * se, bounds_inf)
        ic_high = np.minimum(res.x + z * se, bounds_sup)

        return ic_low, ic_high, se, sigma2

    @classmethod
    def _calcular_metricas_y_escalas(cls, x_opt: np.ndarray, series: Dict) -> Tuple[Dict, Dict]:
        """Calcula métricas y escalas para cada especie"""
        p_opt = cls._construir_parametros(x_opt)
        metricas = {}
        escalas = {}

        for hosp, data in series.items():
            hembras = cls._simular_hembras_en_fechas(p_opt, data["fechas"])
            escala = max(np.mean(data["casos"]) / max(np.mean(hembras), 1e-6), 1e-6)
            pred = escala * hembras

            metricas[hosp] = {
                "escala": float(escala),
                "pred": pred.tolist(),
                "hembras": hembras.tolist(),
                "casos_obs": data["casos"].tolist(),
                "fechas": [f.isoformat() for f in data["fechas"]],
                "rmse_log": float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(data["casos"]))**2))),
                "mae": float(np.mean(np.abs(pred - data["casos"]))),
            }
            escalas[hosp] = float(escala)

        return metricas, escalas

    @classmethod
    def _guardar_resultados(cls, res, ic_low, ic_high, se, sigma2, metricas, escalas):
        """Guarda los resultados de calibración en archivos CSV"""
        x_opt = res.x

        # 1. Parámetros + IC Laplace
        rows_params = []
        for i, nombre in enumerate(cls.NOMBRES_PARAMS):
            rows_params.append({
                "parametro": nombre,
                "valor_optimo": x_opt[i],
                "error_estandar": se[i],
                "ic_2.5": ic_low[i],
                "ic_97.5": ic_high[i],
                "bound_low": cls.LIMITES_INF[i],
                "bound_high": cls.LIMITES_SUP[i],
                "base_literatura": 0.1 if nombre != "c" else 1.0,
            })

        df_params = pd.DataFrame(rows_params)
        df_params.to_csv(cls.ARCHIVO_PARAMS, index=False, encoding="utf-8-sig")

        # 2. Escalas
        rows_escalas = []
        for hosp, m in metricas.items():
            rows_escalas.append({
                "especie": hosp,
                "escala_empirica": m["escala"],
                "rmse_log": m["rmse_log"],
                "mae": m["mae"],
                "n_meses": len(m["casos_obs"]),
            })

        df_escalas = pd.DataFrame(rows_escalas)
        df_escalas.to_csv(cls.ARCHIVO_ESCALAS, index=False, encoding="utf-8-sig")