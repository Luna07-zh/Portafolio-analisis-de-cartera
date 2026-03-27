"""
Fase 2 — Motor de riesgo robusto para backtest walk-forward.

Qué hace este módulo:
1) Recibe la ventana histórica de log-retornos seleccionada en el rebalanceo.
2) Estima media y covarianza anualizadas con métodos más robustos que la covarianza muestral pura.
3) Corre Monte Carlo imponiendo la cuota RV/RF del perfil.
4) Devuelve el mejor portafolio con la MISMA estructura de salida de la Fase 1,
   para que pueda reemplazarse con mínimos cambios en el script principal.

Métodos soportados para covarianza:
- sample: covarianza muestral clásica
- shrink_diag: shrinkage manual hacia matriz diagonal (robusto y simple)
- ewma: covarianza EWMA (pondera más el pasado reciente)

Uso esperado desde tu script principal:
    from fase2_risk_engine import monte_carlo_window_fase2

    best_w_sel, exp_ret, exp_vol, exp_sh, _, _, _ = monte_carlo_window_fase2(
        window_log_ret=window_log,
        rv_tickers=rv_tickers,
        rf_tickers=rf_tickers,
        perfil=perfil,
        perfiles_riesgo=perfiles_riesgo,
        rf_annual=rf_annual,
        num_ports=NUM_PORTS,
        cov_method="shrink_diag",
        shrinkage=0.35,
        seed=42
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# =========================================================
# 1) Utilidades internas
# =========================================================
def _annualized_mean(window_log_ret: pd.DataFrame) -> pd.Series:
    """
    Media anualizada de log-retornos.
    """
    return window_log_ret.mean() * 252


def _sample_cov_annualized(window_log_ret: pd.DataFrame) -> pd.DataFrame:
    """
    Covarianza muestral anualizada.
    """
    return window_log_ret.cov() * 252


def _shrink_diag_cov_annualized(window_log_ret: pd.DataFrame, shrinkage: float = 0.35) -> pd.DataFrame:
    """
    Shrinkage manual hacia la diagonal:
        Sigma_shrunk = (1-delta) * S + delta * F
    donde:
        S = covarianza muestral
        F = diag(diag(S))

    Intuición:
    - conserva estructura básica de riesgo
    - reduce ruido en covarianzas cruzadas
    - suele ser más estable que usar S pura
    """
    delta = float(np.clip(shrinkage, 0.0, 1.0))
    s = window_log_ret.cov().values
    f = np.diag(np.diag(s))
    sigma = (1.0 - delta) * s + delta * f
    sigma = sigma * 252
    return pd.DataFrame(sigma, index=window_log_ret.columns, columns=window_log_ret.columns)


def _ewma_cov_annualized(window_log_ret: pd.DataFrame, decay: float = 0.94) -> pd.DataFrame:
    """
    Covarianza EWMA anualizada.

    decay alto => más suavizado, más peso al pasado reciente.
    Similar en espíritu a RiskMetrics.
    """
    if window_log_ret.empty:
        return pd.DataFrame(index=window_log_ret.columns, columns=window_log_ret.columns, dtype=float)

    x = window_log_ret.values
    n_obs, n_assets = x.shape

    # Centrado simple por media muestral.
    x_centered = x - np.nanmean(x, axis=0, keepdims=True)

    # Pesos EWMA normalizados.
    powers = np.arange(n_obs - 1, -1, -1)
    weights = (1.0 - decay) * (decay ** powers)
    weights = weights / weights.sum()

    # Reemplazo defensivo de NaN por 0 tras centrar.
    # Se asume que antes hubo un filtro razonable de cobertura.
    x_centered = np.nan_to_num(x_centered, nan=0.0)

    sigma = np.zeros((n_assets, n_assets), dtype=float)
    for i in range(n_obs):
        row = x_centered[i, :].reshape(-1, 1)
        sigma += weights[i] * (row @ row.T)

    sigma = sigma * 252
    return pd.DataFrame(sigma, index=window_log_ret.columns, columns=window_log_ret.columns)


def build_cov_matrix(window_log_ret: pd.DataFrame,
                     cov_method: str = "shrink_diag",
                     shrinkage: float = 0.35,
                     ewma_decay: float = 0.94) -> pd.DataFrame:
    """
    Constructor unificado de matriz de covarianza anualizada.
    """
    cov_method = cov_method.lower().strip()

    if cov_method == "sample":
        return _sample_cov_annualized(window_log_ret)
    if cov_method == "shrink_diag":
        return _shrink_diag_cov_annualized(window_log_ret, shrinkage=shrinkage)
    if cov_method == "ewma":
        return _ewma_cov_annualized(window_log_ret, decay=ewma_decay)

    raise ValueError("cov_method inválido. Usa: 'sample', 'shrink_diag' o 'ewma'.")


def portfolio_metrics_from_mu_cov(weights: np.ndarray,
                                  mu_a: pd.Series,
                                  cov_a: pd.DataFrame,
                                  rf_annual: float = 0.03) -> tuple[float, float, float]:
    """
    Retorno esperado anual, volatilidad anual y Sharpe anual.
    """
    w = np.array(weights, dtype=float)
    ret = float(np.sum(mu_a.values * w))
    vol = float(np.sqrt(w.T @ cov_a.values @ w))

    if vol == 0 or np.isnan(vol):
        sh = np.nan
    else:
        sh = (ret - rf_annual) / vol

    return ret, vol, sh


# =========================================================
# 2) Filtro de elegibilidad por cobertura
# =========================================================
def filtrar_activos_por_cobertura(window_log_ret: pd.DataFrame,
                                  tickers: list[str],
                                  min_valid_ratio: float = 0.95,
                                  min_obs_abs: int = 60) -> list[str]:
    """
    Evita que un activo con pocos datos te destruya la ventana completa.

    Regla:
    - se conserva solo el ticker que tenga al menos:
      * min_valid_ratio de observaciones válidas dentro de la ventana, y
      * min_obs_abs observaciones válidas absolutas.
    """
    if not tickers:
        return []

    sub = window_log_ret[tickers].copy()
    n = len(sub)
    valid_counts = sub.notna().sum()

    eligible = valid_counts[
        (valid_counts >= int(np.ceil(n * min_valid_ratio))) &
        (valid_counts >= min_obs_abs)
    ].index.tolist()

    return eligible


# =========================================================
# 3) Motor principal Fase 2
# =========================================================
def monte_carlo_window_fase2(window_log_ret: pd.DataFrame,
                             rv_tickers: list[str],
                             rf_tickers: list[str],
                             perfil: str,
                             perfiles_riesgo: dict,
                             rf_annual: float = 0.03,
                             num_ports: int = 5000,
                             cov_method: str = "shrink_diag",
                             shrinkage: float = 0.35,
                             ewma_decay: float = 0.94,
                             min_valid_ratio: float = 0.95,
                             min_obs_abs: int = 60,
                             seed: int | None = None):
    """
    Versión Fase 2 del optimizador en ventana.

    Diferencia respecto a Fase 1:
    - ya no usa por defecto la covarianza muestral pura;
    - puede usar shrinkage diagonal o EWMA;
    - filtra activos con cobertura insuficiente antes de construir la matriz.

    Devuelve exactamente:
        best_weights (Series index=tickers),
        best_ret, best_vol, best_sharpe,
        ret_arr, vol_arr, sharpe_arr
    para que puedas enchufarlo a tu backtest actual sin reescribir todo.
    """
    rng = np.random.default_rng(seed)

    rv_tickers = list(rv_tickers) if rv_tickers is not None else []
    rf_tickers = list(rf_tickers) if rf_tickers is not None else []

    # 1) Universo candidato dentro de la ventana
    candidate_tickers = [t for t in (rv_tickers + rf_tickers) if t in window_log_ret.columns]
    if len(candidate_tickers) == 0:
        return None, np.nan, np.nan, np.nan, None, None, None

    # 2) Filtrar por cobertura para no mutilar la ventana por un ticker "malo"
    rv_ok = filtrar_activos_por_cobertura(
        window_log_ret=window_log_ret,
        tickers=[t for t in rv_tickers if t in candidate_tickers],
        min_valid_ratio=min_valid_ratio,
        min_obs_abs=min_obs_abs
    )
    rf_ok = filtrar_activos_por_cobertura(
        window_log_ret=window_log_ret,
        tickers=[t for t in rf_tickers if t in candidate_tickers],
        min_valid_ratio=min_valid_ratio,
        min_obs_abs=min_obs_abs
    )

    tickers = rv_ok + rf_ok
    if len(tickers) == 0:
        return None, np.nan, np.nan, np.nan, None, None, None

    # 3) Datos alineados solo para los activos elegibles
    window_sel = window_log_ret[tickers].dropna(how="any")
    if window_sel.empty or len(window_sel) < min_obs_abs:
        return None, np.nan, np.nan, np.nan, None, None, None

    # 4) Objetivos RV/RF del perfil
    objetivo_rv = perfiles_riesgo[perfil]["RV"]
    objetivo_rf = perfiles_riesgo[perfil]["RF"]

    n_rv = len(rv_ok)
    n_rf = len(rf_ok)

    # Si una canasta quedó vacía, reasignamos todo el peso al bloque disponible.
    if n_rv == 0 and n_rf == 0:
        return None, np.nan, np.nan, np.nan, None, None, None
    elif n_rv == 0:
        objetivo_rv, objetivo_rf = 0.0, 1.0
    elif n_rf == 0:
        objetivo_rv, objetivo_rf = 1.0, 0.0

    # 5) Parámetros estimados con motor robusto
    mu_a = _annualized_mean(window_sel)
    cov_a = build_cov_matrix(
        window_log_ret=window_sel,
        cov_method=cov_method,
        shrinkage=shrinkage,
        ewma_decay=ewma_decay
    )

    ret_arr = np.full(num_ports, np.nan)
    vol_arr = np.full(num_ports, np.nan)
    sharpe_arr = np.full(num_ports, np.nan)
    all_w = np.zeros((num_ports, len(tickers)))

    # 6) Monte Carlo con cuota RV/RF impuesta
    for i in range(num_ports):
        if n_rv > 0:
            w_rv = rng.random(n_rv)
            w_rv = w_rv / w_rv.sum()
            w_rv = w_rv * objetivo_rv
        else:
            w_rv = np.array([])

        if n_rf > 0:
            w_rf = rng.random(n_rf)
            w_rf = w_rf / w_rf.sum()
            w_rf = w_rf * objetivo_rf
        else:
            w_rf = np.array([])

        w = np.concatenate([w_rv, w_rf])
        all_w[i, :] = w

        ret_i, vol_i, sh_i = portfolio_metrics_from_mu_cov(
            weights=w,
            mu_a=mu_a,
            cov_a=cov_a,
            rf_annual=rf_annual
        )

        ret_arr[i] = ret_i
        vol_arr[i] = vol_i
        sharpe_arr[i] = sh_i

    # 7) Criterio de elección por perfil
    if perfil == "Conservador":
        idx = np.nanargmin(vol_arr)
    elif perfil == "Moderado":
        idx = np.nanargmax(sharpe_arr)
    elif perfil == "Agresivo":
        idx = np.nanargmax(ret_arr)
    else:
        raise ValueError("Perfil inválido. Usa 'Conservador', 'Moderado' o 'Agresivo'.")

    best_w = pd.Series(all_w[idx, :], index=tickers)
    best_ret = float(ret_arr[idx])
    best_vol = float(vol_arr[idx])
    best_sh = float(sharpe_arr[idx])

    return best_w, best_ret, best_vol, best_sh, ret_arr, vol_arr, sharpe_arr
