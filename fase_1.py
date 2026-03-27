# =========================================================
# FASE 1 (WALK-FORWARD) — Selección por sector + Monte Carlo
# =========================================================
# Qué hace este script:
# 1) Descarga precios UNA sola vez para todo el universo
# 2) Cada fin de mes t:
#    - usa SOLO la ventana histórica [t-L, t]
#    - selecciona 1 ETF por sector (Sharpe en la ventana)
#    - optimiza pesos con Monte Carlo (en la ventana)
#    - impone cuota RV/RF del perfil (Conservador/Moderado/Agresivo)
#    - aplica esos pesos durante el mes siguiente (out-of-sample)
# 3) Devuelve performance out-of-sample + historial de pesos + stats por rebalanceo
#
# Nota: Macro + señales quedan como “módulo aparte” (tiempo real), no backtesteado aquí.

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from fase2_risk_engine import monte_carlo_window_fase2

# =========================================================
# 1) PARÁMETROS GENERALES
# =========================================================
while True: 
    perfil = input("Ingrese el perfil de riesgo ('conservador', 'moderado', 'agresivo'): ").strip().capitalize()
    if perfil not in ["Conservador", "Moderado", "Agresivo"]:
        break
    else:
        print("Perfil inválido. Intente nuevamente.")
# raise ValueError("Perfil inválido. Use: 'conservador', 'moderado' o 'agresivo'.")
    
start_date = "2018-01-01"
end_date   = "2026-03-20"   # puedes actualizarlo cuando quieras

# Cuotas RV/RF por perfil (esto aquí SÍ se impone en pesos)
perfiles_riesgo = {
    "Conservador": {"RV": 0.30, "RF": 0.70},
    "Moderado":    {"RV": 0.60, "RF": 0.40},
    "Agresivo":    {"RV": 0.80, "RF": 0.20}
}

sectores = {
    "Sector 1 - mercado amplio": ["VOO"],
    "Sector 2 - oro y banca": ["IAU", "SAN"],
    "Sector 3 - cybersecurity": ["IHAK"],
    "Sector 4 - health": ["IHI", "XLV"],
    "Sector 5 - clean energy": ["ICLN", "PBD", "CNRG", "QCLN"],
    "Sector 6 - water": ["CGW"]
}

renta_fija = ["TIP"]

rf_annual = 0.03      # tasa libre de riesgo ANUAL (fija en Fase 1)
LOOKBACK_DAYS = 756   # ~ 3 años de días hábiles
NUM_PORTS = 5000      # simulaciones Monte Carlo por rebalanceo (sube a 10000 si tu PC aguanta)
COST_BPS = 10         # costo por rebalanceo (bps * turnover). Si no quieres, pon 0.


# =========================================================
# 2) FUNCIONES BASE: Data
# =========================================================
def get_prices(universe, start, end):
    """
    Descarga precios ajustados (Close) UNA sola vez para todos los tickers.
    """
    universe = sorted(list(set(universe)))
    px = yf.download(
        tickers=universe,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )["Close"]

    if isinstance(px, pd.Series):
        px = px.to_frame()

    px = px.sort_index()
    px = px.dropna(how="all")

    # Forward-fill suave para alinear calendarios (ETFs suelen estar bien alineados)
    px = px.ffill()
    return px


def get_returns_simple(prices):
    """
    Retornos simples diarios para performance real: W_t = W_{t-1}(1+r_t)
    """
    rets = prices.pct_change().dropna(how="all")
    rets = rets.sort_index()
    return rets


def get_returns_log(prices):
    """
    Log-retornos diarios para estimación de medias/cov (económicamente estándar).
    """
    log_ret = np.log(prices / prices.shift(1)).dropna(how="all")
    log_ret = log_ret.sort_index()
    return log_ret


def month_end_dates(index):
    """
    Extrae los últimos días disponibles de cada mes (fechas de rebalanceo).
    """
    s = index.to_series()
    return s.groupby([s.index.year, s.index.month]).tail(1).index


# =========================================================
# 3) MÉTRICAS: Sharpe activo (para selección) y portafolio (para Monte Carlo)
# =========================================================
def sharpe_ratio_anual(returns_daily, rf_annual=0.03):
    """
    Sharpe anualizado usando retornos diarios (simple o log).
    S = (E[R]-rf)/Vol, con E[R] y Vol anualizados.
    """
    r = returns_daily.dropna()
    if r.empty:
        return np.nan

    ret_anual = r.mean() * 252
    vol_anual = r.std() * np.sqrt(252)

    if vol_anual == 0 or pd.isna(vol_anual):
        return np.nan

    return (ret_anual - rf_annual) / vol_anual


def portfolio_metrics(weights, window_log_ret, rf_annual=0.03):
    """
    Métricas del portafolio estimadas con la ventana (log_ret):
    - Retorno esperado anual
    - Volatilidad anual
    - Sharpe anual
    """
    mu_a = window_log_ret.mean() * 252
    cov_a = window_log_ret.cov() * 252

    w = np.array(weights, dtype=float)
    ret = float(np.sum(mu_a.values * w))
    vol = float(np.sqrt(w.T @ cov_a.values @ w))

    if vol == 0 or np.isnan(vol):
        sh = np.nan
    else:
        sh = (ret - rf_annual) / vol

    return ret, vol, sh

# ====== AJUSTE DE COBERTURA ======
# para que no suceda:
# si uno de los activos tiene huecos, fuerzas a que toda la muestra se reduzca a la intersección perfecta.

def filtrar_activos_por_cobertura(window_log_ret, tickers, min_valid_ratio=0.95, min_obs_abs=60):
    """
    Conserva solo activos con cobertura suficiente dentro de la ventana.
    Evita que un ticker con pocos datos destruya la muestra al usar dropna(how="any").
    """
    if not tickers:
        return []

    sub = window_log_ret[tickers].copy()
    n = len(sub)
    valid_counts = sub.notna().sum()

    elegibles = valid_counts[
        (valid_counts >= int(np.ceil(n * min_valid_ratio))) &
        (valid_counts >= min_obs_abs)
    ].index.tolist()

    return elegibles

# =========================================================
# 4) SELECCIÓN POR SECTOR (EN VENTANA)
# =========================================================
def seleccionar_mejor_activo_por_ventana(sectores, window_log_ret, rf_annual=0.03):
    """
    Elige 1 activo por sector, basado en Sharpe calculado SOLO con window_log_ret.
    Devuelve:
      - dict: {sector: ticker_elegido}
      - list: tickers_elegidos_RV
    """
    mejores = {}
    rv_elegidos = []

    for sector, activos in sectores.items():
        activos_validos = [a for a in activos if a in window_log_ret.columns]
        if not activos_validos:
            continue

        if len(activos_validos) == 1:
            elegido = activos_validos[0]
        else:
            scores = {}
            for a in activos_validos:
                sr = sharpe_ratio_anual(window_log_ret[a], rf_annual=rf_annual)
                scores[a] = sr if pd.notna(sr) else -np.inf
            elegido = max(scores, key=scores.get)

        mejores[sector] = elegido
        rv_elegidos.append(elegido)

    return mejores, rv_elegidos


# =========================================================
# 5) MONTE CARLO (EN VENTANA) CON CUOTA RV/RF
# =========================================================
def monte_carlo_window(window_log_ret, rv_tickers, rf_tickers, perfil,
                       perfiles_riesgo, rf_annual=0.03, num_ports=5000, seed=None):
    """
    Monte Carlo dentro de la ventana:
    - genera pesos aleatorios
    - impone cuota RV/RF del perfil
    - calcula ret/vol/sharpe con window_log_ret
    - elige según perfil (min vol / max sharpe / max ret)

    Devuelve:
      best_weights (Series index=tickers),
      best_ret, best_vol, best_sharpe,
      arrays (ret_arr, vol_arr, sharpe_arr) opcionales para graficar si quieres
    """
    rng = np.random.default_rng(seed)
    
    rv_tickers = [t for t in rv_tickers if t in window_log_ret.columns]
    rf_tickers = [t for t in rf_tickers if t in window_log_ret.columns]

    rv_tickers = filtrar_activos_por_cobertura(
        window_log_ret, rv_tickers, min_valid_ratio=0.95, min_obs_abs=60
    )
    rf_tickers = filtrar_activos_por_cobertura(
        window_log_ret, rf_tickers, min_valid_ratio=0.95, min_obs_abs=60
    )

    tickers = rv_tickers + rf_tickers
    if len(tickers) == 0:
        return None, np.nan, np.nan, np.nan, None, None, None

    window_sel = window_log_ret[tickers].dropna(how="any")
    if window_sel.empty or len(window_sel) < 60:
        return None, np.nan, np.nan, np.nan, None, None, None
    
    
    objetivo_rv = perfiles_riesgo[perfil]["RV"]
    objetivo_rf = perfiles_riesgo[perfil]["RF"]

    n_rv = len(rv_tickers)
    n_rf = len(rf_tickers)

    ret_arr = np.zeros(num_ports)
    vol_arr = np.zeros(num_ports)
    sharpe_arr = np.zeros(num_ports)
    all_w = np.zeros((num_ports, len(tickers)))

    # Para mapear: primero RV, luego RF
    # Aseguramos que rv_tickers y rf_tickers estén en tickers y en ese orden
    rv_tickers = [x for x in rv_tickers if x in tickers]
    rf_tickers = [x for x in rf_tickers if x in tickers]
    tickers = rv_tickers + rf_tickers

    for i in range(num_ports):
        # Pesos aleatorios para RV
        if len(rv_tickers) > 0:
            w_rv = rng.random(len(rv_tickers))
            w_rv = w_rv / w_rv.sum()
            w_rv = w_rv * objetivo_rv
        else:
            w_rv = np.array([])

        # Pesos aleatorios para RF
        if len(rf_tickers) > 0:
            w_rf = rng.random(len(rf_tickers))
            w_rf = w_rf / w_rf.sum()
            w_rf = w_rf * objetivo_rf
        else:
            w_rf = np.array([])

        w = np.concatenate([w_rv, w_rf])
        all_w[i, :] = w

        ret_i, vol_i, sh_i = portfolio_metrics(w, window_sel[tickers], rf_annual=rf_annual)

        ret_arr[i] = ret_i
        vol_arr[i] = vol_i
        sharpe_arr[i] = sh_i

    # Selección según perfil
    if perfil == "Conservador":
        idx = np.nanargmin(vol_arr)
    elif perfil == "Moderado":
        idx = np.nanargmax(sharpe_arr)
    else:
        idx = np.nanargmax(ret_arr)

    best_w = pd.Series(all_w[idx, :], index=tickers)
    best_ret = float(ret_arr[idx])
    best_vol = float(vol_arr[idx])
    best_sh = float(sharpe_arr[idx])

    return best_w, best_ret, best_vol, best_sh, ret_arr, vol_arr, sharpe_arr


# =========================================================
# 6) BACKTEST WALK-FORWARD (FASE 1)
# =========================================================
def backtest_walk_forward(prices, sectores, renta_fija, perfil, perfiles_riesgo,
                          rf_annual=0.03, lookback_days=756, num_ports=5000, cost_bps=0):
    """
    Walk-forward mensual:
    - define rebalanceos fin de mes
    - en cada t:
      * ventana [t-L, t] en log-retornos
      * selección por sector (Sharpe en ventana)
      * Monte Carlo en ventana (cuota RV/RF)
      * aplica pesos durante (t, t_next]
    """
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="first")]

    rets_simple = get_returns_simple(prices)
    log_ret_all = get_returns_log(prices)

    reb_dates = month_end_dates(rets_simple.index)
    full_universe = list(prices.columns)

    w_prev = pd.Series(0.0, index=full_universe)

    port_rets_list = []
    weights_hist = {}
    stats_hist = {}     # para guardar ret/vol/sharpe esperados por rebalanceo
    picks_hist = {}     # qué tickers eligió por sector

    for step, t in enumerate(reb_dates):
        window_log = log_ret_all.loc[:t].tail(lookback_days)
        if len(window_log) < lookback_days:
            continue

        # 1) Selección RV por sector en ventana
        mejores_sector, rv_tickers = seleccionar_mejor_activo_por_ventana(sectores, window_log, rf_annual=rf_annual)

        # 2) RF disponible en ventana
        rf_tickers = [x for x in renta_fija if x in window_log.columns]

        # 3) Optimización Monte Carlo (ventana)
        best_w_sel, exp_ret, exp_vol, exp_sh, _, _, _ = monte_carlo_window_fase2(
                    window_log_ret=window_log,
                    rv_tickers=rv_tickers,
                    rf_tickers=rf_tickers,
                    perfil=perfil,
                    perfiles_riesgo=perfiles_riesgo,
                    rf_annual=rf_annual,
                    num_ports=num_ports,
                    cov_method="shrink_diag",
                    shrinkage=0.35,
                    seed=42 + step
                )

        if best_w_sel is None or best_w_sel.empty:
            continue

        # 4) Expandir pesos al universo completo
        w_full = pd.Series(0.0, index=full_universe)
        w_full.loc[best_w_sel.index] = best_w_sel.values

        # 5) Turnover y costo
        turnover = float(np.abs(w_full - w_prev).sum())
        cost = (cost_bps / 10000.0) * turnover if cost_bps else 0.0

        # 6) Periodo out-of-sample: (t, t_next]
        next_dates = reb_dates[reb_dates > t]
        if len(next_dates) == 0:
            break

        t_next = next_dates[0]
        oos = rets_simple.loc[t:t_next].iloc[1:]  # (t, t_next]
        if oos.empty:
            continue

        # Retorno diario cartera
        r_p = (oos * w_full).sum(axis=1)

        # Aplicar costo el primer día
        if len(r_p) > 0 and cost != 0:
            r_p.iloc[0] -= cost

        port_rets_list.append(r_p)

        weights_hist[t] = w_full
        stats_hist[t] = {
            "exp_ret_anual": exp_ret,
            "exp_vol_anual": exp_vol,
            "exp_sharpe": exp_sh,
            "turnover": turnover
        }
        picks_hist[t] = mejores_sector

        w_prev = w_full

    port_rets = pd.concat(port_rets_list).sort_index() if port_rets_list else pd.Series(dtype=float)
    weights_hist = pd.DataFrame(weights_hist).T.sort_index() if weights_hist else pd.DataFrame()
    stats_hist = pd.DataFrame(stats_hist).T.sort_index() if stats_hist else pd.DataFrame()

    return port_rets, weights_hist, stats_hist, picks_hist


def performance_summary(port_rets, rf_annual=0.03):
    """
    Resumen de performance out-of-sample (serie diaria).
    """
    if port_rets is None or len(port_rets) == 0:
        return {"ret_anual": np.nan, "vol_anual": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}

    rf_daily = (1 + rf_annual) ** (1/252) - 1
    excess = port_rets - rf_daily

    ann_ret = (1 + port_rets).prod() ** (252 / len(port_rets)) - 1
    ann_vol = port_rets.std() * np.sqrt(252)
    sharpe = (excess.mean() / port_rets.std()) * np.sqrt(252) if port_rets.std() != 0 else np.nan

    wealth = (1 + port_rets).cumprod()
    dd = wealth / wealth.cummax() - 1
    mdd = dd.min()

    return {
        "ret_anual": float(ann_ret),
        "vol_anual": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(mdd)
    }


# =========================================================
# 7) EJECUCIÓN FASE 1
# =========================================================
universe = [a for lst in sectores.values() for a in lst] + renta_fija

prices = get_prices(universe, start_date, end_date)
if prices.empty:
    raise ValueError("No se descargaron precios. Revisa conexión, tickers o fechas.")

port_rets, weights_hist, stats_hist, picks_hist = backtest_walk_forward(
    prices=prices,
    sectores=sectores,
    renta_fija=renta_fija,
    perfil=perfil,
    perfiles_riesgo=perfiles_riesgo,
    rf_annual=rf_annual,
    lookback_days=LOOKBACK_DAYS,
    num_ports=NUM_PORTS,
    cost_bps=COST_BPS
)

print("\n==============================")
print("FASE 1 — RESULTADOS OUT-OF-SAMPLE")
print("==============================")
res = performance_summary(port_rets, rf_annual=rf_annual)
for k, v in res.items():
    print(f"{k:>14}: {v:.6f}")

if not weights_hist.empty:
    print("\nÚltimos pesos (último rebalanceo):")
    print(weights_hist.iloc[-1].sort_values(ascending=False).head(12).to_frame("Peso"))

if not stats_hist.empty:
    print("\nÚltimas stats de optimización (ventana del último rebalanceo):")
    print(stats_hist.tail(1).T)

# Curva de riqueza
if port_rets is not None and len(port_rets) > 0:
    wealth = (1 + port_rets).cumprod()
    plt.figure(figsize=(10, 4))
    plt.plot(wealth.index, wealth.values)
    plt.title("Curva de riqueza (out-of-sample) — Fase 1")
    plt.xlabel("Fecha")
    plt.ylabel("Crecimiento de 1.0")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =========================================================
# 8) (OPCIONAL) Macro + “señal táctica” solo TIEMPO REAL
#     (No backtesteado en Fase 1)
# =========================================================
# Si quieres, puedes comentar todo este bloque.

def sma(series, length=200):
    return series.rolling(window=length, min_periods=length).mean()

def puntaje_tecnico(precios_serie):
    precios_serie = precios_serie.dropna()
    if precios_serie.empty:
        return 0
    price = precios_serie.iloc[-1]
    sma200 = sma(precios_serie, 200).iloc[-1] if len(precios_serie) >= 200 else np.nan
    momentum = (price / precios_serie.iloc[-252] - 1) if len(precios_serie) >= 252 else 0

    score = 0
    if pd.notna(sma200) and price > sma200:
        score += 1
    if momentum > 0:
        score += 1
    return score

print("\n==============================")
print("MÓDULO TÁCTICO (TIEMPO REAL) — solo informativo")
print("==============================")
fear_greed_raw = input("Fear & Greed Index (0-100) o 'none': ").strip().lower()
try:
    fear_greed_val = float(fear_greed_raw)
except ValueError:
    fear_greed_val = None

sent_score = 0.0
if fear_greed_val is not None:
    if fear_greed_val < 30:
        sent_score += 0.5
    elif fear_greed_val > 70:
        sent_score -= 0.5

# Nota: aquí no descargo FRED (para no mezclar web en esta fase),
# pero si quieres lo integramos después de estabilizar Fase 1.
macro_score = 0.0  # placeholder

if not weights_hist.empty:
    last_reb_date = weights_hist.index[-1]
    last_weights = weights_hist.iloc[-1]
    activos_actuales = last_weights[last_weights > 0].index.tolist()
else:
    activos_actuales = []

if activos_actuales:
    print(f"\nActivos actuales (último rebalanceo): {activos_actuales}")
    for tkr in activos_actuales:
        serie_p = prices[tkr].dropna()
        tech = puntaje_tecnico(serie_p)
        total = tech + macro_score + sent_score
        if total >= 1.5:
            decision = "COMPRAR (señal teórica)"
        elif total <= -1.5:
            decision = "VENDER (señal teórica)"
        else:
            decision = "MANTENER (señal teórica)"
        print(f"{tkr}: técnico={tech}, macro={macro_score:.1f}, sent={sent_score:.1f} -> {decision}")
else:
    print("No hay activos para señal táctica (quizá no hubo rebalanceos suficientes).")