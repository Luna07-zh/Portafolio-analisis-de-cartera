import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


# =========================================================
# 1. PARÁMETROS GENERALES
# =========================================================
perfil = input("Ingrese el perfil de riesgo ('conservador', 'moderado', 'agresivo'): ").strip().capitalize()

if perfil not in ["Conservador", "Moderado", "Agresivo"]:
    raise ValueError("Perfil inválido. Use: 'conservador', 'moderado' o 'agresivo'.")

start_date = "2018-01-01"
end_date = "2026-03-16"   # puedes cambiarlo si deseas

# Proporción objetivo renta variable / renta fija por perfil
perfiles_riesgo = {
    "Conservador": {"RV": 0.30, "RF": 0.70},
    "Moderado": {"RV": 0.60, "RF": 0.40},
    "Agresivo": {"RV": 0.80, "RF": 0.20}
}

# Activos por sector
sectores = {
    "Sector 1 - mercado amplio": ["VOO"],
    "Sector 2 - oro": ["IAU"],
    "Sector 3 - cybersecurity": ["IHAK"],
    "Sector 4 - health": ["IHI", "XLV"],
    "Sector 5 - clean energy": ["ICLN", "PBD", "CNRG", "QCLN"],
    "Sector 6 - water": ["CGW"]
}

# Activos de renta fija
renta_fija = ["TIP"]

# Tasa libre de riesgo anual usada en métricas
rf_annual = 0.03


# =========================================================
# 2. FUNCIONES AUXILIARES
# =========================================================
def sma(series, length=200):
    """Media móvil simple usando pandas."""
    return series.rolling(window=length, min_periods=length).mean()


def obtener_datos(activos, start_date, end_date):
    """
    Descarga precios de cierre ajustados con yfinance.
    Devuelve un DataFrame con una columna por activo.
    """
    datos = pd.DataFrame()

    for activo in activos:
        try:
            serie = yf.download(
                activo,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True
            )["Close"]

            if serie.empty:
                print(f"Advertencia: no se encontraron datos para {activo}")
                continue

            datos[activo] = serie.dropna()

        except Exception as e:
            print(f"Advertencia: error al descargar {activo} -> {e}")

    return datos


def eliminar_outliers_serie(serie):
    """
    Elimina outliers con IQR y luego rellena hacia adelante.
    """
    serie = serie.dropna().copy()

    if serie.empty:
        return serie

    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    serie_filtrada = serie.where((serie >= lower) & (serie <= upper))
    serie_filtrada = serie_filtrada.ffill()

    return serie_filtrada


def obtener_datos_limpios(activos, start_date, end_date):
    """
    Descarga datos y aplica limpieza básica de outliers.
    """
    datos = pd.DataFrame()

    for activo in activos:
        try:
            serie = yf.download(
                activo,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True
            )["Close"]

            if serie.empty:
                print(f"Advertencia: no se encontraron datos para {activo}")
                continue

            serie = eliminar_outliers_serie(serie)
            datos[activo] = serie

        except Exception as e:
            print(f"Advertencia: error al descargar/limpiar {activo} -> {e}")

    # Alinea series y exige observaciones comunes
    datos = datos.ffill().dropna(how="any")
    return datos


def calcular_log_returns(precios):
    return np.log(precios / precios.shift(1)).dropna(how="any")


def sharpe_ratio_anual(retornos, rf_annual=0.03):
    """
    Sharpe anualizado.
    """
    retornos = retornos.dropna()
    if retornos.empty:
        return np.nan

    ret_anual = retornos.mean() * 252
    vol_anual = retornos.std() * np.sqrt(252)

    if vol_anual == 0 or pd.isna(vol_anual):
        return np.nan

    return (ret_anual - rf_annual) / vol_anual


def sortino_ratio_anual(retornos, rf_annual=0.03):
    """
    Sortino anualizado.
    """
    retornos = retornos.dropna()
    if retornos.empty:
        return np.nan

    ret_anual = retornos.mean() * 252
    downside = retornos[retornos < 0].std() * np.sqrt(252)

    if downside == 0 or pd.isna(downside):
        return np.nan

    return (ret_anual - rf_annual) / downside


def seleccionar_mejor_activo_por_sector(sectores, log_ret, rf_annual=0.03):
    """
    Elige el mejor activo de cada sector según Sharpe.
    """
    mejores = {}

    for sector, activos in sectores.items():
        activos_validos = [a for a in activos if a in log_ret.columns]

        if not activos_validos:
            print(f"Advertencia: ningún activo con datos en {sector}")
            continue

        if len(activos_validos) == 1:
            mejores[sector] = activos_validos[0]
            continue

        metricas = {}
        for activo in activos_validos:
            sr = sharpe_ratio_anual(log_ret[activo], rf_annual)
            metricas[activo] = sr

        mejores[sector] = max(metricas, key=metricas.get)

    return mejores


def obtener_ponderaciones_perfil(perfil):
    """
    Ponderaciones para el puntaje interno por activo.
    """
    ponderaciones = {
        "Conservador": {"sortino": 0.60, "sharpe": 0.20, "rendimiento": 0.20},
        "Moderado": {"sortino": 0.20, "sharpe": 0.30, "rendimiento": 0.50},
        "Agresivo": {"sortino": 0.10, "sharpe": 0.10, "rendimiento": 0.80}
    }
    return ponderaciones[perfil]


def calcular_puntajes_activos(activos, log_ret, perfil, rf_annual=0.03):
    """
    Calcula puntajes ponderados por perfil para cada activo.
    """
    ponderaciones = obtener_ponderaciones_perfil(perfil)
    puntajes = {}

    for activo in activos:
        serie = log_ret[activo].dropna()
        if serie.empty:
            puntajes[activo] = np.nan
            continue

        ret_anual = serie.mean() * 252
        sharpe = sharpe_ratio_anual(serie, rf_annual)
        sortino = sortino_ratio_anual(serie, rf_annual)

        # Reemplazos defensivos
        if pd.isna(sharpe):
            sharpe = -999
        if pd.isna(sortino):
            sortino = -999

        puntaje = (
            sortino * ponderaciones["sortino"] +
            sharpe * ponderaciones["sharpe"] +
            ret_anual * ponderaciones["rendimiento"]
        )

        puntajes[activo] = puntaje

    return puntajes


def normalizar_puntajes(puntajes):
    """
    Convierte puntajes a pesos positivos que sumen 1.
    """
    s = pd.Series(puntajes, dtype=float).dropna()

    if s.empty:
        return pd.Series(dtype=float)

    # Desplaza si hay negativos
    minimo = s.min()
    if minimo <= 0:
        s = s - minimo + 1e-6

    total = s.sum()
    if total == 0:
        return pd.Series(1 / len(s), index=s.index)

    return s / total


def metricas_portafolio(pesos, log_ret, rf_annual=0.03):
    """
    Retorno, volatilidad y Sharpe del portafolio.
    """
    media_anual = log_ret.mean() * 252
    cov_anual = log_ret.cov() * 252

    ret = np.sum(media_anual * pesos)
    vol = np.sqrt(np.dot(pesos.T, np.dot(cov_anual, pesos)))

    if vol == 0 or pd.isna(vol):
        sharpe = np.nan
    else:
        sharpe = (ret - rf_annual) / vol

    return ret, vol, sharpe


def fred_series(code, start_date, yoy_periods=None):
    """
    Descarga una serie desde FRED usando CSV directo.
    Evita pandas_datareader.
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"

    try:
        df = pd.read_csv(url)
        df["DATE"] = pd.to_datetime(df["DATE"])
        df = df[df["DATE"] >= pd.to_datetime(start_date)].copy()
        df[code] = pd.to_numeric(df[code], errors="coerce")

        serie = df.set_index("DATE")[code].dropna()

        if yoy_periods is not None:
            serie = serie.pct_change(periods=yoy_periods) * 100

        return serie.dropna()

    except Exception as e:
        print(f"⚠️ FRED: {code} no disponible -> {e}")
        return None


def puntaje_tecnico(serie):
    """
    Puntaje técnico simple:
    +1 si precio > SMA200
    +1 si momentum 12m > 0
    """
    serie = serie.dropna()

    if serie.empty:
        return 0

    price = serie.iloc[-1]

    sma200 = sma(serie, 200).iloc[-1] if len(serie) >= 200 else np.nan
    momentum = (price / serie.iloc[-252] - 1) if len(serie) >= 252 else 0

    score = 0
    if pd.notna(sma200) and price > sma200:
        score += 1
    if momentum > 0:
        score += 1

    return score


# =========================================================
# 3. DESCARGA INICIAL Y SELECCIÓN DEL MEJOR ACTIVO POR SECTOR
# =========================================================
todos_los_activos = [a for lista in sectores.values() for a in lista] + renta_fija
datos_brutos = obtener_datos(todos_los_activos, start_date, end_date)

if datos_brutos.empty:
    raise ValueError("No se descargaron datos. Revisa conexión, fechas o tickers.")

log_ret_brutos = calcular_log_returns(datos_brutos)

mejores_por_sector = seleccionar_mejor_activo_por_sector(sectores, log_ret_brutos, rf_annual)
mejores_activos_rv = list(mejores_por_sector.values())
etfs = mejores_activos_rv + renta_fija

print("\nMejores activos por sector:")
for sector, activo in mejores_por_sector.items():
    print(f"  {sector}: {activo}")

print("\nLista final de activos seleccionados:")
print(etfs)


# =========================================================
# 4. DESCARGA LIMPIA PARA ACTIVOS SELECCIONADOS
# =========================================================
datos_etfs = obtener_datos_limpios(etfs, start_date, end_date)

if datos_etfs.empty or datos_etfs.shape[1] == 0:
    raise ValueError("No hay datos limpios suficientes para los activos seleccionados.")

log_ret = calcular_log_returns(datos_etfs)


# =========================================================
# 5. PONDERACIÓN TEÓRICA SEGÚN PERFIL
# =========================================================
objetivo_rv = perfiles_riesgo[perfil]["RV"]
objetivo_rf = perfiles_riesgo[perfil]["RF"]

puntajes_rv = calcular_puntajes_activos(mejores_activos_rv, log_ret, perfil, rf_annual)
pesos_rv_base = normalizar_puntajes(puntajes_rv)

pesos_base = pd.Series(0.0, index=etfs)

# Renta variable: según puntajes internos
for activo, peso in pesos_rv_base.items():
    pesos_base[activo] = peso * objetivo_rv

# Renta fija: reparto simple entre activos RF
for activo in renta_fija:
    pesos_base[activo] = objetivo_rf / len(renta_fija)

ret_base, vol_base, sharpe_base = metricas_portafolio(pesos_base.values, log_ret, rf_annual)

print(f"\nPerfil seleccionado: {perfil}")
print("\nPesos base teóricos por perfil:")
print(pesos_base.to_frame("Peso"))

print("\nMétricas del portafolio base:")
print(f"  Retorno esperado anual: {ret_base:.4f}")
print(f"  Volatilidad anual:      {vol_base:.4f}")
print(f"  Sharpe anual:           {sharpe_base:.4f}")


# =========================================================
# 6. MONTE CARLO RESTRINGIDO POR PERFIL RV/RF
# =========================================================
num_ports = 10000

ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sharpe_arr = np.zeros(num_ports)
all_weights = np.zeros((num_ports, len(etfs)))

n_rv = len(mejores_activos_rv)
n_rf = len(renta_fija)

for i in range(num_ports):
    # Pesos aleatorios en renta variable
    if n_rv > 0:
        w_rv = np.random.random(n_rv)
        w_rv = w_rv / w_rv.sum()
        w_rv = w_rv * objetivo_rv
    else:
        w_rv = np.array([])

    # Pesos aleatorios en renta fija
    if n_rf > 0:
        w_rf = np.random.random(n_rf)
        w_rf = w_rf / w_rf.sum()
        w_rf = w_rf * objetivo_rf
    else:
        w_rf = np.array([])

    pesos = np.concatenate([w_rv, w_rf])

    all_weights[i, :] = pesos
    ret_i, vol_i, sharpe_i = metricas_portafolio(pesos, log_ret, rf_annual)

    ret_arr[i] = ret_i
    vol_arr[i] = vol_i
    sharpe_arr[i] = sharpe_i

# Criterio de selección según perfil
if perfil == "Conservador":
    criterio = np.nanargmin(vol_arr)
elif perfil == "Moderado":
    criterio = np.nanargmax(sharpe_arr)
else:  # Agresivo
    criterio = np.nanargmax(ret_arr)

best_weights = all_weights[criterio, :]
best_ret = ret_arr[criterio]
best_vol = vol_arr[criterio]
best_sharpe = sharpe_arr[criterio]

print("\nMejor cartera Monte Carlo:")
print(pd.DataFrame(best_weights, index=etfs, columns=["Peso"]))

print("\nMétricas de la cartera óptima:")
print(f"  Retorno esperado anual: {best_ret:.4f}")
print(f"  Volatilidad anual:      {best_vol:.4f}")
print(f"  Sharpe anual:           {best_sharpe:.4f}")


# =========================================================
# 7. GRÁFICO MONTE CARLO
# =========================================================
plt.figure(figsize=(10, 6))
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap="viridis", alpha=0.6)
plt.colorbar(label="Sharpe Ratio")
plt.scatter(best_vol, best_ret, c="red", s=120, edgecolor="black", label="Cartera óptima")
plt.title(f"Carteras simuladas - Perfil {perfil}")
plt.xlabel("Volatilidad anual")
plt.ylabel("Retorno anual esperado")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# =========================================================
# 8. MÓDULO MACRO (FRED) + SENTIMIENTO
# =========================================================
fred_start = "2010-01-01"

# PIB real trimestral -> variación interanual (4 trimestres)
gdp = fred_series("GDPC1", fred_start, yoy_periods=4)

# Desempleo mensual -> nivel
unemp = fred_series("UNRATE", fred_start, yoy_periods=None)

# CPI mensual -> inflación interanual (12 meses)
cpi = fred_series("CPIAUCSL", fred_start, yoy_periods=12)

# Curva de rendimientos
dgs10 = fred_series("DGS10", fred_start, yoy_periods=None)
dgs2 = fred_series("DGS2", fred_start, yoy_periods=None)

curve = None
if dgs10 is not None and dgs2 is not None:
    curve = dgs10 - dgs2

gdp_yoy = None if gdp is None or gdp.dropna().empty else gdp.dropna().iloc[-1]
unemp_lvl = None if unemp is None or unemp.dropna().empty else unemp.dropna().iloc[-1]
cpi_yoy = None if cpi is None or cpi.dropna().empty else cpi.dropna().iloc[-1]
curve_last = None if curve is None or curve.dropna().empty else curve.dropna().iloc[-1]

fear_greed_raw = input("\nFear & Greed Index (0-100) o 'none': ").strip().lower()
try:
    fear_greed_val = float(fear_greed_raw)
except ValueError:
    fear_greed_val = None

# Puntaje macro
macro_score = 0.0

if gdp_yoy is not None and gdp_yoy > 1:
    macro_score += 1.0

if curve_last is not None and curve_last < 0:
    macro_score -= 1.0

if cpi_yoy is not None and cpi_yoy > 4:
    macro_score -= 0.5

if unemp_lvl is not None and unemp_lvl < 4:
    macro_score += 0.5

# Puntaje sentimiento
sent_score = 0.0
if fear_greed_val is not None:
    if fear_greed_val < 30:
        sent_score += 0.5
    elif fear_greed_val > 70:
        sent_score -= 0.5

print("\nContexto macro actual:")
print(f"  GDP YoY:      {gdp_yoy}")
print(f"  Unemployment: {unemp_lvl}")
print(f"  CPI YoY:      {cpi_yoy}")
print(f"  Yield Curve:  {curve_last}")
print(f"  Macro score:  {macro_score:.1f}")
print(f"  Sent score:   {sent_score:.1f}")


# =========================================================
# 9. DECISIÓN TÁCTICA POR ACTIVO
# =========================================================
print("\nSeñal táctica por activo de renta variable:")

for ticker in mejores_activos_rv:
    serie = datos_etfs[ticker].dropna()
    tech_score = puntaje_tecnico(serie)
    total_score = tech_score + macro_score + sent_score

    if total_score >= 1.5:
        decision = "COMPRAR"
    elif total_score <= -1.5:
        decision = "VENDER"
    else:
        decision = "MANTENER"

    print(
        f"{ticker}: "
        f"Técnico={tech_score}, "
        f"Macro={macro_score:.1f}, "
        f"Sentimiento={sent_score:.1f} "
        f"-> {decision}"
    )