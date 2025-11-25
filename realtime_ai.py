#!/usr/bin/env python3
"""
REAL-TIME K8s AI FORECASTER (config-driven)
All URLs & tunables are read from environment variables.
"""

import os
import requests
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import time
import json
from datetime import datetime, timedelta
import logging

# ----------------------------------------------------------------------
# 1. CONFIG – READ FROM ENV (with safe defaults)
# ----------------------------------------------------------------------
VM_URL = os.getenv(
    "VM_URL",
    "http://vm.london.local/api/v1/query_range"   # fallback – never hard-coded in image
)

PUSHGATEWAY = os.getenv(
    "PUSHGATEWAY",
    "http://pushgateway.monitoring:9091"
)

# Forecast horizon (minutes)
HORIZON_MIN = int(os.getenv("HORIZON_MIN", "15"))

# How far back the anomaly model looks (hours)
LOOKBACK_HOURS = int(os.getenv("LOOKBACK_HOURS", "24"))

# Expected fraction of anomalous nodes (0.0 – 1.0)
CONTAMINATION = float(os.getenv("CONTAMINATION", "0.12"))

# How often we pull data / push metrics
STEP = os.getenv("STEP", "60s")          # VictoriaMetrics step
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))   # seconds between cycles

# Model files (must be mounted)
HOST_MODEL = os.getenv("HOST_MODEL", "host_forecast.pkl")
POD_MODEL  = os.getenv("POD_MODEL",  "pod_forecast.pkl")

# ----------------------------------------------------------------------
# 2. LOGGING
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

log.info(f"Config → VM_URL={VM_URL} | PUSHGATEWAY={PUSHGATEWAY}")
log.info(f"Tunables → HORIZON={HORIZON_MIN}m | LOOKBACK={LOOKBACK_HOURS}h | CONTAM={CONTAMINATION}")

# ----------------------------------------------------------------------
# 3. FETCH LIVE DATA
# ----------------------------------------------------------------------
def fetch_live(query, hours=1):
    end   = int(datetime.now().timestamp())
    start = int((datetime.now() - timedelta(hours=hours)).timestamp())
    try:
        r = requests.get(
            VM_URL,
            params={'query': query, 'start': start, 'end': end, 'step': STEP},
            timeout=10
        )
        r.raise_for_status()
        data = r.json()['data']['result']
        rows = []
        for s in data:
            df = pd.DataFrame(s['values'], columns=['ts', 'value'])
            df['ts'] = pd.to_datetime(df['ts'], unit='s')
            df['value'] = pd.to_numeric(df['value'])
            for k, v in s['metric'].items():
                df[k] = v
            rows.append(df)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    except Exception as e:
        log.error(f"Query failed [{query}]: {e}")
        return pd.DataFrame()

# ----------------------------------------------------------------------
# 4. FORECAST (single value – 15-min ahead)
# ----------------------------------------------------------------------
def run_forecast(model_path, df_cpu, df_mem, layer_name):
    models = joblib.load(model_path)
    prophet_model = models['prophet']
    arima_model   = models['arima']

    # Combine CPU + MEM → target = memory (normalized)
    if df_mem is not None and not df_mem.empty:
        mem = df_mem.groupby('timestamp')['value'].mean()
        ts  = mem
    else:
        ts = df_cpu.groupby('timestamp')['value'].mean()

    # Prophet
    future = prophet_model.make_future_dataframe(periods=HORIZON_MIN, freq='min')
    future['hour'] = future['ds'].dt.hour
    future['is_weekend'] = (future['ds'].dt.dayofweek >= 5).astype(int)
    f_prophet = prophet_model.predict(future)['yhat'].tail(HORIZON_MIN)

    # ARIMA
    try:
        f_arima = arima_model.forecast(steps=HORIZON_MIN)
        f_arima.index = pd.date_range(start=ts.index[-1] + timedelta(minutes=1),
                                      periods=HORIZON_MIN, freq='min')
    except Exception:
        f_arima = pd.Series([ts.iloc[-1]] * HORIZON_MIN, index=f_prophet.index)

    ensemble = (f_prophet.values + f_arima.values) / 2
    return float(ensemble[-1])   # final 15-min forecast

# ----------------------------------------------------------------------
# 5. ANOMALY CLASSIFICATION
# ----------------------------------------------------------------------
def classify_anomaly(df_hcpu, df_hmem, df_pcpu, df_pmem):
    now = datetime.now()
    start = now - timedelta(hours=LOOKBACK_HOURS)

    def recent(df):
        return df[df['timestamp'] >= start] if not df.empty else df

    hcpu = recent(df_hcpu).groupby('instance')['value'].mean().rename('host_cpu')
    hmem = recent(df_hmem).groupby('instance')['value'].mean().rename('host_mem')
    pcpu = recent(df_pcpu).groupby('instance')['value'].mean().rename('pod_cpu')
    pmem = recent(df_pmem).groupby('instance')['value'].mean().rename('pod_mem')

    feats = pd.concat([hcpu, hmem, pcpu, pmem], axis=1).fillna(0).reset_index()
    X = StandardScaler().fit_transform(feats[['host_cpu','host_mem','pod_cpu','pod_mem']])
    iso = IsolationForest(contamination=CONTAMINATION, random_state=42)
    labels = iso.fit_predict(X)
    feats['anomaly'] = labels
    return feats[feats['anomaly'] == -1]

# ----------------------------------------------------------------------
# 6. PUSH TO PROMETHEUS
# ----------------------------------------------------------------------
def push_metrics(host_forecast, pod_forecast, divergence, anomaly_count):
    job = "realtime_ai_forecaster"
    metrics = f"""
# HELP k8s_ai_memory_forecast_15min 15-min memory forecast (normalized)
# TYPE k8s_ai_memory_forecast_15min gauge
k8s_ai_memory_forecast_15min{{layer="host"}} {host_forecast}
k8s_ai_memory_forecast_15min{{layer="pod"}} {pod_forecast}
# HELP k8s_ai_divergence Host vs Pod memory divergence
# TYPE k8s_ai_divergence gauge
k8s_ai_divergence {divergence}
# HELP k8s_ai_anomaly_count Number of nodes with non-K8s load
# TYPE k8s_ai_anomaly_count gauge
k8s_ai_anomaly_count {anomaly_count}
"""
    try:
        requests.post(f"{PUSHGATEWAY}/metrics/job/{job}", data=metrics, timeout=5)
        log.info(f"Pushed → host={host_forecast:.3f}, pod={pod_forecast:.3f}, div={divergence:.3f}, anomalies={anomaly_count}")
    except Exception as e:
        log.error(f"Push failed: {e}")

# ----------------------------------------------------------------------
# 7. MAIN LOOP
# ----------------------------------------------------------------------
def main():
    log.info("Real-time AI forecaster started")
    while True:
        try:
            # 1. Pull live data (last hour is enough for forecast)
            df_hcpu = fetch_live('1 - avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) by (instance)', hours=1)
            df_hmem = fetch_live('(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes', hours=1)
            df_pcpu = fetch_live('sum(rate(container_cpu_usage_seconds_total{container!="POD",container!=""}[5m])) by (instance)', hours=1)
            df_pmem = fetch_live('sum(container_memory_working_set_bytes{container!="POD",container!=""}[5m]) by (instance)', hours=1)

            if df_hmem.empty or df_pmem.empty:
                log.warning("No data – skipping cycle")
                time.sleep(POLL_INTERVAL)
                continue

            # 2. Forecast
            host_f = run_forecast(HOST_MODEL, df_hcpu, df_hmem, "host")
            pod_f  = run_forecast(POD_MODEL,  df_pcpu, df_pmem, "pod")
            div    = abs(host_f - pod_f)

            # 3. Anomaly
            anomalies = classify_anomaly(df_hcpu, df_hmem, df_pcpu, df_pmem)

            # 4. Push
            push_metrics(host_f, pod_f, div, len(anomalies))

            # 5. Alert (optional)
            if len(anomalies) > 0:
                log.warning(f"ALERT: {len(anomalies)} anomalous nodes → {list(anomalies['instance'])}")

            time.sleep(POLL_INTERVAL)

        except Exception as e:
            log.error(f"Cycle error: {e}")
            time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
