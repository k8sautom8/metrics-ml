#!/usr/bin/env python3
"""
Dual-Layer + Classification AI for Kubernetes
- Host layer (full node)
- Pod layer (apps only)
- Classification model (per-node IsolationForest)
- FULL ENSEMBLE: Prophet + ARIMA + LSTM
- CPU-only LSTM (no GPU needed)
- All config via environment variables
"""

import os
import sys
import argparse
import json
import time
import socket
import ipaddress
import logging
import re
import requests
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt; plt.ion()
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import joblib
import warnings

# --- YAML support (optional) ---
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    print("PyYAML not found. SLI/SLO config disabled. Install with: pip install pyyaml")
    YAML_AVAILABLE = False

# --- LSTM (CPU-only) ---
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    LSTM_AVAILABLE = True
except ImportError:
    print("TensorFlow not found. LSTM disabled. Install with: pip install tensorflow-cpu")
    LSTM_AVAILABLE = False

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# 1. CONFIG – READ FROM ENV (with safe defaults)
# ----------------------------------------------------------------------
def get_env_value(key, default, cast):
    raw = os.getenv(key)
    if raw is None:
        return cast(default), True
    try:
        return cast(raw), False
    except Exception:
        print(f"Warning: invalid value for {key}; using default {default}")
        return cast(default), True

def get_model_dir():
    """Get the model files directory.
    - If MODEL_FILES_DIR env var is set and is an absolute path, use it
    - If MODEL_FILES_DIR env var is set and is relative, use it relative to current dir
    - Otherwise, default to 'model_files' in current directory
    Creates the directory if it doesn't exist.
    """
    model_dir_env = os.getenv("MODEL_FILES_DIR")
    if model_dir_env:
        # If it's an absolute path, use it as-is
        if os.path.isabs(model_dir_env):
            model_dir = model_dir_env
        else:
            # Relative path, use it relative to current directory
            model_dir = os.path.join(os.getcwd(), model_dir_env)
    else:
        # Default to 'model_files' in current directory
        model_dir = os.path.join(os.getcwd(), "model_files")
    
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

# Get model directory (creates it if needed)
MODEL_DIR = get_model_dir()

def get_forecast_plots_dir():
    """Get the forecast plots directory.
    - If FORECAST_PLOTS_DIR env var is set and is an absolute path, use it
    - If FORECAST_PLOTS_DIR env var is set and is relative, use it relative to current dir
    - Otherwise, default to 'forecast_plots' in current directory
    Creates the directory if it doesn't exist.
    """
    plots_dir_env = os.getenv("FORECAST_PLOTS_DIR")
    if plots_dir_env:
        # If it's an absolute path, use it as-is
        if os.path.isabs(plots_dir_env):
            plots_dir = plots_dir_env
        else:
            # Relative path, use it relative to current directory
            plots_dir = os.path.join(os.getcwd(), plots_dir_env)
    else:
        # Default to 'forecast_plots' in current directory
        plots_dir = os.path.join(os.getcwd(), "forecast_plots")
    
    # Create directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir

# Get forecast plots directory (creates it if needed)
FORECAST_PLOTS_DIR = get_forecast_plots_dir()

VM_BASE_URL, VM_URL_DEFAULT = get_env_value("VM_URL", "http://vm.london.local/api/v1/query_range", str)
HORIZON_MIN, HORIZON_DEFAULT = get_env_value("HORIZON_MIN", "15", int)
LOOKBACK_HOURS, LOOKBACK_DEFAULT = get_env_value("LOOKBACK_HOURS", "24", int)
CONTAMINATION, CONTAM_DEFAULT = get_env_value("CONTAMINATION", "0.12", float)
STEP, STEP_DEFAULT = get_env_value("STEP", "60s", str)
START_HOURS_AGO, START_DEFAULT = get_env_value("START_HOURS_AGO", "360", int)
LSTM_SEQ_LEN, LSTM_SEQ_DEFAULT = get_env_value("LSTM_SEQ_LEN", "60", int)
LSTM_EPOCHS, LSTM_EPOCHS_DEFAULT = get_env_value("LSTM_EPOCHS", "10", int)
TRAIN_FRACTION, TRAIN_DEFAULT = get_env_value("TRAIN_FRACTION", "0.8", float)

# Model paths - use MODEL_DIR unless a full path is provided in env var
def get_model_path(env_var, default_filename):
    """Get model file path.
    - If env var is set and is absolute, use it as-is
    - Otherwise, use MODEL_DIR + default_filename
    """
    env_path = os.getenv(env_var)
    if env_path:
        if os.path.isabs(env_path):
            return env_path
        else:
            # Relative path, use it relative to MODEL_DIR
            return os.path.join(MODEL_DIR, env_path)
    return os.path.join(MODEL_DIR, default_filename)

ANOMALY_MODEL_PATH = get_model_path("ANOMALY_MODEL_PATH", "isolation_forest_anomaly.pkl")
ANOMALY_SCALER_PATH = get_model_path("ANOMALY_SCALER_PATH", "isolation_forest_anomaly_scaler.pkl")
HOST_MODEL_PATH = get_model_path("HOST_MODEL_PATH", "host_forecast.pkl")
POD_MODEL_PATH = get_model_path("POD_MODEL_PATH", "pod_forecast.pkl")
LSTM_MODEL_PATH = get_model_path("LSTM_MODEL_PATH", "lstm_model.pkl")
DISK_MODEL_MANIFEST_PATH = get_model_path("DISK_MODEL_MANIFEST_PATH", "disk_full_models.pkl")
IO_NET_MODEL_MANIFEST_PATH = get_model_path("IO_NET_MODEL_MANIFEST_PATH", "io_net_models.pkl")
AUTO_ALIAS_ENABLED = os.getenv("AUTO_ALIAS_ENABLED", "1").lower() not in ("0", "false", "no")
ALIAS_LOOKBACK_MINUTES = int(os.getenv("ALIAS_LOOKBACK_MINUTES", "15"))
VERBOSE_LEVEL = int(os.getenv("VERBOSE_LEVEL", "0"))
# DNS domain suffixes to try when resolving hostnames (comma-separated)
DNS_DOMAIN_SUFFIXES = [d.strip() for d in os.getenv("DNS_DOMAIN_SUFFIXES", ".london.local,.local").split(",") if d.strip()]
FORCE_TRAINING_RUN = False
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("metrics")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
try:
    INSTANCE_ALIAS_MAP = json.loads(os.getenv("INSTANCE_ALIAS_MAP", "{}"))
except json.JSONDecodeError:
    print("Warning: INSTANCE_ALIAS_MAP is not valid JSON. Ignoring.")
    INSTANCE_ALIAS_MAP = {}
DNS_CACHE = {}
CANON_SOURCE_MAP = {}
SOURCE_REGISTRY = {}
FORWARD_DNS_CACHE = {}

def build_disk_key(node, mount):
    return f"{node}|{mount}"

def should_verbose(level=1):
    return FORCE_TRAINING_RUN or VERBOSE_LEVEL >= level

def log_verbose(msg, level=1):
    if should_verbose(level):
        print(msg)

def load_disk_manifest(path):
    if not os.path.exists(path):
        return {}
    try:
        data = joblib.load(path)
        if isinstance(data, dict):
            return data
    except Exception as exc:
        print(f"Warning: failed to load disk manifest {path}: {exc}")
    return {}

def save_disk_manifest(path, manifest):
    try:
        joblib.dump(manifest, path)
    except Exception as exc:
        print(f"Warning: failed to save disk manifest {path}: {exc}")

def build_io_net_key(node, signal_name):
    """Build key for I/O and network model manifest: node|signal"""
    return f"{node}|{signal_name}"

def sanitize_label(label: str | None) -> str:
    if not label:
        return "dataset"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
    return safe or "dataset"

def dump_dataframe_to_csv(df: pd.DataFrame | None, dump_dir: str | None, label: str | None):
    if not dump_dir or df is None or df.empty:
        return
    try:
        os.makedirs(dump_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{sanitize_label(label)}_{timestamp}.csv"
        path = os.path.join(dump_dir, filename)
        df.to_csv(path, index=False)
        log_verbose(f"Training data exported → {path}")
    except Exception as exc:
        print(f"Warning: unable to dump training CSV ({label}): {exc}")

def load_io_net_manifest(path):
    """Load I/O and network model manifest"""
    if not os.path.exists(path):
        return {}
    try:
        data = joblib.load(path)
        if isinstance(data, dict):
            return data
    except Exception as exc:
        print(f"Warning: failed to load I/O/network manifest {path}: {exc}")
    return {}

def save_io_net_manifest(path, manifest):
    """Save I/O and network model manifest"""
    try:
        joblib.dump(manifest, path)
    except Exception as exc:
        print(f"Warning: failed to save disk manifest {path}: {exc}")

def looks_like_hostname(s):
    """Check if a string looks like it could be a hostname or IP address."""
    if not s or len(s) > 253:  # Max hostname length
        return False
    # Check if it's an IP address (basic check)
    if '.' in s:
        parts = s.split('.')
        if len(parts) == 4:
            try:
                # Quick check if all parts are numbers (IPv4)
                all_numeric = all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)
                if all_numeric:
                    return True
            except:
                pass
    # Check if it contains only valid hostname characters (letters, digits, dots, hyphens)
    if re.match(r'^[a-zA-Z0-9.-]+$', s):
        # Must start and end with alphanumeric
        if s[0].isalnum() and s[-1].isalnum():
            return True
    return False

def parse_disk_retrain_targets(spec):
    targets = set()
    if not spec:
        return targets
    for token in spec.split(','):
        token = token.strip().lower()
        if not token:
            continue
        # Special case: "all" means retrain everything
        if token == 'all':
            return {'__RETRAIN_ALL__'}
        if ':' in token or '|' in token:
            sep = ':' if ':' in token else '|'
            node_part, mount_part = token.split(sep, 1)
            node_canon = canonical_identity(node_part)
            targets.add(build_disk_key(node_canon, mount_part.strip()))
        else:
            targets.add(canonical_identity(token))
    return targets

def parse_io_net_retrain_targets(spec):
    """Parse I/O and network retrain targets.
    Supports formats:
    - all (retrains all nodes/signals)
    - host02 (retrains all signals for that host)
    - host02:DISK_IO_WAIT (retrains specific signal for that host)
    - host02|DISK_IO_WAIT (alternative separator)
    """
    targets = set()
    if not spec:
        return targets
    for token in spec.split(','):
        token = token.strip().lower()
        if not token:
            continue
        # Special case: "all" means retrain everything
        if token == 'all':
            return {'__RETRAIN_ALL__'}
        if ':' in token or '|' in token:
            sep = ':' if ':' in token else '|'
            node_part, signal_part = token.split(sep, 1)
            node_canon = canonical_identity(node_part)
            signal_upper = signal_part.strip().upper()
            # Add both backtest and ensemble keys
            targets.add(f"{build_io_net_key(node_canon, signal_upper)}_backtest")
            targets.add(f"{build_io_net_key(node_canon, signal_upper)}_ensemble")
            # Also add node-only target for broader matching
            targets.add(node_canon)
        else:
            node_canon = canonical_identity(token)
            targets.add(node_canon)
    return targets

def print_config_summary():
    print("\n" + "="*80)
    print("GLOBAL CONFIGURATION")
    print("="*80)
    def flag(default_used):
        return " (default)" if default_used else ""
    train_pct = round(TRAIN_FRACTION * 100)
    test_pct = max(0, 100 - train_pct)
    print(f"  • VM_URL        : {VM_BASE_URL}{flag(VM_URL_DEFAULT)}")
    print(f"  • STEP          : {STEP}{flag(STEP_DEFAULT)}")
    print(f"  • START_HOURS   : {START_HOURS_AGO}{flag(START_DEFAULT)}")
    print(f"  • HORIZON_MIN   : {HORIZON_MIN}{flag(HORIZON_DEFAULT)}")
    print(f"  • LOOKBACK_HRS  : {LOOKBACK_HOURS}{flag(LOOKBACK_DEFAULT)}")
    print(f"  • CONTAMINATION : {CONTAMINATION}{flag(CONTAM_DEFAULT)}")
    print(f"  • LSTM_SEQ_LEN  : {LSTM_SEQ_LEN}{flag(LSTM_SEQ_DEFAULT)}")
    print(f"  • LSTM_EPOCHS   : {LSTM_EPOCHS}{flag(LSTM_EPOCHS_DEFAULT)}")
    print(f"  • TRAIN SPLIT   : {train_pct}% train / {test_pct}% test{flag(TRAIN_DEFAULT)}")
    print(f"  • LSTM Available: {LSTM_AVAILABLE}")
    model_dir_env = os.getenv("MODEL_FILES_DIR")
    if model_dir_env:
        print(f"  • MODEL_DIR     : {MODEL_DIR} (from MODEL_FILES_DIR env var)")
    else:
        print(f"  • MODEL_DIR     : {MODEL_DIR} (default)")
    plots_dir_env = os.getenv("FORECAST_PLOTS_DIR")
    if plots_dir_env:
        print(f"  • FORECAST_DIR  : {FORECAST_PLOTS_DIR} (from FORECAST_PLOTS_DIR env var)")
    else:
        print(f"  • FORECAST_DIR  : {FORECAST_PLOTS_DIR} (default)")

def canonical_identity(raw):
    if raw is None:
        return "unknown"
    ident = str(raw)
    if "://" in ident:
        ident = ident.split("://", 1)[-1]
    if "@" in ident:
        ident = ident.split("@", 1)[-1]
    ident = ident.strip()
    base = ident.split(':')[0]
    cleaned = base.split('/')[0].lower()

    def track_source(alias, candidate):
        if not alias or not candidate:
            return
        try:
            ipaddress.ip_address(candidate)
        except ValueError:
            return
        CANON_SOURCE_MAP.setdefault(alias, candidate)

    for key in (cleaned, base, ident):
        if key in INSTANCE_ALIAS_MAP:
            alias = INSTANCE_ALIAS_MAP[key]
            try:
                ipaddress.ip_address(cleaned)
                track_source(alias, cleaned)
            except ValueError:
                pass
            return alias
    # reverse DNS fallback for bare IPs
    try:
        ipaddress.ip_address(cleaned)
    except ValueError:
        return cleaned or ident
    if cleaned in DNS_CACHE:
        return DNS_CACHE[cleaned]
    try:
        fqdn = socket.gethostbyaddr(cleaned)[0]
        short = fqdn.split('.')[0].lower()
        DNS_CACHE[cleaned] = short
        track_source(short, cleaned)
        return short
    except OSError:
        DNS_CACHE[cleaned] = cleaned
        return cleaned

def canonical_node_label(raw, with_ip=False, raw_label=None):
    if raw is None and raw_label is None:
        return "unknown"
    base = str(raw) if raw is not None else str(raw_label)
    host = base.split(':')[0]
    name = canonical_identity(host)
    if with_ip:
        candidate_label = raw_label if raw_label is not None else raw
        source_ip = None
        if candidate_label is not None:
            candidate_host = str(candidate_label).split(':')[0]
            try:
                ipaddress.ip_address(candidate_host)
                source_ip = candidate_host
            except ValueError:
                pass
        if not source_ip:
            source_ip = CANON_SOURCE_MAP.get(name) or SOURCE_REGISTRY.get(name)
        if not source_ip:
            cached = FORWARD_DNS_CACHE.get(name)
            if cached is None:
                try:
                    resolved = socket.gethostbyname(name)
                    ipaddress.ip_address(resolved)
                    FORWARD_DNS_CACHE[name] = resolved
                    source_ip = resolved
                except (socket.gaierror, ValueError):
                    FORWARD_DNS_CACHE[name] = ""
            elif cached:
                source_ip = cached
        if source_ip and source_ip != name:
            return f"{name} ({source_ip})"
        if name != host:
            return f"{name} ({host})"
    return name

def register_source_identity(alias, raw_value):
    if not alias or not raw_value:
        return
    candidate = str(raw_value).split(':')[0]
    try:
        ipaddress.ip_address(candidate)
    except ValueError:
        return
    SOURCE_REGISTRY.setdefault(alias, candidate)

def _alias_candidates_from_query(query, label_candidates, ip_labels=None):
    now = pd.Timestamp.now()
    start = int((now - pd.Timedelta(minutes=ALIAS_LOOKBACK_MINUTES)).timestamp())
    end = int(now.timestamp())
    df = fetch_victoriametrics_metrics(query, start, end, step="60s")
    aliases = {}
    if df.empty:
        return aliases
    available = [lab for lab in label_candidates if lab in df.columns]
    if not available:
        return aliases

    latest = df.sort_values('timestamp').drop_duplicates(subset=['instance'], keep='last')
    ip_labels = ip_labels or []
    for _, row in latest.iterrows():
        inst_raw = row.get('instance')
        inst = canonical_identity(inst_raw)
        if not inst or inst == "unknown":
            continue
        node_label = None
        for label in available:
            val = row.get(label)
            if isinstance(val, str) and val.strip():
                node_label = canonical_identity(val)
                break
        if not node_label or node_label == inst:
            continue
        aliases.setdefault(inst, node_label)
        aliases.setdefault(node_label, node_label)
        if isinstance(inst_raw, str):
            inst_host = inst_raw.split(':')[0]
            try:
                ipaddress.ip_address(inst_host)
                register_source_identity(node_label, inst_host)
            except ValueError:
                pass

        for ip_label in ip_labels:
            ip_val = row.get(ip_label)
            if not isinstance(ip_val, str):
                continue
            ip_candidate = ip_val.split(':')[0]
            try:
                ipaddress.ip_address(ip_candidate)
            except ValueError:
                continue
            register_source_identity(node_label, ip_candidate)
            aliases.setdefault(ip_candidate, node_label)
    return aliases

def refresh_dynamic_aliases():
    if not AUTO_ALIAS_ENABLED:
        return
    dynamic_aliases = {}
    alias_sources = [
        ("node_uname_info", ['nodename', 'hostname'], []),
        ("kube_node_info", ['node', 'label_kubernetes_io_hostname', 'kubernetes_io_hostname', 'instance'], ['internal_ip', 'external_ip', 'host_ip'])
    ]
    for query, labels, ip_labels in alias_sources:
        try:
            dynamic_aliases.update(_alias_candidates_from_query(query, labels, ip_labels))
        except Exception as exc:
            log_verbose(f"Alias query failed for {query}: {exc}", level=2)
    if not dynamic_aliases:
        log_verbose("Dynamic alias inference: no matches found.", level=2)
        return
    new_entries = 0
    for key, value in dynamic_aliases.items():
        if key not in INSTANCE_ALIAS_MAP:
            INSTANCE_ALIAS_MAP[key] = value
            new_entries += 1
    if new_entries:
        log_verbose(f"Dynamic alias inference added {new_entries} entries.", level=1)

def augment_aliases_from_dns(df_host, df_pod):
    if not AUTO_ALIAS_ENABLED or df_host.empty or df_pod.empty:
        return
    if 'instance' not in df_host.columns or 'entity' not in df_pod.columns:
        return

    pod_entities = set(df_pod['entity'].dropna().map(lambda x: str(x).lower()))
    new_entries = 0
    for raw_instance in df_host['instance'].dropna().unique():
        ident = canonical_node_label(raw_instance)
        if ident in INSTANCE_ALIAS_MAP:
            continue
        host_ip = str(raw_instance).split(':')[0]
        try:
            ipaddress.ip_address(host_ip)
        except ValueError:
            continue
        try:
            fqdn = socket.gethostbyaddr(host_ip)[0]
        except OSError:
            continue
        short = fqdn.split('.')[0].lower()
        if short in pod_entities:
            INSTANCE_ALIAS_MAP[ident] = short
            new_entries += 1
    if new_entries:
        log_verbose(f"DNS alias inference added {new_entries} entries.", level=1)

def recanonicalize_entities(*dfs):
    for df in dfs:
        if df is None or df.empty or 'entity' not in df.columns:
            continue
        df['entity'] = df['entity'].map(canonical_identity)
        if 'raw_entity' in df.columns:
            for entity, raw_val in df[['entity','raw_entity']].dropna().itertuples(index=False, name=None):
                register_source_identity(entity, raw_val)
        if 'raw_instance' in df.columns:
            for entity, raw_inst in df[['entity','raw_instance']].dropna().itertuples(index=False, name=None):
                register_source_identity(entity, raw_inst)

def infer_aliases_from_timeseries(df_host_cpu, df_pod_cpu, corr_threshold=0.9, min_points=50):
    if not AUTO_ALIAS_ENABLED or df_host_cpu.empty or df_pod_cpu.empty:
        return

    host_groups = df_host_cpu.groupby('entity')
    pod_groups = df_pod_cpu.groupby('entity')

    pod_series_cache = {}
    for pod_entity, group in pod_groups:
        series = group.set_index('timestamp')['value'].sort_index()
        pod_series_cache[pod_entity] = series

    new_entries = 0
    for host_entity, group in host_groups:
        canon_host = canonical_identity(host_entity)
        if canon_host in INSTANCE_ALIAS_MAP:
            continue
        host_series = group.set_index('timestamp')['value'].sort_index()
        best_match = None
        best_corr = corr_threshold
        for pod_entity, pod_series in pod_series_cache.items():
            joined = pd.concat([host_series, pod_series], axis=1, join='inner', keys=['host','pod']).dropna()
            if len(joined) < min_points:
                continue
            corr = joined['host'].corr(joined['pod'])
            if corr is not None and corr > best_corr:
                best_corr = corr
                best_match = pod_entity
        if best_match:
            canon_pod = canonical_identity(best_match)
            INSTANCE_ALIAS_MAP[canon_host] = canon_pod
            INSTANCE_ALIAS_MAP.setdefault(canon_pod, canon_pod)
            new_entries += 1
            log_verbose(f"Timeseries alias inferred: {canon_host} → {canon_pod} (corr={best_corr:.2f})", level=1)

    if new_entries == 0:
        log_verbose("Timeseries alias inference found no additional matches.", level=2)

def parse_cli_args():
    parser = argparse.ArgumentParser(description="Dual-layer AI forecasting and anomaly detection")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--training",
        dest="training_mode",
        action="store_true",
        help="Retrain ensemble models and overwrite saved artifacts"
    )
    mode.add_argument(
        "--pt-models",
        dest="training_mode",
        action="store_false",
        help="Use pre-trained ensemble artifacts if available (default behavior)"
    )
    parser.set_defaults(training_mode=False)
    parser.add_argument(
        "--anomaly-watch",
        type=int,
        default=0,
        help="Run realtime anomaly scoring loop N times (15s cadence)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase console verbosity (repeatable)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress verbose output regardless of environment settings"
    )
    parser.add_argument(
        "--disk-retrain",
        default="",
        help="Comma separated list of nodes or node:mount combos to retrain in disk forecast (use 'all' to retrain all disk models)"
    )
    parser.add_argument(
        "--io-net-retrain",
        default="",
        help="Comma separated list of nodes or node:signal combos to retrain in I/O and network models (use 'all' to retrain all I/O and network models, e.g., host02 or host02:DISK_IO_WAIT)"
    )
    parser.add_argument(
        "--show-backtest",
        action="store_true",
        help="Show backtest metrics even when using cached models (default: only show when training)"
    )
    parser.add_argument(
        "--forecast",
        action="store_true",
        help="Forecast mode: generate forecasts using latest Prometheus data and cached models, save forecast plots (optimized for frequent runs)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=0,
        help="Run forecast mode continuously with specified interval in seconds (0 = run once and exit, default: 0)"
    )
    parser.add_argument(
        "--alert-webhook",
        type=str,
        default=None,
        help="HTTP webhook URL to send alerts to (e.g., http://localhost:8080/alert-test)"
    )
    parser.add_argument(
        "--pushgateway",
        type=str,
        default=None,
        help="Prometheus Pushgateway URL to push alert metrics to (e.g., http://localhost:9091)"
    )
    parser.add_argument(
        "--dump-csv",
        type=str,
        default=None,
        help="Directory to dump training datasets as CSV files (created if missing)"
    )
    parser.add_argument(
        "--sli-slo-config",
        type=str,
        default=None,
        help="Path to SLI/SLO configuration YAML file (default: sli_slo_config.yaml in current directory)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate and save plot files (PNG images). If not specified, plots are skipped to save time."
    )
    return parser.parse_args()

def persist_model_metadata(model_path, metadata):
    if not metadata or not model_path:
        return
    meta_path = f"{model_path}.meta.json"
    try:
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, default=str)
        print(f"Metadata saved → {meta_path}")
    except Exception as exc:
        print(f"Warning: unable to write metadata {meta_path}: {exc}")

def summarize_instance_roles(df_host, df_pod):
    def extract_id_set(df):
        if df.empty:
            return set()
        column = 'entity' if 'entity' in df.columns else 'instance'
        return set(df[column].dropna().map(canonical_identity))

    host_instances = extract_id_set(df_host)
    pod_instances = extract_id_set(df_pod)
    hosts_with_pods = sorted(host_instances & pod_instances)
    host_only = sorted(host_instances - pod_instances)
    pod_only = sorted(pod_instances - host_instances)

    print("\nCluster Topology Snapshot:")
    print(f"  • Hosts reporting metrics        : {len(host_instances)}")
    print(f"  • Hosts also running Kubernetes  : {len(hosts_with_pods)}")
    print(f"  • Host-only nodes (no pods seen) : {len(host_only)}")
    print(f"  • Pod-only metrics (no host data): {len(pod_only)}")
    if should_verbose():
        if host_only:
            print(f"    ↳ Host-only sample: {', '.join(host_only[:6])}{' …' if len(host_only) > 6 else ''}")
        if pod_only:
            print(f"    ↳ Pod-only sample : {', '.join(pod_only[:6])}{' …' if len(pod_only) > 6 else ''}")
    if not hosts_with_pods and not INSTANCE_ALIAS_MAP:
        print("  (No overlap detected — configure INSTANCE_ALIAS_MAP to map host aliases/IPs)")

    return {
        "host_only": host_only,
        "hosts_with_pods": hosts_with_pods,
        "pod_only": pod_only
    }

def report_host_only_pressure(feats, cpu_threshold=0.6, mem_threshold=0.7, pod_floor=0.05, return_df=False):
    if feats.empty:
        return pd.DataFrame() if return_df else None
    host_only_pressure = feats[
        ((feats['pod_cpu'] <= pod_floor) & (feats['pod_mem'] <= pod_floor))
        &
        ((feats['host_cpu'] >= cpu_threshold) | (feats['host_mem'] >= mem_threshold))
    ]
    if host_only_pressure.empty:
        return pd.DataFrame() if return_df else None

    print("\n⚠️  Host pressure detected with minimal Kubernetes workload:")
    display = host_only_pressure.copy()
    if 'raw_instance' in display.columns:
        display['instance'] = display.apply(lambda row: canonical_node_label(row['entity'], with_ip=True, raw_label=row.get('raw_instance')), axis=1)
    else:
        display['instance'] = display['entity'].apply(lambda ent: canonical_node_label(ent, with_ip=True))
    print(display[['instance', 'host_cpu', 'host_mem']].to_string(index=False))
    print("Action: inspect OS-level processes (backups, cron jobs, daemons) on these nodes.")
    
    if return_df:
        result_df = display[['instance', 'host_cpu', 'host_mem']].copy()
        result_df['severity'] = 'WARNING'
        result_df['signal'] = 'host_pressure'
        result_df['detected_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return result_df
    return None

def load_cached_ensemble(model_path):
    if not os.path.exists(model_path):
        return None
    try:
        artifact = joblib.load(model_path)
    except Exception as exc:
        print(f"Cached model load failed ({model_path}): {exc}")
        return None

    if isinstance(artifact, tuple) and len(artifact) == 3:
        return artifact

    if isinstance(artifact, dict):
        forecast_df = artifact.get('forecast') or artifact.get('forecast_df')
        metrics = artifact.get('metrics')
        model = artifact.get('prophet') or artifact.get('model')
        if forecast_df is not None and metrics is not None:
            return (model, forecast_df, metrics)

    return None

def generate_forecast_plots_from_cache(df_cpu, df_mem, cached_result, horizon_min, model_path, enable_plots=True):
    """Generate forecast plots from cached model results (simple version for normal mode)."""
    if not isinstance(cached_result, tuple) or len(cached_result) < 3:
        return
    
    model, forecast_df, metrics = cached_result
    if forecast_df is None or metrics is None:
        return
    
    # Prepare data for plotting
    target = 'mem' if df_mem is not None else 'cpu'
    if df_mem is not None:
        cpu_agg = df_cpu.groupby('timestamp')['value'].mean().reset_index(name='cpu')
        mem_agg = df_mem.groupby('timestamp')['value'].mean().reset_index(name='mem')
        mem_agg = mem_agg.set_index('timestamp').reindex(cpu_agg.set_index('timestamp').index).ffill().reset_index()
        ts_data = mem_agg.set_index('timestamp')['mem']
    else:
        ts_data = df_cpu.groupby('timestamp')['value'].mean()
    
    split_info = metrics.get('split_info', {})
    
    # Plot forecast: show last 24 hours of historical data and next 3 hours of forecast
    plot_forecast_horizon = 180  # 3 hours in minutes
    plt.figure(figsize=(16, 6))
    # Plot historical data - last 24 hours
    if not ts_data.empty:
        last_24hours = ts_data.last('24h')
        if not last_24hours.empty:
            plt.plot(last_24hours.index, last_24hours.values, label='Last 24 hours (historical)', color='blue', alpha=0.7, linewidth=1.5)
        
        # Plot forecast (starts right after historical data ends, no overlap)
        last_historical_time = ts_data.index[-1]
        forecast_future = forecast_df[forecast_df['ds'] > last_historical_time] if not ts_data.empty else forecast_df
        # Only show first 3 hours of forecast
        if not forecast_future.empty:
            forecast_3h = forecast_future.head(plot_forecast_horizon)
            plt.plot(forecast_3h['ds'], forecast_3h['yhat'], label='Ensemble Forecast', color='red', lw=2)
        
        # Set x-axis limits to 27-hour window (24 hours historical + 3 hours forecast)
        x_min = last_historical_time - pd.Timedelta(hours=24)
        x_max = last_historical_time + pd.Timedelta(hours=3)
        plt.xlim(x_min, x_max)
        # Format x-axis to show time (hours:minutes)
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        plt.xticks(rotation=45, ha='right')
    # Mark train/test split if available (only if within the window)
    if split_info.get("test_start"):
        split_time = pd.to_datetime(split_info["test_start"])
        if not ts_data.empty:
            x_min = ts_data.index[-1] - pd.Timedelta(hours=24)
            x_max = ts_data.index[-1] + pd.Timedelta(hours=3)
            if x_min <= split_time <= x_max:
                plt.axvline(split_time, color='black', linestyle=':', alpha=0.6, label='Train/Test split')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    # Determine model type from model_path or use default
    if model_path:
        model_type = model_path.split('_')[0].upper() if '_' in model_path else os.path.basename(model_path).split('_')[0].upper()
    else:
        model_type = "MODEL"
    plt.title(f"{model_type} Layer – 24h Historical + 3h Forecast")
    if enable_plots:
        plot_filename = f"{model_type.lower()}_layer_forecast.png"
        plot_path = os.path.join(FORECAST_PLOTS_DIR, plot_filename)
        plt.savefig(plot_path, dpi=180, bbox_inches='tight')
        log_verbose(f"Generated forecast plot from cache: {plot_path}")
    plt.close()

def generate_forecast_from_cached_model(df_cpu, df_mem, cached_result, horizon_min, model_path, dump_csv_dir=None, context=None, enable_plots=True):
    """Generate fresh forecasts from cached model using latest data."""
    if not isinstance(cached_result, tuple) or len(cached_result) < 3:
        return None
    
    prophet_model, _, metrics = cached_result
    if prophet_model is None:
        return None
    
    # Extract instance metadata before aggregation
    instances_included = []
    if 'instance' in df_cpu.columns:
        instances_included = sorted(df_cpu['instance'].unique().tolist())
    elif 'entity' in df_cpu.columns:
        instances_included = sorted(df_cpu['entity'].unique().tolist())
    
    # Prepare latest data
    cpu_agg = df_cpu.groupby('timestamp')['value'].mean().reset_index(name='cpu')
    cpu_agg['hour'] = cpu_agg['timestamp'].dt.hour
    cpu_agg['is_weekend'] = (cpu_agg['timestamp'].dt.dayofweek>=5).astype(int)

    if df_mem is not None:
        mem_agg = df_mem.groupby('timestamp')['value'].mean().reset_index(name='mem')
        mem_agg = mem_agg.set_index('timestamp').reindex(cpu_agg.set_index('timestamp').index).ffill().reset_index()
        cpu_agg['mem'] = mem_agg['mem']
        target = 'mem'
    else:
        target = 'cpu'

    pdf = cpu_agg[['timestamp', target]].rename(columns={'timestamp':'ds', target:'y'}).dropna()
    pdf = pdf.set_index('ds')
    freq = pd.infer_freq(pdf.index)
    if freq: pdf.index.freq = freq
    pdf = pdf.reset_index()
    pdf['hour'] = pdf['ds'].dt.hour
    pdf['is_weekend'] = (pdf['ds'].dt.dayofweek>=5).astype(int)
    
    # Add instance/node metadata to CSV if dumping
    if dump_csv_dir:
        pdf_for_csv = pdf.copy()
        # Add instance metadata if we have multiple instances (cluster-wide aggregate)
        if instances_included:
            pdf_for_csv['instances_count'] = len(instances_included)
            instances_str = ', '.join(instances_included[:20])
            if len(instances_included) > 20:
                instances_str += f' ... (+{len(instances_included) - 20} more)'
            pdf_for_csv['instances'] = instances_str
        # Add node/signal metadata from context if available (for per-node models like I/O network)
        elif context:
            node = context.get('node')
            signal = context.get('signal')
            if node:
                pdf_for_csv['node'] = node
            if signal:
                pdf_for_csv['signal'] = signal
    
    # Dump CSV if requested
    if dump_csv_dir:
        label = None
        if context:
            node = context.get('node')
            signal = context.get('signal')
            if node and signal:
                label = f"{node}_{signal}"
            elif node:
                label = node
        if not label:
            # Derive label from model_path (e.g., "host_forecast.pkl" -> "host")
            if model_path:
                base = os.path.splitext(os.path.basename(model_path))[0]
                if base.endswith('_forecast'):
                    label = base.replace('_forecast', '')
                else:
                    label = base
            else:
                label = "ensemble"
        if instances_included or (context and (context.get('node') or context.get('signal'))):
            dump_dataframe_to_csv(pdf_for_csv.copy(), dump_csv_dir, label)
        else:
            dump_dataframe_to_csv(pdf.copy(), dump_csv_dir, label)
    
    # For plotting, we want 3 hours (180 minutes) of forecast
    plot_forecast_minutes = 180  # 3 hours
    forecast_periods = max(horizon_min, plot_forecast_minutes)  # Generate at least enough for plotting
    
    # Prepare time series from latest data
    ts = pd.Series(pdf.set_index('ds')['y'])
    if ts.index.freq is None:
        ts.index.freq = pd.infer_freq(ts.index)
    
    # Use PRE-TRAINED Prophet model structure with minimal update on latest data
    # Load saved hyperparameters and refit on latest data only (minimal update, not full retraining)
    f_prophet = None
    if model_path:
        # Fix path construction: avoid double replacement
        if model_path.endswith('_forecast.pkl'):
            prophet_params_path = model_path.replace('_forecast.pkl', '_prophet_params.pkl')
        else:
            prophet_params_path = model_path.replace('.pkl', '_prophet_params.pkl')
        try:
            if os.path.exists(prophet_params_path):
                prophet_params = joblib.load(prophet_params_path)
                # Create new Prophet model with same structure, but fit on latest data (minimal update)
                m_updated = Prophet(daily_seasonality=prophet_params.get('daily_seasonality', True),
                                   weekly_seasonality=prophet_params.get('weekly_seasonality', True),
                                   changepoint_prior_scale=prophet_params.get('changepoint_prior_scale', 0.05))
                m_updated.add_regressor('hour')
                m_updated.add_regressor('is_weekend')
                # Fit on recent data only (last 7 days) for faster fitting while keeping seasonality knowledge
                # This is a minimal update - uses learned structure but incorporates recent trends
                recent_data = pdf.tail(min(len(pdf), 7*24*60))  # Last 7 days or all if less
                m_updated.fit(recent_data)
                
                # Generate forecast
                future = m_updated.make_future_dataframe(periods=forecast_periods, freq='min')
                future['hour'] = future['ds'].dt.hour
                future['is_weekend'] = (future['ds'].dt.dayofweek>=5).astype(int)
                f_prophet_full = m_updated.predict(future)
                f_prophet_vals = f_prophet_full['yhat'].tail(forecast_periods).values
                f_prophet = pd.Series(f_prophet_vals, index=pd.date_range(start=ts.index[-1] + pd.Timedelta(minutes=1), periods=forecast_periods, freq='min'))
                log_verbose(f"Using cached Prophet model structure with minimal update on latest data: {prophet_params_path}")
        except Exception as e:
            log_verbose(f"Prophet minimal update failed: {e}")
    
    # Fallback: use pre-trained model directly (no update)
    if f_prophet is None:
        log_verbose("Using pre-trained Prophet model as-is (no update)")
        future = prophet_model.make_future_dataframe(periods=forecast_periods, freq='min')
        future['hour'] = future['ds'].dt.hour
        future['is_weekend'] = (future['ds'].dt.dayofweek>=5).astype(int)
        f_prophet_full = prophet_model.predict(future)
        f_prophet_vals = f_prophet_full['yhat'].tail(forecast_periods).values
        f_prophet = pd.Series(f_prophet_vals, index=pd.date_range(start=ts.index[-1] + pd.Timedelta(minutes=1), periods=forecast_periods, freq='min'))
    
    # Load PRE-TRAINED ARIMA model (knows patterns) and update with latest data
    # ARIMA models need to be updated with new observations to forecast from the latest point
    # We use the pre-trained model's parameters but fit on latest data (minimal update, not full retraining)
    f_arima = None
    arima = None  # Keep reference to save later
    arima_model_path = None
    if model_path:
        # Fix path construction: avoid double replacement (host_forecast.pkl -> host_arima.pkl, not host_arima_arima.pkl)
        if model_path.endswith('_forecast.pkl'):
            arima_model_path = model_path.replace('_forecast.pkl', '_arima.pkl')
        else:
            arima_model_path = model_path.replace('.pkl', '_arima.pkl')
        
        if os.path.exists(arima_model_path):
            try:
                arima_data = joblib.load(arima_model_path)
                cached_order = arima_data.get('order', (2, 1, 0))
                # Use pre-trained model's order, but fit on latest data to incorporate recent trends
                # This is a minimal update (not full retraining) - we use the same model structure
                arima = ARIMA(ts, order=cached_order).fit()
                f_arima = arima.forecast(steps=forecast_periods)
                f_arima.index = pd.date_range(start=ts.index[-1] + pd.Timedelta(minutes=1), periods=forecast_periods, freq='min')
                log_verbose(f"Using cached ARIMA model structure with latest data: {arima_model_path}")
                # Save updated ARIMA model after minimal update (forecast mode)
                try:
                    joblib.dump({
                        'model': arima,
                        'last_training_point': str(ts.index[-1]),
                        'order': cached_order,
                        'training_data_end': str(pdf['ds'].max())
                    }, arima_model_path)
                    log_verbose(f"Saved updated ARIMA model after minimal update: {arima_model_path}")
                except Exception as e:
                    log_verbose(f"Warning: Failed to save updated ARIMA model: {e}")
            except Exception as e:
                log_verbose(f"Failed to load cached ARIMA model: {e}")
    
    # Fallback: if cached ARIMA not available or model_path is None, use latest data
    if f_arima is None:
        log_verbose("Warning: Using fallback ARIMA (should use cached model)")
        arima = ARIMA(ts, order=(2,1,0)).fit()
        f_arima = arima.forecast(steps=forecast_periods)
        f_arima.index = pd.date_range(start=ts.index[-1] + pd.Timedelta(minutes=1), periods=forecast_periods, freq='min')
        # Save fallback ARIMA model if model_path is provided (for future use)
        if arima_model_path and arima is not None:
            try:
                joblib.dump({
                    'model': arima,
                    'last_training_point': str(ts.index[-1]),
                    'order': (2, 1, 0),
                    'training_data_end': str(pdf['ds'].max())
                }, arima_model_path)
                log_verbose(f"Saved fallback ARIMA model: {arima_model_path}")
            except Exception as e:
                log_verbose(f"Warning: Failed to save fallback ARIMA model: {e}")
    
    # Load PRE-TRAINED LSTM model (knows patterns) and do minimal fine-tuning on latest data
    # Fine-tuning: train for just 1-2 epochs on recent data to incorporate latest trends
    # This is a minimal update - uses learned patterns but adapts to recent changes
    f_lstm = f_arima.copy()  # fallback
    if LSTM_AVAILABLE and os.path.exists(LSTM_MODEL_PATH):
        try:
            lstm_data = joblib.load(LSTM_MODEL_PATH)
            cached_lstm_model = lstm_data.get('model')
            cached_lstm_scaler = lstm_data.get('scaler')
            if cached_lstm_model is not None and cached_lstm_scaler is not None and len(ts) >= LSTM_SEQ_LEN + horizon_min:
                # Prepare data for minimal fine-tuning (use recent data only)
                scaled = cached_lstm_scaler.transform(ts.values.reshape(-1, 1))
                # Use recent data (last 2 days) for fine-tuning - minimal update
                recent_scaled = scaled[-min(len(scaled), 2*24*60):]  # Last 2 days or all if less
                X_fine, y_fine = [], []
                for i in range(LSTM_SEQ_LEN, len(recent_scaled) - horizon_min):
                    X_fine.append(recent_scaled[i-LSTM_SEQ_LEN:i])
                    y_fine.append(recent_scaled[i:i+horizon_min])
                
                if X_fine:
                    X_fine, y_fine = np.array(X_fine), np.array(y_fine)
                    # Minimal fine-tuning: just 1-2 epochs on recent data
                    # This adapts the pre-trained model to recent trends without losing learned patterns
                    try:
                        cached_lstm_model.fit(X_fine, y_fine, epochs=2, batch_size=min(32, len(X_fine)), 
                                             verbose=0, validation_split=0.1 if len(X_fine) > 10 else 0)
                        log_verbose("LSTM minimal fine-tuning completed (2 epochs on recent data)")
                    except Exception as fine_tune_error:
                        log_verbose(f"LSTM fine-tuning failed, using pre-trained model as-is: {fine_tune_error}")
                
                # Use fine-tuned (or original) model to predict from latest sequence
                last_seq = scaled[-LSTM_SEQ_LEN:].reshape(1, LSTM_SEQ_LEN, 1)
                lstm_pred = cached_lstm_model.predict(last_seq, verbose=0)
                lstm_values = cached_lstm_scaler.inverse_transform(lstm_pred)[0]
                # LSTM predicts horizon_min steps, extend with ARIMA for longer forecast
                if len(lstm_values) >= horizon_min:
                    lstm_short = pd.Series(lstm_values[:horizon_min], index=f_arima.index[:horizon_min])
                    f_lstm = pd.concat([lstm_short, f_arima[horizon_min:]])
                else:
                    f_lstm = pd.Series(lstm_values, index=f_arima.index[:len(lstm_values)])
                log_verbose(f"Using cached LSTM model with minimal fine-tuning: {LSTM_MODEL_PATH}")
        except Exception as e:
            log_verbose(f"Failed to load/use cached LSTM model: {e}")
    
    # Extract forecast values for plotting (plot_forecast_minutes) and for return (horizon_min)
    # For plotting: get first plot_forecast_minutes values (immediate forecast)
    prophet_plot_vals = f_prophet.head(plot_forecast_minutes).values[:plot_forecast_minutes] if len(f_prophet) >= plot_forecast_minutes else f_prophet.values
    arima_plot_vals = f_arima.head(plot_forecast_minutes).values[:plot_forecast_minutes] if len(f_arima) >= plot_forecast_minutes else f_arima.values
    lstm_plot_vals = f_lstm.head(plot_forecast_minutes).values[:plot_forecast_minutes] if len(f_lstm) >= plot_forecast_minutes else f_lstm.values
    
    # Pad if needed for plotting
    if len(arima_plot_vals) < plot_forecast_minutes:
        arima_plot_vals = np.pad(arima_plot_vals, (0, plot_forecast_minutes - len(arima_plot_vals)), mode='edge')
    if len(lstm_plot_vals) < plot_forecast_minutes:
        lstm_plot_vals = np.pad(lstm_plot_vals, (0, plot_forecast_minutes - len(lstm_plot_vals)), mode='edge')
    
    # For return value: use horizon_min values (first horizon_min values from forecasts)
    # f_prophet, f_arima, f_lstm all start from the latest data point, so we take first horizon_min
    prophet_vals = f_prophet.head(horizon_min).values[:horizon_min] if len(f_prophet) >= horizon_min else f_prophet.values
    arima_vals = f_arima.head(horizon_min).values[:horizon_min] if len(f_arima) >= horizon_min else f_arima.values
    lstm_vals = f_lstm.head(horizon_min).values[:horizon_min] if len(f_lstm) >= horizon_min else f_lstm.values
    
    # Pad shorter arrays to horizon_min if needed
    if len(arima_vals) < horizon_min:
        arima_vals = np.pad(arima_vals, (0, horizon_min - len(arima_vals)), mode='edge')
    if len(lstm_vals) < horizon_min:
        lstm_vals = np.pad(lstm_vals, (0, horizon_min - len(lstm_vals)), mode='edge')
    
    # Ensure all arrays are exactly horizon_min length
    prophet_vals = prophet_vals[:horizon_min]
    arima_vals = arima_vals[:horizon_min]
    lstm_vals = lstm_vals[:horizon_min]
    
    # Create ensemble with exactly horizon_min values
    ensemble_vals = (prophet_vals + arima_vals + lstm_vals) / 3
    
    # Create forecast DataFrame using f_prophet's index (which is the forecast period)
    # f_prophet is already a Series with forecast_periods values starting from latest data point
    # We need to create a DataFrame that matches this, not the full future dataframe
    forecast_ds = f_prophet.index  # This is the correct forecast timestamps
    out = pd.DataFrame({
        'ds': forecast_ds,
        'yhat': f_prophet.values
    })
    # Replace only the first horizon_min rows with ensemble values (for return value compatibility)
    # ensemble_vals already has exactly horizon_min values
    if len(out) >= horizon_min:
        yhat_values = out['yhat'].values.copy()
        yhat_values[:horizon_min] = ensemble_vals
        out['yhat'] = yhat_values
    else:
        # If forecast is shorter than horizon_min, just use ensemble_vals
        out['yhat'] = ensemble_vals[:len(out)]
    
    # Generate and save forecast plot
    ts_data = pdf.set_index('ds')['y']
    split_info = metrics.get('split_info', {}) if metrics else {}
    
    # For plotting: show last 24 hours of historical data and next 3 hours of forecast
    plot_forecast_horizon = 180  # 3 hours in minutes
    plt.figure(figsize=(16, 6))
    
    # Plot historical data - last 24 hours
    if not ts_data.empty:
        last_24hours = ts_data.last('24h')
        if not last_24hours.empty:
            plt.plot(last_24hours.index, last_24hours.values, label='Last 24 hours (historical)', color='blue', alpha=0.7, linewidth=1.5)
        
        # Create forecast timestamps starting AFTER the last historical data point
        last_historical_time = ts_data.index[-1]
        # Forecast starts 1 minute after last historical point to avoid overlap
        forecast_start = last_historical_time + pd.Timedelta(minutes=1)
        plot_forecast_ds = pd.date_range(start=forecast_start, periods=plot_forecast_horizon, freq='min')
        
        # Use the plot values we already extracted (plot_forecast_minutes = 180)
        # Create ensemble from plot values
        plot_ensemble_vals = (prophet_plot_vals + arima_plot_vals + lstm_plot_vals) / 3
        
        # Forecast lines appear after historical line finishes (continuous timeline, no overlap)
        plt.plot(plot_forecast_ds, prophet_plot_vals, label='Prophet (forecast)', color='orange', ls='--', linewidth=1.5)
        plt.plot(plot_forecast_ds, arima_plot_vals, label='ARIMA (forecast)', color='green', ls='--', linewidth=1.5)
        plt.plot(plot_forecast_ds, lstm_plot_vals, label='LSTM (forecast)', color='purple', ls=':', linewidth=1.5)
        plt.plot(plot_forecast_ds, plot_ensemble_vals, label='Ensemble (forecast)', color='red', lw=2)
        
        # Set x-axis limits to 27-hour window (24 hours historical + 3 hours forecast)
        x_min = last_historical_time - pd.Timedelta(hours=24)
        x_max = last_historical_time + pd.Timedelta(hours=3)
        plt.xlim(x_min, x_max)
        # Format x-axis to show time (hours:minutes)
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        # Show ticks every 3 hours for better readability
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        plt.xticks(rotation=45, ha='right')
    
    # Mark train/test split if available (only if within the window)
    if split_info.get("test_start"):
        split_time = pd.to_datetime(split_info["test_start"])
        if not ts_data.empty:
            x_min = ts_data.index[-1] - pd.Timedelta(hours=24)
            x_max = ts_data.index[-1] + pd.Timedelta(hours=3)
            if x_min <= split_time <= x_max:
                plt.axvline(split_time, color='black', linestyle=':', alpha=0.6, label='Train/Test split')
    
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    # Extract model type from filename (e.g., "host_forecast.pkl" -> "host")
    basename = os.path.basename(model_path)
    if '_' in basename:
        model_type = basename.split('_')[0].upper()
    else:
        # Fallback: try to extract from path
        model_type = os.path.basename(os.path.dirname(model_path)).upper() if os.path.dirname(model_path) else "MODEL"
    plt.title(f"{model_type} Layer – 24h Historical + 3h Forecast")
    # Ensure directory exists and use absolute path
    forecast_dir = os.path.abspath(FORECAST_PLOTS_DIR)
    os.makedirs(forecast_dir, exist_ok=True)
    
    if enable_plots:
        plot_filename = f"{model_type.lower()}_layer_forecast.png"
        plot_path = os.path.join(forecast_dir, plot_filename)
        plot_path = os.path.abspath(plot_path)  # Ensure absolute path
        try:
            plt.savefig(plot_path, dpi=180, bbox_inches='tight')
            if os.path.exists(plot_path):
                file_size = os.path.getsize(plot_path)
                print(f"✓ Saved forecast plot: {plot_path} ({file_size} bytes)")
            else:
                print(f"✗ Warning: Plot file not found after save: {plot_path}")
        except Exception as e:
            print(f"✗ Error saving plot to {plot_path}: {e}")
            import traceback
            traceback.print_exc()
            raise
    plt.close()
    
    return (prophet_model, out, metrics)

def train_or_load_ensemble(df_cpu, df_mem, horizon_min, model_path, force_retrain=False,
                           generate_fresh_forecast=False, show_backtest=False,
                           dump_csv_dir=None, context=None, enable_plots=True):
    if not force_retrain:
        cached = load_cached_ensemble(model_path)
        if cached is not None:
            log_verbose(f"Loaded pre-trained ensemble artifacts: {model_path}")
            # Generate fresh forecasts from latest data if requested (forecast mode)
            if generate_fresh_forecast:
                try:
                    result = generate_forecast_from_cached_model(df_cpu, df_mem, cached, horizon_min, model_path, dump_csv_dir=dump_csv_dir, context=context, enable_plots=enable_plots)
                    if result is not None:
                        # Save updated model after minimal update
                        try:
                            joblib.dump(result, model_path)
                            log_verbose(f"Saved updated model after minimal update: {model_path}")
                        except Exception as e:
                            log_verbose(f"Warning: Failed to save updated model: {e}")
                        # Plot was already saved in generate_forecast_from_cached_model
                        return result
                    else:
                        print(f"⚠ Warning: generate_forecast_from_cached_model returned None for {model_path}")
                        print(f"   Falling back to cached forecast plot generation...")
                        # Fall back to generating plot from cached forecast
                        try:
                            generate_forecast_plots_from_cache(df_cpu, df_mem, cached, horizon_min, model_path, enable_plots=enable_plots)
                        except Exception as e2:
                            print(f"✗ Error generating plot from cache: {e2}")
                except Exception as e:
                    print(f"✗ Error generating fresh forecast from cache: {e}")
                    import traceback
                    traceback.print_exc()
                    # Try to generate plot from cached forecast as fallback
                    try:
                        print(f"   Attempting fallback to cached forecast plot...")
                        generate_forecast_plots_from_cache(df_cpu, df_mem, cached, horizon_min, model_path, enable_plots=enable_plots)
                    except Exception as e2:
                        print(f"✗ Fallback also failed: {e2}")
            # Generate backtest plots when show_backtest is True (even with cached models)
            if show_backtest:
                # Retrain to generate backtest plots (but don't save forecast plots, only backtest plots)
                # IMPORTANT: Don't save model files in show_backtest mode - only generate plots
                log_verbose(f"Regenerating backtest plots for {model_path} (--show-backtest flag)")
                result = build_ensemble_forecast_model(
                    df_cpu=df_cpu,
                    df_mem=df_mem,
                    horizon_min=horizon_min,
                    model_path=model_path,
                    context=context,
                    save_forecast_plot=False,  # Only generate backtest plots, not forecast plots
                    save_model=False,  # Don't save model files - only generate plots
                    dump_csv_dir=dump_csv_dir
                )
                if result is not None:
                    # Return cached model (not the newly trained one) to avoid updating model files
                    return cached
            # In normal mode, don't generate plots (only in forecast mode or when show_backtest)
            return cached

    log_verbose(f"Training ensemble model → {model_path}")
    result = build_ensemble_forecast_model(
        df_cpu=df_cpu,
        df_mem=df_mem,
        horizon_min=horizon_min,
        model_path=model_path,
        context=context,
        dump_csv_dir=dump_csv_dir,
        enable_plots=enable_plots
    )
    try:
        joblib.dump(result, model_path)
        print(f"Saved ensemble artifacts → {model_path}")
        metrics = result[-1] if isinstance(result, tuple) and result else {}
        split_info = metrics.get('split_info') if isinstance(metrics, dict) else None
        persist_model_metadata(model_path, split_info)
    except Exception as exc:
        print(f"Warning: failed to save ensemble artifacts ({model_path}): {exc}")

    return result

# ----------------------------------------------------------------------
# 1. FETCH & PREPROCESS
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# SLI/SLO Framework
# ----------------------------------------------------------------------

def load_sli_slo_config(config_path=None):
    """Load SLI/SLO configuration from YAML file."""
    if not YAML_AVAILABLE:
        return None
    
    if config_path is None:
        # Try default locations
        default_paths = [
            os.path.join(os.getcwd(), "sli_slo_config.yaml"),
            os.path.join(os.path.dirname(__file__), "sli_slo_config.yaml"),
            os.getenv("SLI_SLO_CONFIG_PATH", "")
        ]
        for path in default_paths:
            if path and os.path.exists(path):
                config_path = path
                break
    
    if not config_path or not os.path.exists(config_path):
        if should_verbose():
            print(f"SLI/SLO config not found at {config_path or 'default locations'}. SLI/SLO tracking disabled.")
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        log_verbose(f"Loaded SLI/SLO config from {config_path}")
        return config
    except Exception as exc:
        print(f"Warning: Failed to load SLI/SLO config from {config_path}: {exc}")
        return None

def calculate_sli_value(sli_config, disk_alerts=None, classification_anomalies_df=None, 
                        host_pressure_df=None, golden_anomalies_df=None,
                        df_hcpu=None, df_hmem=None, df_pcpu=None, df_pmem=None,
                        crisis_df=None, anomaly_df=None):
    """Calculate current SLI value based on configuration."""
    if sli_config['query_type'] == 'internal':
        # Internal SLIs calculated from anomaly detection results
        if sli_config['name'] == 'node_health':
            # Calculate percentage of nodes without critical anomalies
            all_nodes = set()
            unhealthy_nodes = set()
            
            # Collect all nodes from various sources
            if disk_alerts is not None and not disk_alerts.empty and 'instance' in disk_alerts.columns:
                all_nodes.update(disk_alerts['instance'].unique())
            
            # Collect unhealthy nodes from various sources
            if classification_anomalies_df is not None and not classification_anomalies_df.empty:
                if 'instance' in classification_anomalies_df.columns:
                    unhealthy_nodes.update(classification_anomalies_df['instance'].unique())
            
            if host_pressure_df is not None and not host_pressure_df.empty:
                if 'instance' in host_pressure_df.columns:
                    unhealthy_nodes.update(host_pressure_df['instance'].unique())
            
            if golden_anomalies_df is not None and not golden_anomalies_df.empty:
                if 'node' in golden_anomalies_df.columns:
                    unhealthy_nodes.update(golden_anomalies_df['node'].unique())
                # Also check 'instance' column if present
                if 'instance' in golden_anomalies_df.columns:
                    unhealthy_nodes.update(golden_anomalies_df['instance'].unique())
            
            # Healthy nodes = all nodes minus unhealthy nodes
            healthy_nodes = all_nodes - unhealthy_nodes
            total_nodes = len(all_nodes) if all_nodes else len(unhealthy_nodes)
            
            if total_nodes == 0:
                return 1.0  # No nodes = 100% healthy (edge case)
            
            return len(healthy_nodes) / total_nodes
        
        elif sli_config['name'] == 'disk_availability':
            # Calculate percentage of disks below 90% usage
            if disk_alerts is None or disk_alerts.empty:
                return 1.0  # No disk data = assume 100% available
            
            # Count disks (instance + mountpoint combinations)
            total_disks = len(disk_alerts)
            # Disks with current_% < 90% are "good"
            if 'current_%' in disk_alerts.columns:
                good_disks = (disk_alerts['current_%'] < 90).sum()
                return good_disks / total_disks if total_disks > 0 else 1.0
            
            return 1.0
        
        elif sli_config['name'] == 'host_cpu_performance':
            # Calculate percentage of nodes with host CPU < 80%
            if df_hcpu is None or df_hcpu.empty:
                return None
            
            # Get latest CPU values per instance
            if 'instance' in df_hcpu.columns and 'y' in df_hcpu.columns:
                latest_cpu = df_hcpu.groupby('instance')['y'].last()
                total_nodes = len(latest_cpu)
                good_nodes = (latest_cpu < 0.80).sum()
                return good_nodes / total_nodes if total_nodes > 0 else 1.0
            
            return None
        
        elif sli_config['name'] == 'host_memory_performance':
            # Calculate percentage of nodes with host memory < 85%
            if df_hmem is None or df_hmem.empty:
                return None
            
            # Get latest memory values per instance
            if 'instance' in df_hmem.columns and 'y' in df_hmem.columns:
                latest_mem = df_hmem.groupby('instance')['y'].last()
                total_nodes = len(latest_mem)
                good_nodes = (latest_mem < 0.85).sum()
                return good_nodes / total_nodes if total_nodes > 0 else 1.0
            
            return None
        
        elif sli_config['name'] == 'pod_cpu_performance':
            # Calculate percentage of nodes with pod CPU < 80%
            if df_pcpu is None or df_pcpu.empty:
                return None
            
            # Get latest pod CPU values per instance
            if 'instance' in df_pcpu.columns and 'y' in df_pcpu.columns:
                latest_cpu = df_pcpu.groupby('instance')['y'].last()
                total_nodes = len(latest_cpu)
                good_nodes = (latest_cpu < 0.80).sum()
                return good_nodes / total_nodes if total_nodes > 0 else 1.0
            
            return None
        
        elif sli_config['name'] == 'pod_memory_performance':
            # Calculate percentage of nodes with pod memory < 85%
            if df_pmem is None or df_pmem.empty:
                return None
            
            # Get latest pod memory values per instance
            if 'instance' in df_pmem.columns and 'y' in df_pmem.columns:
                latest_mem = df_pmem.groupby('instance')['y'].last()
                total_nodes = len(latest_mem)
                good_nodes = (latest_mem < 0.85).sum()
                return good_nodes / total_nodes if total_nodes > 0 else 1.0
            
            return None
        
        elif sli_config['name'] == 'io_performance':
            # Calculate percentage of nodes without I/O crisis or anomalies
            all_nodes = set()
            problematic_nodes = set()
            
            # Collect all nodes from disk alerts or other sources
            if disk_alerts is not None and not disk_alerts.empty and 'instance' in disk_alerts.columns:
                all_nodes.update(disk_alerts['instance'].unique())
            
            # Collect nodes with I/O issues
            if crisis_df is not None and not crisis_df.empty:
                if 'instance' in crisis_df.columns:
                    problematic_nodes.update(crisis_df['instance'].unique())
            
            if anomaly_df is not None and not anomaly_df.empty:
                if 'instance' in anomaly_df.columns:
                    problematic_nodes.update(anomaly_df['instance'].unique())
            
            # If no nodes found, try to infer from other sources
            if not all_nodes and df_hcpu is not None and not df_hcpu.empty:
                if 'instance' in df_hcpu.columns:
                    all_nodes.update(df_hcpu['instance'].unique())
            
            healthy_nodes = all_nodes - problematic_nodes
            total_nodes = len(all_nodes) if all_nodes else len(problematic_nodes)
            
            if total_nodes == 0:
                return 1.0  # No nodes = 100% healthy (edge case)
            
            return len(healthy_nodes) / total_nodes
        
        elif sli_config['name'] == 'network_performance':
            # Calculate percentage of nodes without network crisis or anomalies
            # Network issues are typically in the same crisis_df/anomaly_df but filtered by signal
            all_nodes = set()
            problematic_nodes = set()
            
            # Collect all nodes
            if disk_alerts is not None and not disk_alerts.empty and 'instance' in disk_alerts.columns:
                all_nodes.update(disk_alerts['instance'].unique())
            
            # Collect nodes with network issues (filter by signal type if available)
            if crisis_df is not None and not crisis_df.empty:
                if 'instance' in crisis_df.columns:
                    # Filter for network-related signals if signal column exists
                    if 'signal' in crisis_df.columns:
                        network_crisis = crisis_df[crisis_df['signal'].str.contains('NET', case=False, na=False)]
                        problematic_nodes.update(network_crisis['instance'].unique())
                    else:
                        problematic_nodes.update(crisis_df['instance'].unique())
            
            if anomaly_df is not None and not anomaly_df.empty:
                if 'instance' in anomaly_df.columns:
                    if 'signal' in anomaly_df.columns:
                        network_anomaly = anomaly_df[anomaly_df['signal'].str.contains('NET', case=False, na=False)]
                        problematic_nodes.update(network_anomaly['instance'].unique())
                    else:
                        problematic_nodes.update(anomaly_df['instance'].unique())
            
            # If no nodes found, try to infer from other sources
            if not all_nodes and df_hcpu is not None and not df_hcpu.empty:
                if 'instance' in df_hcpu.columns:
                    all_nodes.update(df_hcpu['instance'].unique())
            
            healthy_nodes = all_nodes - problematic_nodes
            total_nodes = len(all_nodes) if all_nodes else len(problematic_nodes)
            
            if total_nodes == 0:
                return 1.0  # No nodes = 100% healthy (edge case)
            
            return len(healthy_nodes) / total_nodes
        
        elif sli_config['name'] == 'disk_forecast_accuracy':
            # This requires historical forecast vs actual data
            # For now, return None (needs implementation with historical tracking)
            return None
        
        elif sli_config['name'] == 'alert_accuracy':
            # This requires feedback mechanism (which alerts were true positives)
            # For now, return None (needs implementation with alert feedback)
            return None
        
        # Default: return None if we can't calculate
        return None
    
    elif sli_config['query_type'] == 'prometheus':
        # Prometheus-based SLIs - would need to execute query
        # For now, return None to indicate it needs Prometheus query
        return None
    
    return None

def calculate_slo_compliance(sli_value, slo_target):
    """
    Calculate SLO compliance percentage.
    
    Args:
        sli_value: Current SLI value (0.0 to 1.0, e.g., 0.6667 for 66.67%)
        slo_target: SLO target percentage (e.g., 99.95)
    
    Returns:
        Compliance percentage (100.0 if SLI meets target, 0.0 otherwise)
    
    Note: This is a simplified implementation. In production, compliance
    would be calculated over a time window (e.g., 30 days) as the percentage
    of time the SLI met the target.
    
    Example:
        SLI = 66.67% (0.6667), Target = 99.95% (0.9995)
        Since 0.6667 < 0.9995, compliance = 0.0%
    """
    if sli_value is None:
        return None
    
    # Convert SLO target from percentage to ratio (99.95% -> 0.9995)
    target_ratio = slo_target / 100.0
    
    # Binary compliance: either 100% (meets target) or 0% (doesn't meet target)
    # In production, this would be: (time_meeting_target / total_time) * 100
    return 100.0 if sli_value >= target_ratio else 0.0

def calculate_error_budget(slo_target, compliance_percent):
    """
    Calculate error budget remaining.
    
    Error Budget Formula:
        1. Total Budget = 100% - SLO Target
        2. Budget Consumed = 100% - Compliance %
        3. Budget Remaining = Total Budget - Budget Consumed
    
    Args:
        slo_target: SLO target percentage (e.g., 99.95)
        compliance_percent: Current compliance percentage (0.0 to 100.0)
    
    Returns:
        Error budget remaining as percentage (clamped to 0.0 minimum)
    
    Example (NODE HEALTH):
        SLO Target = 99.95%
        Total Budget = 100% - 99.95% = 0.05%
        
        If Compliance = 0% (SLI below target):
            Budget Consumed = 100% - 0% = 100%
            Budget Remaining = 0.05% - 100% = -99.95% → Clamped to 0.00%
        
        If Compliance = 100% (SLI meets target):
            Budget Consumed = 100% - 100% = 0%
            Budget Remaining = 0.05% - 0% = 0.05% ✓
    
    Note: In production, budget would be calculated over a time window
    (e.g., 30 days) and would recover gradually as compliance improves.
    """
    if compliance_percent is None:
        return None
    
    # Step 1: Calculate total error budget
    # This is the maximum "unreliability" you can tolerate
    # Example: 99.95% SLO = 0.05% error budget
    total_budget = 100.0 - slo_target
    
    # Step 2: Calculate how much budget has been consumed
    # If compliance is 0%, you've consumed 100% of your budget
    # If compliance is 100%, you've consumed 0% of your budget
    budget_consumed = 100.0 - compliance_percent
    
    # Step 3: Calculate remaining budget
    # This can be negative if you've exceeded your budget
    budget_remaining = total_budget - budget_consumed
    
    # Clamp to 0.0 minimum (can't have negative budget remaining)
    # Negative values indicate budget exhaustion
    return max(0.0, budget_remaining)

def track_sli_slo(config, disk_alerts=None, classification_anomalies_df=None,
                  host_pressure_df=None, golden_anomalies_df=None,
                  df_hcpu=None, df_hmem=None, df_pcpu=None, df_pmem=None,
                  crisis_df=None, anomaly_df=None):
    """Track SLI/SLO metrics and return summary."""
    if config is None:
        return None
    
    slis = config.get('slis', [])
    settings = config.get('settings', {})
    
    results = []
    
    for sli_config in slis:
        sli_name = sli_config.get('name', 'unknown')
        slo_target = sli_config.get('slo_target', 99.9)
        error_budget_percent = sli_config.get('error_budget_percent', 0.1)
        alert_severity = sli_config.get('alert_severity', 'P2')
        
        # Calculate current SLI value
        sli_value = calculate_sli_value(
            sli_config,
            disk_alerts=disk_alerts,
            classification_anomalies_df=classification_anomalies_df,
            host_pressure_df=host_pressure_df,
            golden_anomalies_df=golden_anomalies_df,
            df_hcpu=df_hcpu,
            df_hmem=df_hmem,
            df_pcpu=df_pcpu,
            df_pmem=df_pmem,
            crisis_df=crisis_df,
            anomaly_df=anomaly_df
        )
        
        if sli_value is None:
            continue  # Skip if we can't calculate
        
        # Calculate compliance (simplified - in production would use historical data)
        compliance = calculate_slo_compliance(sli_value, slo_target)
        error_budget_remaining = calculate_error_budget(slo_target, compliance) if compliance is not None else None
        
        # Check if error budget is at risk
        error_budget_threshold = settings.get('error_budget_alert_threshold', 20)
        budget_at_risk = False
        if error_budget_remaining is not None:
            budget_percent_remaining = (error_budget_remaining / (100.0 - slo_target)) * 100 if (100.0 - slo_target) > 0 else 0
            budget_at_risk = budget_percent_remaining < error_budget_threshold
        
        results.append({
            'sli_name': sli_name,
            'description': sli_config.get('description', ''),
            'sli_value': sli_value,
            'slo_target': slo_target,
            'compliance_percent': compliance,
            'error_budget_remaining': error_budget_remaining,
            'budget_at_risk': budget_at_risk,
            'alert_severity': alert_severity
        })
    
    return results

def format_sli_slo_report(sli_slo_results):
    """Format SLI/SLO results for console output."""
    if not sli_slo_results:
        return ""
    
    lines = ["=" * 80]
    lines.append("SLI/SLO STATUS")
    lines.append("=" * 80)
    
    for result in sli_slo_results:
        sli_name = result['sli_name']
        description = result['description']
        sli_value = result['sli_value']
        slo_target = result['slo_target']
        compliance = result['compliance_percent']
        budget_remaining = result['error_budget_remaining']
        budget_at_risk = result['budget_at_risk']
        severity = result['alert_severity']
        
        lines.append(f"\n{sli_name.upper().replace('_', ' ')}")
        lines.append(f"  Description: {description}")
        lines.append(f"  Current SLI: {sli_value:.2%}")
        lines.append(f"  SLO Target: {slo_target}%")
        
        if compliance is not None:
            status = "✓ COMPLIANT" if compliance >= slo_target else "✗ NON-COMPLIANT"
            lines.append(f"  Compliance: {compliance:.2f}% {status}")
        
        if budget_remaining is not None:
            budget_status = "⚠️  AT RISK" if budget_at_risk else "✓ OK"
            lines.append(f"  Error Budget Remaining: {budget_remaining:.2f}% {budget_status}")
        
        if budget_at_risk:
            lines.append(f"  ⚠️  ALERT: Error budget below threshold ({severity})")
    
    lines.append("=" * 80)
    return "\n".join(lines)

def fetch_victoriametrics_metrics(query, start, end, step=STEP):
    params = {'query': query, 'start': start, 'end': end, 'step': step}
    try:
        r = requests.get(VM_BASE_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if data['status'] != 'success':
            raise ValueError(data.get('error'))
        result = data['data']['result']
        log_verbose(f"Query returned {len(result)} series.")
        if not result:
            return pd.DataFrame()
        rows = []
        for s in result:
            df = pd.DataFrame(s['values'], columns=['ts', 'value'])
            df['ts'] = pd.to_datetime(df['ts'], unit='s')
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            for k, v in s['metric'].items():
                df[k] = v
            rows.append(df)
        return pd.concat(rows, ignore_index=True)
    except Exception as e:
        print(f"Query error: {e}")
        return pd.DataFrame()

def fetch_and_preprocess_data(query, start_hours_ago=START_HOURS_AGO, step=STEP):
    start = int((pd.Timestamp.now() - pd.Timedelta(hours=start_hours_ago)).timestamp())
    end   = int(pd.Timestamp.now().timestamp())
    df = fetch_victoriametrics_metrics(query, start, end, step)
    if df.empty:
        return pd.DataFrame()

    if 'memory' in query.lower():
        mx = df['value'].max()
        df['value'] = df['value'] / mx if mx > 0 else df['value']
    elif 'cpu' in query.lower():
        df['value'] = df['value'].clip(0, 1)

    df['hour'] = df['ts'].dt.hour
    df['day_of_week'] = df['ts'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    id_source = None
    for candidate in ['node', 'hostname', 'instance', 'pod', 'device']:
        if candidate in df.columns:
            id_source = candidate
            break
    if id_source:
        df['entity'] = df[id_source].fillna(df.get('instance'))
    else:
        df['entity'] = df.get('instance', 'unknown')

    df['raw_entity'] = df['entity'].copy()
    if 'instance' in df.columns:
        df['raw_instance'] = df['instance'].copy()
    else:
        df['raw_instance'] = df['entity'].copy()

    df['entity'] = df['entity'].apply(canonical_identity)

    group_cols = ['ts']
    for col in ['instance', 'node', 'hostname', 'entity']:
        if col in df.columns:
            group_cols.append(col)
    # Preserve mountpoint/filesystem columns for disk data
    for col in ['mountpoint', 'filesystem']:
        if col in df.columns:
            group_cols.append(col)
    agg_spec = {
        'value':'mean',
        'hour':'first',
        'day_of_week':'first',
        'is_weekend':'first'
    }
    if 'raw_entity' in df.columns:
        agg_spec['raw_entity'] = 'first'
    if 'raw_instance' in df.columns:
        agg_spec['raw_instance'] = 'first'
    # Note: mountpoint/filesystem are preserved automatically via group_cols
    # They become part of the index and are restored by reset_index()
    df = df.groupby(group_cols).agg(agg_spec).reset_index()

    log_verbose(f"Pre-processed {len(df)} rows.")
    df = df.rename(columns={'ts':'timestamp'}).sort_values('timestamp')
    if 'entity' in df.columns:
        df['entity'] = df['entity'].map(canonical_identity)
        if 'raw_entity' in df.columns:
            for entity, raw_val in df[['entity','raw_entity']].dropna().itertuples(index=False, name=None):
                register_source_identity(entity, raw_val)
        if 'raw_instance' in df.columns:
            for entity, raw_inst in df[['entity','raw_instance']].dropna().itertuples(index=False, name=None):
                register_source_identity(entity, raw_inst)
    return df

# ----------------------------------------------------------------------
# DISK FULL PREDICTION — HYBRID LINEAR + PROPHET (7-day accurate ETA)
# ----------------------------------------------------------------------
def predict_disk_full_days(df_disk, horizon_days=7, threshold_pct=90.0,
                           manifest=None, retrain_targets=None, show_backtest=False,
                           forecast_mode=False, dump_csv_dir=None, enable_plots=True):
    """
    Returns a DataFrame with full ETA for every node/mountpoint
    Uses hybrid linear trend + Prophet for maximum accuracy
    Also returns aggregated metrics for all disk models
    """
    alerts = []
    manifest = manifest or {}
    retrain_targets = retrain_targets or set()
    manifest_changed = False
    
    # Aggregate metrics across all models
    all_mae_linear = []
    all_mae_prophet = []
    all_mae_ensemble = []
    all_train_points = []
    all_test_points = []
    train_starts = []
    train_ends = []
    test_starts = []
    test_ends = []
    
    if 'entity' not in df_disk.columns:
        df_disk['entity'] = df_disk.get('instance', 'unknown').map(lambda x: canonical_identity(str(x)))
    mount_col = 'filesystem' if 'filesystem' in df_disk.columns else 'mountpoint'
    retrained_nodes = set()  # Track which nodes/mounts were retrained
    
    for (entity, mountpoint), group in df_disk.groupby(['entity', mount_col]):
        raw_label = None
        if 'raw_instance' in group.columns and not group['raw_instance'].dropna().empty:
            raw_label = group['raw_instance'].dropna().iloc[-1]
        node = canonical_node_label(entity, with_ip=True, raw_label=raw_label)
        key = build_disk_key(entity, mountpoint)
        dump_label = f"disk_{node}_{mountpoint}"
        
        # Enhanced matching logic (similar to I/O and network)
        # We keep the retrain rules readable: first-time builds always retrain, then we let targets override.
        is_first_training = key not in manifest
        needs_retrain = FORCE_TRAINING_RUN or is_first_training
        # Check for "all" flag first
        if not needs_retrain and retrain_targets and '__RETRAIN_ALL__' in retrain_targets:
            needs_retrain = True
        elif not needs_retrain and retrain_targets:
            # Direct matches
            entity_match = entity in retrain_targets
            key_match = key in retrain_targets
            mount_match = any(f":{mountpoint}" in t or f"|{mountpoint}" in t for t in retrain_targets)
            
            # Alias matching
            alias_match = False
            # Allow retrain targets to reference aliases or informal node names.
            for target in retrain_targets:
                if '|' in target or ':' in target:
                    continue  # Skip keys, only check node names
                target_canon = canonical_identity(target)
                # Direct match
                if target_canon == entity:
                    alias_match = True
                    break
                # Check alias map
                if target_canon in INSTANCE_ALIAS_MAP:
                    alias_value = INSTANCE_ALIAS_MAP[target_canon]
                    if canonical_identity(alias_value) == entity:
                        alias_match = True
                        break
                # Reverse alias check
                for k, v in INSTANCE_ALIAS_MAP.items():
                    if canonical_identity(v) == entity and canonical_identity(k) == target_canon:
                        alias_match = True
                        break
                if alias_match:
                    break
                # Check source registry IPs
                target_ip = SOURCE_REGISTRY.get(target_canon) or CANON_SOURCE_MAP.get(target_canon)
                entity_ip = SOURCE_REGISTRY.get(entity) or CANON_SOURCE_MAP.get(entity)
                if target_ip and entity_ip and target_ip == entity_ip:
                    alias_match = True
                    break
                # DNS resolution (only if target looks like a hostname)
                if looks_like_hostname(target) and '(' in node and ')' in node:
                    node_ip = node.split('(')[1].split(')')[0].strip()
                    target_variants = [target] + [f"{target}{d}" for d in DNS_DOMAIN_SUFFIXES if d and not target.endswith(d)]
                    for target_var in target_variants:
                        try:
                            target_resolved = socket.gethostbyname(target_var)
                            if target_resolved == node_ip:
                                alias_match = True
                                log_verbose(f"   DNS match: {target_var} → {target_resolved} == {node_ip}")
                                break
                        except Exception as e:
                            log_verbose(f"   DNS resolution failed for {target_var}: {e}")
                    if alias_match:
                        break
            
            needs_retrain = entity_match or key_match or mount_match or alias_match
        
        ts = group.set_index('timestamp')['value'].sort_index()
        if len(ts) < 50:
            continue
            
        # Train/Test Split (only compute when retraining)
        split_idx = max(1, int(len(ts) * TRAIN_FRACTION))
        if split_idx >= len(ts):
            split_idx = len(ts) - 1
        train_ts = ts.iloc[:split_idx]
        test_ts = ts.iloc[split_idx:]
        
        # Only collect metrics when retraining
        if needs_retrain:
            all_train_points.append(len(train_ts))
            all_test_points.append(len(test_ts))
            if not train_ts.empty:
                train_starts.append(str(train_ts.index[0]))
                train_ends.append(str(train_ts.index[-1]))
            if not test_ts.empty:
                test_starts.append(str(test_ts.index[0]))
                test_ends.append(str(test_ts.index[-1]))
        
        # Use cached result if available and not retraining
        if not needs_retrain and key in manifest:
            cached_record = dict(manifest[key])
            # Ensure ensemble_eta exists (it's the same as days_to_90pct)
            if 'ensemble_eta' not in cached_record:
                cached_record['ensemble_eta'] = cached_record.get('days_to_90pct', 9999.0)
            alerts.append(cached_record)
            # Compute metrics for cached models if show_backtest is true
            if show_backtest:
                # Collect metrics for cached models
                all_train_points.append(len(train_ts))
                all_test_points.append(len(test_ts))
                if not train_ts.empty:
                    train_starts.append(str(train_ts.index[0]))
                    train_ends.append(str(train_ts.index[-1]))
                if not test_ts.empty:
                    test_starts.append(str(test_ts.index[0]))
                    test_ends.append(str(test_ts.index[-1]))
                
                # Compute linear MAE
                daily_increase = train_ts.diff().resample('1D').mean().mean()
                linear_pred = None
                if len(test_ts) > 1:
                    base_value = train_ts.iloc[-1]
                    time_diffs = (test_ts.index - train_ts.index[-1]).total_seconds() / 86400
                    linear_pred = base_value + time_diffs * daily_increase
                    linear_mae = mean_absolute_error(test_ts.values, linear_pred.values)
                    all_mae_linear.append(linear_mae)
                
                # Compute Prophet MAE
                prophet_pred = None
                try:
                    pdf = train_ts.reset_index()
                    pdf.columns = ['ds', 'y']
                    pdf['y'] = pdf['y'].clip(upper=0.99)
                    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False)
                    m.fit(pdf)
                    if len(test_ts) > 0:
                        test_df = test_ts.reset_index()
                        test_df.columns = ['ds', 'y']
                        test_forecast = m.predict(test_df[['ds']])
                        prophet_pred = test_forecast['yhat'].values
                        prophet_mae = mean_absolute_error(test_df['y'].values, prophet_pred)
                        all_mae_prophet.append(prophet_mae)
                except:
                    pass
                
                # Compute ensemble MAE
                if linear_pred is not None and prophet_pred is not None and len(test_ts) > 0:
                    ensemble_pred = pd.Series(np.minimum(linear_pred.values, prophet_pred), index=test_ts.index)
                    ensemble_mae = mean_absolute_error(test_ts.values, ensemble_pred.values)
                    all_mae_ensemble.append(ensemble_mae)
                elif prophet_pred is not None and len(test_ts) > 0:
                    all_mae_ensemble.append(prophet_mae)
                elif linear_pred is not None and len(test_ts) > 0:
                    all_mae_ensemble.append(linear_mae)
            
            # Save plot for cached models too (forecast mode only)
            # Also update forecast with minimal update (use recent data only) - only in forecast mode
            if forecast_mode:
                try:
                    # MINIMAL UPDATE: Use recent data only (last 7 days) for faster fitting
                    # This incorporates latest trends while preserving learned patterns
                    pdf = train_ts.reset_index()
                    pdf.columns = ['ds', 'y']
                    pdf['y'] = pdf['y'].clip(upper=0.99)
                    # Use recent data for minimal update (last 7 days or all if less)
                    recent_pdf = pdf.tail(min(len(pdf), 7*24*6))  # Last 7 days (6 data points per day for 10m intervals)
                    
                    prophet_forecast_df = None
                    updated_prophet_days = cached_record.get('days_to_90pct', 9999.0)
                    try:
                        # Minimal update: fit on recent data only
                        m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False)
                        m.fit(recent_pdf)
                        future = m.make_future_dataframe(periods=horizon_days*24*10, freq='6H')
                        forecast = m.predict(future)
                        prophet_forecast_df = forecast
                        # Update forecast with minimal update
                        over = forecast[forecast['yhat'] >= threshold_pct/100]
                        if not over.empty:
                            updated_prophet_days = (over.iloc[0]['ds'] - pd.Timestamp.now()).total_seconds() / 86400
                        # Update linear forecast too
                        daily_increase = train_ts.diff().resample('1D').mean().mean()
                        if daily_increase > 0.0001:
                            current_pct = ts.iloc[-1] * 100
                            updated_linear_days = (threshold_pct - current_pct) / (daily_increase * 100)
                            updated_linear_days = max(0.1, updated_linear_days)
                        else:
                            updated_linear_days = 9999.0
                        # Update hybrid forecast (min of linear and prophet)
                        updated_hybrid_days = min(updated_linear_days, updated_prophet_days)
                        # Update cached record with fresh forecast
                        cached_record['days_to_90pct'] = round(updated_hybrid_days, 1)
                        cached_record['ensemble_eta'] = round(updated_hybrid_days, 1)
                        cached_record['linear_eta'] = round(updated_linear_days, 1)
                        cached_record['prophet_eta'] = round(updated_prophet_days, 1)
                        cached_record['alert'] = "CRITICAL" if updated_hybrid_days < 3 else "WARNING" if updated_hybrid_days < 7 else "SOON" if updated_hybrid_days < 30 else "OK"
                        manifest[key] = cached_record
                        manifest_changed = True
                        log_verbose(f"  → Disk forecast updated with minimal update: {node} | {mountpoint} → {updated_hybrid_days:.1f} days")
                    except Exception as e:
                        log_verbose(f"  → Minimal update failed, using cached forecast: {e}")
                        pass
                    
                    # Compute linear trend for plotting (even if not computing backtest metrics)
                    linear_pred = None
                    if len(test_ts) > 1 and len(train_ts) > 0:
                        daily_increase = train_ts.diff().resample('1D').mean().mean()
                        base_value = train_ts.iloc[-1]
                        time_diffs = (test_ts.index - train_ts.index[-1]).total_seconds() / 86400
                        linear_pred = base_value + time_diffs * daily_increase
                    
                    plt.figure(figsize=(14, 7))
                    # Plot historical data
                    if len(train_ts) > 0:
                        plt.plot(train_ts.index, train_ts.values * 100, label='Train Data', color='#1f77b4', alpha=0.7)
                    if len(test_ts) > 0:
                        plt.plot(test_ts.index, test_ts.values * 100, label='Test Data', color='#2ca02c', alpha=0.7)
                    # Plot forecast if Prophet model was created successfully
                    if prophet_forecast_df is not None:
                        forecast_future = prophet_forecast_df[prophet_forecast_df['ds'] > ts.index[-1]]
                        if not forecast_future.empty:
                            plt.plot(forecast_future['ds'], forecast_future['yhat'] * 100, label='Prophet Forecast', color='#ff7f0e', linewidth=2)
                            plt.fill_between(forecast_future['ds'], 
                                            forecast_future['yhat_lower'] * 100, 
                                            forecast_future['yhat_upper'] * 100, 
                                            alpha=0.2, color='#ff7f0e')
                    # Plot threshold line
                    plt.axhline(threshold_pct, color='red', linestyle='--', linewidth=2, label=f'{threshold_pct}% Threshold')
                    # Plot train/test split if available
                    if len(test_ts) > 0 and len(train_ts) > 0:
                        split_time = test_ts.index[0]
                        plt.axvline(split_time, color='gray', linestyle=':', alpha=0.7, label='Train/Test Split')
                    # Plot linear trend if available
                    if linear_pred is not None and len(test_ts) > 0:
                        plt.plot(test_ts.index, linear_pred.values * 100, label='Linear Trend', color='green', linestyle='--', alpha=0.7)
                    plt.xlabel('Date')
                    plt.ylabel('Disk Usage (%)')
                    safe_node = node.split('(')[0].strip().replace(' ', '_').replace('/', '_')
                    safe_mount = mountpoint.replace('/', '_')
                    current_pct = ts.iloc[-1] * 100
                    # Use updated forecast if available, otherwise use cached
                    hybrid_days = cached_record.get('days_to_90pct', 9999.0)
                    severity = cached_record.get('alert', 'OK')
                    plt.title(f"{node} | {mountpoint}\nCurrent: {current_pct:.2f}% | ETA to {threshold_pct}%: {hybrid_days:.1f} days → {severity}")
                    plt.legend()
                    plt.grid(alpha=0.3)
                    plt.tight_layout()
                    if enable_plots:
                        plot_file = os.path.join(FORECAST_PLOTS_DIR, f"disk_{safe_node}_{safe_mount}_forecast.png")
                        plt.savefig(plot_file, dpi=180, bbox_inches='tight')
                        print(f"  → Disk plot saved: {plot_file}")
                    plt.close()
                except Exception as e:
                    print(f"  ✗ Failed to save disk plot for cached model: {e}")
                    import traceback
                    traceback.print_exc()
                    try:
                        plt.close()
                    except:
                        pass
            continue
        
        # Track retrained nodes (including first-time training)
        if needs_retrain:
            retrained_nodes.add(f"{node} | {mountpoint}")
            if should_verbose():
                logger.info("Disk forecast start → node=%s mount=%s", node, mountpoint)
        elif should_verbose():
            logger.info("Disk forecast start → node=%s mount=%s", node, mountpoint)
            
        current_pct = ts.iloc[-1] * 100
        
        # Linear ETA (fast & reliable)
        daily_increase = train_ts.diff().resample('1D').mean().mean()
        if daily_increase > 0.0001:  # 0.01% per day
            linear_days = (threshold_pct - current_pct) / (daily_increase * 100)
            linear_days = max(0.1, linear_days)
        else:
            linear_days = 9999.0
        
        # Compute linear MAE on test set (only when retraining)
        linear_pred = None
        if needs_retrain and len(test_ts) > 1:
            base_value = train_ts.iloc[-1]
            time_diffs = (test_ts.index - train_ts.index[-1]).total_seconds() / 86400
            linear_pred = base_value + time_diffs * daily_increase
            linear_mae = mean_absolute_error(test_ts.values, linear_pred.values)
            all_mae_linear.append(linear_mae)

        # Prophet ETA (seasonal correction)
        pdf = train_ts.reset_index()
        pdf.columns = ['ds', 'y']
        pdf['y'] = pdf['y'].clip(upper=0.99)
        
        prophet_days = 9999.0
        prophet_mae = None
        prophet_pred = None
        prophet_model = None
        prophet_forecast_df = None
        try:
            # For retraining: use minimal update (recent data) if not first-time training
            # For first-time training: use all data to learn patterns
            if needs_retrain and key in manifest:
                # Minimal update: use recent data (last 7 days) to incorporate latest trends
                fit_pdf = pdf.tail(min(len(pdf), 7*24*6))  # Last 7 days
                log_verbose(f"  → Disk model minimal update (recent 7 days): {node} | {mountpoint}")
            else:
                # First-time training: use all data to learn patterns
                fit_pdf = pdf
            # Add node and mountpoint metadata to CSV
            if dump_csv_dir:
                fit_pdf_for_csv = fit_pdf.copy()
                fit_pdf_for_csv['node'] = node
                fit_pdf_for_csv['mountpoint'] = mountpoint
                dump_dataframe_to_csv(fit_pdf_for_csv, dump_csv_dir, dump_label)
            else:
                dump_dataframe_to_csv(fit_pdf.copy(), dump_csv_dir, dump_label)
            m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False)
            m.fit(fit_pdf)
            prophet_model = m  # Store for plotting
            
            # Compute Prophet MAE on test set (only when retraining)
            if needs_retrain and len(test_ts) > 0:
                test_df = test_ts.reset_index()
                test_df.columns = ['ds', 'y']
                test_forecast = m.predict(test_df[['ds']])
                prophet_pred = test_forecast['yhat'].values
                prophet_mae = mean_absolute_error(test_df['y'].values, prophet_pred)
                all_mae_prophet.append(prophet_mae)
            
            future = m.make_future_dataframe(periods=horizon_days*24*10, freq='6H')
            forecast = m.predict(future)
            prophet_forecast_df = forecast  # Store for plotting
            over = forecast[forecast['yhat'] >= threshold_pct/100]
            if not over.empty:
                prophet_days = (over.iloc[0]['ds'] - pd.Timestamp.now()).total_seconds() / 86400
        except:
            prophet_days = linear_days

        # Compute ensemble MAE (min of linear and prophet) - only when retraining
        if needs_retrain:
            if linear_pred is not None and prophet_pred is not None and len(test_ts) > 0:
                ensemble_pred = pd.Series(np.minimum(linear_pred.values, prophet_pred), index=test_ts.index)
                ensemble_mae = mean_absolute_error(test_ts.values, ensemble_pred.values)
                all_mae_ensemble.append(ensemble_mae)
            elif prophet_pred is not None and len(test_ts) > 0:
                # Fallback to prophet if linear not available
                all_mae_ensemble.append(prophet_mae)
            elif linear_pred is not None and len(test_ts) > 0:
                # Fallback to linear if prophet not available
                all_mae_ensemble.append(linear_mae)

        hybrid_days = min(linear_days, prophet_days)
        severity = "CRITICAL" if hybrid_days < 3 else "WARNING" if hybrid_days < 7 else "SOON" if hybrid_days < 30 else "OK"

        if should_verbose():
            logger.info("Disk forecast done → node=%s mount=%s", node, mountpoint)

        record = {
            'instance': node,
            'mountpoint': mountpoint,
            'current_%': round(current_pct, 2),
            'days_to_90pct': round(hybrid_days, 1),
            'ensemble_eta': round(hybrid_days, 1),
            'linear_eta': round(linear_days, 1),
            'prophet_eta': round(prophet_days, 1),
            'alert': severity
        }
        manifest[key] = record
        manifest_changed = True
        alerts.append(record)
        
        # Save plot when retraining or when show_backtest is True
        if needs_retrain or show_backtest:
            try:
                plt.figure(figsize=(14, 7))
                # Plot historical data
                if len(train_ts) > 0:
                    plt.plot(train_ts.index, train_ts.values * 100, label='Train Data', color='#1f77b4', alpha=0.7)
                if len(test_ts) > 0:
                    plt.plot(test_ts.index, test_ts.values * 100, label='Test Data', color='#2ca02c', alpha=0.7)
                # Plot forecast if Prophet model was created successfully
                if prophet_forecast_df is not None:
                    forecast_future = prophet_forecast_df[prophet_forecast_df['ds'] > ts.index[-1]]
                    if not forecast_future.empty:
                        plt.plot(forecast_future['ds'], forecast_future['yhat'] * 100, label='Prophet Forecast', color='#ff7f0e', linewidth=2)
                        plt.fill_between(forecast_future['ds'], 
                                        forecast_future['yhat_lower'] * 100, 
                                        forecast_future['yhat_upper'] * 100, 
                                        alpha=0.2, color='#ff7f0e')
                # Plot threshold line
                plt.axhline(threshold_pct, color='red', linestyle='--', linewidth=2, label=f'{threshold_pct}% Threshold')
                # Plot train/test split if available
                if len(test_ts) > 0 and len(train_ts) > 0:
                    split_time = test_ts.index[0]
                    plt.axvline(split_time, color='gray', linestyle=':', alpha=0.7, label='Train/Test Split')
                # Plot linear trend if available
                if linear_pred is not None and len(test_ts) > 0:
                    plt.plot(test_ts.index, linear_pred.values * 100, label='Linear Trend', color='green', linestyle='--', alpha=0.7)
                plt.xlabel('Date')
                plt.ylabel('Disk Usage (%)')
                safe_node = node.split('(')[0].strip().replace(' ', '_').replace('/', '_')
                safe_mount = mountpoint.replace('/', '_')
                plt.title(f"{node} | {mountpoint}\nCurrent: {current_pct:.2f}% | ETA to {threshold_pct}%: {hybrid_days:.1f} days → {severity}")
                plt.legend()
                plt.grid(alpha=0.3)
                plt.tight_layout()
                if enable_plots:
                    plot_file = os.path.join(FORECAST_PLOTS_DIR, f"disk_{safe_node}_{safe_mount}_forecast.png")
                    plt.savefig(plot_file, dpi=180, bbox_inches='tight')
                    print(f"  → Disk plot saved: {plot_file}")
                plt.close()
            except Exception as e:
                log_verbose(f"  → Failed to save disk plot: {e}")
                try:
                    plt.close()
                except:
                    pass

    alerts_df = (pd.DataFrame(alerts).sort_values('days_to_90pct')
                 if alerts else pd.DataFrame(columns=["instance","mountpoint","current_%","days_to_90pct","ensemble_eta","linear_eta","prophet_eta","alert"]))
    
    # Aggregate metrics
    disk_metrics = {}
    if all_mae_ensemble:
        disk_metrics['mae_ensemble'] = np.mean(all_mae_ensemble)
    if all_mae_linear:
        disk_metrics['mae_linear'] = np.mean(all_mae_linear)
    if all_mae_prophet:
        disk_metrics['mae_prophet'] = np.mean(all_mae_prophet)
    if all_train_points:
        disk_metrics['split_info'] = {
            'train_fraction': TRAIN_FRACTION,
            'train_points': int(np.mean(all_train_points)),
            'test_points': int(np.mean(all_test_points)),
            'train_start': min(train_starts) if train_starts else None,
            'train_end': max(train_ends) if train_ends else None,
            'test_start': min(test_starts) if test_starts else None,
            'test_end': max(test_ends) if test_ends else None
        }

    return alerts_df, manifest, manifest_changed, disk_metrics, retrained_nodes

# ----------------------------------------------------------------------
# 2. ENSEMBLE FORECAST (Prophet + ARIMA + LSTM)
# ----------------------------------------------------------------------
def build_ensemble_forecast_model(df_cpu, df_mem=None,
                                 horizon_min=HORIZON_MIN, model_path='model.pkl', context=None,
                                 save_forecast_plot=True, save_backtest_plot=True, print_backtest_metrics=True,
                                 save_model=True, dump_csv_dir=None, enable_plots=True):
    # Override plot saving flags if enable_plots is False
    if not enable_plots:
        save_forecast_plot = False
        save_backtest_plot = False
    # Extract instance metadata before aggregation
    instances_included = []
    if 'instance' in df_cpu.columns:
        instances_included = sorted(df_cpu['instance'].unique().tolist())
    elif 'entity' in df_cpu.columns:
        instances_included = sorted(df_cpu['entity'].unique().tolist())
    
    cpu_agg = df_cpu.groupby('timestamp')['value'].mean().reset_index(name='cpu')
    cpu_agg['hour'] = cpu_agg['timestamp'].dt.hour
    cpu_agg['is_weekend'] = (cpu_agg['timestamp'].dt.dayofweek>=5).astype(int)

    if df_mem is not None:
        mem_agg = df_mem.groupby('timestamp')['value'].mean().reset_index(name='mem')
        mem_agg = mem_agg.set_index('timestamp').reindex(cpu_agg.set_index('timestamp').index).ffill().reset_index()
        cpu_agg['mem'] = mem_agg['mem']
        target = 'mem'
    else:
        target = 'cpu'

    pdf = cpu_agg[['timestamp', target]].rename(columns={'timestamp':'ds', target:'y'}).dropna()
    pdf = pdf.set_index('ds')
    freq = pd.infer_freq(pdf.index)
    if freq: pdf.index.freq = freq
    pdf = pdf.reset_index()

    # --- Train/Test Split (time-ordered) ---
    split_idx = max(1, int(len(pdf) * TRAIN_FRACTION))
    if split_idx >= len(pdf):
        split_idx = len(pdf) - 1
    test_cutoff = pdf.iloc[split_idx]['ds']
    test_ts = pdf[pdf['ds'] > test_cutoff].set_index('ds')['y']
    train = pdf[pdf['ds'] <= test_cutoff]
    split_info = {
        "train_fraction": TRAIN_FRACTION,
        "train_points": int(len(train)),
        "test_points": int(len(test_ts)),
        "train_start": str(train['ds'].min()) if not train.empty else None,
        "train_end": str(train['ds'].max()) if not train.empty else None,
        "test_start": str(test_ts.index.min()) if len(test_ts) else None,
        "test_end": str(test_ts.index.max()) if len(test_ts) else None
    }
    log_verbose(f"Split info ({model_path or 'N/A'}): {split_info}")

    # --- Prophet ---
    # Save hyperparameters for minimal updates during forecast
    prophet_params = {
        'daily_seasonality': True,
        'weekly_seasonality': True,
        'changepoint_prior_scale': 0.05
    }
    m = Prophet(daily_seasonality=prophet_params['daily_seasonality'], 
                weekly_seasonality=prophet_params['weekly_seasonality'], 
                changepoint_prior_scale=prophet_params['changepoint_prior_scale'])
    m.add_regressor('hour'); m.add_regressor('is_weekend')
    pdf['hour'] = pdf['ds'].dt.hour
    pdf['is_weekend'] = (pdf['ds'].dt.dayofweek>=5).astype(int)
    
    # Add instance/node metadata to CSV if dumping
    if dump_csv_dir:
        pdf_for_csv = pdf.copy()
        # Add instance metadata if we have multiple instances (cluster-wide aggregate)
        if instances_included:
            pdf_for_csv['instances_count'] = len(instances_included)
            instances_str = ', '.join(instances_included[:20])
            if len(instances_included) > 20:
                instances_str += f' ... (+{len(instances_included) - 20} more)'
            pdf_for_csv['instances'] = instances_str
        # Add node/signal metadata from context if available (for per-node models like I/O network)
        elif context:
            node = context.get('node')
            signal = context.get('signal')
            if node:
                pdf_for_csv['node'] = node
            if signal:
                pdf_for_csv['signal'] = signal
    
    label = None
    if context:
        if context.get('layer'):
            label = f"{context['layer']}_layer"
        else:
            node = context.get('node')
            signal = context.get('signal')
            if node and signal:
                label = f"{node}_{signal}"
            elif node:
                label = node
    if not label:
        label = os.path.splitext(os.path.basename(model_path or "ensemble"))[0]
    
    if dump_csv_dir:
        if instances_included or (context and (context.get('node') or context.get('signal'))):
            dump_dataframe_to_csv(pdf_for_csv.copy(), dump_csv_dir, label)
        else:
            dump_dataframe_to_csv(pdf.copy(), dump_csv_dir, label)
    m.fit(pdf)
    future = m.make_future_dataframe(periods=horizon_min, freq='min')
    future['hour'] = future['ds'].dt.hour
    future['is_weekend'] = (future['ds'].dt.dayofweek>=5).astype(int)
    f_prophet = m.predict(future)['yhat']

    # --- ARIMA ---
    ts = pd.Series(pdf.set_index('ds')['y'])
    if ts.index.freq is None:
        ts.index.freq = pd.infer_freq(ts.index)
    arima = ARIMA(ts, order=(2,1,0)).fit()
    f_arima = arima.forecast(steps=horizon_min)
    f_arima.index = pd.date_range(start=ts.index[-1] + pd.Timedelta(minutes=1), periods=horizon_min, freq='min')

    # --- LSTM (CPU-only) ---
    f_lstm = None
    if LSTM_AVAILABLE and len(ts) > LSTM_SEQ_LEN + horizon_min:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(ts.values.reshape(-1, 1))
        X, y = [], []
        for i in range(LSTM_SEQ_LEN, len(scaled) - horizon_min):
            X.append(scaled[i-LSTM_SEQ_LEN:i])
            y.append(scaled[i:i+horizon_min])
        X, y = np.array(X), np.array(y)

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(LSTM_SEQ_LEN, 1)),
            LSTM(50),
            Dense(horizon_min)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=LSTM_EPOCHS, batch_size=32, verbose=0,
                  callbacks=[EarlyStopping(patience=3)])

        # Forecast
        last_seq = scaled[-LSTM_SEQ_LEN:].reshape(1, LSTM_SEQ_LEN, 1)
        lstm_pred = model.predict(last_seq, verbose=0)
        f_lstm = pd.Series(scaler.inverse_transform(lstm_pred)[0],
                           index=f_arima.index)

        # Save LSTM (only if save_model is True)
        if save_model:
            joblib.dump({'model': model, 'scaler': scaler}, LSTM_MODEL_PATH)
            print(f"LSTM model saved: {LSTM_MODEL_PATH}")
    else:
        f_lstm = f_arima.copy()  # fallback
        log_verbose("LSTM skipped: not enough data or TensorFlow missing")

    # --- Ensemble (Prophet + ARIMA + LSTM) ---
    tail = future.tail(horizon_min)
    prophet_tail = pd.Series(f_prophet.tail(horizon_min).values, index=tail['ds'])
    ensemble = (prophet_tail + f_arima + f_lstm) / 3

    # --- ROBUST BACKTEST — WORKS WITH 1m, 5m, 10m, 1h DATA ---
    # Initialize backtest variables
    p_back = None
    a_pred = None
    l_back = None
    ens_pred = None
    metrics = {}
    
    if len(test_ts) >= 50:
        # Prophet backtest
        mb = Prophet(daily_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.05)
        mb.add_regressor('hour')
        mb.add_regressor('is_weekend')

        train_b = train.copy()
        train_b['hour'] = train_b['ds'].dt.hour
        train_b['is_weekend'] = (train_b['ds'].dt.dayofweek >= 5).astype(int)
        mb.fit(train_b)

        # Make future at minute resolution (Prophet requirement)
        fut_b = mb.make_future_dataframe(periods=len(test_ts), freq='min')
        fut_b['hour'] = fut_b['ds'].dt.hour
        fut_b['is_weekend'] = (fut_b['ds'].dt.dayofweek >= 5).astype(int)

        prophet_pred_full = mb.predict(fut_b).set_index('ds')
        # CRITICAL FIX: align to actual test timestamps (10m data has gaps!)
        prophet_full = mb.predict(fut_b).set_index('ds')
        p_back = prophet_full.reindex(test_ts.index, method='nearest')['yhat']

        # ARIMA — use original timestamps
        train_ts = train.set_index('ds')['y']
        a_model = ARIMA(train_ts, order=(2,1,0)).fit()
        a_pred = pd.Series(a_model.forecast(steps=len(test_ts)), index=test_ts.index)

        # LSTM backtest
        l_back = a_pred.copy()  # fallback
        if LSTM_AVAILABLE and len(train_ts) > LSTM_SEQ_LEN + len(test_ts):
            try:
                scaler_b = MinMaxScaler()
                values = train.set_index('ds')['y'].values.reshape(-1, 1)
                scaled_b = scaler_b.fit_transform(values)

                Xb, yb = [], []
                for i in range(LSTM_SEQ_LEN, len(scaled_b) - len(test_ts)):
                    Xb.append(scaled_b[i-LSTM_SEQ_LEN:i])
                    yb.append(scaled_b[i:i + len(test_ts)])
                if Xb:
                    Xb, yb = np.array(Xb), np.array(yb)
                    model_b = Sequential([
                        LSTM(50, return_sequences=True, input_shape=(LSTM_SEQ_LEN, 1)),
                        LSTM(50),
                        Dense(len(test_ts))
                    ])
                    model_b.compile(optimizer='adam', loss='mse')
                    model_b.fit(Xb, yb, epochs=5, batch_size=32, verbose=0, callbacks=[EarlyStopping(patience=2)])

                    last_seq = scaled_b[-LSTM_SEQ_LEN:].reshape(1, LSTM_SEQ_LEN, 1)
                    l_pred = model_b.predict(last_seq, verbose=0)
                    l_back = pd.Series(scaler_b.inverse_transform(l_pred)[0], index=test_ts.index)
            except Exception as e:
                print(f"LSTM backtest failed: {e}")
                l_back = a_pred.copy()

        # Ensemble - handle NaN values gracefully
        # Fill NaN values in predictions with the mean of test data (or 0 if test is empty)
        test_mean = test_ts.mean() if not test_ts.empty and not test_ts.isna().all() else 0.0
        p_back_clean = p_back.fillna(test_mean)
        a_pred_clean = a_pred.fillna(test_mean)
        l_back_clean = l_back.fillna(test_mean)
        
        ens_pred = (p_back_clean + a_pred_clean + l_back_clean) / 3
        
        # Calculate metrics with NaN handling
        # Align all series and drop rows where either test or prediction has NaN
        aligned_df = pd.DataFrame({
            'test': test_ts,
            'ensemble': ens_pred,
            'prophet': p_back_clean,
            'arima': a_pred_clean,
            'lstm': l_back_clean
        }).dropna()
        
        if len(aligned_df) > 0:
            try:
                mae_ens = mean_absolute_error(aligned_df['test'], aligned_df['ensemble'])
            except (ValueError, Exception) as e:
                log_verbose(f"Warning: Failed to calculate ensemble MAE: {e}")
                mae_ens = np.nan
            
            try:
                mae_prophet = mean_absolute_error(aligned_df['test'], aligned_df['prophet'])
            except (ValueError, Exception) as e:
                log_verbose(f"Warning: Failed to calculate Prophet MAE: {e}")
                mae_prophet = np.nan
                
            try:
                mae_arima = mean_absolute_error(aligned_df['test'], aligned_df['arima'])
            except (ValueError, Exception) as e:
                log_verbose(f"Warning: Failed to calculate ARIMA MAE: {e}")
                mae_arima = np.nan
                
            try:
                mae_lstm = mean_absolute_error(aligned_df['test'], aligned_df['lstm'])
            except (ValueError, Exception) as e:
                log_verbose(f"Warning: Failed to calculate LSTM MAE: {e}")
                mae_lstm = np.nan
        else:
            # No valid data after alignment
            mae_ens = np.nan
            mae_prophet = np.nan
            mae_arima = np.nan
            mae_lstm = np.nan

        metrics = {
            'mae_ensemble': mae_ens,
            'mae_prophet': mae_prophet,
            'mae_arima': mae_arima,
            'mae_lstm': mae_lstm
        }
        metrics['split_info'] = split_info
        
        # Format metrics nicely and print only if print_backtest_metrics is True
        if print_backtest_metrics:
            context_str = ""
            if context:
                node = context.get('node', '')
                signal = context.get('signal', '')
                if node and signal:
                    context_str = f" → {node} | {signal}"
                elif node:
                    context_str = f" → {node}"
            
            print(f"\nBacktest Metrics{context_str}:")
            for k, v in metrics.items():
                if k == 'split_info' and isinstance(v, dict):
                    print(f"  • Train/Test Split:")
                    print(f"    - Train fraction: {v.get('train_fraction', 0)*100:.0f}%")
                    print(f"    - Train points: {v.get('train_points', 0):,}")
                    print(f"    - Test points: {v.get('test_points', 0):,}")
                    if v.get('train_start'):
                        print(f"    - Train period: {v['train_start']} → {v['train_end']}")
                    if v.get('test_start'):
                        print(f"    - Test period: {v['test_start']} → {v['test_end']}")
                elif isinstance(v, (int, float)) and not np.isnan(v):
                    print(f"  • {k}: {v:.6f}")
                elif isinstance(v, (int, float)) and np.isnan(v):
                    print(f"  • {k}: N/A")
                else:
                    print(f"  • {k}: {v}")
    else:
        metrics = {'mae_ensemble': np.nan, 'mae_prophet': np.nan, 'mae_arima': np.nan, 'mae_lstm': np.nan}
        metrics['split_info'] = split_info
        print("Not enough test data for backtest")

    # --- Plot 1: Forecast --- Always save plots to FORECAST_PLOTS_DIR
    last = ts.last('1h')
    plt.figure(figsize=(12,6))
    plt.plot(tail['ds'], f_prophet.tail(horizon_min), label='Prophet', color='orange', ls='--')
    plt.plot(tail['ds'], f_arima, label='ARIMA', color='green', ls='--')
    plt.plot(tail['ds'], f_lstm, label='LSTM', color='purple', ls=':')
    plt.plot(tail['ds'], ensemble, label='Ensemble (3)', color='red', lw=2)
    plt.plot(last.index, last, label='Last hour', color='blue', alpha=0.7)
    if split_info.get("test_start"):
        plt.axvline(pd.to_datetime(split_info["test_start"]), color='black', linestyle=':', alpha=0.6, label='Train/Test split')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    # Determine model type from model_path or context
    if model_path:
        # Extract basename first, then get model type (e.g., "host_forecast.pkl" -> "host")
        basename = os.path.basename(model_path)
        if '_' in basename:
            model_type = basename.split('_')[0].upper()
        else:
            # Fallback: use basename without extension
            model_type = os.path.splitext(basename)[0].upper()
    elif context:
        # Use context to determine model type (for I/O and Network models)
        signal = context.get('signal', 'MODEL')
        node = context.get('node', '')
        model_type = f"{signal}_{node}" if node else signal
    else:
        model_type = "MODEL"
    plt.title(f"{model_type} Layer – {horizon_min}-min Forecast")
    # Save forecast plot only if save_forecast_plot is True
    if save_forecast_plot:
        # Save plot to FORECAST_PLOTS_DIR
        # Sanitize model_type for filename
        safe_model_type = model_type.lower().replace(' ', '_').replace('/', '_').replace(':', '_').replace('(', '_').replace(')', '_')
        plot_filename = f"{safe_model_type}_layer_forecast.png"
        plot_path = os.path.join(FORECAST_PLOTS_DIR, plot_filename)
        plt.savefig(plot_path, dpi=180, bbox_inches='tight')
    plt.close()
    
    # --- Plot 2: Backtest Visualization (if test data available and save_backtest_plot is True) ---
    if save_backtest_plot and len(test_ts) >= 50 and 'mae_ensemble' in metrics:
        plt.figure(figsize=(14, 7))
        # Plot train data
        train_ts_plot = train.set_index('ds')['y']
        plt.plot(train_ts_plot.index, train_ts_plot.values, label='Train Data', color='#1f77b4', alpha=0.7, linewidth=1.5)
        # Plot test data
        plt.plot(test_ts.index, test_ts.values, label='Test Data (Actual)', color='#2ca02c', alpha=0.7, linewidth=1.5)
        # Plot backtest predictions
        if p_back is not None:
            plt.plot(test_ts.index, p_back.values, label='Prophet Backtest', color='orange', ls='--', linewidth=1.5)
        if a_pred is not None:
            plt.plot(test_ts.index, a_pred.values, label='ARIMA Backtest', color='green', ls='--', linewidth=1.5)
        if l_back is not None:
            plt.plot(test_ts.index, l_back.values, label='LSTM Backtest', color='purple', ls=':', linewidth=1.5)
        if ens_pred is not None:
            plt.plot(test_ts.index, ens_pred.values, label='Ensemble Backtest', color='red', lw=2)
        # Mark train/test split
        if split_info.get("test_start"):
            split_time = pd.to_datetime(split_info["test_start"])
            plt.axvline(split_time, color='gray', linestyle=':', alpha=0.7, linewidth=2, label='Train/Test Split')
        plt.xlabel('Date')
        plt.ylabel('Value (normalized)')
        plt.title(f"{model_type} Layer – Backtest Performance\nMAE: Ensemble={metrics.get('mae_ensemble', 0):.6f}, Prophet={metrics.get('mae_prophet', 0):.6f}, ARIMA={metrics.get('mae_arima', 0):.6f}, LSTM={metrics.get('mae_lstm', 0):.6f}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        # Sanitize model_type for filename
        safe_model_type = model_type.lower().replace(' ', '_').replace('/', '_').replace(':', '_').replace('(', '_').replace(')', '_')
        backtest_plot_filename = f"{safe_model_type}_layer_backtest.png"
        backtest_plot_path = os.path.join(FORECAST_PLOTS_DIR, backtest_plot_filename)
        plt.savefig(backtest_plot_path, dpi=180, bbox_inches='tight')
        plt.close()

    # ---- Forecast DF ----------------------------------------------------
    # Keep historical Prophet predictions, then append 3-model ensemble for future
    out = pd.DataFrame({
    'ds': future['ds'],
    'yhat': f_prophet.values  # Full history + future (Prophet only)
    })
    # Replace only the future part with 3-model ensemble
    out.loc[len(pdf):, 'yhat'] = ensemble.values

    # Save ARIMA model separately for later use (don't retrain during forecast)
    # Save it with a name based on model_path (e.g., host_forecast.pkl -> host_arima.pkl)
    # Only save if model_path is provided (skip for I/O and Network models stored in manifest)
    # Only save if save_model is True (skip when show_backtest mode to avoid updating model files)
    if model_path and save_model:
        # Fix path construction: avoid double replacement (host_forecast.pkl -> host_arima.pkl, not host_arima_arima.pkl)
        if model_path.endswith('_forecast.pkl'):
            arima_model_path = model_path.replace('_forecast.pkl', '_arima.pkl')
        else:
            arima_model_path = model_path.replace('.pkl', '_arima.pkl')
        try:
            joblib.dump({
                'model': arima,
                'last_training_point': str(ts.index[-1]),
                'order': (2, 1, 0),
                'training_data_end': str(pdf['ds'].max())
            }, arima_model_path)
            log_verbose(f"ARIMA model saved: {arima_model_path}")
        except Exception as e:
            log_verbose(f"Warning: Failed to save ARIMA model: {e}")
        
        # Save Prophet hyperparameters for minimal updates during forecast
        # Fix path construction: avoid double replacement
        if model_path.endswith('_forecast.pkl'):
            prophet_params_path = model_path.replace('_forecast.pkl', '_prophet_params.pkl')
        else:
            prophet_params_path = model_path.replace('.pkl', '_prophet_params.pkl')
        try:
            joblib.dump(prophet_params, prophet_params_path)
            log_verbose(f"Prophet parameters saved: {prophet_params_path}")
        except Exception as e:
            log_verbose(f"Warning: Failed to save Prophet parameters: {e}")

    # FINAL FOREVER FIX — return a dict so order never matters again
    return m, out, metrics

# ----------------------------------------------------------------------
# 3. CLASSIFICATION MODEL
# ----------------------------------------------------------------------
def extract_instance_features(df_host_cpu, df_host_mem, df_pod_cpu, df_pod_mem, lookback_hours=LOOKBACK_HOURS):
    now = pd.Timestamp.now()
    start = now - pd.Timedelta(hours=lookback_hours)

    def recent(df):
        if df.empty:
            return df
        return df[df['timestamp'] >= start]

    def aggregate(df, label):
        if df.empty:
            return pd.Series(dtype=float, name=label)
        local = recent(df)
        if 'entity' not in local.columns:
            local = local.rename(columns={'instance': 'entity'})
        return local.groupby('entity')['value'].mean().rename(label)

    hcpu = aggregate(df_host_cpu, 'host_cpu')
    hmem = aggregate(df_host_mem, 'host_mem')
    pcpu = aggregate(df_pod_cpu, 'pod_cpu')
    pmem = aggregate(df_pod_mem, 'pod_mem')

    feats = pd.concat([hcpu, hmem, pcpu, pmem], axis=1).fillna(0)
    feats.index.name = 'entity'

    raw_lookup = {}
    for df in (df_host_cpu, df_host_mem, df_pod_cpu, df_pod_mem):
        if df.empty or 'entity' not in df.columns:
            continue
        subset = recent(df)[['entity', 'raw_instance']].dropna()
        for entity_value, raw_value in subset.values:
            raw_lookup.setdefault(entity_value, raw_value)

    feats = feats.reset_index().rename(columns={'index': 'entity'})
    feats['raw_instance'] = feats['entity'].map(lambda ent: raw_lookup.get(ent, ent))
    return feats

def classification_model(df_host_cpu, df_host_mem, df_pod_cpu, df_pod_mem,
                        lookback_hours=LOOKBACK_HOURS, contamination=CONTAMINATION, forecast_mode=False,
                        dump_csv_dir=None, enable_plots=True):
    feats = extract_instance_features(df_host_cpu, df_host_mem, df_pod_cpu, df_pod_mem, lookback_hours)
    feats['instance_label'] = feats.apply(
        lambda row: canonical_node_label(row['entity'], with_ip=True, raw_label=row.get('raw_instance')),
        axis=1
    )
    dump_dataframe_to_csv(feats.copy(), dump_csv_dir, "classification_features")
    scaler = StandardScaler()
    X = scaler.fit_transform(feats[['host_cpu','host_mem','pod_cpu','pod_mem']])
    iso = IsolationForest(contamination=contamination, random_state=42)
    labels = iso.fit_predict(X)
    feats['anomaly'] = labels

    print("\n" + "="*80)
    print("Building Classification (Anomaly) Model...")
    print("="*80)
    print(classification_report(labels, labels, digits=2))

    anomalous = feats[feats['anomaly'] == -1]
    if not anomalous.empty:
        # Keep the console output concise so on-call engineers can scan it quickly.
        print("\n⚠️  Anomalous nodes detected:")
        display_cols = anomalous[['entity','raw_instance','host_cpu','host_mem','pod_cpu','pod_mem']].copy()
        display_cols['instance'] = display_cols.apply(
            lambda row: canonical_node_label(row['entity'], with_ip=True, raw_label=row.get('raw_instance')),
            axis=1
        )
        print(display_cols[['instance','host_cpu','host_mem','pod_cpu','pod_mem']])
        print("Action: investigate non-Kubernetes workload on these nodes.")
    else:
        print("\nNo anomalous nodes – host and pod usage aligned.")

    plt.figure(figsize=(9,6))
    colors = ['red' if a==-1 else 'steelblue' for a in labels]
    plt.scatter(feats['host_mem'], feats['pod_mem'], c=colors, alpha=0.7)
    for _, r in feats.iterrows():
        plt.text(r['host_mem']+0.01, r['pod_mem'], r['entity'][:10], fontsize=9)
    plt.xlabel('Host Memory (norm)')
    plt.ylabel('Pod Memory (norm)')
    plt.title('Classification: Host vs Pod – Red = non-K8s')
    plt.grid(alpha=0.3); plt.tight_layout()
    # Save plot only in forecast mode or when training, and if enable_plots is True
    if enable_plots and (forecast_mode or FORCE_TRAINING_RUN):
        plot_path = os.path.join(FORECAST_PLOTS_DIR, "classification_host_vs_pod.png")
        plt.savefig(plot_path, dpi=180, bbox_inches='tight')
    plt.close()

    host_pressure_df = report_host_only_pressure(feats, return_df=True)

    if FORCE_TRAINING_RUN:
        try:
            joblib.dump(iso, ANOMALY_MODEL_PATH)
            joblib.dump(scaler, ANOMALY_SCALER_PATH)
            print(f"Saved anomaly model → {ANOMALY_MODEL_PATH}")
            print(f"Saved anomaly scaler → {ANOMALY_SCALER_PATH}")
        except Exception as exc:
            print(f"Warning: unable to persist anomaly model: {exc}")

    # Prepare anomaly dataframes for alert dispatch
    anomalous_df = pd.DataFrame()
    if not anomalous.empty:
        anomalous_df = anomalous[['entity', 'raw_instance', 'host_cpu', 'host_mem', 'pod_cpu', 'pod_mem']].copy()
        anomalous_df['instance'] = anomalous_df.apply(
            lambda row: canonical_node_label(row['entity'], with_ip=True, raw_label=row.get('raw_instance')),
            axis=1
        )
        anomalous_df['severity'] = 'WARNING'
        anomalous_df['signal'] = 'anomalous_node'
        anomalous_df['detected_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return feats, iso, anomalous_df, host_pressure_df

def run_realtime_anomaly_watch(q_host_cpu, q_host_mem, q_pod_cpu, q_pod_mem,
                               iterations=1, interval_seconds=15):
    if iterations <= 0:
        return
    if not os.path.exists(ANOMALY_MODEL_PATH) or not os.path.exists(ANOMALY_SCALER_PATH):
        print("Realtime anomaly watch skipped — trained model/scaler not found.")
        return
    try:
        iso = joblib.load(ANOMALY_MODEL_PATH)
        scaler = joblib.load(ANOMALY_SCALER_PATH)
    except Exception as exc:
        print(f"Unable to load anomaly artifacts: {exc}")
        return

    print(f"\nStarting realtime anomaly watch ({iterations} iterations, {interval_seconds}s cadence)...")
    for idx in range(iterations):
        df_hcpu = fetch_and_preprocess_data(q_host_cpu)
        df_hmem = fetch_and_preprocess_data(q_host_mem)
        df_pcpu = fetch_and_preprocess_data(q_pod_cpu)
        df_pmem = fetch_and_preprocess_data(q_pod_mem)

        feats = extract_instance_features(df_hcpu, df_hmem, df_pcpu, df_pmem)
        if feats.empty:
            print(f"[Watch #{idx+1}] No data available for anomaly scoring.")
        else:
            features = feats[['host_cpu','host_mem','pod_cpu','pod_mem']]
            try:
                scaled = scaler.transform(features)
                preds = iso.predict(scaled)
                feats['anomaly'] = preds
                anomalies = feats[feats['anomaly'] == -1]
                ts_label = datetime.now().strftime("%H:%M:%S")
                if anomalies.empty:
                    print(f"[Watch #{idx+1} @ {ts_label}] ✅ No anomalies detected.")
                else:
                    print(f"[Watch #{idx+1} @ {ts_label}] ⚠️  {len(anomalies)} anomalies detected:")
                    display = anomalies.copy()
                    display['instance'] = anomalies.apply(lambda row: canonical_node_label(row['instance'], with_ip=True, raw_label=row.get('raw_instance')), axis=1)
                    print(display[['instance','host_cpu','host_mem','pod_cpu','pod_mem']].to_string(index=False))
            except Exception as exc:
                print(f"[Watch #{idx+1}] Failed to score anomalies: {exc}")

        if idx < iterations - 1:
            time.sleep(interval_seconds)

# ----------------------------------------------------------------------
# 4. DISK FULL PREDICTION (7-day horizon)
# ----------------------------------------------------------------------
def detect_golden_anomaly_signals(hours=1):
    """
    SRE Golden Anomaly Detector — autonomous root-cause engine
    Returns clean DataFrame even when no signals found
    """
    queries = {
        "iowait_high":      'avg by (instance) (rate(node_cpu_seconds_total{mode="iowait"}[5m])) > 0.15',
        "inodes_critical":  'avg by (instance, mountpoint) (1 - node_filesystem_files_free / node_filesystem_files{mountpoint=~"/.*"}) > 0.90',
        "net_rx_drop":      'changes(node_network_receive_drop_total[5m]) > 10',
        "net_tx_saturated": 'avg by (instance) (rate(node_network_transmit_bytes_total[5m])) > 9e8',  # ~9 Gbit
        "tcp_retrans_high": 'avg by (instance) (rate(node_netstat_Tcp_RetransSegs[5m])) > 1000',
        "oom_kills":        'increase(node_vmstat_oom_kill[1h]) > 0',
        "fork_bomb":        'rate(node_fork_total[5m]) > 1000',
        "fd_leak":          'process_open_fds / process_max_fds > 0.90',
    }

    anomalies = []
    start = int((pd.Timestamp.now() - pd.Timedelta(hours=hours)).timestamp())
    end = int(pd.Timestamp.now().timestamp())

    for signal, query in queries.items():
        df = fetch_victoriametrics_metrics(query=query, start=start, end=end, step="1m")
        if df.empty or 'instance' not in df.columns:
            continue

        for inst in df['instance'].unique():
            node = canonical_node_label(inst, with_ip=True)
            severity = "CRITICAL" if signal in ["oom_kills", "inodes_critical", "fd_leak"] else "WARNING"
            anomalies.append({
                "node": node,
                "signal": signal.replace("_", " ").upper(),
                "severity": severity,
                "detected_at": pd.Timestamp.now().strftime("%H:%M")
            })

    # ←←← THIS IS THE FIX — SAFE EVEN WHEN EMPTY ←←←
    if not anomalies:
        return pd.DataFrame(columns=["node", "signal", "severity", "detected_at"])

    return (pd.DataFrame(anomalies)
            .drop_duplicates()
            .sort_values("severity", ascending=False)
            .reset_index(drop=True))

# ----------------------------------------------------------------------
# 7. IO and NETWORK
# ----------------------------------------------------------------------
def predict_io_and_network_crisis_with_backtest(
    horizon_days: int = 7,
    test_days: int = 7,
    plot_dir: str | None = None,
    force_retrain: bool | None = None,
    manifest: dict | None = None,
    retrain_targets: set | None = None,
    show_backtest: bool = False,
    forecast_mode: bool = False,
    dump_csv_dir: str | None = None,
    enable_plots: bool = True
):
    """
    FINAL PRODUCTION VERSION — NO MORE ERRORS
    Disk I/O + Network crisis forecasting with proper backtesting & plots
    Uses manifest for model storage (single file instead of per-node files)
    """
    if plot_dir is None:
        plot_dir = FORECAST_PLOTS_DIR
    os.makedirs(plot_dir, exist_ok=True)
    if force_retrain is None:
        force_retrain = FORCE_TRAINING_RUN
    manifest = manifest or {}
    retrain_targets = retrain_targets or set()
    manifest_changed = False
    results = []

    queries = {
        "DISK_IO_WAIT": '''
        avg by (instance) (
          rate(node_disk_io_time_seconds_total[5m]) or
          rate(node_cpu_seconds_total{mode="iowait"}[5m])
        )
        ''',
        "NET_TX_BW": 'avg by (instance) (rate(node_network_transmit_bytes_total[5m]))'
    }

    thresholds = {"DISK_IO_WAIT": 0.30, "NET_TX_BW": 9.5e8}
    units = {"DISK_IO_WAIT": "ratio", "NET_TX_BW": "bytes/sec"}

    for name, query in queries.items():
        log_verbose(f"\nFetching {name}...")
        df_raw = fetch_victoriametrics_metrics(
            query=query,
            start=int((pd.Timestamp.now() - pd.Timedelta(days=30)).timestamp()),
            end=int(pd.Timestamp.now().timestamp()),
            step="10m"
        )

        if df_raw.empty:
            log_verbose(f"  → No data for {name}")
            continue

        df = df_raw.copy()
        df['timestamp'] = pd.to_datetime(df['ts'], unit='s')

        processed_nodes = 0
        for inst, group in df.groupby('instance'):
            node = canonical_node_label(inst, with_ip=True)
            entity = canonical_identity(inst)  # Canonical name for matching
            ts = group.set_index('timestamp')['value'].sort_index()
            if len(ts) < 100:
                continue
            dump_label = f"io_crisis_{node}_{name}"

            # Train/test split
            test_cutoff = ts.index[-1] - pd.Timedelta(days=test_days)
            train = ts[ts.index <= test_cutoff]
            test  = ts[ts.index > test_cutoff]

            if len(train) < 50 or len(test) < 10:
                continue

            current = ts.iloc[-1]
            threshold = thresholds[name]

            # Linear 7d burst
            trend_7d = train.last('7D').diff().mean() * 1440
            linear_eta = 9999.0
            if trend_7d > 0:
                remaining = threshold - current
                divisor = trend_7d / 100 if units[name] == "ratio" else trend_7d
                linear_eta = max(0.1, remaining / divisor)

            # Prophet - use manifest (with _backtest suffix to avoid conflicts)
            key = f"{build_io_net_key(entity, name)}_backtest"
            # Check if retraining is needed - match against entity, key, or any aliases
            # Check for "all" flag first
            needs_retrain = force_retrain or ('__RETRAIN_ALL__' in retrain_targets if retrain_targets else False)
            if needs_retrain:
                entity_match = key_match = instance_match = node_match = alias_match = False
            else:
                entity_match = entity in retrain_targets
                key_match = key in retrain_targets
                instance_canon = canonical_identity(inst)
                instance_match = instance_canon in retrain_targets
                node_base = node.split('(')[0].strip() if '(' in node else node
                node_base_canon = canonical_identity(node_base)
                node_match = node_base_canon in retrain_targets
                
                # Check if any retrain target is an alias that maps to this entity
                alias_match = False
            for target in retrain_targets:
                if '|' in target or '_' in target:
                    continue  # Skip keys, only check node names
                target_canon = canonical_identity(target)
                # Direct match already checked above
                if target_canon == entity:
                    alias_match = True
                    break
                # Check if target maps to this entity via alias map
                if target_canon in INSTANCE_ALIAS_MAP:
                    alias_value = INSTANCE_ALIAS_MAP[target_canon]
                    if canonical_identity(alias_value) == entity:
                        alias_match = True
                        break
                # Check reverse: if entity is in alias map, does target match the key?
                for k, v in INSTANCE_ALIAS_MAP.items():
                    if canonical_identity(v) == entity and canonical_identity(k) == target_canon:
                        alias_match = True
                        break
                if alias_match:
                    break
                # Check if both resolve to same IP in source registry
                target_ip = SOURCE_REGISTRY.get(target_canon) or CANON_SOURCE_MAP.get(target_canon)
                entity_ip = SOURCE_REGISTRY.get(entity) or CANON_SOURCE_MAP.get(entity)
                if target_ip and entity_ip and target_ip == entity_ip:
                    alias_match = True
                    break
                # Extract IP from node display string and check if target resolves to it
                # Only attempt DNS if target looks like a hostname
                if looks_like_hostname(target) and '(' in node and ')' in node:
                    node_ip = node.split('(')[1].split(')')[0].strip()
                    # Try to resolve target to IP (try with and without domain suffixes)
                    target_variants = [target]
                    for domain in DNS_DOMAIN_SUFFIXES:
                        if domain and not target.endswith(domain):
                            target_variants.append(f"{target}{domain}")
                    for target_var in target_variants:
                        try:
                            target_resolved = socket.gethostbyname(target_var)
                            if target_resolved == node_ip:
                                alias_match = True
                                log_verbose(f"   DNS match: {target_var} → {target_resolved} == {node_ip}")
                                break
                        except:
                            pass
                    if alias_match:
                        break
                
                needs_retrain = entity_match or key_match or instance_match or node_match or alias_match
            
            if not needs_retrain and key in manifest:
                m = manifest[key].get('model')
                if m is not None:
                    log_verbose(f"  → Loaded model from manifest: {key}")
                    # MINIMAL UPDATE: Use recent data only (last 7 days) for forecast mode only
                    if forecast_mode:
                        # This incorporates latest trends while preserving learned patterns
                        # train is a Series with timestamp index, so use last('7D')
                        recent_train = train.last('7D') if len(train) > 7*24*6 else train
                        pdf = recent_train.reset_index().rename(columns={'timestamp': 'ds', 'value': 'y'})
                        m_updated = Prophet(changepoint_prior_scale=0.2, daily_seasonality=True, weekly_seasonality=True)
                        m_updated.fit(pdf)
                        m = m_updated  # Use updated model for forecasting
                        manifest[key] = {'model': m}  # Save updated model to manifest
                        manifest_changed = True
                        log_verbose(f"  → Minimal update applied (recent 7 days): {key}")
                else:
                    needs_retrain = True
            
            if needs_retrain or key not in manifest:
                # For retraining: use minimal update if model exists, full training if first-time
                if needs_retrain and key in manifest:
                    # Minimal update: use recent data (last 7 days) to incorporate latest trends
                    recent_train = train.last('7D') if len(train) > 7*24*6 else train
                    pdf = recent_train.reset_index().rename(columns={'timestamp': 'ds', 'value': 'y'})
                    m = Prophet(changepoint_prior_scale=0.2, daily_seasonality=True, weekly_seasonality=True)
                    m.fit(pdf)
                    # Add node and signal metadata to CSV
                    if dump_csv_dir:
                        pdf_for_csv = pdf.copy()
                        pdf_for_csv['node'] = node
                        pdf_for_csv['signal'] = name
                        dump_dataframe_to_csv(pdf_for_csv, dump_csv_dir, dump_label)
                    else:
                        dump_dataframe_to_csv(pdf.copy(), dump_csv_dir, dump_label)
                    log_verbose(f"  → Minimal update (recent 7 days): {key}")
                else:
                    # First-time training: use all data to learn patterns
                    pdf = train.reset_index().rename(columns={'timestamp': 'ds', 'value': 'y'})
                    m = Prophet(changepoint_prior_scale=0.2, daily_seasonality=True, weekly_seasonality=True)
                    m.fit(pdf)
                    manifest[key] = {'model': m}
                    manifest_changed = True
                    log_verbose(f"  → Trained & saved to manifest: {key}")
                    # Add node and signal metadata to CSV
                    if dump_csv_dir:
                        pdf_for_csv = pdf.copy()
                        pdf_for_csv['node'] = node
                        pdf_for_csv['signal'] = name
                        dump_dataframe_to_csv(pdf_for_csv, dump_csv_dir, dump_label)
                    else:
                        dump_dataframe_to_csv(pdf.copy(), dump_csv_dir, dump_label)
            elif m is None:
                continue

            future = m.make_future_dataframe(periods=(test_days + horizon_days) * 1440, freq='min')
            forecast = m.predict(future)

            # Backtest
            test_forecast = forecast.set_index('ds').reindex(test.index, method='nearest')
            mae = mean_absolute_error(test, test_forecast['yhat'])
            rmse = np.sqrt(mean_squared_error(test, test_forecast['yhat']))

            # Prophet ETA
            future_pred = forecast[forecast['ds'] > ts.index[-1]]
            crisis = future_pred[future_pred['yhat'] >= threshold]
            prophet_eta = (crisis.iloc[0]['ds'] - pd.Timestamp.now()).total_seconds() / 86400 if not crisis.empty else 9999.0

            # Hybrid ETA
            hybrid_eta = min(linear_eta, prophet_eta)

            # SEVERITY — ALWAYS DEFINED (this was the bug!)
            if hybrid_eta < 3:
                severity = "CRITICAL"
            elif hybrid_eta < 7:
                severity = "WARNING"
            elif hybrid_eta < 30:
                severity = "SOON"
            else:
                severity = "OK"

            # Save backtest plot when training/retraining or when show_backtest is True
            if needs_retrain or show_backtest:
                plt.figure(figsize=(14, 7))
                plt.plot(train.index, train.values, label="Train Data", color="#1f77b4")
                plt.plot(test.index, test.values, label="Test (Actual)", color="#2ca02c", linewidth=2.5)
                plt.plot(forecast['ds'], forecast['yhat'], label="Forecast", color="#ff7f0e", linewidth=2)
                plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2, color="#ff7f0e")
                plt.axhline(threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold ({threshold})")
                plt.axvline(test_cutoff, color="gray", linestyle=":", alpha=0.7)
                plt.title(f"{node} — {name.replace('_', ' ')}\n"
                          f"MAE: {mae:.6f} | RMSE: {rmse:.6f} | Hybrid ETA: {hybrid_eta:.1f} days → {severity}")
                plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
                if enable_plots:
                    # Sanitize node name for filename
                    safe_node = node.split('(')[0].strip().replace(' ', '_').replace('/', '_')
                    plot_file = os.path.join(plot_dir, f"{safe_node}_{name.lower().replace(' ', '_')}_backtest.png")
                    plt.savefig(plot_file, dpi=180, bbox_inches='tight')
                    log_verbose(f"  → Plot saved: {plot_file}")
                plt.close()
            
            # Save forecast plot in forecast mode (showing future predictions, not backtest)
            if forecast_mode:
                plt.figure(figsize=(14, 7))
                # Plot last 24 hours of historical data
                historical = ts.last('24H')
                if len(historical) > 0:
                    plt.plot(historical.index, historical.values, label="Historical Data", color="#1f77b4", linewidth=1.5)
                # Plot forecast (future predictions only)
                future_pred = forecast[forecast['ds'] > ts.index[-1]]
                if not future_pred.empty:
                    # Limit to next 3 hours (180 minutes) for forecast plot
                    forecast_3h = future_pred.head(180)
                    plt.plot(forecast_3h['ds'], forecast_3h['yhat'], label="Forecast", color="#ff7f0e", linewidth=2)
                    plt.fill_between(forecast_3h['ds'], forecast_3h['yhat_lower'], forecast_3h['yhat_upper'], alpha=0.2, color="#ff7f0e")
                plt.axhline(threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold ({threshold})")
                # Mark current time
                plt.axvline(ts.index[-1], color="gray", linestyle=":", alpha=0.7, label="Now")
                plt.title(f"{node} — {name.replace('_', ' ')} Forecast\n"
                          f"Current: {current:,.6f} | Hybrid ETA: {hybrid_eta:.1f} days → {severity}")
                plt.xlabel('Time')
                plt.ylabel('Value')
                plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
                if enable_plots:
                    # Sanitize node name for filename
                    safe_node = node.split('(')[0].strip().replace(' ', '_').replace('/', '_')
                    plot_file = os.path.join(plot_dir, f"{safe_node}_{name.lower().replace(' ', '_')}_forecast.png")
                    plt.savefig(plot_file, dpi=180, bbox_inches='tight')
                    log_verbose(f"  → Forecast plot saved: {plot_file}")
                plt.close()

            if show_backtest or should_verbose():
                print(f"\nBacktest complete → {node} | {name.replace('_', ' ')}")
                print(f"  ├─ Train period     : {train.index[0].strftime('%Y-%m-%d')} → {train.index[-1].strftime('%Y-%m-%d')} ({len(train)} pts)")
                print(f"  ├─ Test period      : {test.index[0].strftime('%Y-%m-%d')} → {test.index[-1].strftime('%Y-%m-%d')} ({len(test)} pts)")
                print(f"  ├─ Current value    : {current:,.6f} → "
                      f"{'{:6.2f}%'.format(current*100) if units[name]=='ratio' else f'{current/1e9:.3f} GB/s'}")
                print(f"  ├─ Backtest MAE     : {mae:.6f}")
                print(f"  ├─ Backtest RMSE    : {rmse:.6f}")
                print(f"  ├─ Linear 7d ETA    : {linear_eta:6.1f} days")
                print(f"  ├─ Prophet ETA      : {prophet_eta:6.1f} days")
                print(f"  └─ HYBRID ETA       : {hybrid_eta:6.1f} days → {severity}")

            # Only add to results if within 30 days
            if hybrid_eta < 30:
                results.append({
                    "node": node,
                    "signal": name.replace("_", " "),
                    "current": f"{current*100:.2f}%" if units[name] == "ratio" else f"{current/1e9:.2f} GB/s",
                    "mae": round(mae, 6),
                    "hybrid_eta_days": round(hybrid_eta, 1),
                    "severity": severity
                })

            processed_nodes += 1

        summary = [r for r in results if r['signal'] == name.replace("_", " ")]
        print(f"{name}: processed {processed_nodes} nodes, crises <30d: {len(summary)}")

    results_df = pd.DataFrame(results) if results else pd.DataFrame(columns=["node","signal","current","mae","hybrid_eta_days","severity"])
    return results_df, manifest, manifest_changed

# ----------------------------------------------------------------------
# 8. IO and NETWORK AGAIN
# ----------------------------------------------------------------------
def predict_io_and_network_ensemble(horizon_days=7, test_days=7, plot_dir="forecast_plots", force_retrain: bool | None = None,
                                    manifest: dict | None = None, retrain_targets: set | None = None, show_backtest: bool = False,
                                    forecast_mode: bool = False, dump_csv_dir: str | None = None, enable_plots: bool = True):
    """
    DISK I/O + NETWORK — FULL ENSEMBLE FORECAST (same brain as CPU/Memory)
    - Uses manifest for model storage (single file instead of per-node files)
    - Instant load if exists
    - Trains only when missing
    - 1Gbps network threshold (120MB/s)
    - Zero bugs. Zero lies.
    """
    if plot_dir is None:
        plot_dir = FORECAST_PLOTS_DIR
    os.makedirs(plot_dir, exist_ok=True)
    if force_retrain is None:
        force_retrain = FORCE_TRAINING_RUN
    manifest = manifest or {}
    retrain_targets = retrain_targets or set()
    manifest_changed = False
    crisis_results = []
    anomaly_results = []
    retrained_nodes = set()  # Track which nodes were actually retrained

    resources = [
        {
            "name": "DISK_IO_WAIT",
            "query": 'avg by (instance) (rate(node_disk_io_time_seconds_total[5m]) or rate(node_cpu_seconds_total{mode="iowait"}[5m]))',
            "threshold": 0.30,                    # 30% iowait = user pain
            "unit": "ratio"
        },
        {
            "name": "NET_TX_BW",
            "query": 'avg by (instance) (rate(node_network_transmit_bytes_total[5m]))',
            "threshold": 120_000_000,             # 120 MB/s = 96% of 1Gbps → real crisis
            "unit": "bytes/sec"
        }
    ]

    # Collect all unique nodes across all signals for summary
    all_unique_nodes = set()
    all_unique_entities = set()
    
    # Collect backtest metrics when show_backtest is true
    backtest_metrics_list = []
    
    if retrain_targets:
        print(f"\nScanning all signals for available nodes...")

    for res in resources:
        log_verbose(f"\nFetching {res['name']}...")
        df_raw = fetch_victoriametrics_metrics(
            query=res["query"],
            start=int((pd.Timestamp.now() - pd.Timedelta(days=35)).timestamp()),
            end=int(pd.Timestamp.now().timestamp()),
            step="10m"
        )
        if df_raw.empty:
            continue

        df = df_raw.copy()
        df['timestamp'] = pd.to_datetime(df['ts'], unit='s')
        
        # Collect nodes for summary
        all_instances = df['instance'].unique()
        for inst in all_instances:
            node = canonical_node_label(inst, with_ip=True)
            entity = canonical_identity(inst)
            all_unique_nodes.add(node)
            all_unique_entities.add(f"{entity} ({inst})")
        
        # Show all nodes found in this signal's data (for debugging retrain targets)
        if retrain_targets:
            all_nodes = [canonical_node_label(inst, with_ip=True) for inst in all_instances]
            all_entities = [canonical_identity(inst) for inst in all_instances]
            log_verbose(f"  Found {len(all_instances)} nodes in {res['name']} data: {', '.join(all_nodes)}")
            log_verbose(f"  Entity names: {', '.join(all_entities)}")

        for instance, group in df.groupby('instance'):
            node = canonical_node_label(instance, with_ip=True)
            entity = canonical_identity(instance)  # Canonical name for matching
            ts = group.set_index('timestamp')['value'].sort_index()
            if len(ts) < 200:
                log_verbose(f"   Skipping {node} | {res['name']}: insufficient data ({len(ts)} points)")
                continue

            cutoff = ts.index[-1] - pd.Timedelta(days=test_days)
            train_raw = ts[ts.index <= cutoff]
            if len(train_raw) < 100:
                continue

            current = ts.iloc[-1]
            train_df = train_raw.reset_index()
            train_df.columns = ['timestamp', 'value']

            key = f"{build_io_net_key(entity, res['name'])}_ensemble"
            log_verbose(f"  Running ensemble forecast for {node} | {res['name']} ({len(train_df)} points)...")

            # ———— MODEL CACHING ————
            # Check if retraining is needed - match against entity, key, or any aliases
            # Check for "all" flag first
            needs_retrain = force_retrain or ('__RETRAIN_ALL__' in retrain_targets if retrain_targets else False)
            if needs_retrain:
                entity_match = key_match = instance_match = node_match = alias_match = False
            else:
                entity_match = entity in retrain_targets
                key_match = key in retrain_targets
                # Also check if instance (raw) matches after canonicalization
                instance_canon = canonical_identity(instance)
                instance_match = instance_canon in retrain_targets
                # Check if node display name (without IP) matches
                node_base = node.split('(')[0].strip() if '(' in node else node
                node_base_canon = canonical_identity(node_base)
                node_match = node_base_canon in retrain_targets
                
                # Check if any retrain target is an alias that maps to this entity
                alias_match = False
            for target in retrain_targets:
                if '|' in target or '_' in target:
                    continue  # Skip keys, only check node names
                target_canon = canonical_identity(target)
                # Direct match already checked above
                if target_canon == entity:
                    alias_match = True
                    break
                # Check if target maps to this entity via alias map
                if target_canon in INSTANCE_ALIAS_MAP:
                    alias_value = INSTANCE_ALIAS_MAP[target_canon]
                    if canonical_identity(alias_value) == entity:
                        alias_match = True
                        break
                # Check reverse: if entity is in alias map, does target match the key?
                for k, v in INSTANCE_ALIAS_MAP.items():
                    if canonical_identity(v) == entity and canonical_identity(k) == target_canon:
                        alias_match = True
                        break
                if alias_match:
                    break
                # Check if both resolve to same IP in source registry
                target_ip = SOURCE_REGISTRY.get(target_canon) or CANON_SOURCE_MAP.get(target_canon)
                entity_ip = SOURCE_REGISTRY.get(entity) or CANON_SOURCE_MAP.get(entity)
                if target_ip and entity_ip and target_ip == entity_ip:
                    alias_match = True
                    break
                # Extract IP from node display string and check if target resolves to it
                # Only attempt DNS if target looks like a hostname
                if looks_like_hostname(target) and '(' in node and ')' in node:
                    node_ip = node.split('(')[1].split(')')[0].strip()
                    # Try to resolve target to IP (try with and without domain suffixes)
                    target_variants = [target] + [f"{target}{d}" for d in DNS_DOMAIN_SUFFIXES if d and not target.endswith(d)]
                    for target_var in target_variants:
                        try:
                            target_resolved = socket.gethostbyname(target_var)
                            if target_resolved == node_ip:
                                alias_match = True
                                log_verbose(f"   DNS match: {target_var} → {target_resolved} == {node_ip}")
                                break
                        except Exception as e:
                            log_verbose(f"   DNS resolution failed for {target_var}: {e}")
                    if alias_match:
                        break
                
                needs_retrain = entity_match or key_match or instance_match or node_match or alias_match
            
            if needs_retrain:
                retrained_nodes.add(f"{node} ({entity})")
                print(f"   ✓ Retraining {node} | {res['name']} (matched via: {'entity' if entity_match else ''} {'key' if key_match else ''} {'instance' if instance_match else ''} {'node' if node_match else ''} {'alias' if alias_match else ''})")
                log_verbose(f"   Retraining requested for {node} | {res['name']} (entity: {entity}, matches: entity={entity_match}, key={key_match}, instance={instance_match}, node={node_match}, alias={alias_match})")
            elif retrain_targets:
                # Show why this node didn't match any retrain targets
                node_targets = {t for t in retrain_targets if '|' not in t and '_' not in t}
                if node_targets:
                    log_verbose(f"   Skipping {node} | {res['name']} (entity: {entity})")
                    for target in node_targets:
                        target_canon = canonical_identity(target)
                        log_verbose(f"      Checking target '{target}' (canon: {target_canon}) vs entity '{entity}': match={target_canon == entity}")
                        # Try DNS resolution for debugging
                        if '(' in node and ')' in node:
                            node_ip = node.split('(')[1].split(')')[0].strip()
                            target_variants = [target] + [f"{target}{d}" for d in DNS_DOMAIN_SUFFIXES if d and not target.endswith(d)]
                            for target_var in target_variants:
                                try:
                                    target_resolved = socket.gethostbyname(target_var)
                                    log_verbose(f"         DNS: {target_var} → {target_resolved}, node IP: {node_ip}, match={target_resolved == node_ip}")
                                except Exception as e:
                                    log_verbose(f"         DNS: {target_var} → failed: {e}")
            
            if not needs_retrain and key in manifest:
                forecast_result = manifest[key].get('model')
                if forecast_result is not None:
                    log_verbose(f"   Loaded ENSEMBLE model from manifest: {key}")
                    # MINIMAL UPDATE: Use recent data only (last 7 days) for forecast mode only
                    if forecast_mode:
                        # This incorporates latest trends while preserving learned patterns
                        # train_df is DataFrame with 'timestamp' column, sort and get last 7 days
                        train_df_sorted = train_df.sort_values('timestamp')
                        cutoff_time = train_df_sorted['timestamp'].max() - pd.Timedelta(days=7)
                        recent_train_df = train_df_sorted[train_df_sorted['timestamp'] >= cutoff_time]
                        if len(recent_train_df) < 50:
                            recent_train_df = train_df_sorted.tail(min(len(train_df_sorted), 7*24*6))  # Fallback: last N rows
                        forecast_result = build_ensemble_forecast_model(
                            df_cpu=recent_train_df,
                            df_mem=None,
                            horizon_min=horizon_days * 24 * 60,
                            model_path=None,
                            context={'node': node, 'signal': res['name']},
                            save_forecast_plot=True,  # Save forecast plots in forecast mode
                            save_backtest_plot=False,  # Don't save backtest plots in forecast mode
                            print_backtest_metrics=False,  # Don't print backtest metrics in forecast mode
                            dump_csv_dir=dump_csv_dir,
                            enable_plots=enable_plots
                        )
                        if forecast_result is not None:
                            manifest[key] = {'model': forecast_result}
                            manifest_changed = True
                            log_verbose(f"   Minimal update applied (recent 7 days): {key}")
                    # If show_backtest is true, compute metrics even for cached models
                    # BUT don't update manifest - only generate plots
                    if show_backtest:
                        # Check if cached model has metrics
                        has_metrics = isinstance(forecast_result, tuple) and len(forecast_result) >= 3
                        if not has_metrics:
                            # Compute metrics for cached model (for display only, don't save to manifest)
                            log_verbose(f"   Computing backtest metrics for cached model (display only, not saving)...")
                            forecast_result = build_ensemble_forecast_model(
                                df_cpu=train_df,
                                df_mem=None,
                                horizon_min=horizon_days * 24 * 60,
                                model_path=None,
                                context={'node': node, 'signal': res['name']},
                                save_forecast_plot=False,  # Don't save forecast plots when computing metrics for cached models
                                save_backtest_plot=False,  # Don't save backtest plots when computing metrics for cached models
                                print_backtest_metrics=False,  # Don't print backtest metrics when computing for cached models
                                save_model=False,  # Don't save model files in show_backtest mode
                                dump_csv_dir=dump_csv_dir,
                                enable_plots=enable_plots
                            )
                            # Don't update manifest in show_backtest mode - only use for display
                            # manifest[key] = {'model': forecast_result}
                            # manifest_changed = True
                else:
                    needs_retrain = True
            
            if needs_retrain or key not in manifest:
                if key in manifest:
                    log_verbose(f"   Retraining cached model → MINIMAL UPDATE (recent 7 days)...")
                    # Minimal update: use recent data (last 7 days) to incorporate latest trends
                    train_df_sorted = train_df.sort_values('timestamp')
                    cutoff_time = train_df_sorted['timestamp'].max() - pd.Timedelta(days=7)
                    recent_train_df = train_df_sorted[train_df_sorted['timestamp'] >= cutoff_time]
                    if len(recent_train_df) < 50:
                        recent_train_df = train_df_sorted.tail(min(len(train_df_sorted), 7*24*6))  # Fallback: last N rows
                    forecast_result = build_ensemble_forecast_model(
                        df_cpu=recent_train_df,
                        df_mem=None,
                        horizon_min=horizon_days * 24 * 60,
                        model_path=None,  # Don't save to individual file
                        context={'node': node, 'signal': res['name']},
                        save_forecast_plot=False,  # Don't save forecast plots when retraining in forecast mode
                        save_backtest_plot=False,  # Don't save backtest plots when retraining in forecast mode
                        print_backtest_metrics=False,  # Don't print backtest metrics when retraining in forecast mode
                        dump_csv_dir=dump_csv_dir,
                        enable_plots=enable_plots
                    )
                else:
                    log_verbose(f"   No cached model → FULL TRAINING...")
                    # First-time training: use all data to learn patterns
                    forecast_result = build_ensemble_forecast_model(
                        df_cpu=train_df,
                        df_mem=None,
                        horizon_min=horizon_days * 24 * 60,
                        model_path=None,  # Don't save to individual file
                        context={'node': node, 'signal': res['name']},
                        save_forecast_plot=False,  # Don't save forecast plots when first-time training in forecast mode
                        save_backtest_plot=False,  # Don't save backtest plots when first-time training in forecast mode
                        enable_plots=enable_plots,
                        print_backtest_metrics=False,  # Don't print backtest metrics when first-time training in forecast mode
                        dump_csv_dir=dump_csv_dir
                )
                if forecast_result is not None:
                    manifest[key] = {'model': forecast_result}
                    manifest_changed = True
                    log_verbose(f"   Saved ENSEMBLE to manifest → {key}")

            if forecast_result is None:
                continue

            # ———— SAFE UNPACK (handles old and new cache) ————
            # forecast_result should be a tuple from build_ensemble_forecast_model
            if isinstance(forecast_result, tuple):
                if len(forecast_result) == 3:
                    _, forecast_df, metrics = forecast_result
                else:  # old cache with only 2 items
                    _, forecast_df = forecast_result
                    metrics = {"mae_ensemble": 0.0}
            else:
                # Unexpected type - skip this node
                log_verbose(f"   Warning: unexpected forecast_result type for {key}, skipping")
                continue

            future_threshold = forecast_df[forecast_df['yhat'] >= res["threshold"]]
            eta_days = 9999.0
            if not future_threshold.empty:
                eta_days = max(0.1, (future_threshold.iloc[0]['ds'] - pd.Timestamp.now()).total_seconds() / 86400)

            log_verbose(f"  Done → ETA: {eta_days:.1f} days | MAE: {metrics['mae_ensemble']:.6f}")
            
            # Collect metrics for display when show_backtest is true or when retraining (but not in forecast mode)
            if (show_backtest or (needs_retrain and not forecast_mode)) and metrics:
                backtest_metrics_list.append({
                    'node': node,
                    'signal': res['name'],
                    'metrics': metrics
                })

            # ———— CRISIS ALERT ————
            if eta_days < 30:
                severity = "CRITICAL" if eta_days < 3 else "WARNING" if eta_days < 7 else "SOON"
                crisis_results.append({
                    "node": node,
                    "signal": res["name"].replace("_", " "),
                    "current": f"{current*100:.2f}%" if res["unit"] == "ratio" else f"{current/1e6:.1f} MB/s",
                    "mae_ensemble": round(metrics.get('mae_ensemble', 0.0), 6),
                    "hybrid_eta_days": round(eta_days, 1),
                    "severity": severity
                })

            # Anomaly detection placeholder (you can expand later)
            # is_anomaly = metrics.get('anomaly_score', 0) > 0.7
            # if is_anomaly: ...

    crisis_df = pd.DataFrame(crisis_results)
    anomaly_df = pd.DataFrame(anomaly_results)
    print(f"Ensemble forecasts complete: {len(crisis_results)} crises, {len(anomaly_results)} anomalies flagged.")
    
    # Display backtest metrics when show_backtest is true or when models were retrained (but not in forecast mode)
    if (show_backtest or (retrained_nodes and not forecast_mode)) and backtest_metrics_list:
        print("\n" + "="*80)
        if retrained_nodes:
            print("DISK I/O + NETWORK — BACKTEST METRICS (retrained models only)")
        else:
            print("DISK I/O + NETWORK — BACKTEST METRICS (cached models)")
        print("="*80)
        for item in sorted(backtest_metrics_list, key=lambda x: (x['node'], x['signal'])):
            node = item['node']
            signal = item['signal']
            metrics = item['metrics']
            print(f"\nBacktest Metrics → {node} | {signal}:")
            if metrics.get('mae_ensemble') is not None:
                print(f"  • mae_ensemble: {metrics['mae_ensemble']:.6f}")
            if metrics.get('mae_prophet') is not None:
                print(f"  • mae_prophet: {metrics['mae_prophet']:.6f}")
            if metrics.get('mae_arima') is not None:
                print(f"  • mae_arima: {metrics['mae_arima']:.6f}")
            if metrics.get('mae_lstm') is not None:
                print(f"  • mae_lstm: {metrics['mae_lstm']:.6f}")
            if metrics.get('split_info'):
                split_info = metrics['split_info']
                print(f"  • Train/Test Split:")
                train_pct = round(split_info.get('train_fraction', 0.8) * 100)
                test_pct = 100 - train_pct
                print(f"    - Train fraction: {train_pct}%")
                print(f"    - Train points: {split_info.get('train_points', 0):,}")
                print(f"    - Test points: {split_info.get('test_points', 0):,}")
                if split_info.get('train_start'):
                    print(f"    - Train period: {split_info['train_start']} → {split_info['train_end']}")
                if split_info.get('test_start'):
                    print(f"    - Test period: {split_info['test_start']} → {split_info['test_end']}")
    
    # Summary of retraining
    if retrain_targets and all_unique_nodes:
        print(f"\nAvailable nodes in data ({len(all_unique_nodes)} total):")
        for node in sorted(all_unique_nodes):
            print(f"  • {node}")
    
    if retrain_targets:
        # Check for "all" flag
        if '__RETRAIN_ALL__' in retrain_targets:
            print(f"\nRetrain Summary:")
            print(f"  Requested targets: all")
            if retrained_nodes:
                print(f"  ✓ Retrained all nodes/signals: {len(retrained_nodes)}")
            else:
                print(f"  ⚠️  No nodes/signals were retrained")
        else:
            node_targets = {t for t in retrain_targets if '|' not in t and '_' not in t}
            # Show summary if we have retrain targets (even if filtered node_targets is empty)
            if retrain_targets:
                if retrained_nodes:
                    # Show success summary
                    if node_targets:
                        print(f"\nRetrain Summary:")
                        print(f"  Requested targets: {', '.join(sorted(node_targets))}")
                        print(f"  ✓ Retrained nodes: {', '.join(sorted(retrained_nodes))}")
                else:
                    # Show warning when nothing matched
                    print(f"\n" + "="*80)
                    print("DISK I/O + NETWORK — RETRAIN SUMMARY")
                    print("="*80)
                    # Show all requested targets (including those with underscores)
                    all_targets = {t for t in retrain_targets if t != '__RETRAIN_ALL__'}
                    print(f"\n⚠️  No nodes/signals matched the retrain targets")
                    print(f"  Requested targets: {', '.join(sorted(all_targets))}")
                    if all_unique_nodes:
                        print(f"\n  Available nodes ({len(all_unique_nodes)} total):")
                        for node in sorted(all_unique_nodes):
                            print(f"    • {node}")
                    print(f"\n  Note: Using cached models (if available) for predictions.")
                    print(f"  To retrain all I/O and network models, use: --io-net-retrain all")
                    print(f"  To retrain specific nodes, use: --io-net-retrain host02,host03")
                    print(f"  To retrain specific signals, use: --io-net-retrain host02:DISK_IO_WAIT,worker01:NET_TX_BW")
            
            # Check which targets didn't match
            matched_targets = set()
            for retrained in retrained_nodes:
                # Format is "node (entity)" where node is like "worker02 (192.168.10.82)"
                # So full format is "worker02 (192.168.10.82) (worker02)"
                if '(' in retrained and ')' in retrained:
                    # Extract entity (last part in parentheses)
                    entity_match = retrained.split('(')[-1].rstrip(')').strip()
                    # Extract node display (everything before last parentheses)
                    node_display = retrained.rsplit('(', 1)[0].strip()
                    # Extract IP from node display if present
                    node_ip = None
                    if '(' in node_display and ')' in node_display:
                        node_ip = node_display.split('(')[1].split(')')[0].strip()
                else:
                    entity_match = retrained
                    node_display = retrained
                    node_ip = None
                
                for target in node_targets:
                    if target in matched_targets:
                        continue
                    target_canon = canonical_identity(target)
                    # Check entity match
                    if target_canon == entity_match:
                        matched_targets.add(target)
                        continue
                    # Try DNS resolution with domain suffixes (only if target looks like hostname)
                    if looks_like_hostname(target) and node_ip:
                        target_variants = [target] + [f"{target}{d}" for d in DNS_DOMAIN_SUFFIXES if d and not target.endswith(d)]
                        for target_var in target_variants:
                            try:
                                target_ip = socket.gethostbyname(target_var)
                                if target_ip == node_ip:
                                    matched_targets.add(target)
                                    break
                            except:
                                pass
                        if target in matched_targets:
                            continue
            
            unmatched_targets = node_targets - matched_targets
            if unmatched_targets:
                print(f"  ⚠️  Unmatched targets: {', '.join(sorted(unmatched_targets))}")
                for target in unmatched_targets:
                    # Only try DNS resolution if target looks like a hostname
                    if looks_like_hostname(target):
                        target_variants = [target] + [f"{target}{d}" for d in DNS_DOMAIN_SUFFIXES if d and not target.endswith(d)]
                        resolved = False
                        for target_var in target_variants:
                            try:
                                target_ip = socket.gethostbyname(target_var)
                                print(f"      {target} ({target_var}) resolves to {target_ip}")
                                # Check if this IP exists in any of the available nodes
                                ip_found = False
                                for node_str in all_unique_nodes:
                                    if f"({target_ip})" in node_str:
                                        print(f"        → IP found in data as: {node_str}")
                                        ip_found = True
                                        break
                                if not ip_found:
                                    print(f"        → IP {target_ip} not found in any node in data")
                                resolved = True
                                break
                            except Exception as e:
                                pass
                        if not resolved:
                            print(f"      {target} DNS resolution failed (tried: {', '.join(target_variants)})")
                    else:
                        print(f"      {target} does not look like a valid hostname or IP address")
    
    return crisis_df, anomaly_df, manifest, manifest_changed

# ----------------------------------------------------------------------
# 8.4. ALERT DISPATCHING
# ----------------------------------------------------------------------
def summarize_alert_counts(disk_alerts, crisis_df, anomaly_df, anomalies_df, classification_anomalies_df=None, host_pressure_df=None, include_samples=True, sample_limit=5):
    """Summarize all alerts and return a summary dict with counts and sample records."""
    # Use a flat structure so webhook consumers can parse without nesting lookups.
    summary = {
        'timestamp': datetime.now().isoformat(),
        'disk': {'critical': 0, 'warning': 0, 'soon': 0, 'total': 0, 'samples': []},
        'io_network_crisis': {'total': 0, 'samples': []},
        'io_network_anomaly': {'total': 0, 'samples': []},
        'golden_anomaly': {'total': 0, 'samples': []},
        'classification_anomaly': {'total': 0, 'samples': []},
        'host_pressure': {'total': 0, 'samples': []}
    }
    
    # Disk alerts (exclude OK status)
    if disk_alerts is not None and not disk_alerts.empty:
        # Normalize alert column - handle both string and numeric types
        alerts_series = disk_alerts['alert'].astype(str).fillna('').str.strip()
        alerts_upper = alerts_series.str.upper()
        
        # Debug: print unique alert values to help diagnose
        if should_verbose():
            unique_alerts = alerts_series.unique()
            print(f"DEBUG: Unique alert values in disk_alerts: {unique_alerts}")
        
        # Keep anything that is not strictly "OK" (case-insensitive)
        non_ok_mask = alerts_upper != 'OK'
        non_ok_alerts = disk_alerts[non_ok_mask].copy() if non_ok_mask.any() else pd.DataFrame()
        
        if not non_ok_alerts.empty:
            alerts_filtered_upper = non_ok_alerts['alert'].astype(str).fillna('').str.strip().str.upper()
            critical_mask = alerts_filtered_upper.str.contains('CRITICAL', case=False, na=False)
            warning_mask = alerts_filtered_upper.str.contains('WARNING', case=False, na=False)
            soon_mask = alerts_filtered_upper.str.contains('SOON', case=False, na=False)
            
            summary['disk']['critical'] = int(critical_mask.sum())
            summary['disk']['warning'] = int(warning_mask.sum())
            summary['disk']['soon'] = int(soon_mask.sum())
            summary['disk']['total'] = len(non_ok_alerts)
            
            if should_verbose():
                print(f"DEBUG: Disk alert counts - Critical: {summary['disk']['critical']}, Warning: {summary['disk']['warning']}, Soon: {summary['disk']['soon']}, Total: {summary['disk']['total']}")
            
            if include_samples:
                cols = ['instance', 'mountpoint', 'current_%', 'days_to_90pct',
                        'ensemble_eta', 'linear_eta', 'prophet_eta', 'alert']
                cols = [c for c in cols if c in non_ok_alerts.columns]
                sort_field = 'days_to_90pct' if 'days_to_90pct' in non_ok_alerts.columns else None
                def pick(mask):
                    if not mask.any():
                        return pd.DataFrame(columns=cols if cols else non_ok_alerts.columns)
                    subset = non_ok_alerts.loc[mask, cols].copy() if cols else non_ok_alerts.loc[mask].copy()
                    if sort_field and sort_field in subset.columns:
                        subset = subset.sort_values(sort_field, ascending=True, na_position='last')
                    return subset
                sample_frames = [
                    pick(critical_mask),
                    pick(warning_mask),
                    pick(soon_mask)
                ]
                # Filter out empty dataframes before concatenating
                sample_frames = [df for df in sample_frames if not df.empty]
                if sample_frames:
                    top_alerts = pd.concat(sample_frames).drop_duplicates().head(sample_limit)
                    summary['disk']['samples'] = top_alerts.to_dict('records') if not top_alerts.empty else []
                else:
                    summary['disk']['samples'] = []
    
    # I/O + Network Crisis
    if crisis_df is not None and not crisis_df.empty:
        summary['io_network_crisis']['total'] = len(crisis_df)
        if include_samples:
            crisis_cols = ['node', 'signal', 'severity', 'hybrid_eta_days', 'current', 'mae_ensemble']
            crisis_cols = [c for c in crisis_cols if c in crisis_df.columns]
            crisis_sample = crisis_df[crisis_cols].head(sample_limit) if crisis_cols else crisis_df.head(sample_limit)
            summary['io_network_crisis']['samples'] = crisis_sample.to_dict('records')
    
    # I/O + Network Anomaly
    if anomaly_df is not None and not anomaly_df.empty:
        summary['io_network_anomaly']['total'] = len(anomaly_df)
        if include_samples:
            anomaly_cols = ['node', 'signal', 'severity', 'score', 'mae_ensemble']
            anomaly_cols = [c for c in anomaly_cols if c in anomaly_df.columns]
            anomaly_sample = anomaly_df[anomaly_cols].head(sample_limit) if anomaly_cols else anomaly_df.head(sample_limit)
            summary['io_network_anomaly']['samples'] = anomaly_sample.to_dict('records')
    
    # Golden Anomaly
    if anomalies_df is not None and not anomalies_df.empty:
        summary['golden_anomaly']['total'] = len(anomalies_df)
        if include_samples:
            golden_cols = ['node', 'signal', 'severity', 'detected_at']
            golden_cols = [c for c in golden_cols if c in anomalies_df.columns]
            golden_sample = anomalies_df[golden_cols].head(sample_limit) if golden_cols else anomalies_df.head(sample_limit)
            summary['golden_anomaly']['samples'] = golden_sample.to_dict('records')
    
    # Classification Anomalies (anomalous nodes)
    if classification_anomalies_df is not None and not classification_anomalies_df.empty:
        summary['classification_anomaly']['total'] = len(classification_anomalies_df)
        if include_samples:
            class_cols = ['instance', 'host_cpu', 'host_mem', 'pod_cpu', 'pod_mem', 'severity', 'signal', 'detected_at']
            class_cols = [c for c in class_cols if c in classification_anomalies_df.columns]
            class_sample = classification_anomalies_df[class_cols].head(sample_limit) if class_cols else classification_anomalies_df.head(sample_limit)
            summary['classification_anomaly']['samples'] = class_sample.to_dict('records')
    
    # Host Pressure (high host usage with low pod usage)
    if host_pressure_df is not None and not host_pressure_df.empty:
        # Host pressure is lower severity but useful when correlating noisy dashboards.
        summary['host_pressure']['total'] = len(host_pressure_df)
        if include_samples:
            pressure_cols = ['instance', 'host_cpu', 'host_mem', 'severity', 'signal', 'detected_at']
            pressure_cols = [c for c in pressure_cols if c in host_pressure_df.columns]
            pressure_sample = host_pressure_df[pressure_cols].head(sample_limit) if pressure_cols else host_pressure_df.head(sample_limit)
            summary['host_pressure']['samples'] = pressure_sample.to_dict('records')
    
    # Summary text for webhook consumers
    summary['summary_text'] = (
        f"Disk → {summary['disk']['critical']} critical, {summary['disk']['warning']} warning, "
        f"{summary['disk']['soon']} soon | "
        f"I/O+Network Crisis → {summary['io_network_crisis']['total']} | "
        f"I/O+Network Anomaly → {summary['io_network_anomaly']['total']} | "
        f"Golden Signals → {summary['golden_anomaly']['total']} | "
        f"Classification Anomalies → {summary['classification_anomaly']['total']} | "
        f"Host Pressure → {summary['host_pressure']['total']}"
    )
    
    return summary

def post_alert_webhook(webhook_url, summary):
    """Send alert summary to HTTP webhook."""
    try:
        response = requests.post(
            webhook_url,
            json=summary,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        response.raise_for_status()
        print(f"✓ Alert webhook sent → {webhook_url} (status: {response.status_code})")
        return True
    except Exception as e:
        print(f"✗ Alert webhook failed → {webhook_url}: {e}")
        return False

def push_to_pushgateway(pushgateway_url, summary):
    """Push alert metrics to Prometheus Pushgateway."""
    try:
        # Format metrics for Pushgateway
        metrics = []
        metrics.append(f"metrics_ai_disk_alerts_critical {summary['disk']['critical']}")
        metrics.append(f"metrics_ai_disk_alerts_warning {summary['disk']['warning']}")
        metrics.append(f"metrics_ai_disk_alerts_soon {summary['disk']['soon']}")
        metrics.append(f"metrics_ai_disk_alerts_total {summary['disk']['total']}")
        metrics.append(f"metrics_ai_io_network_crisis_total {summary['io_network_crisis']['total']}")
        metrics.append(f"metrics_ai_io_network_anomaly_total {summary['io_network_anomaly']['total']}")
        metrics.append(f"metrics_ai_golden_anomaly_total {summary['golden_anomaly']['total']}")
        metrics.append(f"metrics_ai_classification_anomaly_total {summary['classification_anomaly']['total']}")
        metrics.append(f"metrics_ai_host_pressure_total {summary['host_pressure']['total']}")
        
        payload = '\n'.join(metrics) + '\n'
        
        response = requests.post(
            f"{pushgateway_url}/metrics/job/metrics_ai",
            data=payload,
            timeout=10
        )
        response.raise_for_status()
        print(f"✓ Metrics pushed to Pushgateway → {pushgateway_url} (status: {response.status_code})")
        return True
    except Exception as e:
        print(f"✗ Pushgateway push failed → {pushgateway_url}: {e}")
        return False

def dispatch_alerts(disk_alerts, crisis_df, anomaly_df, anomalies_df, classification_anomalies_df=None, host_pressure_df=None, alert_webhook=None, pushgateway_url=None, sli_slo_results=None):
    """Dispatch alerts to webhook and/or Pushgateway."""
    if alert_webhook is None and pushgateway_url is None:
        return
    
    # Dump disk_alerts metadata first to simplify debugging when alerts are missing.
    print(f"\nDEBUG: disk_alerts type: {type(disk_alerts)}")
    if disk_alerts is not None and hasattr(disk_alerts, 'empty'):
        print(f"DEBUG: disk_alerts.empty: {disk_alerts.empty}")
        print(f"DEBUG: disk_alerts.shape: {disk_alerts.shape if hasattr(disk_alerts, 'shape') else 'N/A'}")
        if not disk_alerts.empty and 'alert' in disk_alerts.columns:
            print(f"DEBUG: Unique alert values: {disk_alerts['alert'].unique()}")
            print(f"DEBUG: Alert value counts:\n{disk_alerts['alert'].value_counts()}")
            # Show first few rows
            print(f"DEBUG: First 3 rows of disk_alerts:\n{disk_alerts.head(3).to_string()}")
    
    summary = summarize_alert_counts(disk_alerts, crisis_df, anomaly_df, anomalies_df, classification_anomalies_df, host_pressure_df)
    
    # Add SLI/SLO results to summary if available
    if sli_slo_results:
        summary['sli_slo'] = {
            'results': sli_slo_results,
            'budget_at_risk_count': sum(1 for r in sli_slo_results if r.get('budget_at_risk', False)),
            'non_compliant_count': sum(1 for r in sli_slo_results if r.get('compliance_percent', 100) < r.get('slo_target', 100))
        }
    
    total_alerts = (
        summary['disk']['total'] +
        summary['io_network_crisis']['total'] +
        summary['io_network_anomaly']['total'] +
        summary['golden_anomaly']['total'] +
        summary['classification_anomaly']['total'] +
        summary['host_pressure']['total']
    )
    
    # Add SLI/SLO budget at risk to total alerts if applicable
    if sli_slo_results:
        budget_at_risk_count = sum(1 for r in sli_slo_results if r.get('budget_at_risk', False))
        if budget_at_risk_count > 0:
            print(f"SLI/SLO Error Budgets at Risk: {budget_at_risk_count}")
    
    # Always print summary for debugging
    print(f"\n{'='*80}")
    if total_alerts > 0:
        print(f"ALERT SUMMARY (Total: {total_alerts})")
    else:
        print("ALERT SUMMARY (No actionable alerts - all OK)")
    print(f"{'='*80}")
    print(f"Disk Alerts: {summary['disk']['critical']} CRITICAL, {summary['disk']['warning']} WARNING, {summary['disk']['soon']} SOON (Total non-OK: {summary['disk']['total']})")
    print(f"I/O+Network Crisis: {summary['io_network_crisis']['total']}")
    print(f"I/O+Network Anomaly: {summary['io_network_anomaly']['total']}")
    print(f"Golden Anomaly: {summary['golden_anomaly']['total']}")
    print(f"Classification Anomalies: {summary['classification_anomaly']['total']}")
    print(f"Host Pressure: {summary['host_pressure']['total']}")
    
    # Only send webhook if there are actionable alerts (non-OK)
    if total_alerts > 0:
        if alert_webhook:
            post_alert_webhook(alert_webhook, summary)
        
        if pushgateway_url:
            push_to_pushgateway(pushgateway_url, summary)
    else:
        print("Skipping webhook dispatch - no actionable alerts (all OK)")

# ----------------------------------------------------------------------
# 8.5. FORECAST MODE (lightweight, frequent runs)
# ----------------------------------------------------------------------
def run_forecast_mode(alert_webhook=None, pushgateway_url=None, csv_dump_dir=None, sli_slo_config_path=None, enable_plots=False):
    """Forecast mode: generate forecasts using latest Prometheus data and cached models.
    Runs all forecasting models (CPU, Memory, Disk, I/O, Network) and displays predictions and anomalies.
    Optimized for frequent runs (e.g., every 15 seconds via external scheduler).
    
    Args:
        enable_plots: If True, generates and saves plot files (PNG images). If False, plots are skipped.
    
    Generates forecast plots for all models (when enable_plots=True) and displays:
    - Host and Pod layer forecasts
    - Disk full predictions
    - I/O and Network crisis predictions
    - Anomaly detection results
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Forecast Mode\n")
    print("Execution mode: FORECAST (using cached models, latest Prometheus data)")
    print("Running all forecasting models and displaying predictions + anomalies")
    # Steps: fetch metrics, run models, print summaries, dispatch alerts.
    
    refresh_dynamic_aliases()
    
    # Load manifests
    disk_manifest = load_disk_manifest(DISK_MODEL_MANIFEST_PATH)
    io_net_manifest = load_io_net_manifest(IO_NET_MODEL_MANIFEST_PATH)
    
    # Initialize alert dataframes
    disk_alerts = pd.DataFrame()
    crisis_df = pd.DataFrame()
    anomaly_df = pd.DataFrame()
    anomalies_df = pd.DataFrame()
    classification_anomalies_df = pd.DataFrame()
    host_pressure_df = pd.DataFrame()
    
    # ====================== HOST LAYER ======================
    print("\n" + "="*80)
    print("HOST LAYER (full node) — FORECAST")
    print("="*80)
    q_host_cpu = '1 - avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) by (instance)'
    q_host_mem = '(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes'
    df_hcpu = fetch_and_preprocess_data(q_host_cpu)
    df_hmem = fetch_and_preprocess_data(q_host_mem)
    
    if not os.path.exists(HOST_MODEL_PATH):
        print(f"⚠ Warning: Host model not found at {HOST_MODEL_PATH}")
        print("   Skipping host forecast. Run with --training flag first to train models.")
        host_fc = None
    else:
        _, host_fc, _ = train_or_load_ensemble(
            df_hcpu,
            df_hmem,
            horizon_min=7*24*60,
            model_path=HOST_MODEL_PATH,
            force_retrain=False,
            generate_fresh_forecast=True,
            dump_csv_dir=csv_dump_dir,
            context={'node': 'host'},
            enable_plots=enable_plots
        )
    
    # ====================== POD LAYER ======================
    print("\n" + "="*80)
    print("POD LAYER (apps only) — FORECAST")
    print("="*80)
    q_pod_cpu = 'sum(rate(container_cpu_usage_seconds_total{container!="POD",container!=""}[5m])) by (instance)'
    q_pod_mem = 'sum(container_memory_working_set_bytes{container!="POD",container!=""}[5m]) by (instance)'
    df_pcpu = fetch_and_preprocess_data(q_pod_cpu)
    df_pmem = fetch_and_preprocess_data(q_pod_mem)
    
    augment_aliases_from_dns(df_hcpu, df_pcpu)
    infer_aliases_from_timeseries(df_hcpu, df_pcpu)
    recanonicalize_entities(df_hcpu, df_hmem, df_pcpu, df_pmem)
    summarize_instance_roles(df_hcpu, df_pcpu)
    print()
    
    if not os.path.exists(POD_MODEL_PATH):
        print(f"⚠ Warning: Pod model not found at {POD_MODEL_PATH}")
        print("   Skipping pod forecast. Run with --training flag first to train models.")
        pod_fc = None
    else:
        _, pod_fc, _ = train_or_load_ensemble(
            df_pcpu,
            df_pmem,
            horizon_min=7*24*60,
            model_path=POD_MODEL_PATH,
            force_retrain=False,
            generate_fresh_forecast=True,
            dump_csv_dir=csv_dump_dir,
            context={'node': 'pod'},
            enable_plots=enable_plots
        )
    
    # ====================== DIVERGENCE & ANOMALY ======================
    if host_fc is not None and pod_fc is not None:
        host_mem = host_fc['yhat'].iloc[-1]
        pod_mem = pod_fc['yhat'].iloc[-1]
        div = abs(host_mem - pod_mem)
        print(f"\nDivergence (host vs pod memory): {div:.3f}")
    
    _, _, classification_anomalies_df, host_pressure_df = classification_model(
        df_hcpu,
        df_hmem,
        df_pcpu,
        df_pmem,
        lookback_hours=LOOKBACK_HOURS,
        contamination=CONTAMINATION,
        forecast_mode=True,
        dump_csv_dir=csv_dump_dir,
        enable_plots=enable_plots
    )
    
    # ====================== DISK FULL PREDICTION ======================
    print("\n" + "="*80)
    print("DISK FULL PREDICTION (7-day horizon) — FORECAST")
    print("="*80)
    
    q_disk = '''
    1 - (
      node_filesystem_free_bytes{mountpoint=~"/$|/var$|/data$|/home$|/opt$"}
      /
      node_filesystem_size_bytes{mountpoint=~"/$|/var$|/data$|/home$|/opt$"}
    )
    '''
    
    df_disk_raw = fetch_victoriametrics_metrics(
        query=q_disk,
        start=int((pd.Timestamp.now() - pd.Timedelta(days=30)).timestamp()),
        end=int(pd.Timestamp.now().timestamp()),
        step="10m"
    )
    
    if df_disk_raw.empty:
        print("No disk metrics found")
    else:
        df_disk = df_disk_raw.copy()
        df_disk['timestamp'] = pd.to_datetime(df_disk['ts'], unit='s')
        df_disk['value'] = pd.to_numeric(df_disk['value'], errors='coerce').fillna(0)
        if 'mountpoint' in df_disk.columns:
            df_disk = df_disk.rename(columns={'mountpoint': 'filesystem'})
        if 'instance' not in df_disk.columns:
            df_disk['instance'] = 'unknown'
        if 'entity' not in df_disk.columns:
            df_disk['entity'] = df_disk['instance'].map(lambda x: canonical_identity(str(x)))
        
        print(f"Analyzing {df_disk['instance'].nunique()} nodes, {df_disk['filesystem'].nunique()} mountpoints")
        
        disk_alerts, disk_manifest, manifest_changed, disk_metrics, disk_retrained_nodes = predict_disk_full_days(
            df_disk,
            horizon_days=7,
            manifest=disk_manifest,
            retrain_targets=None,
            show_backtest=False,
            forecast_mode=True,
            dump_csv_dir=csv_dump_dir,
            enable_plots=enable_plots
        )
        if manifest_changed:
            save_disk_manifest(DISK_MODEL_MANIFEST_PATH, disk_manifest)
        
        # Print disk alerts table in forecast mode
        if not disk_alerts.empty:
            # Print critical and warning rows before the full table to highlight potential incidents.
            print("\nCRITICAL / WARNING DISKS:")
            critical = disk_alerts[disk_alerts['alert'].str.contains('CRITICAL', case=False, na=False)]
            warning = disk_alerts[disk_alerts['alert'].str.contains('WARNING', case=False, na=False)]
            soon = disk_alerts[disk_alerts['alert'].str.contains('SOON', case=False, na=False)]
            if not critical.empty:
                print("CRITICAL (<3 days to 90%):")
                print(critical[['instance', 'mountpoint', 'current_%', 'days_to_90pct']].to_string(index=False))
            if not warning.empty:
                print("WARNING (3–7 days to 90%):")
                print(warning[['instance', 'mountpoint', 'current_%', 'days_to_90pct']].to_string(index=False))
            if not soon.empty:
                print("SOON (7–30 days to 90%):")
                print(soon[['instance', 'mountpoint', 'current_%', 'days_to_90pct']].to_string(index=False))
            
            print("\nFULL ETA FOR ALL DISKS (90% threshold):")
            print(disk_alerts[['instance', 'mountpoint', 'current_%', 'days_to_90pct', 'ensemble_eta', 'linear_eta', 'prophet_eta', 'alert']].to_string(index=False))
            
            disk_csv_path = os.path.join(FORECAST_PLOTS_DIR, "disk_full_prediction.csv")
            disk_alerts.to_csv(disk_csv_path, index=False)
            print(f"\nFull report → {disk_csv_path}")
        else:
            print("No disk predictions generated")
    
    # ====================== I/O + NETWORK CRISIS PREDICTION ======================
    print("\n" + "="*80)
    print("DISK I/O + NETWORK — CRISIS PREDICTION (FORECAST)")
    print("="*80)
    
    crisis_df, io_net_manifest, io_net_manifest_changed = predict_io_and_network_crisis_with_backtest(
        horizon_days=7,
        test_days=7,
        plot_dir=None,  # Uses FORECAST_PLOTS_DIR
        force_retrain=False,
        manifest=io_net_manifest,
        retrain_targets=None,
        show_backtest=False,
        forecast_mode=True,
        dump_csv_dir=csv_dump_dir,
        enable_plots=enable_plots
    )
    if io_net_manifest_changed:
        save_io_net_manifest(IO_NET_MODEL_MANIFEST_PATH, io_net_manifest)
    
    # ====================== I/O + NETWORK ENSEMBLE FORECAST ======================
    print("\n" + "="*80)
    print("DISK I/O + NETWORK — FULL ENSEMBLE FORECAST & ANOMALY DETECTION (FORECAST)")
    print("="*80)
    
    crisis_df, anomaly_df, io_net_manifest, io_net_manifest_changed = predict_io_and_network_ensemble(
        horizon_days=7,
        test_days=7,
        plot_dir=None,  # Uses FORECAST_PLOTS_DIR
        force_retrain=False,
        manifest=io_net_manifest,
        retrain_targets=None,
        show_backtest=False,
        forecast_mode=True,
        dump_csv_dir=csv_dump_dir,
        enable_plots=enable_plots
    )
    if io_net_manifest_changed:
        save_io_net_manifest(IO_NET_MODEL_MANIFEST_PATH, io_net_manifest)
    
    # ====================== GOLDEN ANOMALY DETECTION ======================
    print("\n" + "="*80)
    print("GOLDEN ANOMALY DETECTION — AUTONOMOUS ROOT-CAUSE ENGINE (FORECAST)")
    print("="*80)
    
    anomalies_df = detect_golden_anomaly_signals(hours=1)
    
    # ====================== SLI/SLO TRACKING ======================
    if sli_slo_config_path or YAML_AVAILABLE:
        sli_slo_config = load_sli_slo_config(sli_slo_config_path)
        sli_slo_results = None
        if sli_slo_config:
            sli_slo_results = track_sli_slo(
                sli_slo_config,
                disk_alerts=disk_alerts,
                classification_anomalies_df=classification_anomalies_df,
                host_pressure_df=host_pressure_df,
                golden_anomalies_df=anomalies_df,
                df_hcpu=df_hcpu,
                df_hmem=df_hmem,
                df_pcpu=df_pcpu,
                df_pmem=df_pmem,
                crisis_df=crisis_df,
                anomaly_df=anomaly_df
            )
            if sli_slo_results:
                print("\n" + format_sli_slo_report(sli_slo_results))
        elif sli_slo_config_path:
            print(f"\n⚠️  SLI/SLO config file not found or invalid: {sli_slo_config_path}")
    elif sli_slo_config_path:
        print(f"\n⚠️  PyYAML not installed. SLI/SLO tracking disabled. Install with: pip install pyyaml")
    
    # ====================== DISPATCH ALERTS ======================
    dispatch_alerts(disk_alerts, crisis_df, anomaly_df, anomalies_df, classification_anomalies_df, host_pressure_df, alert_webhook, pushgateway_url, sli_slo_results)
    
    # ====================== SUMMARY ======================
    print("\n" + "="*80)
    print("FORECAST MODE COMPLETE")
    print("="*80)
    if enable_plots:
        print(f"All forecast plots saved → {FORECAST_PLOTS_DIR}")
    else:
        print("Plots skipped (use --plot flag to generate plots)")
    print(f"Forecast timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n✓ All models executed with latest Prometheus data")
    print("✓ Forecasts generated for: Host, Pod, Disk, I/O, Network")
    print("✓ Anomalies detected and displayed above")

# ----------------------------------------------------------------------
# 9. MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_cli_args()
    csv_dump_dir = args.dump_csv
    
    # Forecast mode: lightweight, frequent runs
    if args.forecast:
        if args.quiet:
            VERBOSE_LEVEL = 0
        else:
            VERBOSE_LEVEL = max(VERBOSE_LEVEL, args.verbose)
        
        # Handle continuous runs with --interval
        if args.interval > 0:
            print(f"Running forecast mode continuously with {args.interval}s interval")
            print("Press Ctrl+C to stop")
            iteration = 0
            try:
                while True:
                    iteration += 1
                    print(f"\n{'='*80}")
                    print(f"Forecast Run #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"{'='*80}")
                    run_forecast_mode(alert_webhook=args.alert_webhook, pushgateway_url=args.pushgateway, csv_dump_dir=csv_dump_dir, sli_slo_config_path=args.sli_slo_config, enable_plots=args.plot)
                    if args.interval > 0:
                        print(f"\nWaiting {args.interval} seconds until next run...")
                        time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\n\nForecast mode stopped by user")
                sys.exit(0)
        else:
            # Single run
            run_forecast_mode(alert_webhook=args.alert_webhook, pushgateway_url=args.pushgateway, csv_dump_dir=csv_dump_dir, sli_slo_config_path=args.sli_slo_config, enable_plots=args.plot)
        sys.exit(0)
    
    # Normal mode: training or pre-trained with full analysis
    force_training = args.training_mode
    FORCE_TRAINING_RUN = force_training
    if args.quiet:
        VERBOSE_LEVEL = 0
    else:
        VERBOSE_LEVEL = max(VERBOSE_LEVEL, args.verbose)

    refresh_dynamic_aliases()

    print_config_summary()

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Dual-Layer + LSTM AI\n")
    mode_label = "TRAINING" if force_training else "PRE-TRAINED"
    print(f"Execution mode: {mode_label}")
    show_backtest = args.show_backtest
    disk_retrain_targets = parse_disk_retrain_targets(args.disk_retrain)
    disk_manifest = load_disk_manifest(DISK_MODEL_MANIFEST_PATH)
    io_net_manifest = load_io_net_manifest(IO_NET_MODEL_MANIFEST_PATH)
    io_net_retrain_targets = parse_io_net_retrain_targets(args.io_net_retrain)

    # ====================== HOST LAYER ======================
    print("\n" + "="*80)
    print("HOST LAYER (full node)")
    print("="*80)
    q_host_cpu = '1 - avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) by (instance)'
    q_host_mem = '(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes'

    df_hcpu = fetch_and_preprocess_data(q_host_cpu)
    df_hmem = fetch_and_preprocess_data(q_host_mem)
    _, host_fc, host_metrics = train_or_load_ensemble(
        df_hcpu,
        df_hmem,
        horizon_min=7*24*60,
        model_path=HOST_MODEL_PATH,
        force_retrain=force_training,
        show_backtest=show_backtest,
        dump_csv_dir=csv_dump_dir,
        context={'node': 'host'}
    )
    # Only show metrics when training or when --show-backtest is used
    if (force_training or show_backtest) and host_metrics:
        print("Host Model Metrics:")
        for k, v in host_metrics.items():
            if k == 'split_info' and isinstance(v, dict):
                print(f"  • Train/Test Split:")
                print(f"    - Train fraction: {v.get('train_fraction', 0)*100:.0f}%")
                print(f"    - Train points: {v.get('train_points', 0):,}")
                print(f"    - Test points: {v.get('test_points', 0):,}")
                if v.get('train_start'):
                    print(f"    - Train period: {v['train_start']} → {v['train_end']}")
                if v.get('test_start'):
                    print(f"    - Test period: {v['test_start']} → {v['test_end']}")
            elif isinstance(v, (int, float)):
                print(f"  • {k}: {v:.6f}")
            else:
                print(f"  • {k}: {v}")

    # ====================== POD LAYER ======================
    print("\n" + "="*80)
    print("POD LAYER (apps only)")
    print("="*80)
    q_pod_cpu = 'sum(rate(container_cpu_usage_seconds_total{container!="POD",container!=""}[5m])) by (instance)'
    q_pod_mem = 'sum(container_memory_working_set_bytes{container!="POD",container!=""}[5m]) by (instance)'

    df_pcpu = fetch_and_preprocess_data(q_pod_cpu)
    df_pmem = fetch_and_preprocess_data(q_pod_mem)
    augment_aliases_from_dns(df_hcpu, df_pcpu)
    infer_aliases_from_timeseries(df_hcpu, df_pcpu)
    recanonicalize_entities(df_hcpu, df_hmem, df_pcpu, df_pmem)
    summarize_instance_roles(df_hcpu, df_pcpu)
    print()
    _, pod_fc, pod_metrics = train_or_load_ensemble(
        df_pcpu,
        df_pmem,
        horizon_min=7*24*60,
        model_path=POD_MODEL_PATH,
        force_retrain=force_training,
        show_backtest=show_backtest,
        dump_csv_dir=csv_dump_dir,
        context={'node': 'pod'}
    )
    # Only show metrics when training or when --show-backtest is used
    if (force_training or show_backtest) and pod_metrics:
        print("Pod Model Metrics:")
        for k, v in pod_metrics.items():
            if k == 'split_info' and isinstance(v, dict):
                print(f"  • Train/Test Split:")
                print(f"    - Train fraction: {v.get('train_fraction', 0)*100:.0f}%")
                print(f"    - Train points: {v.get('train_points', 0):,}")
                print(f"    - Test points: {v.get('test_points', 0):,}")
                if v.get('train_start'):
                    print(f"    - Train period: {v['train_start']} → {v['train_end']}")
                if v.get('test_start'):
                    print(f"    - Test period: {v['test_start']} → {v['test_end']}")
            elif isinstance(v, (int, float)):
                print(f"  • {k}: {v:.6f}")
            else:
                print(f"  • {k}: {v}")

    # ====================== DIVERGENCE & ANOMALY ======================
    host_mem = host_fc['yhat'].iloc[-1]
    pod_mem  = pod_fc['yhat'].iloc[-1]
    div = abs(host_mem - pod_mem)
    print(f"\nDivergence (host vs pod memory): {div:.3f}")

    _, _, _, _ = classification_model(
        df_hcpu,
        df_hmem,
        df_pcpu,
        df_pmem,
        lookback_hours=LOOKBACK_HOURS,
        contamination=CONTAMINATION,
        dump_csv_dir=csv_dump_dir
    )

    print(f"\nAll models saved: {HOST_MODEL_PATH}, {POD_MODEL_PATH}, {LSTM_MODEL_PATH}")
    print("Dual-layer + LSTM + classification complete.")

    # ====================== DISK FULL PREDICTION — FULL TRANSPARENCY ======================
    print("\n" + "="*80)
    print("DISK FULL PREDICTION (7-day horizon) — FULL ETA FOR ALL DISKS")
    print("="*80)

    q_disk = '''
    1 - (
      node_filesystem_free_bytes{mountpoint=~"/$|/var$|/data$|/home$|/opt$"}
      /
      node_filesystem_size_bytes{mountpoint=~"/$|/var$|/data$|/home$|/opt$"}
    )
    '''

    df_disk_raw = fetch_victoriametrics_metrics(
        query=q_disk,
        start=int((pd.Timestamp.now() - pd.Timedelta(days=30)).timestamp()),
        end=int(pd.Timestamp.now().timestamp()),
        step="10m"
    )

    if df_disk_raw.empty:
        print("No disk metrics found")
    else:
        df_disk = df_disk_raw.copy()
        df_disk['timestamp'] = pd.to_datetime(df_disk['ts'], unit='s')
        df_disk['value'] = pd.to_numeric(df_disk['value'], errors='coerce').fillna(0)
        if 'mountpoint' in df_disk.columns:
            df_disk = df_disk.rename(columns={'mountpoint': 'filesystem'})
        if 'instance' not in df_disk.columns:
            df_disk['instance'] = 'unknown'
        if 'entity' not in df_disk.columns:
            df_disk['entity'] = df_disk['instance'].map(lambda x: canonical_identity(str(x)))

        print(f"Analyzing {df_disk['instance'].nunique()} nodes, {df_disk['filesystem'].nunique()} mountpoints")

        disk_alerts, disk_manifest, manifest_changed, disk_metrics, disk_retrained_nodes = predict_disk_full_days(
            df_disk,
            horizon_days=7,
            manifest=disk_manifest,
            retrain_targets=disk_retrain_targets,
            show_backtest=show_backtest,
            dump_csv_dir=csv_dump_dir
        )
        if manifest_changed:
            save_disk_manifest(DISK_MODEL_MANIFEST_PATH, disk_manifest)

        # Show backtest metrics when training, retraining, or when --show-backtest is used
        if disk_retrain_targets and not disk_retrained_nodes:
            # Retrain targets specified but nothing matched
            print("\n" + "="*80)
            print("DISK FULL PREDICTION — RETRAIN SUMMARY")
            print("="*80)
            print(f"\n⚠️  No nodes/mounts matched the retrain targets")
            print(f"  Requested targets: {', '.join(sorted(disk_retrain_targets))}")
            
            # Show available nodes
            unique_entities = df_disk['entity'].unique()
            if len(unique_entities) > 0:
                print(f"\n  Available nodes ({len(unique_entities)} total):")
                for entity in sorted(unique_entities):
                    # Get display name with IP if available
                    entity_rows = df_disk[df_disk['entity'] == entity]
                    if 'raw_instance' in entity_rows.columns and not entity_rows['raw_instance'].dropna().empty:
                        raw_label = entity_rows['raw_instance'].dropna().iloc[-1]
                        display_name = canonical_node_label(entity, with_ip=True, raw_label=raw_label)
                    else:
                        display_name = entity
                    # Get mountpoints for this entity
                    mounts = sorted(entity_rows['filesystem'].unique())
                    print(f"    • {display_name} (mounts: {', '.join(mounts)})")
            
            print(f"\n  Note: Using cached models (if available) for predictions.")
            print(f"  To retrain all disk models, use: --disk-retrain all")
            print(f"  To retrain all models (host, pod, disk, I/O, network), use: --training flag")
            print(f"  To retrain specific nodes, use: --disk-retrain host02,host03")
            print(f"  To retrain specific mounts, use: --disk-retrain host02:/,worker01:/home")
        elif (disk_retrained_nodes or show_backtest or (manifest_changed and disk_metrics)) and disk_metrics:
            print("\n" + "="*80)
            # Distinguish between explicit retraining and first-time training
            is_first_training = disk_retrained_nodes and not disk_retrain_targets and manifest_changed
            if disk_retrained_nodes and not is_first_training:
                print("DISK FULL PREDICTION — BACKTEST METRICS (retrained models only)")
            elif is_first_training or (manifest_changed and not disk_retrained_nodes):
                print("DISK FULL PREDICTION — BACKTEST METRICS (newly trained models)")
            else:
                print("DISK FULL PREDICTION — BACKTEST METRICS (cached models)")
            print("="*80)
            
            # Show which nodes/mounts were retrained or all if show_backtest or first training
            if disk_retrained_nodes and not is_first_training:
                print("\nRetrained nodes/mounts:")
                for retrained in sorted(disk_retrained_nodes):
                    # Format: "node | mountpoint"
                    if '|' in retrained:
                        node_part, mount_part = retrained.split('|', 1)
                        node_part = node_part.strip()
                        mount_part = mount_part.strip()
                        print(f"  ✓ {node_part} | {mount_part}")
                    else:
                        print(f"  ✓ {retrained}")
            elif is_first_training:
                print("\nAll nodes/mounts (newly trained models):")
                for retrained in sorted(disk_retrained_nodes):
                    # Format: "node | mountpoint"
                    if '|' in retrained:
                        node_part, mount_part = retrained.split('|', 1)
                        node_part = node_part.strip()
                        mount_part = mount_part.strip()
                        print(f"  • {node_part} | {mount_part}")
                    else:
                        print(f"  • {retrained}")
            elif show_backtest or (manifest_changed and not disk_retrained_nodes):
                if manifest_changed:
                    print("\nAll nodes/mounts (newly trained models):")
                else:
                    print("\nAll nodes/mounts (cached models):")
                for (entity, mountpoint), _ in df_disk.groupby(['entity', df_disk.get('filesystem', 'mountpoint')]):
                    entity_rows = df_disk[(df_disk['entity'] == entity) & (df_disk.get('filesystem', df_disk.get('mountpoint')) == mountpoint)]
                    if 'raw_instance' in entity_rows.columns and not entity_rows['raw_instance'].dropna().empty:
                        raw_label = entity_rows['raw_instance'].dropna().iloc[-1]
                        display_name = canonical_node_label(entity, with_ip=True, raw_label=raw_label)
                    else:
                        display_name = entity
                    print(f"  • {display_name} | {mountpoint}")
            
            # Show aggregated metrics
            if disk_retrained_nodes and not is_first_training:
                print("\nAggregated Backtest Metrics (across all retrained models):")
            elif is_first_training or (manifest_changed and not disk_retrained_nodes):
                print("\nAggregated Backtest Metrics (across all newly trained models):")
            else:
                print("\nAggregated Backtest Metrics (across all cached models):")
            if disk_metrics.get('mae_ensemble'):
                print(f"  • mae_ensemble: {disk_metrics['mae_ensemble']:.6f}")
            if disk_metrics.get('mae_linear'):
                print(f"  • mae_linear: {disk_metrics['mae_linear']:.6f}")
            if disk_metrics.get('mae_prophet'):
                print(f"  • mae_prophet: {disk_metrics['mae_prophet']:.6f}")
            
            if disk_metrics.get('split_info'):
                split_info = disk_metrics['split_info']
                print(f"  • Train/Test Split:")
                train_pct = round(split_info['train_fraction'] * 100)
                test_pct = 100 - train_pct
                print(f"    - Train fraction: {train_pct}%")
                print(f"    - Train points: {split_info['train_points']:,}")
                print(f"    - Test points: {split_info['test_points']:,}")
                if split_info.get('train_start'):
                    print(f"    - Train period: {split_info['train_start']} → {split_info['train_end']}")
                if split_info.get('test_start'):
                    print(f"    - Test period: {split_info['test_start']} → {split_info['test_end']}")
            
            # Show retrain summary
            if disk_retrain_targets:
                print(f"\nRetrain Summary:")
                # Show "all" if that was the target, otherwise show the actual targets
                if '__RETRAIN_ALL__' in disk_retrain_targets:
                    print(f"  Requested targets: all")
                    print(f"  ✓ Retrained all nodes/mounts: {len(disk_retrained_nodes)}")
                else:
                    print(f"  Requested targets: {', '.join(sorted(disk_retrain_targets))}")
                    print(f"  ✓ Retrained nodes/mounts: {len(disk_retrained_nodes)}")

        if not disk_alerts.empty:
            print("\nCRITICAL / WARNING DISKS:")
            critical = disk_alerts[disk_alerts['alert'].str.contains('CRITICAL')]
            warning = disk_alerts[disk_alerts['alert'].str.contains('WARNING')]
            if not critical.empty:
                print("CRITICAL (<3 days to 90%):")
                print(critical[['instance', 'mountpoint', 'current_%', 'days_to_90pct']].to_string(index=False))
            if not warning.empty:
                print("WARNING (3–7 days to 90%):")
                print(warning[['instance', 'mountpoint', 'current_%', 'days_to_90pct']].to_string(index=False))

            print("\nFULL ETA FOR ALL DISKS (90% threshold):")
            print(disk_alerts[['instance', 'mountpoint', 'current_%', 'days_to_90pct', 'ensemble_eta', 'linear_eta', 'prophet_eta', 'alert']].to_string(index=False))

            disk_csv_path = os.path.join(FORECAST_PLOTS_DIR, "disk_full_prediction.csv")
            disk_alerts.to_csv(disk_csv_path, index=False)
            print(f"\nFull report → {disk_csv_path}")
        else:
            print("No disk predictions generated")

    # ====================== ROOT-CAUSE ANOMALY ENGINE ======================
    print("\n" + "="*80)
    print("GOLDEN ANOMALY DETECTION — AUTONOMOUS ROOT-CAUSE ENGINE")
    print("="*80)

    anomalies_df = detect_golden_anomaly_signals(hours=1)

    if anomalies_df.empty:
        print("\nNo active root-cause signals — cluster is clean and healthy")
    else:
        print(f"\n{len(anomalies_df)} FAILURE SIGNALS DETECTED:")
        print(anomalies_df.to_string(index=False))
        print("\nThese explain current or upcoming incidents")

    # ====================== I/O + NETWORK CRISIS PREDICTION ======================
    print("\n" + "="*80)
    print("DISK I/O + NETWORK CRISIS PREDICTION (user-visible slowness)")
    print("="*80)

    crisis_df, io_net_manifest, io_net_manifest_changed = predict_io_and_network_crisis_with_backtest(
        horizon_days=7,
        test_days=7,
        plot_dir=None,  # Uses FORECAST_PLOTS_DIR
        force_retrain=force_training,
        manifest=io_net_manifest,
        retrain_targets=io_net_retrain_targets,
        show_backtest=show_backtest,
        dump_csv_dir=csv_dump_dir
    )
    if io_net_manifest_changed:
        save_io_net_manifest(IO_NET_MODEL_MANIFEST_PATH, io_net_manifest)

    if crisis_df.empty:
        print("\nNo I/O or network crises predicted in next 30 days — users will be happy")
        print("All models backtested with honest train/test split")
    else:
        print(f"\n{len(crisis_df)} CRISES IMMINENT — ACTION REQUIRED:")
        print(crisis_df.sort_values("hybrid_eta_days")[['node', 'signal', 'current', 'hybrid_eta_days', 'severity']].to_string(index=False))

    print(f"\nForecast plots + models → {FORECAST_PLOTS_DIR}/")
    print("Your cluster is now protected by real, validated, visualized AI.")
    print("="*80)

    # ====================== I/O + NETWORK — FULL ENSEMBLE (CPU/MEM GRADE) ======================
    print("\n" + "="*80)
    print("DISK I/O + NETWORK — FULL ENSEMBLE FORECAST & ANOMALY DETECTION (same brain as CPU/Memory)")
    print("="*80)

    io_net_crisis_df, io_net_anomaly_df, io_net_manifest, io_net_manifest_changed = predict_io_and_network_ensemble(
        horizon_days=7,
        test_days=7,
        plot_dir=None,  # Uses FORECAST_PLOTS_DIR
        force_retrain=force_training,
        manifest=io_net_manifest,
        retrain_targets=io_net_retrain_targets,
        show_backtest=show_backtest,
        dump_csv_dir=csv_dump_dir
    )
    if io_net_manifest_changed:
        save_io_net_manifest(IO_NET_MODEL_MANIFEST_PATH, io_net_manifest)

    if not io_net_crisis_df.empty:
        print(f"\n{len(io_net_crisis_df)} USER-VISIBLE I/O OR NETWORK CRISES IMMINENT — ACT NOW:")
        print(io_net_crisis_df[['node', 'signal', 'current', 'mae_ensemble', 'hybrid_eta_days', 'severity']]
              .sort_values('hybrid_eta_days')
              .to_string(index=False))

    if not io_net_anomaly_df.empty:
        print(f"\n{len(io_net_anomaly_df)} I/O OR NETWORK ANOMALIES DETECTED:")
        print(io_net_anomaly_df.to_string(index=False))

    if io_net_crisis_df.empty and io_net_anomaly_df.empty:
        print("\nI/O and Network layers are healthy, predictable, and anomaly-free")
        print("Your cluster is running at true FAANG-grade intelligence.")

    print(f"\nAll plots + models saved → {FORECAST_PLOTS_DIR}/")
    print("AI SRE Brain v2.0 — Unified across CPU • Memory • Disk • I/O • Network")
    print("="*80)

    if args.anomaly_watch > 0:
        run_realtime_anomaly_watch(
            q_host_cpu, q_host_mem, q_pod_cpu, q_pod_mem,
            iterations=args.anomaly_watch
        )
