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
from joblib import Parallel, delayed
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

# VM_URL must be set via environment variable - no hardcoded default
# Example: export VM_URL="http://vm.example.com/api/v1/query_range"
VM_BASE_URL = os.getenv("VM_URL")
if not VM_BASE_URL:
    print("ERROR: VM_URL environment variable is not set!")
    print("Please set it to your VictoriaMetrics/Prometheus URL, e.g.:")
    print('  export VM_URL="http://vm.example.com/api/v1/query_range"')
    print("Exiting...")
    sys.exit(1)
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
K8S_COMBINED_MODEL_PATH = get_model_path("K8S_COMBINED_MODEL_PATH", "k8s_combined_forecast.pkl")
STANDALONE_MODEL_PATH = get_model_path("STANDALONE_MODEL_PATH", "standalone_forecast.pkl")
LSTM_MODEL_PATH = get_model_path("LSTM_MODEL_PATH", "lstm_model.pkl")
DISK_MODEL_MANIFEST_PATH = get_model_path("DISK_MODEL_MANIFEST_PATH", "disk_full_models.pkl")
IO_NET_MODEL_MANIFEST_PATH = get_model_path("IO_NET_MODEL_MANIFEST_PATH", "io_net_models.pkl")
AUTO_ALIAS_ENABLED = os.getenv("AUTO_ALIAS_ENABLED", "1").lower() not in ("0", "false", "no")
ALIAS_LOOKBACK_MINUTES = int(os.getenv("ALIAS_LOOKBACK_MINUTES", "15"))
VERBOSE_LEVEL = int(os.getenv("VERBOSE_LEVEL", "0"))
# DNS domain suffixes to try when resolving hostnames (comma-separated)
DNS_DOMAIN_SUFFIXES = [d.strip() for d in os.getenv("DNS_DOMAIN_SUFFIXES", ".london.local,.local").split(",") if d.strip()]
FORCE_TRAINING_RUN = False

# ----------------------------------------------------------------------
# CPU Detection and Resource Limits (Kubernetes/Docker aware)
# ----------------------------------------------------------------------
def get_cpu_limit_from_cgroups():
    """
    Detect CPU limits from cgroups (Kubernetes/Docker).
    Returns the CPU limit in cores (float) or None if not found.
    
    Checks:
    - /sys/fs/cgroup/cpu.max (cgroups v2)
    - /sys/fs/cgroup/cpu/cpu.cfs_quota_us and cpu.cfs_period_us (v1)
    """
    cpu_limit = None
    
    # Try cgroups v2 first (newer systems)
    cpu_max_path = "/sys/fs/cgroup/cpu.max"
    if os.path.exists(cpu_max_path):
        try:
            with open(cpu_max_path, 'r') as f:
                content = f.read().strip()
                if content and content != "max":
                    # Format: "quota period" or "max"
                    parts = content.split()
                    if len(parts) == 2 and parts[0] != "max":
                        quota = int(parts[0])
                        period = int(parts[1])
                        if period > 0:
                            cpu_limit = quota / period
        except (ValueError, IOError, OSError):
            pass
    
    # Try cgroups v1 (older systems, Docker)
    if cpu_limit is None:
        quota_path = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
        period_path = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
        if os.path.exists(quota_path) and os.path.exists(period_path):
            try:
                with open(quota_path, 'r') as f:
                    quota = int(f.read().strip())
                with open(period_path, 'r') as f:
                    period = int(f.read().strip())
                # -1 means unlimited
                if quota > 0 and period > 0:
                    cpu_limit = quota / period
            except (ValueError, IOError, OSError):
                pass
    
    return cpu_limit

def get_effective_cpu_count():
    """
    Get effective CPU count for parallel processing.
    
    Strategy:
    1. Check explicit override via MAX_WORKER_THREADS env var
    2. Check container limits (cgroups) - use 80% of limit
    3. Use 80% of physical CPU count (thumb rule)
    4. Minimum of 1 core
    
    Returns:
        int: Number of CPU cores to use for parallel processing
    """
    # Explicit override (highest priority)
    explicit_limit = os.getenv("MAX_WORKER_THREADS")
    if explicit_limit:
        try:
            count = int(explicit_limit)
            if count > 0:
                return max(1, count)
        except ValueError:
            pass
    
    # Get physical CPU count
    physical_cpus = os.cpu_count() or 1
    
    # Check container limits (Kubernetes/Docker)
    container_limit = get_cpu_limit_from_cgroups()
    
    # Determine available CPUs
    if container_limit is not None:
        # In container: use 80% of container limit
        available_cpus = container_limit
        source = "container limit"
    else:
        # On host: use 80% of physical CPUs
        available_cpus = physical_cpus
        source = "physical CPUs"
    
    # Apply 80% thumb rule (leave 20% headroom for OS and other processes)
    effective_cpus = int(available_cpus * 0.8)
    
    # Ensure minimum of 1 core
    effective_cpus = max(1, effective_cpus)
    
    # Log the decision
    if VERBOSE_LEVEL >= 1:
        container_info = f" (container limit: {container_limit:.2f})" if container_limit else ""
        print(f"CPU Detection: {effective_cpus} workers from {source} ({physical_cpus} physical CPUs{container_info})")
    
    return effective_cpus

# Get effective CPU count for parallel processing
# This can be overridden by --parallel CLI flag or MAX_WORKER_THREADS env var
MAX_WORKER_THREADS = get_effective_cpu_count()
# Track CLI override for display purposes
CLI_PARALLEL_OVERRIDE = None

# Print CPU detection details at startup
def print_cpu_info(cli_override=None):
    """Print CPU detection and parallelization configuration."""
    physical_cpus = os.cpu_count() or 1
    container_limit = get_cpu_limit_from_cgroups()
    explicit_limit = os.getenv("MAX_WORKER_THREADS")
    
    print("="*80)
    print("PARALLEL PROCESSING CONFIGURATION")
    print("="*80)
    print(f"  Physical CPUs detected: {physical_cpus}")
    if container_limit:
        print(f"  Container CPU limit: {container_limit:.2f} cores (cgroups)")
    else:
        print(f"  Container CPU limit: None (running on host)")
    if cli_override is not None:
        print(f"  CLI override (--parallel): {cli_override}")
        print(f"  Effective workers: {MAX_WORKER_THREADS} (from --parallel flag)")
    elif explicit_limit:
        print(f"  Environment override (MAX_WORKER_THREADS): {explicit_limit}")
        print(f"  Effective workers: {MAX_WORKER_THREADS} (from environment variable)")
    else:
        print(f"  Auto-detection: Using 80% rule")
        print(f"  Effective workers: {MAX_WORKER_THREADS} (80% of available CPUs)")
    print(f"  Parallelization thresholds:")
    print(f"    ├─ Disk Models: >10 disks")
    print(f"    ├─ I/O Network Crisis: >10 nodes")
    print(f"    └─ I/O Network Ensemble: >10 nodes")
    print("="*80)

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
    
    # Forward DNS lookup for hostnames (before reverse DNS for IPs)
    try:
        ipaddress.ip_address(cleaned)
        is_ip = True
    except ValueError:
        is_ip = False
    
    if not is_ip:
        # It's a hostname, try forward DNS lookup
        if cleaned not in DNS_CACHE:
            try:
                fqdn = socket.gethostbyname_ex(cleaned)[0]
                short = fqdn.split('.')[0].lower()
                DNS_CACHE[cleaned] = short
                # Also cache the IP if we can get it
                try:
                    ip = socket.gethostbyname(cleaned)
                    track_source(short, ip)
                except OSError:
                    pass
                return short
            except (OSError, socket.gaierror):
                DNS_CACHE[cleaned] = cleaned
        return DNS_CACHE.get(cleaned, cleaned) or ident
    
    # reverse DNS fallback for bare IPs
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
    
    # Build a mapping of hostname -> FQDN short name for pod entities
    pod_hostname_to_fqdn = {}
    for pod_entity in pod_entities:
        # Try forward DNS lookup for hostnames
        try:
            # Check if it's already an IP
            ipaddress.ip_address(pod_entity)
            continue  # Skip IPs, handle them via reverse DNS below
        except ValueError:
            pass  # It's a hostname, do forward lookup
        
        try:
            fqdn = socket.gethostbyname_ex(pod_entity)[0]
            short = fqdn.split('.')[0].lower()
            pod_hostname_to_fqdn[pod_entity] = short
        except (OSError, socket.gaierror):
            pass  # DNS lookup failed, skip
    
    # Now match host IPs to pod entities via reverse DNS
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
            short = fqdn.split('.')[0].lower()
            # Check if this FQDN short name matches any pod entity (direct match or via hostname mapping)
            if short in pod_entities:
                INSTANCE_ALIAS_MAP[ident] = short
                new_entries += 1
            elif short in pod_hostname_to_fqdn.values():
                # Find the pod entity that maps to this FQDN short name
                for pod_entity, fqdn_short in pod_hostname_to_fqdn.items():
                    if fqdn_short == short:
                        INSTANCE_ALIAS_MAP[ident] = pod_entity
                        INSTANCE_ALIAS_MAP[pod_entity] = short  # Bidirectional mapping
                        new_entries += 1
                        break
        except OSError:
            continue
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
    """
    Infer aliases by correlating host and pod CPU time series.
    If two entities have highly correlated time series, they're likely the same node.
    """
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
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        metavar="N",
        help="Override automatic CPU detection and use N parallel workers (overrides 80%% rule and MAX_WORKER_THREADS env var). Example: --parallel 4"
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

    print("\nEstate Topology Snapshot:")
    print(f"  • Total hosts reporting metrics        : {len(host_instances)}")
    print(f"  • Hosts in Kubernetes clusters         : {len(hosts_with_pods)}")
    print(f"  • Standalone hosts (no pods detected)  : {len(host_only)}")
    print(f"  • Pod-only metrics (no host data)      : {len(pod_only)}")
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

    display = host_only_pressure.copy()
    if 'raw_instance' in display.columns:
        display['instance'] = display.apply(lambda row: canonical_node_label(row['entity'], with_ip=True, raw_label=row.get('raw_instance')), axis=1)
    else:
        display['instance'] = display['entity'].apply(lambda ent: canonical_node_label(ent, with_ip=True))
    # Convert to percentages and format for readability
    display_output = display[['instance', 'host_cpu', 'host_mem']].copy()
    display_output['Host CPU %'] = (display_output['host_cpu'] * 100).round(1)
    display_output['Host Mem %'] = (display_output['host_mem'] * 100).round(1)
    
    # Separate nodes with pods vs nodes without pods
    if 'has_pod_metrics' in host_only_pressure.columns:
        nodes_with_pods = host_only_pressure[host_only_pressure['has_pod_metrics'] == True].copy()
        nodes_without_pods = host_only_pressure[host_only_pressure['has_pod_metrics'] == False].copy()
        
        if not nodes_without_pods.empty and not nodes_with_pods.empty:
            # Mixed case: some nodes have pods, some don't
            print("\n⚠️  Host pressure detected:")
            print("   Nodes with low pod usage:")
            with_pods_instances = nodes_with_pods['instance'].values if 'instance' in nodes_with_pods.columns else nodes_with_pods['entity'].apply(lambda e: canonical_node_label(e, with_ip=True)).values
            with_pods_display = display_output[display_output['instance'].isin(with_pods_instances)]
            print(with_pods_display[['instance', 'Host CPU %', 'Host Mem %']].to_string(index=False))
            print("\n   Nodes with high host usage:")
            without_pods_instances = nodes_without_pods['instance'].values if 'instance' in nodes_without_pods.columns else nodes_without_pods['entity'].apply(lambda e: canonical_node_label(e, with_ip=True)).values
            without_pods_display = display_output[display_output['instance'].isin(without_pods_instances)]
            print(without_pods_display[['instance', 'Host CPU %', 'Host Mem %']].to_string(index=False))
            print("\nAction:")
            for _, row in with_pods_display.iterrows():
                instance = row['instance']
                cpu_pct = row['Host CPU %']
                mem_pct = row['Host Mem %']
                print(f"  • {instance}: High host usage (CPU: {cpu_pct}%, Mem: {mem_pct}%) with low pod usage - inspect OS-level processes (backups, cron jobs, daemons)")
            for _, row in without_pods_display.iterrows():
                instance = row['instance']
                cpu_pct = row['Host CPU %']
                mem_pct = row['Host Mem %']
                print(f"  • {instance}: High host usage (CPU: {cpu_pct}%, Mem: {mem_pct}%) - inspect OS-level processes (backups, cron jobs, daemons)")
        elif not nodes_without_pods.empty:
            # All nodes have no pods - don't mention Kubernetes
            print("\n⚠️  Host pressure detected:")
            print("   (High host resource usage detected)")
            print(display_output[['instance', 'Host CPU %', 'Host Mem %']].to_string(index=False))
            print("\nAction:")
            for _, row in display_output.iterrows():
                instance = row['instance']
                cpu_pct = row['Host CPU %']
                mem_pct = row['Host Mem %']
                print(f"  • {instance}: High host usage (CPU: {cpu_pct}%, Mem: {mem_pct}%) - inspect OS-level processes (backups, cron jobs, daemons)")
        else:
            # All nodes have pods (minimal usage)
            print("\n⚠️  Host pressure detected with minimal pod usage:")
            print("   (High host resource usage but low pod usage indicates OS-level processes)")
            print(display_output[['instance', 'Host CPU %', 'Host Mem %']].to_string(index=False))
            print("\nAction:")
            for _, row in display_output.iterrows():
                instance = row['instance']
                cpu_pct = row['Host CPU %']
                mem_pct = row['Host Mem %']
                print(f"  • {instance}: High host usage (CPU: {cpu_pct}%, Mem: {mem_pct}%) with low pod usage - inspect OS-level processes (backups, cron jobs, daemons)")
    else:
        # Fallback: can't determine if pods exist, use generic message
        print("\n⚠️  Host pressure detected:")
        print("   (High host resource usage with low pod usage)")
        print(display_output[['instance', 'Host CPU %', 'Host Mem %']].to_string(index=False))
        print("\nAction:")
        for _, row in display_output.iterrows():
            instance = row['instance']
            cpu_pct = row['Host CPU %']
            mem_pct = row['Host Mem %']
            print(f"  • {instance}: High host usage (CPU: {cpu_pct}%, Mem: {mem_pct}%) with low pod usage - inspect OS-level processes (backups, cron jobs, daemons)")
    
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
                # Minimal update using same model structure (not full retraining)
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
    # Minimal update using learned patterns with recent changes
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
    forecast_ds = f_prophet.index
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
    was_saved = False
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
                            was_saved = True
                        except Exception as e:
                            log_verbose(f"Warning: Failed to save updated model: {e}")
                        # Plot was already saved in generate_forecast_from_cached_model
                        # Return result with was_saved flag
                        if isinstance(result, tuple) and len(result) == 3:
                            return (*result, was_saved)
                        else:
                            return (result, None, {}, was_saved)
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
                # Don't save model files in show_backtest mode - only generate plots
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
                    if isinstance(cached, tuple) and len(cached) == 3:
                        return (*cached, False)
                    else:
                        return (cached, None, {}, False)
            # In normal mode, don't generate plots (only in forecast mode or when show_backtest)
            if isinstance(cached, tuple) and len(cached) == 3:
                return (*cached, False)
            else:
                return (cached, None, {}, False)

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
        was_saved = True
        metrics = result[-1] if isinstance(result, tuple) and result else {}
        split_info = metrics.get('split_info') if isinstance(metrics, dict) else None
        persist_model_metadata(model_path, split_info)
    except Exception as exc:
        print(f"Warning: failed to save ensemble artifacts ({model_path}): {exc}")

    # Return result with was_saved flag
    if isinstance(result, tuple) and len(result) == 3:
        return (*result, was_saved)
    else:
        return (result, None, {}, was_saved)

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
    # Maximum unreliability tolerance
    # Example: 99.95% SLO = 0.05% error budget
    total_budget = 100.0 - slo_target
    
    # Step 2: Calculate how much budget has been consumed
    # If compliance is 0%, you've consumed 100% of your budget
    # If compliance is 100%, you've consumed 0% of your budget
    budget_consumed = 100.0 - compliance_percent
    
    # Step 3: Calculate remaining budget
    # May be negative if budget exceeded
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
        r = requests.get(VM_BASE_URL, params=params, timeout=30, verify=False)
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
    # Preserve cluster labels and other important metadata
    cluster_label_cols = ['cluster', 'cluster_id', 'cluster_name', 'cluster_label', 
                         'kubernetes_cluster', 'k8s_cluster']
    for col in cluster_label_cols:
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
    # mountpoint/filesystem and cluster labels are preserved via group_cols
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
def _process_single_disk(entity, mountpoint, group, mount_col, horizon_days, threshold_pct,
                         manifest_snapshot, retrain_targets, FORCE_TRAINING_RUN, forecast_mode,
                         dump_csv_dir, enable_plots, show_backtest):
    """
    Worker function to process a single disk (node/mountpoint) for disk full prediction.
    This function is designed to be called in parallel.
    Returns: dict with result, metrics, and model info, or None if skipped
    """
    try:
        raw_label = None
        if 'raw_instance' in group.columns and not group['raw_instance'].dropna().empty:
            raw_label = group['raw_instance'].dropna().iloc[-1]
        node = canonical_node_label(entity, with_ip=True, raw_label=raw_label)
        key = build_disk_key(entity, mountpoint)
        dump_label = f"disk_{node}_{mountpoint}"
        
        # Enhanced matching logic
        is_first_training = key not in manifest_snapshot
        needs_retrain = FORCE_TRAINING_RUN or is_first_training
        if not needs_retrain and retrain_targets and '__RETRAIN_ALL__' in retrain_targets:
            needs_retrain = True
        elif not needs_retrain and retrain_targets:
            entity_match = entity in retrain_targets
            key_match = key in retrain_targets
            mount_match = any(f":{mountpoint}" in t or f"|{mountpoint}" in t for t in retrain_targets)
            alias_match = False
            for target in retrain_targets:
                if '|' in target or ':' in target:
                    continue
                target_canon = canonical_identity(target)
                if target_canon == entity:
                    alias_match = True
                    break
                if target_canon in INSTANCE_ALIAS_MAP:
                    alias_value = INSTANCE_ALIAS_MAP[target_canon]
                    if canonical_identity(alias_value) == entity:
                        alias_match = True
                        break
                if alias_match:
                    break
            needs_retrain = entity_match or key_match or mount_match or alias_match
        
        ts = group.set_index('timestamp')['value'].sort_index()
        if len(ts) < 50:
            return None
        
        # Train/Test Split
        split_idx = max(1, int(len(ts) * TRAIN_FRACTION))
        if split_idx >= len(ts):
            split_idx = len(ts) - 1
        train_ts = ts.iloc[:split_idx]
        test_ts = ts.iloc[split_idx:]
        
        # Use cached result if available and not retraining
        if not needs_retrain and key in manifest_snapshot:
            cached_record = dict(manifest_snapshot[key])
            if 'ensemble_eta' not in cached_record:
                cached_record['ensemble_eta'] = cached_record.get('days_to_90pct', 9999.0)
            cached_record['days_to_90pct'] = max(0.0, cached_record.get('days_to_90pct', 9999.0))
            cached_record['ensemble_eta'] = max(0.0, cached_record.get('ensemble_eta', 9999.0))
            cached_record['linear_eta'] = max(0.0, cached_record.get('linear_eta', 9999.0))
            cached_record['prophet_eta'] = max(0.0, cached_record.get('prophet_eta', 9999.0))
            
            # Minimal update in forecast mode
            if forecast_mode:
                try:
                    pdf = train_ts.reset_index()
                    pdf.columns = ['ds', 'y']
                    pdf['y'] = pdf['y'].clip(upper=0.99)
                    recent_pdf = pdf.tail(min(len(pdf), 7*24*6))
                    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False)
                    m.fit(recent_pdf)
                    future = m.make_future_dataframe(periods=horizon_days*24*10, freq='6H')
                    forecast = m.predict(future)
                    now_ts = pd.Timestamp.now()
                    future_forecast = forecast[forecast['ds'] > now_ts]
                    over = future_forecast[future_forecast['yhat'] >= threshold_pct/100]
                    updated_prophet_days = max(0.0, (over.iloc[0]['ds'] - now_ts).total_seconds() / 86400) if not over.empty else 9999.0
                    daily_increase = train_ts.diff().resample('1D').mean().mean()
                    current_pct = ts.iloc[-1] * 100
                    if current_pct >= threshold_pct:
                        updated_linear_days = 0.0
                        updated_prophet_days = 0.0
                        updated_hybrid_days = 0.0
                    else:
                        updated_linear_days = max(0.0, (threshold_pct - current_pct) / (daily_increase * 100)) if daily_increase > 0.0001 else 9999.0
                        updated_hybrid_days = max(0.0, min(updated_linear_days, updated_prophet_days))
                    cached_record['days_to_90pct'] = round(updated_hybrid_days, 1)
                    cached_record['ensemble_eta'] = round(updated_hybrid_days, 1)
                    cached_record['linear_eta'] = round(updated_linear_days, 1)
                    cached_record['prophet_eta'] = round(updated_prophet_days, 1)
                    cached_record['alert'] = "CRITICAL" if updated_hybrid_days <= 0 else "CRITICAL" if updated_hybrid_days < 3 else "WARNING" if updated_hybrid_days < 7 else "SOON" if updated_hybrid_days < 30 else "OK"
                except:
                    pass
            
            return {
                'result': cached_record,
                'key': key,
                'needs_retrain': False,
                'metrics': None,
                'train_points': len(train_ts) if show_backtest else None,
                'test_points': len(test_ts) if show_backtest else None
            }
        
        # Training/retraining logic
        current_pct = ts.iloc[-1] * 100
        if current_pct >= threshold_pct:
            linear_days = 0.0
            prophet_days = 0.0
            hybrid_days = 0.0
            severity = "CRITICAL"
            prophet_model = None
            prophet_forecast_df = None
            prophet_mae = None
            prophet_pred = None
            linear_pred = None
            linear_mae = None
        else:
            daily_increase = train_ts.diff().resample('1D').mean().mean()
            if daily_increase > 0.0001:
                linear_days = max(0.0, (threshold_pct - current_pct) / (daily_increase * 100))
                if 0 < linear_days < 0.1:
                    linear_days = 0.1
            else:
                linear_days = 9999.0
            
            # Prophet ETA
            pdf = train_ts.reset_index()
            pdf.columns = ['ds', 'y']
            pdf['y'] = pdf['y'].clip(upper=0.99)
            prophet_days = 9999.0
            prophet_mae = None
            prophet_pred = None
            prophet_model = None
            prophet_forecast_df = None
            try:
                if needs_retrain and key in manifest_snapshot:
                    fit_pdf = pdf.tail(min(len(pdf), 7*24*6))
                else:
                    fit_pdf = pdf
                if dump_csv_dir:
                    fit_pdf_for_csv = fit_pdf.copy()
                    fit_pdf_for_csv['node'] = node
                    fit_pdf_for_csv['mountpoint'] = mountpoint
                    dump_dataframe_to_csv(fit_pdf_for_csv, dump_csv_dir, dump_label)
                else:
                    dump_dataframe_to_csv(fit_pdf.copy(), dump_csv_dir, dump_label)
                m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False)
                m.fit(fit_pdf)
                prophet_model = m
                if needs_retrain and len(test_ts) > 0:
                    test_df = test_ts.reset_index()
                    test_df.columns = ['ds', 'y']
                    test_forecast = m.predict(test_df[['ds']])
                    prophet_pred = test_forecast['yhat'].values
                    prophet_mae = mean_absolute_error(test_df['y'].values, prophet_pred)
                future = m.make_future_dataframe(periods=horizon_days*24*10, freq='6H')
                forecast = m.predict(future)
                prophet_forecast_df = forecast
                now_ts = pd.Timestamp.now()
                future_forecast = forecast[forecast['ds'] > now_ts]
                over = future_forecast[future_forecast['yhat'] >= threshold_pct/100]
                prophet_days = max(0.0, (over.iloc[0]['ds'] - now_ts).total_seconds() / 86400) if not over.empty else 9999.0
            except:
                prophet_days = max(0.0, linear_days) if 'linear_days' in locals() else 9999.0
            
            # Linear MAE
            linear_pred = None
            linear_mae = None
            if needs_retrain and len(test_ts) > 1:
                base_value = train_ts.iloc[-1]
                time_diffs = (test_ts.index - train_ts.index[-1]).total_seconds() / 86400
                linear_pred = base_value + time_diffs * daily_increase
                linear_mae = mean_absolute_error(test_ts.values, linear_pred.values)
            
            hybrid_days = max(0.0, min(linear_days, prophet_days))
            severity = "CRITICAL" if hybrid_days < 3 else "WARNING" if hybrid_days < 7 else "SOON" if hybrid_days < 30 else "OK"
        
        # Ensemble MAE
        ensemble_mae = None
        if needs_retrain:
            if linear_pred is not None and prophet_pred is not None and len(test_ts) > 0:
                ensemble_pred = pd.Series(np.minimum(linear_pred.values, prophet_pred), index=test_ts.index)
                ensemble_mae = mean_absolute_error(test_ts.values, ensemble_pred.values)
            elif prophet_pred is not None and len(test_ts) > 0:
                ensemble_mae = prophet_mae
            elif linear_pred is not None and len(test_ts) > 0:
                ensemble_mae = linear_mae
        
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
        
        metrics = {
            'linear_mae': linear_mae,
            'prophet_mae': prophet_mae,
            'ensemble_mae': ensemble_mae
        } if needs_retrain else None
        
        return {
            'result': record,
            'key': key,
            'needs_retrain': needs_retrain,
            'metrics': metrics,
            'train_points': len(train_ts),
            'test_points': len(test_ts),
            'train_start': str(train_ts.index[0]) if not train_ts.empty else None,
            'train_end': str(train_ts.index[-1]) if not train_ts.empty else None,
            'test_start': str(test_ts.index[0]) if not test_ts.empty else None,
            'test_end': str(test_ts.index[-1]) if not test_ts.empty else None
        }
    except Exception as e:
        log_verbose(f"  Error processing disk {entity}|{mountpoint}: {e}")
        return None

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
    
    # Prepare for parallelization
    disk_groups = list(df_disk.groupby(['entity', mount_col]))
    total_disks = len(disk_groups)
    # If --parallel flag is set, bypass threshold and use parallel processing
    # Otherwise, only parallelize if we have enough items to justify overhead
    use_parallel = (CLI_PARALLEL_OVERRIDE is not None) or (total_disks > 10 and MAX_WORKER_THREADS > 1)
    n_workers = min(total_disks, MAX_WORKER_THREADS) if use_parallel else 1
    
    if total_disks > 5:
        if use_parallel:
            print(f"  Processing {total_disks} disks in PARALLEL mode:")
            print(f"    ├─ Available workers: {MAX_WORKER_THREADS}")
            print(f"    ├─ Workers used: {n_workers} (min({total_disks}, {MAX_WORKER_THREADS}))")
            print(f"    └─ Expected speedup: ~{n_workers}x (vs sequential)")
        else:
            print(f"  Processing {total_disks} disks in SEQUENTIAL mode:")
            print(f"    ├─ Available workers: {MAX_WORKER_THREADS}")
            if CLI_PARALLEL_OVERRIDE is None:
                reason = 'Too few items (<10)' if total_disks <= 10 else 'Single worker only'
            else:
                reason = 'Single worker only (MAX_WORKER_THREADS=1)'
            print(f"    ├─ Reason: {reason}")
            print(f"    └─ Workers used: 1")
    
    # Process disks in parallel or sequentially
    if use_parallel:
        # Parallel processing
        manifest_snapshot = manifest.copy()
        processed_results = Parallel(n_jobs=n_workers, verbose=0)(
            delayed(_process_single_disk)(
                entity, mountpoint, group, mount_col, horizon_days, threshold_pct,
                manifest_snapshot, retrain_targets, FORCE_TRAINING_RUN, forecast_mode,
                dump_csv_dir, enable_plots, show_backtest
            )
            for (entity, mountpoint), group in disk_groups
        )
        
        # Process results and aggregate metrics
        for idx, proc_result in enumerate(processed_results):
            if proc_result is None:
                continue
            
            alerts.append(proc_result['result'])
            manifest[proc_result['key']] = proc_result['result']
            manifest_changed = True
            
            if proc_result['needs_retrain']:
                retrained_nodes.add(f"{proc_result['result']['instance']} | {proc_result['result']['mountpoint']}")
            
            # Collect metrics
            if proc_result['metrics']:
                if proc_result['metrics']['linear_mae'] is not None:
                    all_mae_linear.append(proc_result['metrics']['linear_mae'])
                if proc_result['metrics']['prophet_mae'] is not None:
                    all_mae_prophet.append(proc_result['metrics']['prophet_mae'])
                if proc_result['metrics']['ensemble_mae'] is not None:
                    all_mae_ensemble.append(proc_result['metrics']['ensemble_mae'])
            
            if proc_result['train_points']:
                all_train_points.append(proc_result['train_points'])
            if proc_result['test_points']:
                all_test_points.append(proc_result['test_points'])
            if proc_result.get('train_start'):
                train_starts.append(proc_result['train_start'])
            if proc_result.get('train_end'):
                train_ends.append(proc_result['train_end'])
            if proc_result.get('test_start'):
                test_starts.append(proc_result['test_start'])
            if proc_result.get('test_end'):
                test_ends.append(proc_result['test_end'])
            
            if total_disks > 5 and (idx + 1) % 10 == 0:
                print(f"    → Progress: {idx + 1}/{total_disks} disks processed...", end='\r')
        
        if total_disks > 5:
            print()  # New line after progress
        successful_disks = len([r for r in processed_results if r is not None])
        print(f"    ✓ Parallel execution complete: {successful_disks}/{total_disks} disks processed successfully")
    else:
        # Sequential processing (original code)
        for (entity, mountpoint), group in disk_groups:
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
                
                # Sanitize cached record: ensure all eta values are non-negative
                # This fixes any negative values that might have been stored in previous runs
                cached_record['days_to_90pct'] = max(0.0, cached_record.get('days_to_90pct', 9999.0))
                cached_record['ensemble_eta'] = max(0.0, cached_record.get('ensemble_eta', 9999.0))
                cached_record['linear_eta'] = max(0.0, cached_record.get('linear_eta', 9999.0))
                cached_record['prophet_eta'] = max(0.0, cached_record.get('prophet_eta', 9999.0))
                
                # Update manifest with sanitized values to prevent future negative values
                manifest[key] = cached_record
                manifest_changed = True
                
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
                            # Only look at FUTURE forecast points (not historical)
                            now_ts = pd.Timestamp.now()
                            future_forecast = forecast[forecast['ds'] > now_ts]
                            over = future_forecast[future_forecast['yhat'] >= threshold_pct/100]
                            if not over.empty:
                                updated_prophet_days_calc = (over.iloc[0]['ds'] - now_ts).total_seconds() / 86400
                                # Ensure non-negative (should always be positive for future points, but clamp anyway)
                                updated_prophet_days = max(0.0, updated_prophet_days_calc)
                            else:
                                # No future forecast point exceeds threshold
                                updated_prophet_days = 9999.0
                            # Update linear forecast too
                            daily_increase = train_ts.diff().resample('1D').mean().mean()
                            current_pct = ts.iloc[-1] * 100
                            
                            # Check if already exceeded threshold
                            if current_pct >= threshold_pct:
                                updated_linear_days = 0.0
                                updated_prophet_days = 0.0
                                updated_hybrid_days = 0.0
                            else:
                                if daily_increase > 0.0001:
                                    updated_linear_days = (threshold_pct - current_pct) / (daily_increase * 100)
                                    # Ensure non-negative - if calculation gives negative, disk is not approaching threshold
                                    updated_linear_days = max(0.0, updated_linear_days)
                                    # If very small positive value, set to minimum 0.1 for display
                                    if 0 < updated_linear_days < 0.1:
                                        updated_linear_days = 0.1
                                else:
                                    updated_linear_days = 9999.0
                                # Update hybrid forecast (min of linear and prophet)
                                updated_hybrid_days = min(updated_linear_days, updated_prophet_days)
                                
                                # Ensure all values are non-negative
                                updated_linear_days = max(0.0, updated_linear_days)
                                updated_prophet_days = max(0.0, updated_prophet_days)
                                updated_hybrid_days = max(0.0, updated_hybrid_days)
                            # Update cached record with fresh forecast - ensure all values are non-negative
                            cached_record['days_to_90pct'] = round(max(0.0, updated_hybrid_days), 1)
                            cached_record['ensemble_eta'] = round(max(0.0, updated_hybrid_days), 1)
                            cached_record['linear_eta'] = round(max(0.0, updated_linear_days), 1)
                            cached_record['prophet_eta'] = round(max(0.0, updated_prophet_days), 1)
                            # Set alert severity - if already exceeded (0 days), mark as CRITICAL
                            if updated_hybrid_days <= 0:
                                cached_record['alert'] = "CRITICAL"
                            else:
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
            
            # Check if already exceeded threshold
            if current_pct >= threshold_pct:
                # Already exceeded - set to 0 days
                linear_days = 0.0
                prophet_days = 0.0
                hybrid_days = 0.0
                severity = "CRITICAL"
                # Skip Prophet calculation since already exceeded
                prophet_model = None
                prophet_forecast_df = None
                prophet_mae = None
                prophet_pred = None
            else:
                # Linear ETA (fast & reliable)
                daily_increase = train_ts.diff().resample('1D').mean().mean()
                if daily_increase > 0.0001:  # 0.01% per day
                    linear_days = (threshold_pct - current_pct) / (daily_increase * 100)
                    # Ensure non-negative - if calculation gives negative, disk is not approaching threshold
                    linear_days = max(0.0, linear_days)
                    # If very small positive value, set to minimum 0.1 for display
                    if 0 < linear_days < 0.1:
                        linear_days = 0.1
                else:
                    # No significant increase, disk not approaching threshold
                    linear_days = 9999.0
            
            # Compute linear MAE on test set (only when retraining)
            linear_pred = None
            if needs_retrain and len(test_ts) > 1:
                base_value = train_ts.iloc[-1]
                time_diffs = (test_ts.index - train_ts.index[-1]).total_seconds() / 86400
                linear_pred = base_value + time_diffs * daily_increase
                linear_mae = mean_absolute_error(test_ts.values, linear_pred.values)
                all_mae_linear.append(linear_mae)

            # Prophet ETA (seasonal correction) - only if not already exceeded
            if current_pct < threshold_pct:
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
                    # Only look at FUTURE forecast points (not historical)
                    now_ts = pd.Timestamp.now()
                    future_forecast = forecast[forecast['ds'] > now_ts]
                    over = future_forecast[future_forecast['yhat'] >= threshold_pct/100]
                    if not over.empty:
                        prophet_days_calc = (over.iloc[0]['ds'] - now_ts).total_seconds() / 86400
                        # Ensure non-negative (should always be positive for future points, but clamp anyway)
                        prophet_days = max(0.0, prophet_days_calc)
                    else:
                        # No future forecast point exceeds threshold
                        prophet_days = 9999.0
                except:
                    # Fallback to linear_days, but ensure it's non-negative
                    prophet_days = max(0.0, linear_days) if 'linear_days' in locals() else 9999.0

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

            # Only calculate hybrid_days if not already exceeded
            if current_pct < threshold_pct:
                hybrid_days = min(linear_days, prophet_days)
                severity = "CRITICAL" if hybrid_days < 3 else "WARNING" if hybrid_days < 7 else "SOON" if hybrid_days < 30 else "OK"

            # Ensure all values are non-negative before storing
            linear_days = max(0.0, linear_days)
            prophet_days = max(0.0, prophet_days)
            hybrid_days = max(0.0, hybrid_days)

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
        # Align to actual test timestamps (10m data may have gaps)
        prophet_full = mb.predict(fut_b).set_index('ds')
        p_back = prophet_full.reindex(test_ts.index, method='nearest')['yhat']
        # Ensure p_back is numeric and has no NaN/inf
        p_back = pd.to_numeric(p_back, errors='coerce').replace([np.inf, -np.inf], np.nan)

        # ARIMA — use original timestamps
        train_ts = train.set_index('ds')['y']
        try:
            a_model = ARIMA(train_ts, order=(2,1,0)).fit()
            a_pred = pd.Series(a_model.forecast(steps=len(test_ts)), index=test_ts.index)
            # Ensure a_pred is numeric and has no NaN/inf
            a_pred = pd.to_numeric(a_pred, errors='coerce').replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            log_verbose(f"ARIMA backtest failed: {e}")
            # Fallback: use mean of test_ts or 0
            fallback_value = test_ts.mean() if not test_ts.empty and not test_ts.isna().all() else 0.0
            a_pred = pd.Series([fallback_value] * len(test_ts), index=test_ts.index)

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
                    # Ensure l_back is numeric and has no NaN/inf
                    l_back = pd.to_numeric(l_back, errors='coerce').replace([np.inf, -np.inf], np.nan)
            except Exception as e:
                print(f"LSTM backtest failed: {e}")
                l_back = a_pred.copy()
                # Ensure l_back is numeric
                l_back = pd.to_numeric(l_back, errors='coerce').replace([np.inf, -np.inf], np.nan)

        # Ensemble - handle NaN values gracefully
        # First, ensure all predictions are Series with proper index alignment
        # Convert to numeric and replace inf with NaN, then fill NaN
        test_mean = test_ts.mean() if not test_ts.empty and not test_ts.isna().all() else 0.0
        if pd.isna(test_mean) or np.isinf(test_mean):
            test_mean = 0.0
        
        # Clean predictions: replace inf, fill NaN, ensure numeric
        p_back_clean = pd.to_numeric(p_back, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(test_mean)
        a_pred_clean = pd.to_numeric(a_pred, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(test_mean)
        l_back_clean = pd.to_numeric(l_back, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(test_mean)
        
        # Ensure all have same index as test_ts
        p_back_clean = p_back_clean.reindex(test_ts.index, fill_value=test_mean)
        a_pred_clean = a_pred_clean.reindex(test_ts.index, fill_value=test_mean)
        l_back_clean = l_back_clean.reindex(test_ts.index, fill_value=test_mean)
        
        # Calculate ensemble (ensure no NaN or inf in result)
        ens_pred = (p_back_clean + a_pred_clean + l_back_clean) / 3
        ens_pred = pd.to_numeric(ens_pred, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(test_mean)
        
        # Clean test_ts as well
        test_ts_clean = pd.to_numeric(test_ts, errors='coerce').replace([np.inf, -np.inf], np.nan)
        
        # Align all series and drop rows where test has NaN (but keep predictions)
        valid_mask = ~test_ts_clean.isna()
        if valid_mask.sum() > 0:
            test_valid = test_ts_clean[valid_mask]
            ens_valid = ens_pred[valid_mask]
            p_back_valid = p_back_clean[valid_mask]
            a_pred_valid = a_pred_clean[valid_mask]
            l_back_valid = l_back_clean[valid_mask]
            
            # Final check: ensure no NaN or inf in any array before calling mean_absolute_error
            if (not ens_valid.isna().any() and not np.isinf(ens_valid).any() and 
                not test_valid.isna().any() and not np.isinf(test_valid).any() and
                len(ens_valid) == len(test_valid) and len(ens_valid) > 0):
                try:
                    mae_ens = mean_absolute_error(test_valid, ens_valid)
                    if pd.isna(mae_ens) or np.isinf(mae_ens):
                        mae_ens = np.nan
                except (ValueError, Exception) as e:
                    log_verbose(f"Warning: Failed to calculate ensemble MAE: {e}")
                    mae_ens = np.nan
            else:
                mae_ens = np.nan
            
            if (not p_back_valid.isna().any() and not np.isinf(p_back_valid).any() and
                len(p_back_valid) == len(test_valid) and len(p_back_valid) > 0):
                try:
                    mae_prophet = mean_absolute_error(test_valid, p_back_valid)
                    if pd.isna(mae_prophet) or np.isinf(mae_prophet):
                        mae_prophet = np.nan
                except (ValueError, Exception) as e:
                    log_verbose(f"Warning: Failed to calculate Prophet MAE: {e}")
                    mae_prophet = np.nan
            else:
                mae_prophet = np.nan
                
            if (not a_pred_valid.isna().any() and not np.isinf(a_pred_valid).any() and
                len(a_pred_valid) == len(test_valid) and len(a_pred_valid) > 0):
                try:
                    mae_arima = mean_absolute_error(test_valid, a_pred_valid)
                    if pd.isna(mae_arima) or np.isinf(mae_arima):
                        mae_arima = np.nan
                except (ValueError, Exception) as e:
                    log_verbose(f"Warning: Failed to calculate ARIMA MAE: {e}")
                    mae_arima = np.nan
            else:
                mae_arima = np.nan
                
            if (not l_back_valid.isna().any() and not np.isinf(l_back_valid).any() and
                len(l_back_valid) == len(test_valid) and len(l_back_valid) > 0):
                try:
                    mae_lstm = mean_absolute_error(test_valid, l_back_valid)
                    if pd.isna(mae_lstm) or np.isinf(mae_lstm):
                        mae_lstm = np.nan
                except (ValueError, Exception) as e:
                    log_verbose(f"Warning: Failed to calculate LSTM MAE: {e}")
                    mae_lstm = np.nan
            else:
                mae_lstm = np.nan
        else:
            # No valid test data
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
def identify_clusters(df_host_cpu, df_host_mem, df_pod_cpu, df_pod_mem, lookback_hours=LOOKBACK_HOURS):
    """
    Identify which nodes belong to which Kubernetes cluster.
    
    Strategy:
    1. Try to extract explicit cluster labels from Prometheus data (if available)
    2. If no explicit labels, infer clusters by grouping nodes that share pod instance patterns
    3. Nodes with no pod metrics are considered standalone (not part of any cluster)
    
    Returns:
        dict: Maps entity -> cluster_id (or 'standalone' for non-cluster nodes)
    """
    now = pd.Timestamp.now()
    start = now - pd.Timedelta(hours=lookback_hours)
    
    cluster_map = {}
    
    # Step 1: Try to extract explicit cluster labels from Prometheus data
    # Check if cluster labels exist in any of the dataframes
    # IMPORTANT: Only assign nodes WITH PODS to clusters - nodes without pods are standalone
    cluster_labels_found = False
    
    # First, identify which nodes have pods (these are the only ones that should be in clusters)
    # IMPORTANT: Pod metrics have an 'instance' label that directly tells us which host the pod is on
    # Use that directly instead of trying to match via IPs/DNS
    nodes_with_pods = set()
    nodes_with_pods_raw = {}  # Track raw -> canonical mapping for debug
    pod_instance_to_host = {}  # Map pod entity -> host instance (from instance label)
    if not df_pod_cpu.empty or not df_pod_mem.empty:
        for df in [df_pod_cpu, df_pod_mem]:
            if df.empty:
                continue
            recent_df = df[df['timestamp'] >= start] if 'timestamp' in df.columns else df
            entity_col = 'entity' if 'entity' in recent_df.columns else 'instance'
            
            # Pod metrics 'instance' column contains the hostname where the pod is running
            if 'instance' in recent_df.columns:
                for _, row in recent_df.iterrows():
                    pod_entity = row.get(entity_col)
                    host_instance = row.get('instance')
                    if pod_entity and host_instance:
                        pod_canonical = canonical_identity(pod_entity)
                        host_canonical = canonical_identity(host_instance)
                        nodes_with_pods.add(host_canonical)
                        nodes_with_pods_raw[host_canonical] = host_instance
                        pod_instance_to_host[pod_canonical] = host_canonical
            else:
                # Fallback: if no instance column, use entity (old behavior)
                for entity in recent_df[entity_col].unique():
                    canonical = canonical_identity(entity)
                    nodes_with_pods.add(canonical)
                    nodes_with_pods_raw[canonical] = entity
    
    # Debug output
    if VERBOSE_LEVEL >= 1:
        print(f"\n[DEBUG] Nodes with pods (from pod metrics instance label): {sorted(nodes_with_pods)}")
        print(f"[DEBUG] Raw pod instance -> canonical mapping: {nodes_with_pods_raw}")
        if pod_instance_to_host:
            print(f"[DEBUG] Pod entity -> host instance mapping: {pod_instance_to_host}")
        print(f"[DEBUG] INSTANCE_ALIAS_MAP: {dict(list(INSTANCE_ALIAS_MAP.items())[:20])}")  # Show first 20 entries
    
    # Now check for cluster labels, but only assign nodes that have pods
    for df in [df_host_cpu, df_host_mem, df_pod_cpu, df_pod_mem]:
        if df.empty:
            continue
        recent_df = df[df['timestamp'] >= start] if 'timestamp' in df.columns else df
        if recent_df.empty:
            continue
        
        # Check for common cluster label names
        cluster_label_candidates = ['cluster', 'cluster_id', 'cluster_name', 'cluster_label', 
                                   'kubernetes_cluster', 'k8s_cluster']
        for label in cluster_label_candidates:
            if label in recent_df.columns:
                cluster_labels_found = True
                # Map entities to their cluster, but ONLY if they have pods
                # Use 'entity' column first (already canonicalized by recanonicalize_entities)
                # Fall back to 'instance' only if 'entity' is not available
                # Process unique entities only (not all rows with timestamps)
                found_entities = set()
                missed_entities = set()
                processed_entities = set()  # Track which entities we've already processed
                
                for _, row in recent_df.iterrows():
                    entity = row.get('entity')
                    if not entity:
                        entity = row.get('instance')
                    cluster_id = row.get(label)
                    if entity and cluster_id:
                        entity_normalized = canonical_identity(entity)
                        # Only process each entity once
                        if entity_normalized in processed_entities:
                            continue
                        processed_entities.add(entity_normalized)
                        
                        # Only assign to cluster if node has pods
                        # Check canonicalized identity, and also check alias mapping and IP matching
                        has_pods = False
                        if entity_normalized in nodes_with_pods:
                            has_pods = True
                        else:
                            # Check if this entity maps to a pod entity via alias mapping
                            # e.g., 'pi' might map to 'host01' via INSTANCE_ALIAS_MAP
                            if entity_normalized in INSTANCE_ALIAS_MAP:
                                alias_target = canonical_identity(INSTANCE_ALIAS_MAP[entity_normalized])
                                if alias_target in nodes_with_pods:
                                    has_pods = True
                                    # Update nodes_with_pods to include this entity too
                                    nodes_with_pods.add(entity_normalized)
                            
                            # Also check reverse: if any pod entity maps to this entity
                            if not has_pods:
                                for pod_entity in list(nodes_with_pods):  # Use list to avoid modification during iteration
                                    if pod_entity in INSTANCE_ALIAS_MAP:
                                        if canonical_identity(INSTANCE_ALIAS_MAP[pod_entity]) == entity_normalized:
                                            has_pods = True
                                            nodes_with_pods.add(entity_normalized)
                                            break
                            
                            # Also check IP matching via SOURCE_REGISTRY and CANON_SOURCE_MAP
                            if not has_pods:
                                # Get IP for this entity
                                entity_ip = SOURCE_REGISTRY.get(entity_normalized) or CANON_SOURCE_MAP.get(entity_normalized)
                                if entity_ip:
                                    # Check if any pod entity has the same IP
                                    for pod_entity in nodes_with_pods:
                                        pod_ip = SOURCE_REGISTRY.get(pod_entity) or CANON_SOURCE_MAP.get(pod_entity)
                                        if pod_ip and pod_ip == entity_ip:
                                            has_pods = True
                                            nodes_with_pods.add(entity_normalized)
                                            break
                        
                        if has_pods:
                            cluster_map[entity_normalized] = str(cluster_id)
                            found_entities.add((entity, entity_normalized, cluster_id))
                        else:
                            missed_entities.add((entity, entity_normalized, cluster_id))
                
                # Debug output
                if VERBOSE_LEVEL >= 1:
                    df_name = 'host_cpu' if df is df_host_cpu else 'host_mem' if df is df_host_mem else 'pod_cpu' if df is df_pod_cpu else 'pod_mem' if df is df_pod_mem else 'unknown'
                    print(f"\n[DEBUG] Cluster label '{label}' found in {df_name}")
                    print(f"[DEBUG] Unique entities assigned to cluster: {sorted(found_entities)}")
                    if missed_entities:
                        print(f"[DEBUG] Unique entities NOT assigned (no pods or mismatch): {sorted(missed_entities)}")
                        # Show why they're not matching
                        for raw, canonical, cluster_id in sorted(missed_entities):
                            in_pods = canonical in nodes_with_pods
                            print(f"  - {raw} -> {canonical}: in nodes_with_pods={in_pods}")
                    print(f"[DEBUG] All unique entities in this dataframe: {sorted(processed_entities)}")
                    print(f"[DEBUG] Missing from nodes_with_pods: {sorted(processed_entities - nodes_with_pods)}")
                
                # Assign all nodes with pods to this cluster (including pod-only nodes)
                cluster_id_value = None
                for _, row in recent_df.iterrows():
                    if row.get(label):
                        cluster_id_value = str(row.get(label))
                        break
                
                if cluster_id_value:
                    # Assign all nodes with pods to this cluster (even if they weren't in this dataframe)
                    # These are the authoritative hostnames from pod metrics instance labels
                    nodes_missing_from_df = nodes_with_pods - processed_entities
                    if nodes_missing_from_df:
                        if VERBOSE_LEVEL >= 1:
                            print(f"[DEBUG] Assigning {len(nodes_missing_from_df)} pod-only nodes to cluster '{cluster_id_value}': {sorted(nodes_missing_from_df)}")
                        for node in nodes_missing_from_df:
                            cluster_map[node] = cluster_id_value
                    
                    # Now match host entities to pod hostnames (from instance labels)
                    # The pod instance labels are authoritative - match host entities to them
                    host_only_entities = processed_entities - nodes_with_pods
                    if host_only_entities:
                        if VERBOSE_LEVEL >= 1:
                            print(f"[DEBUG] Matching host entities to pod hostnames (from instance labels): {sorted(host_only_entities)}")
                        
                        for host_entity in host_only_entities:
                            matched = False
                            
                            # First, try reverse DNS on ALL IPs to get the actual hostname
                            host_ips_to_try = set()
                            host_ip_reg = SOURCE_REGISTRY.get(host_entity) or CANON_SOURCE_MAP.get(host_entity)
                            if host_ip_reg:
                                host_ips_to_try.add(host_ip_reg)
                            # Also get IPs from raw_instance
                            for df_check in [df_host_cpu, df_host_mem]:
                                if df_check.empty:
                                    continue
                                if 'timestamp' in df_check.columns:
                                    check_df = df_check.loc[df_check['timestamp'] >= start].copy()
                                else:
                                    check_df = df_check.copy()
                                if check_df.empty:
                                    continue
                                entity_col = 'entity' if 'entity' in check_df.columns else 'instance'
                                matching_rows = check_df[check_df[entity_col].apply(canonical_identity) == host_entity]
                                if not matching_rows.empty:
                                    for col in ['raw_instance', 'raw_entity', 'instance']:
                                        if col in matching_rows.columns:
                                            for val in matching_rows[col].dropna().unique():
                                                try:
                                                    ip = str(val).split(':')[0]
                                                    ipaddress.ip_address(ip)
                                                    host_ips_to_try.add(ip)
                                                except (ValueError, AttributeError):
                                                    pass
                            
                            # Try reverse DNS on each IP
                            for host_ip in host_ips_to_try:
                                try:
                                    reverse_hostname = socket.gethostbyaddr(host_ip)[0]
                                    reverse_short = reverse_hostname.split('.')[0].lower()
                                    reverse_canonical = canonical_identity(reverse_short)
                                    if reverse_canonical in nodes_with_pods:
                                        cluster_map[host_entity] = cluster_id_value
                                        nodes_with_pods.add(host_entity)
                                        matched = True
                                        if VERBOSE_LEVEL >= 1:
                                            print(f"[DEBUG] Reverse DNS matched: {host_entity} (IP {host_ip}) -> {reverse_short} -> {reverse_canonical}")
                                        break
                                except (OSError, socket.herror):
                                    pass
                            
                            if matched:
                                break
                            
                            if matched:
                                continue
                            
                            # Try forward DNS resolution to get the actual hostname
                            try:
                                resolved_info = socket.gethostbyname_ex(host_entity)  # Returns (hostname, aliaslist, ipaddrlist)
                                resolved_hostname = resolved_info[0]
                                resolved_short = resolved_hostname.split('.')[0].lower()
                                resolved_canonical = canonical_identity(resolved_short)
                                if resolved_canonical in nodes_with_pods:
                                    cluster_map[host_entity] = cluster_id_value
                                    nodes_with_pods.add(host_entity)
                                    matched = True
                                    if VERBOSE_LEVEL >= 1:
                                        print(f"[DEBUG] DNS resolved: {host_entity} -> {resolved_short} -> {resolved_canonical}")
                                    break
                            except (OSError, socket.gaierror):
                                pass
                            
                            if matched:
                                continue
                            
                            # Try alias matching - check if host_entity or its aliases match any pod hostname
                            host_aliases = set([host_entity])
                            if host_entity in INSTANCE_ALIAS_MAP:
                                host_aliases.add(INSTANCE_ALIAS_MAP[host_entity])
                            # Also check reverse: if any pod hostname maps to this host entity
                            for pod_hostname, alias_target in INSTANCE_ALIAS_MAP.items():
                                if alias_target == host_entity or canonical_identity(alias_target) == host_entity:
                                    host_aliases.add(pod_hostname)
                            
                            # Check if any alias matches a pod hostname
                            for alias in host_aliases:
                                alias_canonical = canonical_identity(alias)
                                if alias_canonical in nodes_with_pods:
                                    cluster_map[host_entity] = cluster_id_value
                                    nodes_with_pods.add(host_entity)
                                    matched = True
                                    if VERBOSE_LEVEL >= 1:
                                        print(f"[DEBUG] Alias matched: {host_entity} (via alias {alias}) -> {alias_canonical}")
                                    break
                            
                            if matched:
                                continue
                            
                            # Pattern matching removed: If metrics don't return worker01/02/03 names,
                            # we shouldn't create them via DNS and then try to match them.
                            # Instead, we rely on IP-based matching and pod instance labels below.
                            
                            # If not matched, try IP-based matching
                            # Extract ALL IPs from raw_instance for both host and pod entities
                            if not matched:
                                # Get host IPs from multiple sources
                                host_ips = set()
                                host_ip_reg = SOURCE_REGISTRY.get(host_entity) or CANON_SOURCE_MAP.get(host_entity)
                                if host_ip_reg:
                                    host_ips.add(host_ip_reg)
                                
                                # Extract IPs from raw_instance in host dataframes
                                for df_check in [df_host_cpu, df_host_mem]:
                                    if df_check.empty:
                                        continue
                                    if 'timestamp' in df_check.columns:
                                        check_df = df_check.loc[df_check['timestamp'] >= start].copy()
                                    else:
                                        check_df = df_check.copy()
                                    if check_df.empty:
                                        continue
                                    entity_col = 'entity' if 'entity' in check_df.columns else 'instance'
                                    matching_rows = check_df[check_df[entity_col].apply(canonical_identity) == host_entity]
                                    if not matching_rows.empty:
                                        # Extract from raw_instance column - get ALL unique values
                                        if 'raw_instance' in matching_rows.columns:
                                            raw_instances = matching_rows['raw_instance'].dropna().unique()
                                            if VERBOSE_LEVEL >= 1 and host_entity == 'pi':
                                                print(f"[DEBUG] Found {len(raw_instances)} raw_instance values for {host_entity}: {list(raw_instances)}")
                                            for raw_inst in raw_instances:
                                                try:
                                                    raw_ip = str(raw_inst).split(':')[0]
                                                    ipaddress.ip_address(raw_ip)  # Validate it's an IP
                                                    host_ips.add(raw_ip)
                                                except (ValueError, AttributeError):
                                                    pass
                                        # Also try raw_entity column
                                        if 'raw_entity' in matching_rows.columns:
                                            for raw_ent in matching_rows['raw_entity'].dropna().unique():
                                                try:
                                                    raw_ip = str(raw_ent).split(':')[0]
                                                    ipaddress.ip_address(raw_ip)
                                                    host_ips.add(raw_ip)
                                                except (ValueError, AttributeError):
                                                    pass
                                        # Also check if instance column has IPs
                                        if 'instance' in matching_rows.columns:
                                            for inst in matching_rows['instance'].dropna().unique():
                                                try:
                                                    inst_ip = str(inst).split(':')[0]
                                                    ipaddress.ip_address(inst_ip)
                                                    host_ips.add(inst_ip)
                                                except (ValueError, AttributeError):
                                                    pass
                                
                                if VERBOSE_LEVEL >= 1:
                                    print(f"[DEBUG] Checking IPs for {host_entity}: {sorted(host_ips)}")
                                
                                if host_ips:
                                    for pod_entity in nodes_with_pods:
                                        # Get pod IPs from multiple sources
                                        pod_ips = set()
                                        pod_ip_reg = SOURCE_REGISTRY.get(pod_entity) or CANON_SOURCE_MAP.get(pod_entity)
                                        if pod_ip_reg:
                                            pod_ips.add(pod_ip_reg)
                                        
                                        # Extract IP from raw_instance
                                        pod_raw = nodes_with_pods_raw.get(pod_entity)
                                        if pod_raw:
                                            try:
                                                pod_raw_str = str(pod_raw)
                                                if ':' in pod_raw_str:
                                                    pod_ip_candidate = pod_raw_str.split(':')[0]
                                                    ipaddress.ip_address(pod_ip_candidate)
                                                    pod_ips.add(pod_ip_candidate)
                                            except (ValueError, AttributeError):
                                                pass
                                        
                                        # Also extract from pod dataframes
                                        # For pod entities, the 'instance' column contains the hostname
                                        # We need to find that hostname in host dataframes to get its IP
                                        pod_hostname = nodes_with_pods_raw.get(pod_entity)
                                        if pod_hostname:
                                            # Try to find this hostname in host dataframes to get its IP
                                            for df_check in [df_host_cpu, df_host_mem]:
                                                if df_check.empty:
                                                    continue
                                                if 'timestamp' in df_check.columns:
                                                    check_df = df_check.loc[df_check['timestamp'] >= start].copy()
                                                else:
                                                    check_df = df_check.copy()
                                                if check_df.empty:
                                                    continue
                                                entity_col = 'entity' if 'entity' in check_df.columns else 'instance'
                                                # Match by canonical identity
                                                pod_hostname_canonical = canonical_identity(pod_hostname)
                                                matching_rows = check_df[check_df[entity_col].apply(canonical_identity) == pod_hostname_canonical]
                                                if not matching_rows.empty:
                                                    # Extract IPs from this host's raw_instance
                                                    if 'raw_instance' in matching_rows.columns:
                                                        for raw_inst in matching_rows['raw_instance'].dropna().unique():
                                                            try:
                                                                raw_ip = str(raw_inst).split(':')[0]
                                                                ipaddress.ip_address(raw_ip)
                                                                pod_ips.add(raw_ip)
                                                            except (ValueError, AttributeError):
                                                                pass
                                                    if 'raw_entity' in matching_rows.columns:
                                                        for raw_ent in matching_rows['raw_entity'].dropna().unique():
                                                            try:
                                                                raw_ip = str(raw_ent).split(':')[0]
                                                                ipaddress.ip_address(raw_ip)
                                                                pod_ips.add(raw_ip)
                                                            except (ValueError, AttributeError):
                                                                pass
                                                    if 'instance' in matching_rows.columns:
                                                        for inst in matching_rows['instance'].dropna().unique():
                                                            try:
                                                                inst_ip = str(inst).split(':')[0]
                                                                ipaddress.ip_address(inst_ip)
                                                                pod_ips.add(inst_ip)
                                                            except (ValueError, AttributeError):
                                                                pass
                                        
                                        if VERBOSE_LEVEL >= 1:
                                            print(f"[DEBUG]   Comparing with {pod_entity}: {sorted(pod_ips)}")
                                        
                                        # Match if ANY IP overlaps - if there's any common IP, they're the same node
                                        matching_ips = host_ips.intersection(pod_ips)
                                        if matching_ips:
                                            cluster_map[host_entity] = cluster_id_value
                                            nodes_with_pods.add(host_entity)
                                            matched = True
                                            if VERBOSE_LEVEL >= 1:
                                                print(f"[DEBUG] IP matched: {host_entity} ({sorted(matching_ips)}) -> {pod_entity} ({sorted(pod_ips)})")
                                            break
                                
                                # If still not matched and we have raw_instance, try to extract IP from it
                                if not matched:
                                    # Try to find the raw instance value for this entity
                                    for df_check in [df_host_cpu, df_host_mem]:
                                        if df_check.empty:
                                            continue
                                        if 'timestamp' in df_check.columns:
                                            check_df = df_check.loc[df_check['timestamp'] >= start].copy()
                                        else:
                                            check_df = df_check.copy()
                                        if check_df.empty:
                                            continue
                                        # Find rows with this entity
                                        entity_col = 'entity' if 'entity' in check_df.columns else 'instance'
                                        matching_rows = check_df[check_df[entity_col].apply(canonical_identity) == host_entity]
                                        if not matching_rows.empty and 'raw_instance' in matching_rows.columns:
                                            raw_inst = matching_rows['raw_instance'].iloc[0]
                                            if raw_inst:
                                                # Extract IP from raw_instance (format: "IP:PORT", e.g., "192.168.1.100:9100")
                                                try:
                                                    raw_ip = str(raw_inst).split(':')[0]
                                                    ipaddress.ip_address(raw_ip)  # Validate it's an IP
                                                    # Now check if any pod entity has this IP
                                                    for pod_entity in nodes_with_pods:
                                                        pod_raw = nodes_with_pods_raw.get(pod_entity)
                                                        if pod_raw:
                                                            pod_raw_ip = str(pod_raw).split(':')[0] if ':' in str(pod_raw) else str(pod_raw)
                                                            try:
                                                                ipaddress.ip_address(pod_raw_ip)
                                                                if pod_raw_ip == raw_ip:
                                                                    cluster_map[host_entity] = cluster_id_value
                                                                    nodes_with_pods.add(host_entity)
                                                                    matched = True
                                                                    if VERBOSE_LEVEL >= 1:
                                                                        print(f"[DEBUG] Raw IP matched: {host_entity} ({raw_ip}) -> {pod_entity} ({pod_raw_ip})")
                                                                    break
                                                            except ValueError:
                                                                pass
                                                    if matched:
                                                        break
                                                except (ValueError, AttributeError):
                                                    pass
                    
                    # Final fallback: if we have unmatched host entities, try to match them to pod entities
                    # that are already assigned to the cluster (pod-only nodes)
                    if host_only_entities:
                        unmatched_host = [e for e in host_only_entities if e not in cluster_map or cluster_map.get(e) == 'standalone']
                        # Get pod entities that are assigned to this cluster but not in host dataframes
                        pod_only_in_cluster = [e for e in nodes_with_pods if e not in processed_entities and cluster_map.get(e) == cluster_id_value]
                        if len(unmatched_host) > 0 and len(pod_only_in_cluster) > 0:
                            if VERBOSE_LEVEL >= 1:
                                print(f"[DEBUG] Fallback: Attempting to match {len(unmatched_host)} unmatched host entities to {len(pod_only_in_cluster)} pod-only entities")
                            
                            # For each pod-only node, try to get its IP by looking it up in host dataframes
                            # Then match unmatched hosts to pod-only nodes by IP or DNS
                            for pod_hostname in pod_only_in_cluster:
                                pod_ips = set()
                                # Try to find this pod hostname in host dataframes to get its IP
                                for df_check in [df_host_cpu, df_host_mem]:
                                    if df_check.empty:
                                        continue
                                    if 'timestamp' in df_check.columns:
                                        check_df = df_check.loc[df_check['timestamp'] >= start].copy()
                                    else:
                                        check_df = df_check.copy()
                                    if check_df.empty:
                                        continue
                                    entity_col = 'entity' if 'entity' in check_df.columns else 'instance'
                                    pod_canonical = canonical_identity(pod_hostname)
                                    matching_rows = check_df[check_df[entity_col].apply(canonical_identity) == pod_canonical]
                                    if not matching_rows.empty:
                                        # Extract IPs
                                        for col in ['raw_instance', 'raw_entity', 'instance']:
                                            if col in matching_rows.columns:
                                                for val in matching_rows[col].dropna().unique():
                                                    try:
                                                        ip = str(val).split(':')[0]
                                                        ipaddress.ip_address(ip)
                                                        pod_ips.add(ip)
                                                    except (ValueError, AttributeError):
                                                        pass
                                
                                # Also try DNS lookup for pod hostname to get ALL its IPs
                                try:
                                    pod_dns_info = socket.gethostbyname_ex(pod_hostname)
                                    # pod_dns_info[2] is the list of IP addresses
                                    for ip in pod_dns_info[2]:
                                        try:
                                            ipaddress.ip_address(ip)
                                            pod_ips.add(ip)
                                        except ValueError:
                                            pass
                                    # Also try reverse DNS on each IP to see if it matches any unmatched host
                                    for ip in pod_dns_info[2]:
                                        try:
                                            reverse_hostname = socket.gethostbyaddr(ip)[0]
                                            reverse_short = reverse_hostname.split('.')[0].lower()
                                            reverse_canonical = canonical_identity(reverse_short)
                                            # Check if this reverse DNS matches any unmatched host
                                            for host_e_check in unmatched_host:
                                                if reverse_canonical == canonical_identity(host_e_check):
                                                    cluster_map[host_e_check] = cluster_id_value
                                                    nodes_with_pods.add(host_e_check)
                                                    if VERBOSE_LEVEL >= 1:
                                                        print(f"[DEBUG] Fallback reverse DNS matched: {host_e_check} <-> {pod_hostname} (via IP {ip} -> {reverse_short})")
                                                    break
                                        except (OSError, socket.herror):
                                            pass
                                except (OSError, socket.gaierror):
                                    pass
                                
                                # Now try to match unmatched hosts to this pod hostname
                                for host_e in unmatched_host:
                                    if host_e in cluster_map and cluster_map.get(host_e) != 'standalone':
                                        continue
                                    
                                    # Get host IPs
                                    host_ips = set()
                                    host_ip_reg = SOURCE_REGISTRY.get(host_e) or CANON_SOURCE_MAP.get(host_e)
                                    if host_ip_reg:
                                        host_ips.add(host_ip_reg)
                                    # Extract from host dataframes
                                    for df_check in [df_host_cpu, df_host_mem]:
                                        if df_check.empty:
                                            continue
                                        if 'timestamp' in df_check.columns:
                                            check_df = df_check.loc[df_check['timestamp'] >= start].copy()
                                        else:
                                            check_df = df_check.copy()
                                        if check_df.empty:
                                            continue
                                        entity_col = 'entity' if 'entity' in check_df.columns else 'instance'
                                        matching_rows = check_df[check_df[entity_col].apply(canonical_identity) == host_e]
                                        if not matching_rows.empty:
                                            for col in ['raw_instance', 'raw_entity', 'instance']:
                                                if col in matching_rows.columns:
                                                    for val in matching_rows[col].dropna().unique():
                                                        try:
                                                            ip = str(val).split(':')[0]
                                                            ipaddress.ip_address(ip)
                                                            host_ips.add(ip)
                                                        except (ValueError, AttributeError):
                                                            pass
                                    
                                    # Match if any IP overlaps (excluding LB IPs if we have other IPs)
                                    matching_ips = host_ips.intersection(pod_ips)
                                    if matching_ips:
                                        cluster_map[host_e] = cluster_id_value
                                        nodes_with_pods.add(host_e)
                                        if VERBOSE_LEVEL >= 1:
                                            print(f"[DEBUG] Fallback IP matched: {host_e} ({sorted(matching_ips)}) -> {pod_hostname} ({sorted(pod_ips)})")
                                        break
                                
                            # Final fallback: try reverse DNS on host IPs to match pod hostnames
                            for host_e in unmatched_host:
                                if host_e in cluster_map and cluster_map.get(host_e) != 'standalone':
                                    continue
                                
                                # Get all IPs for this host
                                host_ips_to_check = set()
                                host_ip_reg = SOURCE_REGISTRY.get(host_e) or CANON_SOURCE_MAP.get(host_e)
                                if host_ip_reg:
                                    host_ips_to_check.add(host_ip_reg)
                                # Extract from host dataframes
                                for df_check in [df_host_cpu, df_host_mem]:
                                    if df_check.empty:
                                        continue
                                    if 'timestamp' in df_check.columns:
                                        check_df = df_check.loc[df_check['timestamp'] >= start].copy()
                                    else:
                                        check_df = df_check.copy()
                                    if check_df.empty:
                                        continue
                                    entity_col = 'entity' if 'entity' in check_df.columns else 'instance'
                                    matching_rows = check_df[check_df[entity_col].apply(canonical_identity) == host_e]
                                    if not matching_rows.empty:
                                        for col in ['raw_instance', 'raw_entity', 'instance']:
                                            if col in matching_rows.columns:
                                                for val in matching_rows[col].dropna().unique():
                                                    try:
                                                        ip = str(val).split(':')[0]
                                                        ipaddress.ip_address(ip)
                                                        host_ips_to_check.add(ip)
                                                    except (ValueError, AttributeError):
                                                        pass
                                
                                # Try reverse DNS on each IP to see if it resolves to any pod hostname
                                for host_ip in host_ips_to_check:
                                    try:
                                        reverse_hostname = socket.gethostbyaddr(host_ip)[0]
                                        reverse_short = reverse_hostname.split('.')[0].lower()
                                        reverse_canonical = canonical_identity(reverse_short)
                                        # Check if this matches any pod-only node
                                        if reverse_canonical in pod_only_in_cluster:
                                            cluster_map[host_e] = cluster_id_value
                                            nodes_with_pods.add(host_e)
                                            if VERBOSE_LEVEL >= 1:
                                                print(f"[DEBUG] Fallback reverse DNS matched: {host_e} (IP {host_ip}) -> {reverse_short} -> {reverse_canonical}")
                                            break
                                    except (OSError, socket.herror):
                                        pass
                                
                                if host_e in cluster_map and cluster_map.get(host_e) == cluster_id_value:
                                    break
                            
                            # Final fallback: if exactly 1 unmatched host and 1 pod-only node, match them
                            unmatched_after_dns = [e for e in unmatched_host if e not in cluster_map or cluster_map.get(e) == 'standalone']
                            pod_only_remaining = [e for e in pod_only_in_cluster if e not in [cluster_map.get(h) for h in unmatched_host if h in cluster_map]]
                            
                            if len(unmatched_after_dns) == 1 and len(pod_only_remaining) == 1:
                                host_e = unmatched_after_dns[0]
                                pod_e = pod_only_remaining[0]
                                # Only match if host has cluster label (meaning it's supposed to be in cluster)
                                if host_e not in cluster_map or cluster_map.get(host_e) == 'standalone':
                                    cluster_map[host_e] = cluster_id_value
                                    nodes_with_pods.add(host_e)
                                    if VERBOSE_LEVEL >= 1:
                                        print(f"[DEBUG] Fallback 1:1 matched: {host_e} -> {pod_e} (both have cluster label)")
                                    break
                            
                            # Ultimate fallback: if host has cluster label and there are pod-only nodes, match it
                            # This handles cases where all other matching methods fail but the host is definitely in the cluster
                            unmatched_final = [e for e in unmatched_host if e not in cluster_map or cluster_map.get(e) == 'standalone']
                            # Recalculate pod_only_in_cluster to exclude nodes that were just matched
                            # A pod-only node should only be in this list if:
                            # 1. It's in nodes_with_pods (has pods)
                            # 2. It's not in processed_entities (no host metrics) - this is the key: pod-only nodes
                            # 3. It's assigned to this cluster
                            # 4. It's NOT already matched to a HOST entity (i.e., no host entity from processed_entities that matches it)
                            pod_only_candidates = [e for e in nodes_with_pods if e not in processed_entities and cluster_map.get(e) == cluster_id_value]
                            # Filter out pod-only nodes that are already matched to HOST entities (not pod-only nodes)
                            pod_only_remaining_final = []
                            for pod_candidate in pod_only_candidates:
                                pod_canonical = canonical_identity(pod_candidate)
                                is_matched = False
                                # Check if any HOST entity (from processed_entities, not pod-only) in cluster_map matches this pod node
                                for host_entity in cluster_map:
                                    # Only check host entities (those in processed_entities), not pod-only nodes
                                    if (cluster_map.get(host_entity) == cluster_id_value and 
                                        host_entity != pod_candidate and 
                                        host_entity in processed_entities):
                                        host_canonical = canonical_identity(host_entity)
                                        # Direct match
                                        if host_canonical == pod_canonical:
                                            is_matched = True
                                            break
                                if not is_matched:
                                    pod_only_remaining_final.append(pod_candidate)
                            
                            if VERBOSE_LEVEL >= 1:
                                print(f"[DEBUG] Ultimate fallback check: {len(unmatched_final)} unmatched hosts, {len(pod_only_remaining_final)} pod-only nodes remaining")
                                if unmatched_final:
                                    print(f"[DEBUG]   Unmatched hosts: {unmatched_final}")
                                if pod_only_remaining_final:
                                    print(f"[DEBUG]   Pod-only nodes remaining: {pod_only_remaining_final}")
                            
                            if unmatched_final and pod_only_remaining_final:
                                # Find which pod-only nodes are already matched to host entities (to avoid double-matching)
                                # A pod-only node is "matched" if a host entity with the same canonical identity exists in cluster_map
                                matched_pod_nodes = set()
                                # Only check HOST entities (from processed_entities), not pod-only nodes
                                for host_entity in cluster_map:
                                    if (cluster_map.get(host_entity) == cluster_id_value and 
                                        host_entity in processed_entities):  # Only host entities, not pod-only
                                        host_canonical = canonical_identity(host_entity)
                                        for pod_node in pod_only_remaining_final:
                                            pod_canonical = canonical_identity(pod_node)
                                            # Direct canonical match
                                            if host_canonical == pod_canonical:
                                                matched_pod_nodes.add(pod_node)
                                                if VERBOSE_LEVEL >= 1:
                                                    print(f"[DEBUG]   Direct match: {host_entity} ({host_canonical}) == {pod_node} ({pod_canonical})")
                                            # Pattern matching removed: We match based on what metrics actually return,
                                            # not on DNS-derived names that may not exist in the metrics.
                                
                                # Only match to pod-only nodes that aren't already matched
                                remaining_pod_only = [p for p in pod_only_remaining_final if p not in matched_pod_nodes]
                                
                                if VERBOSE_LEVEL >= 1:
                                    print(f"[DEBUG]   Already matched pod nodes: {matched_pod_nodes}")
                                    print(f"[DEBUG]   Remaining pod-only nodes for matching: {remaining_pod_only}")
                                
                                # Match unmatched hosts to remaining pod-only nodes
                                for host_e in unmatched_final:
                                    if remaining_pod_only:
                                        # Match to the first remaining pod-only node
                                        pod_match = remaining_pod_only[0]
                                        # Use the pod_match as the canonical identity to avoid double-counting
                                        # Create alias to prevent double-counting
                                        pod_match_canonical = canonical_identity(pod_match)
                                        cluster_map[host_e] = cluster_id_value
                                        # Also map host_e to pod_match_canonical in alias map for deduplication
                                        INSTANCE_ALIAS_MAP[canonical_identity(host_e)] = pod_match_canonical
                                        nodes_with_pods.add(host_e)
                                        matched_pod_nodes.add(pod_match)
                                        remaining_pod_only.remove(pod_match)
                                        if VERBOSE_LEVEL >= 1:
                                            print(f"[DEBUG] Ultimate fallback matched: {host_e} -> {pod_match} (host has cluster label, alias created)")
                                    else:
                                        # No remaining pod-only nodes, can't match
                                        if VERBOSE_LEVEL >= 1:
                                            print(f"[DEBUG] Ultimate fallback: No remaining pod-only nodes to match {host_e}")
                                        break
                
                break
        if cluster_labels_found:
            break
    
    # Step 2: If no explicit cluster labels, try to query kube_pod_info and kube_node_info for cluster identification
    if not cluster_labels_found:
        try:
            now_ts = int(pd.Timestamp.now().timestamp())
            start_ts = int((pd.Timestamp.now() - pd.Timedelta(hours=lookback_hours)).timestamp())
            
            # First try kube_pod_info - it has pod-to-node mapping and may have cluster labels
            kube_pod_query = 'kube_pod_info'
            kube_pod_df = fetch_victoriametrics_metrics(
                query=kube_pod_query,
                start=start_ts,
                end=now_ts,
                step="60s"
            )
            
            if not kube_pod_df.empty:
                # Check for cluster labels in kube_pod_info
                cluster_label_candidates = ['cluster', 'cluster_id', 'cluster_name', 'cluster_label',
                                           'kubernetes_cluster', 'k8s_cluster', 'label_cluster']
                for label in cluster_label_candidates:
                    if label in kube_pod_df.columns:
                        cluster_labels_found = True
                        # Map nodes to their cluster via pod-to-node mapping
                        # Only assign nodes that have pods (nodes_with_pods set)
                        for _, row in kube_pod_df.iterrows():
                            node_entity = row.get('node') or row.get('instance')
                            cluster_id = row.get(label)
                            if node_entity and cluster_id:
                                node_entity = canonical_identity(node_entity)
                                # Only assign to cluster if node has pods (check canonicalized identity)
                                if node_entity in nodes_with_pods:
                                    cluster_map[node_entity] = str(cluster_id)
                        if cluster_labels_found:
                            break
                
                # If no cluster label but we have pod info, infer clusters by grouping nodes that share pods
                # Nodes in the same cluster typically share system pods (kube-system namespace)
                if not cluster_labels_found and 'node' in kube_pod_df.columns:
                    # Build node-to-pods mapping
                    node_to_pods = {}
                    for _, row in kube_pod_df.iterrows():
                        node = row.get('node')
                        pod = row.get('pod') or row.get('name')
                        namespace = row.get('namespace', '')
                        if node and pod:
                            node_normalized = canonical_identity(node)
                            if node_normalized not in node_to_pods:
                                node_to_pods[node_normalized] = {'pods': set(), 'namespaces': set()}
                            node_to_pods[node_normalized]['pods'].add(str(pod))
                            if namespace:
                                node_to_pods[node_normalized]['namespaces'].add(str(namespace))
                    
                    if len(node_to_pods) > 0:
                        # Group nodes into clusters based on pod sharing
                        # Strategy: Nodes that share pods (especially kube-system namespace pods) are in the same cluster
                        cluster_groups = {}  # cluster_id -> set of nodes
                        cluster_counter = 0
                        
                        # First, identify nodes with kube-system namespace - strongest indicator of cluster membership
                        # Nodes in the same cluster will have pods in kube-system namespace
                        nodes_with_kube_system = {}
                        for node, data in node_to_pods.items():
                            if 'kube-system' in data['namespaces']:
                                nodes_with_kube_system[node] = data
                        
                        # Group nodes by shared kube-system namespace (strongest signal)
                        # If nodes both have kube-system pods, they're in the same cluster
                        for node, data in nodes_with_kube_system.items():
                            assigned = False
                            for cluster_id, cluster_nodes in cluster_groups.items():
                                # Check if any node in this cluster also has kube-system namespace
                                for cluster_node in cluster_nodes:
                                    if cluster_node in nodes_with_kube_system:
                                        # Both have kube-system namespace - same cluster
                                        cluster_groups[cluster_id].add(node)
                                        cluster_map[node] = cluster_id
                                        assigned = True
                                        break
                                if assigned:
                                    break
                            
                            if not assigned:
                                # Create new cluster
                                cluster_id = f"inferred_cluster_{cluster_counter}"
                                cluster_counter += 1
                                cluster_groups[cluster_id] = {node}
                                cluster_map[node] = cluster_id
                        
                        # Now handle nodes without system pods - group by general pod sharing
                        for node, data in node_to_pods.items():
                            if node in cluster_map:  # Already assigned
                                continue
                            
                            assigned = False
                            # Try to find a cluster where nodes share at least some pods
                            for cluster_id, cluster_nodes in cluster_groups.items():
                                for cluster_node in cluster_nodes:
                                    if cluster_node in node_to_pods:
                                        shared = data['pods'] & node_to_pods[cluster_node]['pods']
                                        # If they share at least 2 pods or 10% of pods, likely same cluster
                                        min_shared = max(2, min(len(data['pods']), len(node_to_pods[cluster_node]['pods'])) * 0.1)
                                        if len(shared) >= min_shared:
                                            cluster_groups[cluster_id].add(node)
                                            cluster_map[node] = cluster_id
                                            assigned = True
                                            break
                                if assigned:
                                    break
                            
                            if not assigned:
                                # Create new cluster for isolated nodes
                                cluster_id = f"inferred_cluster_{cluster_counter}"
                                cluster_counter += 1
                                cluster_groups[cluster_id] = {node}
                                cluster_map[node] = cluster_id
                        
                        if cluster_map:
                            cluster_labels_found = True  # Mark as found so we don't override
            
            # If still no cluster found, try kube_node_info
            if not cluster_labels_found:
                kube_node_query = 'kube_node_info'
                kube_node_df = fetch_victoriametrics_metrics(
                    query=kube_node_query,
                    start=start_ts,
                    end=now_ts,
                    step="60s"
                )
                if not kube_node_df.empty:
                    cluster_label_candidates = ['cluster', 'cluster_id', 'cluster_name', 'cluster_label',
                                               'kubernetes_cluster', 'k8s_cluster', 'label_cluster']
                    for label in cluster_label_candidates:
                        if label in kube_node_df.columns:
                            cluster_labels_found = True
                            for _, row in kube_node_df.iterrows():
                                node_entity = row.get('node') or row.get('instance')
                                cluster_id = row.get(label)
                                if node_entity and cluster_id:
                                    node_entity = canonical_identity(node_entity)
                                    # Only assign to cluster if node has pods (check canonicalized identity)
                                    if node_entity in nodes_with_pods:
                                        cluster_map[node_entity] = str(cluster_id)
                            if cluster_labels_found:
                                break
        except Exception as e:
            log_verbose(f"Failed to query kube_pod_info/kube_node_info for cluster identification: {e}", level=2)
    
    # Step 3: If still no cluster labels found, group all nodes with pods into a single inferred cluster
    # This is the safest default: if we can't determine clusters explicitly, assume all K8s nodes are in one cluster
    # Only split into multiple clusters if we have explicit evidence (labels) that they're different
    if not cluster_labels_found:
        # Get all nodes that have pod metrics (these are Kubernetes nodes)
        # Use canonical_identity to normalize both hostnames and IPs for proper matching
        nodes_with_pods = set()
        if not df_pod_cpu.empty or not df_pod_mem.empty:
            for df in [df_pod_cpu, df_pod_mem]:
                if df.empty:
                    continue
                recent_df = df[df['timestamp'] >= start] if 'timestamp' in df.columns else df
                entity_col = 'entity' if 'entity' in recent_df.columns else 'instance'
                # Canonicalize all pod entities to ensure proper matching with host entities
                for entity in recent_df[entity_col].unique():
                    nodes_with_pods.add(canonical_identity(entity))
        
        # If we have nodes with pods but no explicit cluster labels, put them all in one inferred cluster
        # This is better than creating separate clusters per node - we assume they're all in the same cluster
        # until we have explicit evidence (labels) that they're different
        if nodes_with_pods:
            for node in nodes_with_pods:
                node_normalized = canonical_identity(node)
                if node_normalized not in cluster_map:
                    # All nodes with pods go into a single inferred cluster when we can't determine clusters
                    # Use 'inferred_cluster_0' to indicate this is an inferred cluster (not unknown)
                    cluster_map[node_normalized] = 'inferred_cluster_0'
    
    # Step 4: Mark nodes without pod metrics as standalone (not in any Kubernetes cluster)
    all_host_entities = set()
    for df in [df_host_cpu, df_host_mem]:
        if df.empty:
            continue
        recent_df = df[df['timestamp'] >= start] if 'timestamp' in df.columns else df
        entity_col = 'entity' if 'entity' in recent_df.columns else 'instance'
        all_host_entities.update(recent_df[entity_col].unique())
    
    standalone_nodes = []
    cluster_nodes = []
    for entity in all_host_entities:
        entity_normalized = canonical_identity(entity)
        if entity_normalized not in cluster_map:
            # No pods = standalone (not a Kubernetes cluster node)
            cluster_map[entity_normalized] = 'standalone'
            standalone_nodes.append((entity, entity_normalized))
        else:
            cluster_nodes.append((entity, entity_normalized, cluster_map[entity_normalized]))
    
    # Debug output
    if VERBOSE_LEVEL >= 1:
        print(f"\n[DEBUG] Final cluster_map: {dict(sorted(cluster_map.items()))}")
        print(f"[DEBUG] Nodes assigned to clusters: {cluster_nodes}")
        print(f"[DEBUG] Standalone nodes (no pods or not matched): {standalone_nodes}")
        print(f"[DEBUG] Total nodes in cluster_map: {len(cluster_map)}")
    
    return cluster_map

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
    
    # Identify clusters - this groups nodes by their Kubernetes cluster membership
    cluster_map = identify_clusters(df_host_cpu, df_host_mem, df_pod_cpu, df_pod_mem, lookback_hours)
    feats['cluster_id'] = feats['entity'].map(lambda e: cluster_map.get(e, 'standalone'))
    
    # Track which nodes actually have pod metrics (not just filled with 0)
    # This helps distinguish "no pods" from "low pod usage"
    now = pd.Timestamp.now()
    start = now - pd.Timedelta(hours=lookback_hours)
    def has_pod_metrics(entity):
        """Check if entity has actual pod metrics (not just NaN filled with 0)"""
        if df_pod_cpu.empty and df_pod_mem.empty:
            return False
        has_cpu = False
        has_mem = False
        if not df_pod_cpu.empty:
            pod_cpu_data = df_pod_cpu[(df_pod_cpu['timestamp'] >= start)]
            if not pod_cpu_data.empty:
                if 'entity' not in pod_cpu_data.columns:
                    pod_cpu_data = pod_cpu_data.rename(columns={'instance': 'entity'})
                has_cpu = entity in pod_cpu_data['entity'].values
        if not df_pod_mem.empty:
            pod_mem_data = df_pod_mem[(df_pod_mem['timestamp'] >= start)]
            if not pod_mem_data.empty:
                if 'entity' not in pod_mem_data.columns:
                    pod_mem_data = pod_mem_data.rename(columns={'instance': 'entity'})
                has_mem = entity in pod_mem_data['entity'].values
        return has_cpu or has_mem
    
    feats['has_pod_metrics'] = feats['entity'].apply(has_pod_metrics)
    feats['instance_label'] = feats.apply(
        lambda row: canonical_node_label(row['entity'], with_ip=True, raw_label=row.get('raw_instance')),
        axis=1
    )
    dump_dataframe_to_csv(feats.copy(), dump_csv_dir, "classification_features")
    
    # Train separate IsolationForest models per cluster
    # Compare nodes only against their own cluster baseline
    feats['anomaly'] = 1  # Default to normal
    all_labels = []
    all_feats_indices = []
    
    cluster_summary = {}
    for cluster_id in feats['cluster_id'].unique():
        cluster_feats = feats[feats['cluster_id'] == cluster_id].copy()
        
        # Skip anomaly detection for:
        # - Standalone nodes (no pods, not in clusters)
        # - Unknown clusters (have pods but cluster can't be determined - legacy)
        # - Single-node clusters (need at least 2 nodes to compare)
        # inferred_cluster_0 is a valid cluster and should be processed
        if cluster_id == 'standalone' or (cluster_id == 'unknown_cluster' and not cluster_id.startswith('inferred_cluster')) or len(cluster_feats) < 2:
            cluster_feats['anomaly'] = 1  # Mark as normal (no anomaly detection)
            feats.loc[cluster_feats.index, 'anomaly'] = 1
            cluster_summary[cluster_id] = {'total': len(cluster_feats), 'anomalous': 0}
            continue
        
        # Train IsolationForest for this cluster
        scaler = StandardScaler()
        X = scaler.fit_transform(cluster_feats[['host_cpu','host_mem','pod_cpu','pod_mem']])
        iso = IsolationForest(contamination=contamination, random_state=42)
        cluster_labels = iso.fit_predict(X)
        cluster_feats['anomaly'] = cluster_labels
        feats.loc[cluster_feats.index, 'anomaly'] = cluster_labels
        
        all_labels.extend(cluster_labels)
        all_feats_indices.extend(cluster_feats.index.tolist())
        cluster_summary[cluster_id] = {
            'total': len(cluster_feats),
            'anomalous': int((cluster_labels == -1).sum())
        }

    print("\n" + "="*80)
    print("Building Classification (Anomaly) Model...")
    print("="*80)
    
    # Print cluster summary
    if len(cluster_summary) > 1:
        print("\nCluster/Group Summary:")
        for cluster_id, stats in sorted(cluster_summary.items()):
            if cluster_id == 'standalone':
                cluster_name = "Standalone nodes (no Kubernetes)"
            elif cluster_id == 'unknown_cluster':
                cluster_name = "Kubernetes nodes (cluster unknown)"
            elif cluster_id.startswith('inferred_cluster'):
                cluster_name = f"Kubernetes cluster (inferred)"
            else:
                cluster_name = f"Cluster: {cluster_id}"
            print(f"  • {cluster_name}: {stats['total']} nodes, {stats['anomalous']} anomalous")
    
    # Print overall classification report if we have labels
    if all_labels:
        print(classification_report(all_labels, all_labels, digits=2))

    anomalous = feats[feats['anomaly'] == -1]
    if not anomalous.empty:
        # Keep the console output concise so on-call engineers can scan it quickly.
        print("\n⚠️  Anomalous nodes detected:")
        print("   (Unusual resource pattern compared to their cluster/group baseline - check values below)")
        display_cols = anomalous[['entity','raw_instance','host_cpu','host_mem','pod_cpu','pod_mem','cluster_id']].copy()
        display_cols['instance'] = display_cols.apply(
            lambda row: canonical_node_label(row['entity'], with_ip=True, raw_label=row.get('raw_instance')),
            axis=1
        )
        # Convert to percentages and format for readability
        display_output = display_cols[['instance','host_cpu','host_mem','pod_cpu','pod_mem','cluster_id']].copy()
        display_output['Host CPU %'] = (display_output['host_cpu'] * 100).round(1)
        display_output['Host Mem %'] = (display_output['host_mem'] * 100).round(1)
        display_output['Pod CPU %'] = (display_output['pod_cpu'] * 100).round(1)
        display_output['Pod Mem %'] = (display_output['pod_mem'] * 100).round(1)
        
        # Group by cluster for better readability
        if len(display_output['cluster_id'].unique()) > 1:
            for cluster_id in sorted(display_output['cluster_id'].unique()):
                cluster_nodes = display_output[display_output['cluster_id'] == cluster_id]
                if cluster_id == 'standalone':
                    cluster_name = "Standalone nodes"
                elif cluster_id == 'unknown_cluster':
                    cluster_name = "Kubernetes nodes (cluster unknown)"
                else:
                    cluster_name = f"Cluster: {cluster_id}"
                print(f"\n  {cluster_name}:")
                print(cluster_nodes[['instance', 'Host CPU %', 'Host Mem %', 'Pod CPU %', 'Pod Mem %']].to_string(index=False))
        else:
            print(display_output[['instance', 'Host CPU %', 'Host Mem %', 'Pod CPU %', 'Pod Mem %']].to_string(index=False))
        
        # Provide context-aware action based on the pattern
        print("\nAction:")
        for _, row in display_output.iterrows():
            host_cpu = row['Host CPU %']
            host_mem = row['Host Mem %']
            pod_cpu = row['Pod CPU %']
            pod_mem = row['Pod Mem %']
            instance = row['instance']
            
            if pod_cpu >= 90 or pod_mem >= 90:
                print(f"  • {instance}: Pod resources saturated (CPU: {pod_cpu}%, Mem: {pod_mem}%) - investigate pod resource limits/requests")
            elif host_cpu >= 60 or host_mem >= 70:
                if pod_cpu < 30 and pod_mem < 30:
                    print(f"  • {instance}: High host usage ({host_cpu}% CPU, {host_mem}% Mem) but low pod usage - likely non-Kubernetes workloads")
                else:
                    print(f"  • {instance}: High host usage ({host_cpu}% CPU, {host_mem}% Mem) - investigate system processes")
            else:
                print(f"  • {instance}: Unusual resource pattern - review workload distribution")
    else:
        print("\nNo anomalous nodes – host and pod usage aligned.")

    plt.figure(figsize=(9,6))
    colors = ['red' if a==-1 else 'steelblue' for a in feats['anomaly']]
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
    # Pass has_pod_metrics info to feats for host_pressure_df
    if 'has_pod_metrics' in feats.columns and not host_pressure_df.empty:
        # Merge has_pod_metrics info if available
        if 'entity' in host_pressure_df.columns:
            pod_metrics_map = feats.set_index('entity')['has_pod_metrics'].to_dict()
            host_pressure_df['has_pod_metrics'] = host_pressure_df['entity'].map(pod_metrics_map).fillna(False)

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

    # Return iso model if it was trained, otherwise None
    # iso is only defined when at least one cluster has 2+ nodes and a model was trained
    iso_model = None
    if all_labels:  # If any models were trained, we have at least one iso instance
        # We don't actually need to return iso since we're doing per-cluster training
        # But for backward compatibility, return None
        iso_model = None
    
    return feats, iso_model, anomalous_df, host_pressure_df

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
                    # Convert to percentages and format for readability
                    display_output = display[['instance','host_cpu','host_mem','pod_cpu','pod_mem']].copy()
                    display_output['Host CPU %'] = (display_output['host_cpu'] * 100).round(1)
                    display_output['Host Mem %'] = (display_output['host_mem'] * 100).round(1)
                    display_output['Pod CPU %'] = (display_output['pod_cpu'] * 100).round(1)
                    display_output['Pod Mem %'] = (display_output['pod_mem'] * 100).round(1)
                    print(display_output[['instance', 'Host CPU %', 'Host Mem %', 'Pod CPU %', 'Pod Mem %']].to_string(index=False))
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
def _process_single_node_io_crisis(inst, group, name, thresholds, units, test_days, horizon_days, 
                                    force_retrain, retrain_targets_set, retrain_all, retrain_targets_canon,
                                    manifest_snapshot, forecast_mode, dump_csv_dir, plot_dir, enable_plots, show_backtest):
    """
    Worker function to process a single node for I/O crisis prediction.
    This function is designed to be called in parallel.
    Returns: (result_dict, updated_model_dict, manifest_changed_bool) or None if skipped
    """
    try:
        node = canonical_node_label(inst, with_ip=True)
        entity = canonical_identity(inst)
        ts = group.set_index('timestamp')['value'].sort_index()
        if len(ts) < 100:
            return None
        
        dump_label = f"io_crisis_{node}_{name}"
        
        # Train/test split
        test_cutoff = ts.index[-1] - pd.Timedelta(days=test_days)
        train = ts[ts.index <= test_cutoff]
        test = ts[ts.index > test_cutoff]
        
        if len(train) < 50 or len(test) < 10:
            return None
        
        current = ts.iloc[-1]
        threshold = thresholds[name]
        
        # Linear 7d burst
        train_last_7d = train.loc[train.index >= (train.index[-1] - pd.Timedelta(days=7))]
        trend_7d = train_last_7d.diff().mean() * 1440
        linear_eta = 9999.0
        if trend_7d > 0:
            remaining = threshold - current
            divisor = trend_7d / 100 if units[name] == "ratio" else trend_7d
            linear_eta_calc = remaining / divisor
            linear_eta = max(0.0, linear_eta_calc)
        
        # Prophet - use manifest (with _backtest suffix to avoid conflicts)
        key = f"{build_io_net_key(entity, name)}_backtest"
        
        # Determine if retraining is needed
        needs_retrain = force_retrain or retrain_all
        if not needs_retrain and retrain_targets_set:
            entity_match = entity in retrain_targets_set
            key_match = key in retrain_targets_set
            instance_canon = canonical_identity(inst)
            instance_match = instance_canon in retrain_targets_set
            node_base = node.split('(')[0].strip() if '(' in node else node
            node_base_canon = canonical_identity(node_base)
            node_match = node_base_canon in retrain_targets_set
            
            alias_match = False
            if not (entity_match or key_match or instance_match or node_match):
                for target, target_canon in retrain_targets_canon.items():
                    if target_canon == entity:
                        alias_match = True
                        break
                    if target_canon in INSTANCE_ALIAS_MAP:
                        alias_value = INSTANCE_ALIAS_MAP[target_canon]
                        if canonical_identity(alias_value) == entity:
                            alias_match = True
                            break
                    if alias_match:
                        break
            
            needs_retrain = entity_match or key_match or instance_match or node_match or alias_match
        
        # Load or train model
        m = None
        model_updated = False
        
        if not needs_retrain and key in manifest_snapshot:
            m = manifest_snapshot[key].get('model')
            if m is not None and forecast_mode:
                recent_train = train.loc[train.index >= (train.index[-1] - pd.Timedelta(days=7))] if len(train) > 7*24*6 else train
                if len(recent_train) >= 50:
                    pdf = recent_train.reset_index().rename(columns={'timestamp': 'ds', 'value': 'y'})
                    m_updated = Prophet(changepoint_prior_scale=0.2, daily_seasonality=True, 
                                      weekly_seasonality=True, n_changepoints=10)
                    m_updated.fit(pdf)
                    m = m_updated
                    model_updated = True
        
        if needs_retrain or key not in manifest_snapshot or m is None:
            if needs_retrain and key in manifest_snapshot:
                recent_train = train.loc[train.index >= (train.index[-1] - pd.Timedelta(days=7))] if len(train) > 7*24*6 else train
                pdf = recent_train.reset_index().rename(columns={'timestamp': 'ds', 'value': 'y'})
                m = Prophet(changepoint_prior_scale=0.2, daily_seasonality=True, 
                           weekly_seasonality=True, n_changepoints=10)
                m.fit(pdf)
                model_updated = True
            else:
                pdf = train.reset_index().rename(columns={'timestamp': 'ds', 'value': 'y'})
                m = Prophet(changepoint_prior_scale=0.2, daily_seasonality=True, 
                           weekly_seasonality=True, n_changepoints=15)
                m.fit(pdf)
                model_updated = True
        
        if m is None:
            return None
        
        # Forecast
        future = m.make_future_dataframe(periods=(test_days + horizon_days) * 1440, freq='min')
        forecast = m.predict(future)
        
        # Backtest
        test_forecast = forecast.set_index('ds').reindex(test.index, method='nearest')
        mae = mean_absolute_error(test, test_forecast['yhat'])
        rmse = np.sqrt(mean_squared_error(test, test_forecast['yhat']))
        
        # Prophet ETA
        future_pred = forecast[forecast['ds'] > ts.index[-1]]
        crisis = future_pred[future_pred['yhat'] >= threshold]
        prophet_eta = max(0.0, (crisis.iloc[0]['ds'] - pd.Timestamp.now()).total_seconds() / 86400) if not crisis.empty else 9999.0
        
        hybrid_eta = max(0.0, min(linear_eta, prophet_eta))
        
        severity = "CRITICAL" if hybrid_eta < 3 else "WARNING" if hybrid_eta < 7 else "SOON" if hybrid_eta < 30 else "OK"
        
        # Build result
        result = {
            "node": node,
            "signal": name.replace("_", " "),
            "current": f"{current*100:.2f}%" if units[name] == "ratio" else f"{current/1e9:.2f} GB/s",
            "mae": round(mae, 6),
            "hybrid_eta_days": round(hybrid_eta, 1),
            "severity": severity
        } if hybrid_eta < 30 else None
        
        # Return result with model update info
        return {
            'result': result,
            'key': key,
            'model': m if model_updated else None,
            'needs_retrain': needs_retrain
        }
    except Exception as e:
        log_verbose(f"  Error processing {inst}: {e}")
        return None

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
        # Pre-compute retrain matching logic once (outside loop) for performance
        # Only do expensive DNS lookups if retrain_targets is not empty
        retrain_targets_set = set(retrain_targets) if retrain_targets else set()
        has_retrain_targets = len(retrain_targets_set) > 0
        retrain_all = '__RETRAIN_ALL__' in retrain_targets_set if has_retrain_targets else False
        
        # Pre-compute canonical identities for retrain targets (avoid repeated canonicalization)
        retrain_targets_canon = {}
        if has_retrain_targets and not retrain_all:
            for target in retrain_targets_set:
                if '|' not in target and '_' not in target:  # Only node names, not keys
                    retrain_targets_canon[target] = canonical_identity(target)
        
        # Progress reporting for large node counts (reduces perceived slowness)
        node_groups = list(df.groupby('instance'))
        total_nodes = len(node_groups)
        show_progress = total_nodes > 20
        # If --parallel flag is set, bypass threshold and use parallel processing
        # Otherwise, only parallelize if we have enough nodes to justify the overhead
        # Use min of available workers and node count to avoid over-subscription
        use_parallel = (CLI_PARALLEL_OVERRIDE is not None) or (total_nodes > 10 and MAX_WORKER_THREADS > 1)
        n_workers = min(total_nodes, MAX_WORKER_THREADS) if use_parallel else 1
        
        if show_progress or use_parallel or total_nodes > 5:
            if use_parallel:
                print(f"  Processing {total_nodes} nodes for {name} in PARALLEL mode:")
                print(f"    ├─ Available workers: {MAX_WORKER_THREADS}")
                print(f"    ├─ Workers used: {n_workers} (min({total_nodes}, {MAX_WORKER_THREADS}))")
                print(f"    └─ Expected speedup: ~{n_workers}x (vs sequential)")
            else:
                print(f"  Processing {total_nodes} nodes for {name} in SEQUENTIAL mode:")
                print(f"    ├─ Available workers: {MAX_WORKER_THREADS}")
                if CLI_PARALLEL_OVERRIDE is None:
                    reason = 'Too few items (<10)' if total_nodes <= 10 else 'Single worker only'
                else:
                    reason = 'Single worker only (MAX_WORKER_THREADS=1)'
                print(f"    ├─ Reason: {reason}")
                print(f"    └─ Workers used: 1")
        
        # Process nodes in parallel or sequentially
        if use_parallel:
            # Parallel processing
            print(f"    → Starting parallel execution with {n_workers} workers...")
            manifest_snapshot = manifest.copy()  # Snapshot for parallel workers
            processed_results = Parallel(n_jobs=n_workers, verbose=0)(
                delayed(_process_single_node_io_crisis)(
                    inst, group, name, thresholds, units, test_days, horizon_days,
                    force_retrain, retrain_targets_set, retrain_all, retrain_targets_canon,
                    manifest_snapshot, forecast_mode, dump_csv_dir, plot_dir, enable_plots, show_backtest
                )
                for inst, group in node_groups
            )
            
            # Process results and update manifest
            for idx, proc_result in enumerate(processed_results):
                if proc_result is None:
                    continue
                
                if proc_result['result']:
                    results.append(proc_result['result'])
                
                # Update manifest with new/updated models
                if proc_result['model'] is not None:
                    manifest[proc_result['key']] = {'model': proc_result['model']}
                    manifest_changed = True
                
                processed_nodes += 1
                if show_progress and (idx + 1) % 10 == 0:
                    print(f"    → Progress: {idx + 1}/{total_nodes} nodes processed...", end='\r')
            
            if use_parallel:
                print()  # New line after progress indicator
                successful_nodes = len([r for r in processed_results if r is not None])
                print(f"    ✓ Parallel execution complete: {successful_nodes}/{total_nodes} nodes processed successfully")
        else:
            # Sequential processing (original code)
            for idx, (inst, group) in enumerate(node_groups, 1):
                if show_progress and idx % 10 == 0:
                    print(f"  Progress: {idx}/{total_nodes} nodes processed...", end='\r')
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
            train_last_7d = train.loc[train.index >= (train.index[-1] - pd.Timedelta(days=7))]
            trend_7d = train_last_7d.diff().mean() * 1440
            linear_eta = 9999.0
            if trend_7d > 0:
                remaining = threshold - current
                divisor = trend_7d / 100 if units[name] == "ratio" else trend_7d
                linear_eta_calc = remaining / divisor
                # Ensure non-negative
                linear_eta = max(0.0, linear_eta_calc)

            # Prophet - use manifest (with _backtest suffix to avoid conflicts)
            key = f"{build_io_net_key(entity, name)}_backtest"
            
            # OPTIMIZED: Fast path when no retraining needed
            needs_retrain = force_retrain or retrain_all
            if not needs_retrain and has_retrain_targets:
                # Fast checks first (no DNS lookups)
                entity_match = entity in retrain_targets_set
                key_match = key in retrain_targets_set
                instance_canon = canonical_identity(inst)
                instance_match = instance_canon in retrain_targets_set
                node_base = node.split('(')[0].strip() if '(' in node else node
                node_base_canon = canonical_identity(node_base)
                node_match = node_base_canon in retrain_targets_set
                
                # Check alias map (fast, no DNS)
                alias_match = False
                if not (entity_match or key_match or instance_match or node_match):
                    # Only check alias map if simple matches failed
                    for target, target_canon in retrain_targets_canon.items():
                        if target_canon == entity:
                            alias_match = True
                            break
                        # Check alias map (cached, no DNS)
                        if target_canon in INSTANCE_ALIAS_MAP:
                            alias_value = INSTANCE_ALIAS_MAP[target_canon]
                            if canonical_identity(alias_value) == entity:
                                alias_match = True
                                break
                        # Check reverse alias map
                        for k, v in INSTANCE_ALIAS_MAP.items():
                            if canonical_identity(v) == entity and canonical_identity(k) == target_canon:
                                alias_match = True
                                break
                        if alias_match:
                            break
                        # Check IP registry (fast, no DNS)
                        target_ip = SOURCE_REGISTRY.get(target_canon) or CANON_SOURCE_MAP.get(target_canon)
                        entity_ip = SOURCE_REGISTRY.get(entity) or CANON_SOURCE_MAP.get(entity)
                        if target_ip and entity_ip and target_ip == entity_ip:
                            alias_match = True
                            break
                
                # Only do expensive DNS lookups as last resort (and only if still no match)
                if not alias_match and not (entity_match or key_match or instance_match or node_match):
                    # DNS lookup only for remaining unmatched targets (expensive, do last)
                    if '(' in node and ')' in node:
                        node_ip = node.split('(')[1].split(')')[0].strip()
                        for target, target_canon in retrain_targets_canon.items():
                            if looks_like_hostname(target):
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
                            if alias_match:
                                break
                
                needs_retrain = entity_match or key_match or instance_match or node_match or alias_match
            
            if not needs_retrain and key in manifest:
                m = manifest[key].get('model')
                if m is not None:
                    log_verbose(f"  → Loaded model from manifest: {key}")
                    # MINIMAL UPDATE: Use recent data only (last 7 days) for forecast mode
                    # This incorporates latest trends while preserving learned patterns
                    # Minimal updates are required in --forecast, --disk-retrain, and --io-net-retrain modes
                    if forecast_mode:
                        # This incorporates latest trends while preserving learned patterns
                        # train is a Series with timestamp index, so use last('7D')
                        recent_train = train.loc[train.index >= (train.index[-1] - pd.Timedelta(days=7))] if len(train) > 7*24*6 else train
                        if len(recent_train) >= 50:  # Only update if we have enough recent data
                            pdf = recent_train.reset_index().rename(columns={'timestamp': 'ds', 'value': 'y'})
                            # OPTIMIZATION: Use faster Prophet settings for minimal updates
                            # Use fewer changepoints and simpler seasonality for speed
                            m_updated = Prophet(
                                changepoint_prior_scale=0.2,
                                daily_seasonality=True,
                                weekly_seasonality=True,
                                n_changepoints=10  # Reduce changepoints from default 25 for faster fitting
                            )
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
                    recent_train = train.loc[train.index >= (train.index[-1] - pd.Timedelta(days=7))] if len(train) > 7*24*6 else train
                    pdf = recent_train.reset_index().rename(columns={'timestamp': 'ds', 'value': 'y'})
                    # OPTIMIZATION: Use faster Prophet settings for minimal updates
                    m = Prophet(changepoint_prior_scale=0.2, daily_seasonality=True, weekly_seasonality=True, n_changepoints=10)
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
                    # OPTIMIZATION: Use faster Prophet settings for first-time training
                    m = Prophet(changepoint_prior_scale=0.2, daily_seasonality=True, weekly_seasonality=True, n_changepoints=15)
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
            prophet_eta_calc = (crisis.iloc[0]['ds'] - pd.Timestamp.now()).total_seconds() / 86400 if not crisis.empty else 9999.0
            # Ensure non-negative
            prophet_eta = max(0.0, prophet_eta_calc)

            # Hybrid ETA - ensure non-negative
            hybrid_eta = max(0.0, min(linear_eta, prophet_eta))

            # Severity is always defined
            if hybrid_eta < 3:
                severity = "CRITICAL"
            elif hybrid_eta < 7:
                severity = "WARNING"
            elif hybrid_eta < 30:
                severity = "SOON"
            else:
                severity = "OK"

            # OPTIMIZATION: Skip plot generation unless explicitly needed (plots are expensive)
            # Save backtest plot when training/retraining or when show_backtest is True
            if (needs_retrain or show_backtest) and enable_plots:
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
                # Sanitize node name for filename
                safe_node = node.split('(')[0].strip().replace(' ', '_').replace('/', '_')
                plot_file = os.path.join(plot_dir, f"{safe_node}_{name.lower().replace(' ', '_')}_backtest.png")
                plt.savefig(plot_file, dpi=180, bbox_inches='tight')
                log_verbose(f"  → Plot saved: {plot_file}")
                plt.close()
            
            # OPTIMIZATION: Skip forecast plots in forecast mode unless enable_plots is True
            # Save forecast plot in forecast mode (showing future predictions, not backtest)
            if forecast_mode and enable_plots:
                plt.figure(figsize=(14, 7))
                # Plot last 24 hours of historical data
                historical = ts.loc[ts.index >= (ts.index[-1] - pd.Timedelta(hours=24))]
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
                    "hybrid_eta_days": round(max(0.0, hybrid_eta), 1),
                    "severity": severity
                })

            processed_nodes += 1

        if show_progress:
            print()  # New line after progress indicator
        
        summary = [r for r in results if r['signal'] == name.replace("_", " ")]
        print(f"{name}: processed {processed_nodes} nodes, crises <30d: {len(summary)}")

    results_df = pd.DataFrame(results) if results else pd.DataFrame(columns=["node","signal","current","mae","hybrid_eta_days","severity"])
    return results_df, manifest, manifest_changed

# ----------------------------------------------------------------------
# 8. I/O AND NETWORK ENSEMBLE FORECASTING
# ----------------------------------------------------------------------
def format_anomaly_description(node, signal, current_val, current_str, mae_ensemble, score, severity, deviation_pct, unit):
    """Create human-readable description of an anomaly."""
    signal_lower = signal.lower()
    
    # Determine what the signal measures
    if "disk io wait" in signal_lower or "iowait" in signal_lower:
        signal_name = "Disk I/O Wait"
        what_it_means = "how much time the CPU spends waiting for disk operations"
        impact = "High I/O wait means the system is waiting for storage, causing slowdowns"
    elif "net tx" in signal_lower or "network" in signal_lower:
        signal_name = "Network Transmit"
        what_it_means = "outgoing network bandwidth"
        impact = "Network issues can cause slow data transfers and connectivity problems"
    else:
        signal_name = signal
        what_it_means = "system performance metric"
        impact = "Anomalies indicate unexpected behavior"
    
    # Interpret the severity and score
    if severity == "CRITICAL":
        urgency = "⚠️  URGENT: Requires immediate attention"
    elif severity == "WARNING":
        urgency = "⚠️  Monitor closely"
    else:
        urgency = "ℹ️  Informational - keep an eye on it"
    
    # Interpret deviation
    if deviation_pct > 50:
        deviation_desc = f"significantly different ({deviation_pct:.1f}% off from expected)"
    elif deviation_pct > 20:
        deviation_desc = f"noticeably different ({deviation_pct:.1f}% off from expected)"
    else:
        deviation_desc = f"slightly different ({deviation_pct:.1f}% off from expected)"
    
    # Interpret MAE
    if unit == "ratio":
        if mae_ensemble > 0.05:
            mae_desc = "Model is struggling to predict this pattern accurately"
        elif mae_ensemble > 0.01:
            mae_desc = "Model has moderate prediction accuracy"
        else:
            mae_desc = "Model predictions are fairly accurate"
    else:  # bytes/sec
        threshold = 120_000_000 * 0.10  # 10% of network threshold
        if mae_ensemble > threshold:
            mae_desc = "Model is struggling to predict this pattern accurately"
        elif mae_ensemble > threshold * 0.5:
            mae_desc = "Model has moderate prediction accuracy"
        else:
            mae_desc = "Model predictions are fairly accurate"
    
    # Build the description
    description = (
        f"{urgency}\n"
        f"  • Node: {node}\n"
        f"  • Metric: {signal_name} ({what_it_means})\n"
        f"  • Current Value: {current_str}\n"
        f"  • What's Wrong: The actual value is {deviation_desc}\n"
        f"  • Prediction Quality: {mae_desc}\n"
        f"  • Impact: {impact}"
    )
    
    return description

def _process_single_node_io_ensemble(instance, group, res, test_days, horizon_days, force_retrain, retrain_targets_set,
                                     retrain_all, retrain_targets_canon, manifest_snapshot, forecast_mode,
                                     dump_csv_dir, plot_dir, enable_plots, show_backtest):
    """
    Worker function to process a single node for I/O and Network ensemble forecasting.
    This function is designed to be called in parallel.
    Returns: dict with crisis_result, anomaly_result, backtest_metrics, retrained_node, key, model, or None if skipped
    """
    try:
        node = canonical_node_label(instance, with_ip=True)
        entity = canonical_identity(instance)
        ts = group.set_index('timestamp')['value'].sort_index()
        if len(ts) < 200:
            return None
        
        cutoff = ts.index[-1] - pd.Timedelta(days=test_days)
        train_raw = ts[ts.index <= cutoff]
        if len(train_raw) < 100:
            return None
        
        current = ts.iloc[-1]
        train_df = train_raw.reset_index()
        train_df.columns = ['timestamp', 'value']
        
        key = f"{build_io_net_key(entity, res['name'])}_ensemble"
        log_verbose(f"  Running ensemble forecast for {node} | {res['name']} ({len(train_df)} points)...")
        
        # ———— RETRAIN MATCHING LOGIC ————
        # Check if retraining is needed - match against entity, key, or any aliases
        needs_retrain = force_retrain or retrain_all
        if not needs_retrain and retrain_targets_set:
            entity_match = entity in retrain_targets_set
            key_match = key in retrain_targets_set
            instance_canon = canonical_identity(instance)
            instance_match = instance_canon in retrain_targets_set
            node_base = node.split('(')[0].strip() if '(' in node else node
            node_base_canon = canonical_identity(node_base)
            node_match = node_base_canon in retrain_targets_set
            alias_match = False
            if not (entity_match or key_match or instance_match or node_match):
                for target, target_canon in retrain_targets_canon.items():
                    if target_canon == entity:
                        alias_match = True
                        break
            needs_retrain = entity_match or key_match or instance_match or node_match or alias_match
        
        retrained_node = None
        if needs_retrain:
            retrained_node = f"{node} ({entity})"
            log_verbose(f"   Retraining requested for {node} | {res['name']} (entity: {entity})")
        
        # ———— MODEL CACHING AND TRAINING ————
        forecast_result = None
        manifest_key_updated = False
        
        if not needs_retrain and key in manifest_snapshot:
            forecast_result = manifest_snapshot[key].get('model')
            if forecast_result is not None:
                log_verbose(f"   Loaded ENSEMBLE model from manifest: {key}")
                # MINIMAL UPDATE: Use recent data only (last 7 days) for forecast mode only
                if forecast_mode:
                    train_df_sorted = train_df.sort_values('timestamp')
                    cutoff_time = train_df_sorted['timestamp'].max() - pd.Timedelta(days=7)
                    recent_train_df = train_df_sorted[train_df_sorted['timestamp'] >= cutoff_time]
                    if len(recent_train_df) < 50:
                        recent_train_df = train_df_sorted.tail(min(len(train_df_sorted), 7*24*6))
                    forecast_result = build_ensemble_forecast_model(
                        df_cpu=recent_train_df,
                        df_mem=None,
                        horizon_min=horizon_days * 24 * 60,
                        model_path=None,
                        context={'node': node, 'signal': res['name']},
                        save_forecast_plot=True,
                        save_backtest_plot=False,
                        print_backtest_metrics=False,
                        dump_csv_dir=dump_csv_dir,
                        enable_plots=enable_plots
                    )
                    if forecast_result is not None:
                        manifest_key_updated = True
                        log_verbose(f"   Minimal update applied (recent 7 days): {key}")
                # If show_backtest is true, compute metrics even for cached models
                if show_backtest:
                    has_metrics = isinstance(forecast_result, tuple) and len(forecast_result) >= 3
                    if not has_metrics:
                        log_verbose(f"   Computing backtest metrics for cached model (display only, not saving)...")
                        forecast_result = build_ensemble_forecast_model(
                            df_cpu=train_df,
                            df_mem=None,
                            horizon_min=horizon_days * 24 * 60,
                            model_path=None,
                            context={'node': node, 'signal': res['name']},
                            save_forecast_plot=False,
                            save_backtest_plot=False,
                            print_backtest_metrics=False,
                            save_model=False,
                            dump_csv_dir=dump_csv_dir,
                            enable_plots=enable_plots
                        )
            else:
                needs_retrain = True
        
        if needs_retrain or key not in manifest_snapshot:
            if key in manifest_snapshot:
                log_verbose(f"   Retraining cached model → MINIMAL UPDATE (recent 7 days)...")
                train_df_sorted = train_df.sort_values('timestamp')
                cutoff_time = train_df_sorted['timestamp'].max() - pd.Timedelta(days=7)
                recent_train_df = train_df_sorted[train_df_sorted['timestamp'] >= cutoff_time]
                if len(recent_train_df) < 50:
                    recent_train_df = train_df_sorted.tail(min(len(train_df_sorted), 7*24*6))
                forecast_result = build_ensemble_forecast_model(
                    df_cpu=recent_train_df,
                    df_mem=None,
                    horizon_min=horizon_days * 24 * 60,
                    model_path=None,
                    context={'node': node, 'signal': res['name']},
                    save_forecast_plot=False,
                    save_backtest_plot=False,
                    print_backtest_metrics=False,
                    dump_csv_dir=dump_csv_dir,
                    enable_plots=enable_plots
                )
            else:
                log_verbose(f"   No cached model → FULL TRAINING...")
                forecast_result = build_ensemble_forecast_model(
                    df_cpu=train_df,
                    df_mem=None,
                    horizon_min=horizon_days * 24 * 60,
                    model_path=None,
                    context={'node': node, 'signal': res['name']},
                    save_forecast_plot=False,
                    save_backtest_plot=False,
                    enable_plots=enable_plots,
                    print_backtest_metrics=False,
                    dump_csv_dir=dump_csv_dir
                )
            if forecast_result is not None:
                manifest_key_updated = True
                log_verbose(f"   Saved ENSEMBLE to manifest → {key}")
        
        if forecast_result is None:
            return None
        
        # ———— SAFE UNPACK ————
        if isinstance(forecast_result, tuple):
            if len(forecast_result) == 3:
                _, forecast_df, metrics = forecast_result
            else:
                _, forecast_df = forecast_result
                metrics = {"mae_ensemble": 0.0}
        else:
            log_verbose(f"   Warning: unexpected forecast_result type for {key}, skipping")
            return None
        
        # ———— CRISIS DETECTION ————
        future_threshold = forecast_df[forecast_df['yhat'] >= res["threshold"]]
        eta_days = 9999.0
        if not future_threshold.empty:
            eta_days_calc = (future_threshold.iloc[0]['ds'] - pd.Timestamp.now()).total_seconds() / 86400
            eta_days = max(0.0, eta_days_calc)
        
        log_verbose(f"  Done → ETA: {eta_days:.1f} days | MAE: {metrics['mae_ensemble']:.6f}")
        
        crisis_result = None
        if eta_days < 30:
            severity = "CRITICAL" if eta_days < 3 else "WARNING" if eta_days < 7 else "SOON"
            crisis_result = {
                "node": node,
                "signal": res["name"].replace("_", " "),
                "current": f"{current*100:.2f}%" if res["unit"] == "ratio" else f"{current/1e6:.1f} MB/s",
                "mae_ensemble": round(metrics.get('mae_ensemble', 0.0), 6),
                "hybrid_eta_days": round(max(0.0, eta_days), 1),
                "severity": severity
            }
        
        # ———— ANOMALY DETECTION ————
        anomaly_result = None
        mae_ensemble = metrics.get('mae_ensemble', 0.0)
        recent_window = pd.Timedelta(hours=24)
        now = pd.Timestamp.now()
        recent_start = now - recent_window
        recent_actual = ts[ts.index >= recent_start]
        
        if len(recent_actual) >= 6:
            forecast_df_sorted = forecast_df.sort_values('ds')
            if not forecast_df_sorted.empty:
                past_forecasts = forecast_df_sorted[forecast_df_sorted['ds'] <= now]
                if not past_forecasts.empty:
                    baseline_forecast = past_forecasts.iloc[-1]['yhat']
                else:
                    baseline_forecast = forecast_df_sorted.iloc[0]['yhat']
                
                recent_mean = recent_actual.mean()
                recent_std = recent_actual.std()
                recent_max = recent_actual.max()
                recent_min = recent_actual.min()
                
                current_deviation_abs = abs(current - baseline_forecast)
                current_deviation_pct = abs((current - baseline_forecast) / baseline_forecast) if baseline_forecast != 0 else (current_deviation_abs if current != 0 else 0)
                mean_deviation_abs = abs(recent_mean - baseline_forecast)
                mean_deviation_pct = abs((recent_mean - baseline_forecast) / baseline_forecast) if baseline_forecast != 0 else (mean_deviation_abs if recent_mean != 0 else 0)
                spike_deviation_abs = abs(recent_max - baseline_forecast)
                spike_deviation = abs((recent_max - baseline_forecast) / baseline_forecast) if baseline_forecast != 0 else (spike_deviation_abs if recent_max != 0 else 0)
                drop_deviation_abs = abs(baseline_forecast - recent_min)
                drop_deviation = abs((baseline_forecast - recent_min) / baseline_forecast) if baseline_forecast != 0 else (drop_deviation_abs if recent_min != 0 else 0)
                
                if res["unit"] == "ratio":
                    min_abs_threshold = 0.01
                    mae_threshold = 0.05
                    concerning_threshold = 0.05
                    baseline_too_small = baseline_forecast < 0.01
                else:
                    min_abs_threshold = 5_000_000
                    mae_threshold = res["threshold"] * 0.10
                    concerning_threshold = res["threshold"] * 0.30
                    baseline_too_small = baseline_forecast < 5_000_000
                
                if res["unit"] == "ratio":
                    mae_confidence_factor = 1.0 if mae_ensemble < 0.005 else (0.7 if mae_ensemble < 0.01 else 0.3)
                else:
                    mae_confidence_factor = 1.0 if mae_ensemble < mae_threshold * 0.1 else (0.7 if mae_ensemble < mae_threshold * 0.3 else 0.3)
                
                is_concerning_value = (
                    (res["unit"] == "ratio" and current >= concerning_threshold) or
                    (res["unit"] != "ratio" and current >= concerning_threshold)
                )
                has_significant_abs_diff = (
                    current_deviation_abs >= min_abs_threshold or
                    mean_deviation_abs >= min_abs_threshold
                )
                
                if baseline_too_small:
                    is_anomaly = has_significant_abs_diff and is_concerning_value
                else:
                    has_significant_pct_diff = (
                        current_deviation_pct > 0.50 or
                        mean_deviation_pct > 0.30
                    )
                    is_anomaly = (
                        has_significant_abs_diff and
                        has_significant_pct_diff and
                        (is_concerning_value or mae_confidence_factor < 0.5 or mae_ensemble > mae_threshold)
                    )
                
                if is_anomaly:
                    anomaly_score = min(1.0, max(
                        current_deviation_pct,
                        mean_deviation_pct,
                        spike_deviation / 2.0,
                        drop_deviation / 2.0,
                        min(1.0, mae_ensemble / mae_threshold) if mae_threshold > 0 else 0
                    ))
                    
                    if anomaly_score > 0.8:
                        severity = "CRITICAL"
                    elif anomaly_score > 0.5:
                        severity = "WARNING"
                    else:
                        severity = "INFO"
                    
                    signal_display = res["name"].replace("_", " ")
                    current_str = f"{current*100:.2f}%" if res["unit"] == "ratio" else f"{current/1e6:.1f} MB/s"
                    
                    description = format_anomaly_description(
                        node, signal_display, current, current_str, mae_ensemble, 
                        anomaly_score, severity, current_deviation_pct * 100, res["unit"]
                    )
                    
                    anomaly_result = {
                        "node": node,
                        "signal": signal_display,
                        "current": current_str,
                        "mae_ensemble": round(mae_ensemble, 6),
                        "score": round(anomaly_score, 3),
                        "severity": severity,
                        "deviation_pct": round(current_deviation_pct * 100, 1),
                        "description": description
                    }
                    
                    log_verbose(f"   ⚠️  Anomaly detected: {node} | {res['name']} (score: {anomaly_score:.3f}, deviation: {current_deviation_pct*100:.1f}%)")
        
        # ———— COLLECT BACKTEST METRICS ————
        backtest_metrics = None
        if (show_backtest or (needs_retrain and not forecast_mode)) and metrics:
            backtest_metrics = {
                'node': node,
                'signal': res['name'],
                'metrics': metrics
            }
        
        return {
            'crisis_result': crisis_result,
            'anomaly_result': anomaly_result,
            'backtest_metrics': backtest_metrics,
            'retrained_node': retrained_node,
            'key': key,
            'model': forecast_result if manifest_key_updated else None,
            'needs_retrain': needs_retrain
        }
    except Exception as e:
        log_verbose(f"  Error processing {instance}: {e}")
        return None

def predict_io_and_network_ensemble(horizon_days=7, test_days=7, plot_dir="forecast_plots", force_retrain: bool | None = None,
                                    manifest: dict | None = None, retrain_targets: set | None = None, show_backtest: bool = False,
                                    forecast_mode: bool = False, dump_csv_dir: str | None = None, enable_plots: bool = True):
    """
    Disk I/O and Network ensemble forecasting using Prophet, ARIMA, and LSTM models.
    
    This function provides comprehensive forecasting and anomaly detection for:
    - Disk I/O wait time (DISK_IO_WAIT): Measures CPU time spent waiting for disk operations
    - Network transmit bandwidth (NET_TX_BW): Measures outgoing network traffic
    
    Features:
    - Manifest-based model storage: Single file storage for all node/signal combinations
    - Lazy model loading: Models are loaded from cache if available, trained only when missing
    - Selective retraining: Supports retraining specific nodes/signals without full retraining
    - Network threshold: 120 MB/s (96% of 1 Gbps) for crisis detection
    - Anomaly detection: Statistical deviation analysis with configurable sensitivity
    
    Returns:
        crisis_df: DataFrame of predicted crises (ETA < 30 days to threshold)
        anomaly_df: DataFrame of detected anomalies (statistical deviations)
        manifest: Updated model manifest dictionary
        manifest_changed: Boolean indicating if manifest was modified
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

        # Prepare for parallelization
        node_groups = list(df.groupby('instance'))
        total_nodes = len(node_groups)
        # If --parallel flag is set, bypass threshold and use parallel processing
        # Otherwise, only parallelize if we have enough nodes to justify the overhead
        use_parallel = (CLI_PARALLEL_OVERRIDE is not None) or (total_nodes > 10 and MAX_WORKER_THREADS > 1)
        n_workers = min(total_nodes, MAX_WORKER_THREADS) if use_parallel else 1
        
        # Pre-compute retrain matching logic
        retrain_targets_set = set(retrain_targets) if retrain_targets else set()
        retrain_all = '__RETRAIN_ALL__' in retrain_targets_set if retrain_targets_set else False
        retrain_targets_canon = {}
        if retrain_targets_set and not retrain_all:
            for target in retrain_targets_set:
                if '|' not in target and '_' not in target:
                    retrain_targets_canon[target] = canonical_identity(target)
        
        if total_nodes > 5:
            if use_parallel:
                print(f"  Processing {total_nodes} nodes for {res['name']} ensemble in PARALLEL mode:")
                print(f"    ├─ Available workers: {MAX_WORKER_THREADS}")
                print(f"    ├─ Workers used: {n_workers} (min({total_nodes}, {MAX_WORKER_THREADS}))")
                print(f"    └─ Expected speedup: ~{n_workers}x (vs sequential)")
            else:
                print(f"  Processing {total_nodes} nodes for {res['name']} ensemble in SEQUENTIAL mode:")
                print(f"    ├─ Available workers: {MAX_WORKER_THREADS}")
                if CLI_PARALLEL_OVERRIDE is None:
                    reason = 'Too few items (<10)' if total_nodes <= 10 else 'Single worker only'
                else:
                    reason = 'Single worker only (MAX_WORKER_THREADS=1)'
                print(f"    ├─ Reason: {reason}")
                print(f"    └─ Workers used: 1")
        
        if use_parallel:
            # Parallel processing
            print(f"    → Starting parallel execution with {n_workers} workers...")
            manifest_snapshot = manifest.copy()  # Snapshot for parallel workers
            processed_results = Parallel(n_jobs=n_workers, verbose=0)(
                delayed(_process_single_node_io_ensemble)(
                    instance, group, res, test_days, horizon_days, force_retrain,
                    retrain_targets_set, retrain_all, retrain_targets_canon, manifest_snapshot,
                    forecast_mode, dump_csv_dir, plot_dir, enable_plots, show_backtest
                )
                for instance, group in node_groups
            )
            
            # Process results and aggregate
            for proc_result in processed_results:
                if proc_result is None:
                    continue
                
                # Update manifest with new/updated models
                if proc_result['model'] is not None:
                    manifest[proc_result['key']] = {'model': proc_result['model']}
                    manifest_changed = True
                
                # Collect crisis results
                if proc_result['crisis_result']:
                    crisis_results.append(proc_result['crisis_result'])
                
                # Collect anomaly results
                if proc_result['anomaly_result']:
                    anomaly_results.append(proc_result['anomaly_result'])
                
                # Collect backtest metrics
                if proc_result['backtest_metrics']:
                    backtest_metrics_list.append(proc_result['backtest_metrics'])
                
                # Track retrained nodes
                if proc_result['retrained_node']:
                    retrained_nodes.add(proc_result['retrained_node'])
            
            print()  # New line after progress indicator
            successful_nodes = len([r for r in processed_results if r is not None])
            print(f"    ✓ Parallel execution complete: {successful_nodes}/{total_nodes} nodes processed successfully")
        else:
            # Sequential processing (original code)
            for instance, group in node_groups:
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
                eta_days_calc = (future_threshold.iloc[0]['ds'] - pd.Timestamp.now()).total_seconds() / 86400
                # Ensure non-negative
                eta_days = max(0.0, eta_days_calc)

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
                    "hybrid_eta_days": round(max(0.0, eta_days), 1),
                    "severity": severity
                })

            # ———— ANOMALY DETECTION ————
            # Compare actual values vs ensemble forecast to detect statistical deviations
            # Method: Check deviation of recent actual values from predicted patterns
            # 
            # IMPROVED LOGIC (to reduce false positives):
            # - Uses BOTH absolute and percentage thresholds (avoids flagging tiny differences)
            # - Only flags if value is ACTUALLY concerning (e.g., I/O wait > 5%, not just different)
            # - Accounts for model confidence (low MAE = trust model more, be conservative)
            # - For very small baselines (< 1% I/O, < 5MB/s network), uses absolute thresholds only
            #   (percentage deviations are misleading when baseline is near zero)
            #
            mae_ensemble = metrics.get('mae_ensemble', 0.0)
            
            # Get recent actual values (last 24 hours) for anomaly detection
            recent_window = pd.Timedelta(hours=24)
            now = pd.Timestamp.now()
            recent_start = now - recent_window
            recent_actual = ts[ts.index >= recent_start]
            
            if len(recent_actual) >= 6:  # Need at least 6 data points (1 hour at 10min intervals)
                # Get forecast values for the same time period (if available in forecast_df)
                # forecast_df contains future predictions, so we need to find the model's prediction
                # for the current/recent time period. We'll use the most recent forecast value as baseline.
                
                # Find the forecast value closest to now (or most recent forecast)
                forecast_df_sorted = forecast_df.sort_values('ds')
                if not forecast_df_sorted.empty:
                    # Get the forecast value at or just before now
                    past_forecasts = forecast_df_sorted[forecast_df_sorted['ds'] <= now]
                    if not past_forecasts.empty:
                        # Use the most recent past forecast as baseline
                        baseline_forecast = past_forecasts.iloc[-1]['yhat']
                    else:
                        # If no past forecasts, use the earliest future forecast as proxy
                        baseline_forecast = forecast_df_sorted.iloc[0]['yhat']
                    
                    # Calculate statistics on recent actual values
                    recent_mean = recent_actual.mean()
                    recent_std = recent_actual.std()
                    recent_max = recent_actual.max()
                    recent_min = recent_actual.min()
                    
                    # Calculate deviation metrics
                    # 1. Current value deviation from baseline forecast (both absolute and percentage)
                    current_deviation_abs = abs(current - baseline_forecast)
                    current_deviation_pct = abs((current - baseline_forecast) / baseline_forecast) if baseline_forecast != 0 else (current_deviation_abs if current != 0 else 0)
                    
                    # 2. Recent mean deviation from baseline
                    mean_deviation_abs = abs(recent_mean - baseline_forecast)
                    mean_deviation_pct = abs((recent_mean - baseline_forecast) / baseline_forecast) if baseline_forecast != 0 else (mean_deviation_abs if recent_mean != 0 else 0)
                    
                    # 3. Check for sudden spikes/drops (recent max/min vs baseline)
                    spike_deviation_abs = abs(recent_max - baseline_forecast)
                    spike_deviation = abs((recent_max - baseline_forecast) / baseline_forecast) if baseline_forecast != 0 else (spike_deviation_abs if recent_max != 0 else 0)
                    drop_deviation_abs = abs(baseline_forecast - recent_min)
                    drop_deviation = abs((baseline_forecast - recent_min) / baseline_forecast) if baseline_forecast != 0 else (drop_deviation_abs if recent_min != 0 else 0)
                    
                    # 4. Define meaningful thresholds to avoid false positives
                    # For I/O wait (ratio): values < 1% are usually fine, > 5% is concerning
                    # For network (bytes/sec): use percentage of threshold
                    if res["unit"] == "ratio":
                        min_abs_threshold = 0.01  # 1% - minimum absolute difference to care about
                        mae_threshold = 0.05  # 5% for ratio signals
                        concerning_threshold = 0.05  # 5% I/O wait is actually concerning
                        baseline_too_small = baseline_forecast < 0.01  # If baseline < 1%, use absolute only
                    else:
                        min_abs_threshold = 5_000_000  # 5 MB/s - minimum absolute difference
                        mae_threshold = res["threshold"] * 0.10  # 10% of threshold
                        concerning_threshold = res["threshold"] * 0.30  # 30% of threshold is concerning
                        baseline_too_small = baseline_forecast < 5_000_000  # If baseline < 5MB/s, use absolute only
                    
                    # 5. Model confidence check - if MAE is very low, model is accurate, be more conservative
                    # If MAE is high, model is struggling, be more lenient (don't trust it as much)
                    if res["unit"] == "ratio":
                        mae_confidence_factor = 1.0 if mae_ensemble < 0.005 else (0.7 if mae_ensemble < 0.01 else 0.3)
                    else:
                        mae_confidence_factor = 1.0 if mae_ensemble < mae_threshold * 0.1 else (0.7 if mae_ensemble < mae_threshold * 0.3 else 0.3)
                    
                    # Anomaly detection logic - be conservative to avoid false positives
                    # Strategy: Only flag if the value is ACTUALLY concerning, not just different
                    
                    # Check if current value is concerning (regardless of deviation)
                    is_concerning_value = (
                        (res["unit"] == "ratio" and current >= concerning_threshold) or
                        (res["unit"] != "ratio" and current >= concerning_threshold)
                    )
                    
                    # Check for significant absolute deviation
                    has_significant_abs_diff = (
                        current_deviation_abs >= min_abs_threshold or
                        mean_deviation_abs >= min_abs_threshold
                    )
                    
                    # For small baselines, only use absolute thresholds (percentage is misleading)
                    # For larger baselines, use both absolute and percentage
                    if baseline_too_small:
                        # Small baseline: only check absolute difference AND if value is concerning
                        is_anomaly = has_significant_abs_diff and is_concerning_value
                    else:
                        # Larger baseline: check both absolute and percentage deviation
                        has_significant_pct_diff = (
                            current_deviation_pct > 0.50 or  # 50% relative deviation
                            mean_deviation_pct > 0.30        # 30% mean deviation
                        )
                        # Anomaly if: (significant absolute + significant percentage) AND (concerning value OR low model confidence)
                        is_anomaly = (
                            has_significant_abs_diff and
                            has_significant_pct_diff and
                            (is_concerning_value or mae_confidence_factor < 0.5 or mae_ensemble > mae_threshold)
                        )
                    
                    if is_anomaly:
                        # Calculate anomaly score (0-1 scale)
                        anomaly_score = min(1.0, max(
                            current_deviation_pct,
                            mean_deviation_pct,
                            spike_deviation / 2.0,
                            drop_deviation / 2.0,
                            min(1.0, mae_ensemble / mae_threshold) if mae_threshold > 0 else 0
                        ))
                        
                        # Determine severity based on score
                        if anomaly_score > 0.8:
                            severity = "CRITICAL"
                        elif anomaly_score > 0.5:
                            severity = "WARNING"
                        else:
                            severity = "INFO"
                        
                        signal_display = res["name"].replace("_", " ")
                        current_str = f"{current*100:.2f}%" if res["unit"] == "ratio" else f"{current/1e6:.1f} MB/s"
                        
                        # Create human-readable description
                        description = format_anomaly_description(
                            node, signal_display, current, current_str, mae_ensemble, 
                            anomaly_score, severity, current_deviation_pct * 100, res["unit"]
                        )
                        
                        anomaly_results.append({
                            "node": node,
                            "signal": signal_display,
                            "current": current_str,
                            "mae_ensemble": round(mae_ensemble, 6),
                            "score": round(anomaly_score, 3),
                            "severity": severity,
                            "deviation_pct": round(current_deviation_pct * 100, 1),
                            "description": description
                        })
                        
                        log_verbose(f"   ⚠️  Anomaly detected: {node} | {res['name']} (score: {anomaly_score:.3f}, deviation: {current_deviation_pct*100:.1f}%)")

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
                # Format is "node (entity)" where node is like "hostname (IP)"
                # So full format is "hostname (IP) (canonical_name)"
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
    print_cpu_info(cli_override=CLI_PARALLEL_OVERRIDE)
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
    
    # ====================== IDENTIFY KUBERNETES CLUSTERS VS STANDALONE NODES ======================
    q_host_cpu = '1 - avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) by (instance)'
    q_host_mem = '(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes'
    q_pod_cpu = 'sum(rate(container_cpu_usage_seconds_total{container!="POD",container!=""}[5m])) by (instance)'
    q_pod_mem = 'sum(container_memory_working_set_bytes{container!="POD",container!=""}[5m]) by (instance)'

    df_hcpu = fetch_and_preprocess_data(q_host_cpu)
    df_hmem = fetch_and_preprocess_data(q_host_mem)
    df_pcpu = fetch_and_preprocess_data(q_pod_cpu)
    df_pmem = fetch_and_preprocess_data(q_pod_mem)
    
    # Alias resolution and entity canonicalization
    augment_aliases_from_dns(df_hcpu, df_pcpu)
    infer_aliases_from_timeseries(df_hcpu, df_pcpu)
    recanonicalize_entities(df_hcpu, df_hmem, df_pcpu, df_pmem)
    summarize_instance_roles(df_hcpu, df_pcpu)
    
    # Identify clusters to understand which nodes belong to which Kubernetes cluster
    cluster_map = identify_clusters(df_hcpu, df_hmem, df_pcpu, df_pmem, lookback_hours=LOOKBACK_HOURS)
    
    # Group entities by cluster
    def get_entity_set(df):
        if df.empty:
            return set()
        entity_col = 'entity' if 'entity' in df.columns else 'instance'
        return set(df[entity_col].unique())
    
    host_entities = get_entity_set(df_hcpu)
    pod_entities = get_entity_set(df_pcpu)
    
    # Group entities by cluster_id
    # Include ALL entities (both host and pod) to catch all nodes with pods
    # But deduplicate by canonical identity AND alias map to avoid double-counting
    # (e.g., pi and host03 are the same node, worker03 and host03 are the same node)
    all_entities = host_entities | pod_entities
    entities_by_cluster = {}
    # Track which canonical identities we've already added (to prevent duplicates via aliases)
    added_canonical = set()
    for entity in all_entities:
        entity_normalized = canonical_identity(entity)
        # Check if this entity has an alias - if so, use the alias target for deduplication
        alias_target = INSTANCE_ALIAS_MAP.get(entity_normalized)
        if alias_target:
            alias_target_canonical = canonical_identity(alias_target)
            # If the alias target is already in the set, skip this entity (it's a duplicate)
            if alias_target_canonical in added_canonical:
                continue
            # Use the alias target as the canonical identity
            entity_normalized = alias_target_canonical
        
        # Skip if we've already added this canonical identity
        if entity_normalized in added_canonical:
            continue
        
        cluster_id = cluster_map.get(entity_normalized, 'standalone')
        # Also check if any alias of this entity is in cluster_map
        if entity_normalized not in cluster_map:
            for alias_key, alias_val in INSTANCE_ALIAS_MAP.items():
                if canonical_identity(alias_val) == entity_normalized:
                    if alias_key in cluster_map:
                        cluster_id = cluster_map[alias_key]
                        break
        
        if cluster_id not in entities_by_cluster:
            entities_by_cluster[cluster_id] = set()
        entities_by_cluster[cluster_id].add(entity_normalized)
        added_canonical.add(entity_normalized)
    
    # Separate Kubernetes clusters from standalone
    # Include inferred_cluster_0 as a valid cluster (nodes with pods but no explicit cluster label)
    k8s_clusters = {cid: ents for cid, ents in entities_by_cluster.items() 
                    if cid != 'standalone' and not cid.startswith('unknown_cluster')}
    standalone_entities = entities_by_cluster.get('standalone', set())
    unknown_cluster_entities = entities_by_cluster.get('unknown_cluster', set())
    
    print(f"\nNode Classification:")
    print(f"  • Kubernetes clusters: {len(k8s_clusters)}")
    for cluster_id, entities in sorted(k8s_clusters.items()):
        cluster_display = "inferred" if cluster_id.startswith('inferred_cluster') else cluster_id
        print(f"    - Cluster '{cluster_display}': {len(entities)} nodes")
    if unknown_cluster_entities:
        print(f"  • Kubernetes nodes (cluster unknown): {len(unknown_cluster_entities)}")
    print(f"  • Standalone nodes (no pods): {len(standalone_entities)}")
    
    # Helper function to combine host + pod data
    def combine_host_pod(host_df, pod_df, metric_name):
        """Combine host and pod data by averaging them per timestamp"""
        if host_df.empty and pod_df.empty:
            return pd.DataFrame()
        
        host_agg = host_df.groupby('timestamp')['value'].mean().reset_index(name='host')
        pod_agg = pod_df.groupby('timestamp')['value'].mean().reset_index(name='pod')
        
        # Merge and average
        combined = pd.merge(host_agg, pod_agg, on='timestamp', how='outer')
        combined['value'] = (combined['host'].fillna(0) + combined['pod'].fillna(0)) / 2
        # If one is missing, use the available one
        combined.loc[combined['host'].isna(), 'value'] = combined.loc[combined['host'].isna(), 'pod']
        combined.loc[combined['pod'].isna(), 'value'] = combined.loc[combined['pod'].isna(), 'host']
        
        return combined[['timestamp', 'value']]
    
    # ====================== KUBERNETES CLUSTER MODELS (Host + Pod per cluster) — FORECAST ======================
    k8s_cluster_forecasts = {}  # cluster_id -> forecast
    k8s_host_fc = None  # Combined host forecast for all K8s nodes (for divergence)
    
    if k8s_clusters:
        print("\n" + "="*80)
        print("KUBERNETES CLUSTER MODELS (Host + Pod data per cluster) — FORECAST")
        print("="*80)
        
        entity_col_h = 'entity' if 'entity' in df_hcpu.columns else 'instance'
        entity_col_p = 'entity' if 'entity' in df_pcpu.columns else 'instance'
        
        # Load and generate forecasts for each Kubernetes cluster
        for cluster_id, cluster_entities in sorted(k8s_clusters.items()):
            print(f"\n  Processing Cluster: {cluster_id} ({len(cluster_entities)} nodes)")
            
            # Filter data to this cluster's nodes
            # cluster_entities contains canonical identities, so we need to match entities via canonical_identity
            df_hcpu_cluster = df_hcpu[df_hcpu[entity_col_h].apply(canonical_identity).isin(cluster_entities)].copy()
            df_hmem_cluster = df_hmem[df_hmem[entity_col_h].apply(canonical_identity).isin(cluster_entities)].copy()
            df_pcpu_cluster = df_pcpu[df_pcpu[entity_col_p].apply(canonical_identity).isin(cluster_entities)].copy()
            df_pmem_cluster = df_pmem[df_pmem[entity_col_p].apply(canonical_identity).isin(cluster_entities)].copy()
            
            # Combine host + pod data for this cluster
            df_combined_cpu = combine_host_pod(df_hcpu_cluster, df_pcpu_cluster, 'cpu')
            df_combined_mem = combine_host_pod(df_hmem_cluster, df_pmem_cluster, 'mem')
            
            if not df_combined_cpu.empty and not df_combined_mem.empty:
                # Create cluster-specific model path
                cluster_model_path = os.path.join(MODEL_DIR, f"k8s_cluster_{sanitize_label(cluster_id)}_forecast.pkl")
                
                if not os.path.exists(cluster_model_path):
                    print(f"    ⚠ Warning: Cluster model not found at {cluster_model_path}")
                    print("       Skipping this cluster. Run with --training flag first to train models.")
                else:
                    _, cluster_fc, _, _ = train_or_load_ensemble(
                        df_combined_cpu,
                        df_combined_mem,
                        horizon_min=7*24*60,
                        model_path=cluster_model_path,
                        force_retrain=False,
                        generate_fresh_forecast=True,
                        dump_csv_dir=csv_dump_dir,
                        context={'node': 'k8s_cluster', 'cluster_id': cluster_id},
                        enable_plots=enable_plots
                    )
                    k8s_cluster_forecasts[cluster_id] = cluster_fc
        
        # Also create combined host forecast for all K8s nodes (for divergence calculation)
        all_k8s_entities = set()
        for entities in k8s_clusters.values():
            all_k8s_entities.update(entities)
        
        if all_k8s_entities:
            df_hcpu_all_k8s = df_hcpu[df_hcpu[entity_col_h].isin(all_k8s_entities)].copy()
            df_hmem_all_k8s = df_hmem[df_hmem[entity_col_h].isin(all_k8s_entities)].copy()
            
            if os.path.exists(HOST_MODEL_PATH):
                _, k8s_host_fc, _, _ = train_or_load_ensemble(
                    df_hcpu_all_k8s,
                    df_hmem_all_k8s,
                    horizon_min=7*24*60,
                    model_path=HOST_MODEL_PATH,
                    force_retrain=False,
                    generate_fresh_forecast=True,
                    dump_csv_dir=None,
                    context={'node': 'k8s_host_all'},
                    enable_plots=enable_plots
                )
    
    # Handle unknown_cluster nodes (have pods but cluster can't be determined)
    if unknown_cluster_entities:
        print("\n" + "="*80)
        print("KUBERNETES UNKNOWN CLUSTER MODEL (Host + Pod data - cluster unknown) — FORECAST")
        print("="*80)
        print(f"  Processing {len(unknown_cluster_entities)} nodes with pods but unknown cluster")
        
        entity_col_h = 'entity' if 'entity' in df_hcpu.columns else 'instance'
        entity_col_p = 'entity' if 'entity' in df_pcpu.columns else 'instance'
        
        df_hcpu_unknown = df_hcpu[df_hcpu[entity_col_h].isin(unknown_cluster_entities)].copy()
        df_hmem_unknown = df_hmem[df_hmem[entity_col_h].isin(unknown_cluster_entities)].copy()
        df_pcpu_unknown = df_pcpu[df_pcpu[entity_col_p].isin(unknown_cluster_entities)].copy()
        df_pmem_unknown = df_pmem[df_pmem[entity_col_p].isin(unknown_cluster_entities)].copy()
        
        df_combined_cpu = combine_host_pod(df_hcpu_unknown, df_pcpu_unknown, 'cpu')
        df_combined_mem = combine_host_pod(df_hmem_unknown, df_pmem_unknown, 'mem')
        
        if not df_combined_cpu.empty and not df_combined_mem.empty:
            unknown_model_path = os.path.join(MODEL_DIR, "k8s_unknown_cluster_forecast.pkl")
            if os.path.exists(unknown_model_path):
                _, unknown_fc, _, _ = train_or_load_ensemble(
                    df_combined_cpu,
                    df_combined_mem,
                    horizon_min=7*24*60,
                    model_path=unknown_model_path,
                    force_retrain=False,
                    generate_fresh_forecast=True,
                    dump_csv_dir=csv_dump_dir,
                    context={'node': 'k8s_unknown_cluster'},
                    enable_plots=enable_plots
                )
                k8s_cluster_forecasts['unknown_cluster'] = unknown_fc
            else:
                print(f"  ⚠ Warning: Unknown cluster model not found. Run with --training flag first.")
    
    # ====================== STANDALONE MODEL (Host only) — FORECAST ======================
    standalone_fc = None
    
    if standalone_entities:
        print("\n" + "="*80)
        print("STANDALONE MODEL (Host data only - no Kubernetes) — FORECAST")
        print("="*80)
        
        # Filter host data to only standalone nodes
        entity_col = 'entity' if 'entity' in df_hcpu.columns else 'instance'
        df_hcpu_standalone = df_hcpu[df_hcpu[entity_col].isin(standalone_entities)].copy()
        df_hmem_standalone = df_hmem[df_hmem[entity_col].isin(standalone_entities)].copy()
        
        if not df_hcpu_standalone.empty and not df_hmem_standalone.empty:
            if not os.path.exists(STANDALONE_MODEL_PATH):
                print(f"⚠ Warning: Standalone model not found at {STANDALONE_MODEL_PATH}")
                print("   Skipping standalone forecast. Run with --training flag first to train models.")
                standalone_fc = None
            else:
                _, standalone_fc, _, _ = train_or_load_ensemble(
                    df_hcpu_standalone,
                    df_hmem_standalone,
                    horizon_min=7*24*60,
                    model_path=STANDALONE_MODEL_PATH,
                    force_retrain=False,
                    generate_fresh_forecast=True,
                    dump_csv_dir=csv_dump_dir,
                    context={'node': 'standalone'},
                    enable_plots=enable_plots
                )
        else:
            print("⚠ Warning: Insufficient data for standalone model")
    else:
        print("\n⚠ Warning: No standalone nodes found (all nodes have pods)")
    
    # For backward compatibility, set host_fc and pod_fc
    # Use first available cluster forecast, or unknown_cluster, or standalone
    pod_fc = None
    if k8s_cluster_forecasts:
        # Use the first cluster's forecast (or unknown_cluster if available)
        if 'unknown_cluster' in k8s_cluster_forecasts:
            pod_fc = k8s_cluster_forecasts['unknown_cluster']
        else:
            pod_fc = list(k8s_cluster_forecasts.values())[0]
    
    host_fc = k8s_host_fc if k8s_host_fc is not None else standalone_fc
    
    # ====================== DIVERGENCE & ANOMALY ======================
    # Divergence only makes sense for Kubernetes nodes (comparing host vs combined host+pod)
    if k8s_host_fc is not None and pod_fc is not None:
        host_mem = k8s_host_fc['yhat'].iloc[-1]
        combined_mem = pod_fc['yhat'].iloc[-1]
        div = abs(host_mem - combined_mem)
        print(f"\nDivergence (K8s host vs combined host+pod memory): {div:.3f}")
    elif host_fc is not None:
        node_type = 'Kubernetes' if pod_fc is not None else 'Standalone'
        print(f"\nForecast available for {node_type} nodes")
    
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
    
    # Override MAX_WORKER_THREADS if --parallel flag is provided
    # Note: We're at module level, so we can modify module-level variables directly
    if args.parallel is not None:
        if args.parallel < 1:
            print(f"⚠️  Warning: --parallel value must be >= 1, got {args.parallel}. Using 1 worker.")
            MAX_WORKER_THREADS = 1
            CLI_PARALLEL_OVERRIDE = 1
        else:
            MAX_WORKER_THREADS = args.parallel
            CLI_PARALLEL_OVERRIDE = args.parallel
    
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
    print_cpu_info(cli_override=CLI_PARALLEL_OVERRIDE)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Dual-Layer + LSTM AI\n")
    mode_label = "TRAINING" if force_training else "PRE-TRAINED"
    print(f"Execution mode: {mode_label}")
    show_backtest = args.show_backtest
    disk_retrain_targets = parse_disk_retrain_targets(args.disk_retrain)
    disk_manifest = load_disk_manifest(DISK_MODEL_MANIFEST_PATH)
    io_net_manifest = load_io_net_manifest(IO_NET_MODEL_MANIFEST_PATH)
    io_net_retrain_targets = parse_io_net_retrain_targets(args.io_net_retrain)

    # ====================== IDENTIFY KUBERNETES VS STANDALONE NODES ======================
    q_host_cpu = '1 - avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) by (instance)'
    q_host_mem = '(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes'
    q_pod_cpu = 'sum(rate(container_cpu_usage_seconds_total{container!="POD",container!=""}[5m])) by (instance)'
    q_pod_mem = 'sum(container_memory_working_set_bytes{container!="POD",container!=""}[5m]) by (instance)'

    df_hcpu = fetch_and_preprocess_data(q_host_cpu)
    df_hmem = fetch_and_preprocess_data(q_host_mem)
    df_pcpu = fetch_and_preprocess_data(q_pod_cpu)
    df_pmem = fetch_and_preprocess_data(q_pod_mem)
    
    # Alias resolution and entity canonicalization
    augment_aliases_from_dns(df_hcpu, df_pcpu)
    infer_aliases_from_timeseries(df_hcpu, df_pcpu)
    recanonicalize_entities(df_hcpu, df_hmem, df_pcpu, df_pmem)
    summarize_instance_roles(df_hcpu, df_pcpu)
    
    # Identify clusters to understand which nodes belong to which Kubernetes cluster
    cluster_map = identify_clusters(df_hcpu, df_hmem, df_pcpu, df_pmem, lookback_hours=LOOKBACK_HOURS)
    
    # Group entities by cluster
    def get_entity_set(df):
        if df.empty:
            return set()
        entity_col = 'entity' if 'entity' in df.columns else 'instance'
        return set(df[entity_col].unique())
    
    host_entities = get_entity_set(df_hcpu)
    pod_entities = get_entity_set(df_pcpu)
    
    # Group entities by cluster_id
    # Include ALL entities (both host and pod) to catch all nodes with pods
    # But deduplicate by canonical identity AND alias map to avoid double-counting
    # (e.g., pi and host03 are the same node, worker03 and host03 are the same node)
    all_entities = host_entities | pod_entities
    entities_by_cluster = {}
    # Track which canonical identities we've already added (to prevent duplicates via aliases)
    added_canonical = set()
    for entity in all_entities:
        entity_normalized = canonical_identity(entity)
        # Check if this entity has an alias - if so, use the alias target for deduplication
        alias_target = INSTANCE_ALIAS_MAP.get(entity_normalized)
        if alias_target:
            alias_target_canonical = canonical_identity(alias_target)
            # If the alias target is already in the set, skip this entity (it's a duplicate)
            if alias_target_canonical in added_canonical:
                continue
            # Use the alias target as the canonical identity
            entity_normalized = alias_target_canonical
        
        # Skip if we've already added this canonical identity
        if entity_normalized in added_canonical:
            continue
        
        cluster_id = cluster_map.get(entity_normalized, 'standalone')
        # Also check if any alias of this entity is in cluster_map
        if entity_normalized not in cluster_map:
            for alias_key, alias_val in INSTANCE_ALIAS_MAP.items():
                if canonical_identity(alias_val) == entity_normalized:
                    if alias_key in cluster_map:
                        cluster_id = cluster_map[alias_key]
                        break
        
        if cluster_id not in entities_by_cluster:
            entities_by_cluster[cluster_id] = set()
        entities_by_cluster[cluster_id].add(entity_normalized)
        added_canonical.add(entity_normalized)
    
    # Separate Kubernetes clusters from standalone
    # Include inferred_cluster_0 as a valid cluster (nodes with pods but no explicit cluster label)
    k8s_clusters = {cid: ents for cid, ents in entities_by_cluster.items() 
                    if cid != 'standalone' and not cid.startswith('unknown_cluster')}
    standalone_entities = entities_by_cluster.get('standalone', set())
    unknown_cluster_entities = entities_by_cluster.get('unknown_cluster', set())
    
    print(f"\nNode Classification:")
    print(f"  • Kubernetes clusters: {len(k8s_clusters)}")
    for cluster_id, entities in sorted(k8s_clusters.items()):
        cluster_display = "inferred" if cluster_id.startswith('inferred_cluster') else cluster_id
        print(f"    - Cluster '{cluster_display}': {len(entities)} nodes")
    if unknown_cluster_entities:
        print(f"  • Kubernetes nodes (cluster unknown): {len(unknown_cluster_entities)}")
    print(f"  • Standalone nodes (no pods): {len(standalone_entities)}")
    
    # Helper function to combine host + pod data
    def combine_host_pod(host_df, pod_df, metric_name):
        """Combine host and pod data by averaging them per timestamp"""
        if host_df.empty and pod_df.empty:
            return pd.DataFrame()
        
        host_agg = host_df.groupby('timestamp')['value'].mean().reset_index(name='host')
        pod_agg = pod_df.groupby('timestamp')['value'].mean().reset_index(name='pod')
        
        # Merge and average
        combined = pd.merge(host_agg, pod_agg, on='timestamp', how='outer')
        combined['value'] = (combined['host'].fillna(0) + combined['pod'].fillna(0)) / 2
        # If one is missing, use the available one
        combined.loc[combined['host'].isna(), 'value'] = combined.loc[combined['host'].isna(), 'pod']
        combined.loc[combined['pod'].isna(), 'value'] = combined.loc[combined['pod'].isna(), 'host']
        
        return combined[['timestamp', 'value']]
    
    # ====================== KUBERNETES CLUSTER MODELS (Host + Pod per cluster) ======================
    k8s_cluster_forecasts = {}  # cluster_id -> forecast
    k8s_cluster_metrics = {}    # cluster_id -> metrics
    k8s_cluster_saved = {}      # cluster_id -> was_saved
    k8s_host_fc = None  # Combined host forecast for all K8s nodes (for divergence)
    
    if k8s_clusters:
        print("\n" + "="*80)
        print("KUBERNETES CLUSTER MODELS (Host + Pod data per cluster)")
        print("="*80)
        
        entity_col_h = 'entity' if 'entity' in df_hcpu.columns else 'instance'
        entity_col_p = 'entity' if 'entity' in df_pcpu.columns else 'instance'
        
        # Train separate model for each Kubernetes cluster
        for cluster_id, cluster_entities in sorted(k8s_clusters.items()):
            print(f"\n  Processing Cluster: {cluster_id} ({len(cluster_entities)} nodes)")
            
            # Filter data to this cluster's nodes
            # cluster_entities contains canonical identities, so we need to match entities via canonical_identity
            df_hcpu_cluster = df_hcpu[df_hcpu[entity_col_h].apply(canonical_identity).isin(cluster_entities)].copy()
            df_hmem_cluster = df_hmem[df_hmem[entity_col_h].apply(canonical_identity).isin(cluster_entities)].copy()
            df_pcpu_cluster = df_pcpu[df_pcpu[entity_col_p].apply(canonical_identity).isin(cluster_entities)].copy()
            df_pmem_cluster = df_pmem[df_pmem[entity_col_p].apply(canonical_identity).isin(cluster_entities)].copy()
            
            # Combine host + pod data for this cluster
            df_combined_cpu = combine_host_pod(df_hcpu_cluster, df_pcpu_cluster, 'cpu')
            df_combined_mem = combine_host_pod(df_hmem_cluster, df_pmem_cluster, 'mem')
            
            if not df_combined_cpu.empty and not df_combined_mem.empty:
                # Create cluster-specific model path
                cluster_model_path = os.path.join(MODEL_DIR, f"k8s_cluster_{sanitize_label(cluster_id)}_forecast.pkl")
                
                _, cluster_fc, cluster_metrics, cluster_saved = train_or_load_ensemble(
                    df_combined_cpu,
                    df_combined_mem,
                    horizon_min=7*24*60,
                    model_path=cluster_model_path,
                    force_retrain=force_training,
                    show_backtest=show_backtest,
                    dump_csv_dir=csv_dump_dir,
                    context={'node': 'k8s_cluster', 'cluster_id': cluster_id}
                )
                
                k8s_cluster_forecasts[cluster_id] = cluster_fc
                k8s_cluster_metrics[cluster_id] = cluster_metrics
                k8s_cluster_saved[cluster_id] = cluster_saved
                
                if (force_training or show_backtest) and cluster_metrics:
                    print(f"    Cluster '{cluster_id}' Model Metrics:")
                    for k, v in cluster_metrics.items():
                        if k == 'split_info' and isinstance(v, dict):
                            print(f"      • Train/Test Split:")
                            print(f"        - Train fraction: {v.get('train_fraction', 0)*100:.0f}%")
                            print(f"        - Train points: {v.get('train_points', 0):,}")
                            print(f"        - Test points: {v.get('test_points', 0):,}")
                        elif isinstance(v, (int, float)):
                            print(f"      • {k}: {v:.6f}")
            else:
                print(f"    ⚠ Warning: Insufficient data for cluster '{cluster_id}'")
        
        # Also create combined host forecast for all K8s nodes (for divergence calculation)
        all_k8s_entities = set()
        for entities in k8s_clusters.values():
            all_k8s_entities.update(entities)
        
        if all_k8s_entities:
            df_hcpu_all_k8s = df_hcpu[df_hcpu[entity_col_h].isin(all_k8s_entities)].copy()
            df_hmem_all_k8s = df_hmem[df_hmem[entity_col_h].isin(all_k8s_entities)].copy()
            
            _, k8s_host_fc, _, _ = train_or_load_ensemble(
                df_hcpu_all_k8s,
                df_hmem_all_k8s,
                horizon_min=7*24*60,
                model_path=HOST_MODEL_PATH,  # Reuse for backward compatibility
                force_retrain=False,
                show_backtest=False,
                dump_csv_dir=None,
                context={'node': 'k8s_host_all'}
            )
    
    # Handle unknown_cluster nodes (have pods but cluster can't be determined)
    if unknown_cluster_entities:
        print("\n" + "="*80)
        print("KUBERNETES UNKNOWN CLUSTER MODEL (Host + Pod data - cluster unknown)")
        print("="*80)
        print(f"  Processing {len(unknown_cluster_entities)} nodes with pods but unknown cluster")
        
        entity_col_h = 'entity' if 'entity' in df_hcpu.columns else 'instance'
        entity_col_p = 'entity' if 'entity' in df_pcpu.columns else 'instance'
        
        df_hcpu_unknown = df_hcpu[df_hcpu[entity_col_h].isin(unknown_cluster_entities)].copy()
        df_hmem_unknown = df_hmem[df_hmem[entity_col_h].isin(unknown_cluster_entities)].copy()
        df_pcpu_unknown = df_pcpu[df_pcpu[entity_col_p].isin(unknown_cluster_entities)].copy()
        df_pmem_unknown = df_pmem[df_pmem[entity_col_p].isin(unknown_cluster_entities)].copy()
        
        df_combined_cpu = combine_host_pod(df_hcpu_unknown, df_pcpu_unknown, 'cpu')
        df_combined_mem = combine_host_pod(df_hmem_unknown, df_pmem_unknown, 'mem')
        
        if not df_combined_cpu.empty and not df_combined_mem.empty:
            unknown_model_path = os.path.join(MODEL_DIR, "k8s_unknown_cluster_forecast.pkl")
            _, unknown_fc, unknown_metrics, unknown_saved = train_or_load_ensemble(
                df_combined_cpu,
                df_combined_mem,
                horizon_min=7*24*60,
                model_path=unknown_model_path,
                force_retrain=force_training,
                show_backtest=show_backtest,
                dump_csv_dir=csv_dump_dir,
                context={'node': 'k8s_unknown_cluster'}
            )
            k8s_cluster_forecasts['unknown_cluster'] = unknown_fc
            k8s_cluster_metrics['unknown_cluster'] = unknown_metrics
            k8s_cluster_saved['unknown_cluster'] = unknown_saved
    
    # ====================== STANDALONE MODEL (Host only) ======================
    standalone_fc = None
    standalone_metrics = None
    standalone_saved = False
    
    if standalone_entities:
        print("\n" + "="*80)
        print("STANDALONE MODEL (Host data only - no Kubernetes)")
        print("="*80)
        
        # Filter host data to only standalone nodes
        entity_col = 'entity' if 'entity' in df_hcpu.columns else 'instance'
        df_hcpu_standalone = df_hcpu[df_hcpu[entity_col].isin(standalone_entities)].copy()
        df_hmem_standalone = df_hmem[df_hmem[entity_col].isin(standalone_entities)].copy()
        
        if not df_hcpu_standalone.empty and not df_hmem_standalone.empty:
            _, standalone_fc, standalone_metrics, standalone_saved = train_or_load_ensemble(
                df_hcpu_standalone,
                df_hmem_standalone,
                horizon_min=7*24*60,
                model_path=STANDALONE_MODEL_PATH,
                force_retrain=force_training,
                show_backtest=show_backtest,
                dump_csv_dir=csv_dump_dir,
                context={'node': 'standalone'}
            )
            
            if (force_training or show_backtest) and standalone_metrics:
                print("Standalone Model Metrics:")
                for k, v in standalone_metrics.items():
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
        else:
            print("⚠ Warning: Insufficient data for standalone model")
    else:
        print("\n⚠ Warning: No standalone nodes found (all nodes have pods)")
    
    # For backward compatibility, set host_fc and pod_fc
    # Use first available cluster forecast, or unknown_cluster, or standalone
    pod_fc = None
    if k8s_cluster_forecasts:
        # Use the first cluster's forecast (or unknown_cluster if available)
        if 'unknown_cluster' in k8s_cluster_forecasts:
            pod_fc = k8s_cluster_forecasts['unknown_cluster']
        else:
            pod_fc = list(k8s_cluster_forecasts.values())[0]
    
    host_fc = k8s_host_fc if k8s_host_fc is not None else standalone_fc

    # ====================== DIVERGENCE & ANOMALY ======================
    # Divergence only makes sense for Kubernetes nodes (comparing host vs combined host+pod)
    if k8s_host_fc is not None and pod_fc is not None:
        host_mem = k8s_host_fc['yhat'].iloc[-1]
        combined_mem = pod_fc['yhat'].iloc[-1]
        div = abs(host_mem - combined_mem)
        print(f"\nDivergence (K8s host vs combined host+pod memory): {div:.3f}")
    elif host_fc is not None:
        node_type = 'Kubernetes' if pod_fc is not None else 'Standalone'
        print(f"\nForecast available for {node_type} nodes")

    _, _, _, _ = classification_model(
        df_hcpu,
        df_hmem,
        df_pcpu,
        df_pmem,
        lookback_hours=LOOKBACK_HOURS,
        contamination=CONTAMINATION,
        dump_csv_dir=csv_dump_dir
    )

    # Print "models saved" message only when models were actually saved
    any_saved = any(k8s_cluster_saved.values()) if k8s_cluster_saved else False
    any_saved = any_saved or standalone_saved
    
    if any_saved:
        saved_models = []
        # Add all saved cluster models
        for cluster_id, was_saved in k8s_cluster_saved.items():
            if was_saved:
                if cluster_id == 'unknown_cluster':
                    saved_models.append("k8s_unknown_cluster_forecast.pkl")
                else:
                    saved_models.append(f"k8s_cluster_{sanitize_label(cluster_id)}_forecast.pkl")
        if standalone_saved:
            saved_models.append(os.path.basename(STANDALONE_MODEL_PATH))
        # LSTM is saved as part of ensemble training, so include it if any model was saved
        if any_saved and LSTM_AVAILABLE:
            saved_models.append(os.path.basename(LSTM_MODEL_PATH))
        if saved_models:
            print(f"\nAll models saved: {', '.join(saved_models)}")
    print("\nDual-layer + LSTM + classification complete.")

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
        print("\nNo active root-cause signals — estate is clean and healthy")
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
    print("Your estate is now protected by real, validated, visualized AI.")
    print("="*80)

    # ====================== I/O + NETWORK — FULL ENSEMBLE (CPU/MEM GRADE) ======================
    print("\n" + "="*80)
    print("DISK I/O + NETWORK — FULL ENSEMBLE FORECAST & ANOMALY DETECTION")
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
        print("\n" + "="*80)
        
        # Group by severity for better readability
        critical = io_net_anomaly_df[io_net_anomaly_df['severity'] == 'CRITICAL'] if 'severity' in io_net_anomaly_df.columns else pd.DataFrame()
        warning = io_net_anomaly_df[io_net_anomaly_df['severity'] == 'WARNING'] if 'severity' in io_net_anomaly_df.columns else pd.DataFrame()
        info = io_net_anomaly_df[io_net_anomaly_df['severity'] == 'INFO'] if 'severity' in io_net_anomaly_df.columns else pd.DataFrame()
        
        # Print human-readable descriptions if available
        if 'description' in io_net_anomaly_df.columns:
            for idx, row in io_net_anomaly_df.iterrows():
                print(f"\n{row['description']}")
                print("-" * 80)
        else:
            # Fallback to technical table if descriptions not available
            print(io_net_anomaly_df[['node', 'signal', 'current', 'severity', 'deviation_pct']].to_string(index=False))
        
        # Also show technical summary table for reference
        print("\n" + "="*80)
        print("TECHNICAL SUMMARY (for detailed analysis):")
        print("="*80)
        tech_cols = ['node', 'signal', 'current', 'severity', 'deviation_pct', 'score', 'mae_ensemble']
        tech_cols = [c for c in tech_cols if c in io_net_anomaly_df.columns]
        print(io_net_anomaly_df[tech_cols].to_string(index=False))

    if io_net_crisis_df.empty and io_net_anomaly_df.empty:
        print("\nI/O and Network layers are healthy, predictable, and anomaly-free")
        print("I/O and Network monitoring is operating within expected parameters.")

    print(f"\nAll plots + models saved → {FORECAST_PLOTS_DIR}/")
    print("Metrics AI — Unified forecasting and anomaly detection across CPU • Memory • Disk • I/O • Network")
    print("="*80)

    if args.anomaly_watch > 0:
        run_realtime_anomaly_watch(
            q_host_cpu, q_host_mem, q_pod_cpu, q_pod_mem,
            iterations=args.anomaly_watch
        )
