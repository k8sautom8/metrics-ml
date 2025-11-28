# Anomaly Detection in Metrics AI

## Overview

Metrics AI detects anomalies at **multiple levels** across the entire Kubernetes infrastructure using a combination of:
1. **Unsupervised Machine Learning** (IsolationForest)
2. **Rule-based Threshold Detection** (Golden Signals)
3. **Statistical Anomaly Detection** (Ensemble Forecast Deviations)
4. **Pattern-Based Detection** (Host/Pod Resource Misalignment)

---

## Detection Levels

### 1. **Node-Level Anomalies** (Per-Instance)

#### A. Classification Anomalies (IsolationForest)
- **Level**: Individual node/host
- **Method**: Unsupervised ML using IsolationForest
- **Features Analyzed**:
  - `host_cpu` - Average host CPU usage (last 24h by default)
  - `host_mem` - Average host memory usage (last 24h)
  - `pod_cpu` - Average pod/container CPU usage (last 24h)
  - `pod_mem` - Average pod/container memory usage (last 24h)

- **How It Works**:
  1. Extracts average resource usage for each node over the lookback window (default: 24 hours)
  2. Normalizes features using StandardScaler
  3. Trains IsolationForest with contamination rate (default: 12%)
  4. Flags nodes where host and pod usage patterns are misaligned

- **What It Detects**:
  - Nodes with high host CPU/memory but low pod usage → **Non-Kubernetes workloads** (backups, cron jobs, daemons)
  - Nodes with low host usage but high pod usage → **Potential resource accounting issues**
  - Nodes with unusual resource patterns compared to cluster baseline

- **Output**: 
  - Anomaly label: `-1` (anomalous) or `1` (normal)
  - Alert type: `classification_anomaly`
  - Severity: `WARNING`
  - Includes: instance name, host_cpu, host_mem, pod_cpu, pod_mem values

- **Scope**: **All nodes in the cluster** - compares each node against the cluster-wide pattern

---

#### B. Host Pressure Detection
- **Level**: Individual node/host
- **Method**: Rule-based threshold detection
- **Criteria**:
  - `host_cpu > 0.6` (60%) OR `host_mem > 0.7` (70%)
  - AND `pod_cpu < 0.05` (5%) AND `pod_mem < 0.05` (5%)

- **What It Detects**:
  - Nodes with high host resource consumption but minimal Kubernetes workload
  - Indicates OS-level processes consuming resources (backups, system daemons, cron jobs)

- **Output**:
  - Alert type: `host_pressure`
  - Severity: `WARNING`
  - Includes: instance name, host_cpu, host_mem values

- **Scope**: **All nodes in the cluster** - evaluated per-node

---

### 2. **Signal-Level Anomalies** (Per-Node, Per-Metric)

#### A. Golden Anomaly Signals (Root-Cause Detection)
- **Level**: Per-node, per-signal
- **Method**: Prometheus query-based threshold detection
- **Signals Detected** (8 types):

| Signal | Query | Threshold | Severity |
|--------|-------|-----------|----------|
| `iowait_high` | CPU iowait rate | > 0.15 (15%) | WARNING |
| `inodes_critical` | Filesystem inode usage | > 0.90 (90%) | CRITICAL |
| `net_rx_drop` | Network receive drops | > 10 changes/5m | WARNING |
| `net_tx_saturated` | Network transmit bandwidth | > 9 Gbps | WARNING |
| `tcp_retrans_high` | TCP retransmissions | > 1000/5m | WARNING |
| `oom_kills` | OOM kill events | > 0 in last hour | CRITICAL |
| `fork_bomb` | Process fork rate | > 1000/5m | WARNING |
| `fd_leak` | File descriptor usage | > 0.90 (90%) | CRITICAL |

- **How It Works**:
  1. Executes Prometheus queries for each signal type
  2. Checks if any instance exceeds the threshold
  3. Flags the node and signal combination

- **Output**:
  - Alert type: `golden_anomaly`
  - Severity: `CRITICAL` or `WARNING` (based on signal type)
  - Includes: node name, signal name, detected_at timestamp

- **Scope**: **All nodes, all signals** - evaluated independently per node per signal

---

#### B. I/O + Network Crisis Detection
- **Level**: Per-node, per-signal (DISK_IO_WAIT, NET_TX_BW)
- **Method**: Ensemble forecast with threshold prediction
- **Signals**:
  - `DISK_IO_WAIT`: Disk I/O wait time (threshold: 30%)
  - `NET_TX_BW`: Network transmit bandwidth (threshold: 120 MB/s)

- **How It Works**:
  1. Trains ensemble model (Prophet + ARIMA + LSTM) for each node/signal
  2. Forecasts 7 days ahead
  3. Predicts when the signal will cross the threshold
  4. Flags if ETA < 30 days

- **Output**:
  - Alert type: `io_network_crisis`
  - Severity: `CRITICAL` (<3 days), `WARNING` (3-7 days), `SOON` (7-30 days)
  - Includes: node, signal, current value, hybrid_eta_days, mae_ensemble

- **Scope**: **All nodes, per signal** - separate model per node per signal

---

#### C. I/O + Network Anomaly Detection
- **Level**: Per-node, per-signal
- **Method**: Statistical deviation from ensemble forecast
- **Status**: Currently a placeholder (line 3066-3068)
  ```python
  # Anomaly detection placeholder (you can expand later)
  # is_anomaly = metrics.get('anomaly_score', 0) > 0.7
  # if is_anomaly: ...
  ```

- **Potential Implementation**:
  - Compare actual values vs ensemble forecast
  - Flag if deviation exceeds threshold (e.g., MAE > 0.7)
  - Detect sudden spikes or drops that don't match predicted patterns

- **Scope**: **All nodes, per signal** - would be evaluated per node per signal

---

### 3. **Disk-Level Anomalies** (Per-Node, Per-Mountpoint)

#### Disk Capacity Crisis Detection
- **Level**: Per-node, per-mountpoint
- **Method**: Hybrid linear + Prophet forecast with 90% threshold
- **How It Works**:
  1. Trains linear trend model and Prophet model for each node/mountpoint
  2. Forecasts 7 days ahead
  3. Predicts when disk usage will reach 90%
  4. Flags based on ETA

- **Output**:
  - Alert type: `disk` (with sub-categories)
  - Severity: 
    - `CRITICAL` - < 3 days to 90%
    - `WARNING` - 3-7 days to 90%
    - `SOON` - 7-30 days to 90%
    - `OK` - > 30 days (filtered out from alerts)
  - Includes: instance, mountpoint, current_%, days_to_90pct, ensemble_eta, linear_eta, prophet_eta

- **Scope**: **All nodes, all mountpoints** - separate model per node per mountpoint

---

## Detection Scope Summary

| Detection Type | Level | Scope | Method |
|----------------|-------|-------|--------|
| **Classification Anomaly** | Node | All nodes in cluster | IsolationForest (ML) |
| **Host Pressure** | Node | All nodes in cluster | Rule-based thresholds |
| **Golden Signals** | Node + Signal | All nodes, 8 signal types | Prometheus query thresholds |
| **I/O Network Crisis** | Node + Signal | All nodes, 2 signals | Ensemble forecast prediction |
| **I/O Network Anomaly** | Node + Signal | All nodes, 2 signals | *Placeholder (not implemented)* |
| **Disk Crisis** | Node + Mountpoint | All nodes, all mountpoints | Hybrid forecast prediction |

---

## How Anomalies Are Found Across the Entire System

### 1. **Cluster-Wide Scanning**
- **Classification Model**: Scans ALL nodes in the cluster simultaneously
  - Fetches host CPU/memory and pod CPU/memory for every node
  - Builds a feature matrix: `[node1_features, node2_features, ..., nodeN_features]`
  - Trains IsolationForest on the entire cluster
  - Flags any node that deviates from cluster-wide patterns

### 2. **Per-Node Evaluation**
- **Golden Signals**: Queries Prometheus for each signal type, evaluates every instance
- **I/O Network**: Trains separate models for each node/signal combination
- **Disk**: Trains separate models for each node/mountpoint combination

### 3. **Time-Window Analysis**
- **Lookback Window**: Default 24 hours (configurable via `LOOKBACK_HOURS`)
- **Golden Signals**: Last 1 hour (configurable)
- **Forecast Horizon**: 7 days ahead

### 4. **Aggregation Strategy**
- **Cluster-wide patterns**: Classification model compares nodes against each other
- **Per-node baselines**: I/O, Network, and Disk models learn individual node patterns
- **Threshold-based**: Golden signals use fixed thresholds across all nodes

---

## Alert Aggregation

All anomalies are aggregated and dispatched via:
- **Webhook**: JSON payload with counts and sample records
- **Pushgateway**: Prometheus metrics (`metrics_ai_*_total`)

The system provides a **unified view** of all anomalies across:
- ✅ All nodes
- ✅ All mountpoints  
- ✅ All signal types
- ✅ All detection methods

---

## Configuration

Key environment variables for anomaly detection:
- `LOOKBACK_HOURS` (default: 24) - Window for classification model
- `CONTAMINATION` (default: 0.12) - IsolationForest contamination rate (12% of nodes expected to be anomalous)
- `HORIZON_MIN` (default: 15) - Forecast horizon in minutes

---

## Example: Complete Anomaly Scan

When `metrics.py --forecast` runs, it:

1. **Fetches data** for all nodes from Prometheus
2. **Classification Model**:
   - Extracts features for all nodes
   - Trains IsolationForest on cluster-wide data
   - Flags anomalous nodes (e.g., "host02 has high host usage but low pod usage")
3. **Host Pressure**:
   - Checks each node individually
   - Flags nodes with host pressure (e.g., "pi has 84% host memory but only 2% pod memory")
4. **Golden Signals**:
   - Queries 8 signal types across all nodes
   - Flags any node/signal combination exceeding thresholds
5. **I/O Network Crisis**:
   - For each node, for each signal (DISK_IO_WAIT, NET_TX_BW):
     - Loads or trains ensemble model
     - Forecasts 7 days ahead
     - Flags if threshold will be crossed within 30 days
6. **Disk Crisis**:
   - For each node, for each mountpoint:
     - Loads or trains hybrid model
     - Forecasts 7 days ahead
     - Flags if 90% threshold will be crossed

**Result**: Comprehensive anomaly detection across the entire infrastructure at multiple levels.


