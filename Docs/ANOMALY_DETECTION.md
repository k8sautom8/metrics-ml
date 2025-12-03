# Anomaly Detection in Metrics AI

## Overview

Metrics AI detects anomalies at **multiple levels** across the entire Kubernetes infrastructure using a combination of:
1. **Unsupervised Machine Learning** (IsolationForest)
2. **Rule-based Threshold Detection** (Golden Signals)
3. **Statistical Anomaly Detection** (Ensemble Forecast Deviations)
4. **Unified Correlated Model Anomaly Detection** (Multi-metric composite anomaly detection)
5. **Pattern-Based Detection** (Host/Pod Resource Misalignment)

---

## Detection Levels

### 1. **Node-Level Anomalies** (Per-Instance)

#### A. Classification Anomalies (IsolationForest) - Temporal-Aware
- **Level**: Individual node/host
- **Method**: Unsupervised ML using IsolationForest with per-cluster training and temporal awareness
- **Features Analyzed** (Temporal-Aware Mode - auto-enabled with 3+ months data):
  - `host_cpu_current_context` - Average host CPU for same-time patterns (hour, day-of-week)
  - `host_mem_current_context` - Average host memory for same-time patterns
  - `pod_cpu_current_context` - Average pod CPU for same-time patterns
  - `pod_mem_current_context` - Average pod memory for same-time patterns
  - Plus overall averages: `host_cpu`, `host_mem`, `pod_cpu`, `pod_mem`
- **Features Analyzed** (Basic Mode - fallback):
  - `host_cpu` - Average host CPU usage (last 24h by default)
  - `host_mem` - Average host memory usage (last 24h)
  - `pod_cpu` - Average pod/container CPU usage (last 24h)
  - `pod_mem` - Average pod/container memory usage (last 24h)

- **How It Works**:
  1. Identifies Kubernetes clusters and groups nodes by cluster membership
  2. **Temporal-Aware Feature Extraction** (when 3+ months data available):
     - Compares current values to same-time historical patterns (hour, day-of-week)
     - Priority: Same hour + same day-of-week > Same hour > Same day-of-week > Weekend/weekday
     - Reduces false positives from normal weekly/daily patterns
  3. **Basic Feature Extraction** (fallback):
     - Extracts average resource usage for each node over the lookback window (default: 24 hours)
  4. Normalizes features using StandardScaler per cluster
  5. Trains separate IsolationForest models for each Kubernetes cluster with contamination rate (default: 12%)
  6. Compares nodes only against their own cluster baseline (not global baseline)
  7. Flags nodes where host and pod usage patterns are misaligned within their cluster context
  8. Standalone nodes (no Kubernetes workloads) are excluded from anomaly detection

- **What It Detects**:
  - Nodes with high host CPU/memory but low pod usage → **Non-Kubernetes workloads** (backups, cron jobs, daemons)
  - Nodes with low host usage but high pod usage → **Potential resource accounting issues**
  - Nodes with unusual resource patterns compared to their cluster baseline (not global baseline)

- **Output**: 
  - Anomaly label: `-1` (anomalous) or `1` (normal)
  - Alert type: `classification_anomaly`
  - Severity: `WARNING`
  - Includes: instance name, host_cpu, host_mem, pod_cpu, pod_mem values

- **Scope**: **Per-cluster evaluation** - trains separate IsolationForest models per Kubernetes cluster, compares nodes only within their cluster. Requires minimum 2 nodes per cluster for comparison. Standalone nodes are excluded.

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
- **Method**: Statistical deviation from ensemble forecast with **temporal-aware** dual-threshold detection
- **Status**: **Fully Implemented** with temporal awareness

- **How It Works**:
  1. **Temporal-Aware Baseline** (when 3+ months of data available):
     - Compares current values to same-time historical patterns (hour, day-of-week)
     - Priority: Same hour + same day-of-week > Same hour > Same day-of-week > Weekend/weekday
     - Accounts for seasonality (e.g., weekend batch jobs, Monday morning spikes)
     - Falls back to forecast baseline if insufficient historical data
  2. Compares recent actual values (last 24 hours) against temporal-aware baseline
  3. Calculates both absolute and percentage deviations from baseline
  4. Applies dual-threshold logic to reduce false positives:
     - **Absolute thresholds**: Minimum meaningful differences (1% for I/O wait, 5 MB/s for network)
     - **Percentage thresholds**: Relative deviations (50% for current, 30% for mean)
  5. Considers model confidence (MAE) - high confidence models trigger fewer false positives
  6. Only flags anomalies when:
     - Significant absolute difference AND significant percentage deviation
     - AND (value is actually concerning OR model confidence is low)
  7. For very small baselines (< 1% I/O, < 5 MB/s network), uses absolute thresholds only (percentage is misleading)

- **Anomaly Conditions**:
  - Current deviation > 50% from forecast baseline
  - Mean deviation > 30% over recent window
  - Spike/drop > 100% from baseline
  - High MAE indicating model struggling to fit pattern
  - **AND** actual value exceeds concerning threshold (5% for I/O wait, 30% of threshold for network)

- **Anomaly Scoring**:
  - Score range: 0.0 to 1.0 (higher = more anomalous)
  - Severity classification:
    - `CRITICAL`: Score > 0.8
    - `WARNING`: Score 0.5-0.8
    - `INFO`: Score < 0.5

- **Output**:
  - Alert type: `io_network_anomaly`
  - Severity: `CRITICAL`, `WARNING`, or `INFO` (based on anomaly score)
  - Includes: node, signal, current value, deviation percentage, anomaly score, MAE, human-readable description

- **Scope**: **All nodes, per signal** - evaluated per node per signal with improved false positive reduction

---

#### D. Unified Correlated Model Anomaly Detection
- **Level**: Per-node (composite system health)
- **Method**: Statistical deviation detection on unified correlated model with metric contribution analysis
- **Status**: **Fully Implemented**

- **How It Works**:
  1. Uses unified correlated model that combines all 6 metrics (Host CPU, Host Memory, Pod CPU, Pod Memory, Disk I/O Wait, Network Transmit Bandwidth) into a single composite target
  2. Compares recent actual composite values (last 24 hours) against forecast baseline
  3. Calculates both absolute and percentage deviations from baseline
  4. Identifies primary contributing metric (single metric contributing most to composite value)
  5. Identifies top 3 contributing metrics by weighted contribution
  6. Provides current unnormalized values for all metrics
  7. Calculates deviations for each individual metric from its recent mean
  8. Detects strong correlations between metrics in the last 24 hours
  9. Applies threshold logic similar to I/O Network Anomaly Detection

- **Anomaly Conditions**:
  - Current composite deviation > 50% from forecast baseline
  - Mean composite deviation > 30% over recent window
  - Spike/drop > 100% from baseline
  - High MAE indicating model struggling to fit pattern
  - **AND** composite value exceeds concerning threshold

- **Anomaly Scoring**:
  - Score range: 0.0 to 1.0 (higher = more anomalous)
  - Severity classification:
    - `CRITICAL`: Score > 0.8
    - `WARNING`: Score 0.5-0.8
    - `INFO`: Score < 0.5

- **Output**:
  - Alert type: `unified_correlated_anomaly`
  - Severity: `CRITICAL`, `WARNING`, or `INFO` (based on anomaly score)
  - Includes:
    - Node identifier
    - Primary contributing metric (which metric is driving the anomaly)
    - Top 3 contributing metrics with their contributions
    - Current actual values for all 6 metrics (unnormalized)
    - Metric deviations from recent mean
    - Strong correlations detected between metrics
    - Anomaly score, MAE, human-readable description

- **Scope**: **All nodes** - evaluated per node using unified composite metric with detailed metric breakdown

---

### 3. **Disk-Level Anomalies** (Per-Node, Per-Mountpoint)

#### Disk Capacity Crisis Detection
- **Level**: Per-node, per-mountpoint
- **Method**: Hybrid linear + Prophet forecast with 90% threshold (temporal-aware)
- **How It Works**:
  1. Trains linear trend model and Prophet model for each node/mountpoint
  2. **Prophet Configuration**: `daily_seasonality=True, weekly_seasonality=True`
     - Learns daily patterns (e.g., daily ETL jobs, hourly backups)
     - Learns weekly patterns (e.g., weekend batch jobs, weekly reports)
  3. Forecasts 7 days ahead (accounts for seasonal patterns)
  4. Predicts when disk usage will reach 90% (considering daily/weekly patterns)
  5. Flags based on ETA (reduces false positives from normal batch job patterns)

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
| **Classification Anomaly** | Node | Per-cluster (min 2 nodes) | IsolationForest (ML, per-cluster) |
| **Host Pressure** | Node | All nodes | Rule-based thresholds |
| **Golden Signals** | Node + Signal | All nodes, 8 signal types | Prometheus query thresholds |
| **I/O Network Crisis** | Node + Signal | All nodes, 2 signals | Ensemble forecast prediction |
| **I/O Network Anomaly** | Node + Signal | All nodes, 2 signals | Statistical deviation (dual-threshold) |
| **Unified Correlated Anomaly** | Node | All nodes | Statistical deviation with metric contribution analysis |
| **Disk Crisis** | Node + Mountpoint | All nodes, all mountpoints | Hybrid forecast prediction |

---

## How Anomalies Are Found Across the Entire System

### 1. **Per-Cluster Scanning** (Temporal-Aware)
- **Classification Model**: Scans nodes within each Kubernetes cluster separately
  - Identifies cluster membership via Prometheus labels or pod instance patterns
  - Fetches host CPU/memory and pod CPU/memory for every node
  - **Temporal-Aware Feature Extraction** (auto-enabled with 3+ months data):
    - Compares current values to same-time historical patterns (hour, day-of-week)
    - Features: `*_current_context` (same-time patterns) + overall averages
    - Reduces false positives from normal weekly/daily patterns
  - **Basic Feature Extraction** (fallback):
    - Simple averages over `LOOKBACK_HOURS` window
  - Groups nodes by cluster and trains separate IsolationForest models per cluster
  - Builds feature matrices per cluster: `[cluster1_nodes], [cluster2_nodes], ...`
  - Compares nodes only against their own cluster baseline (not global)
  - Requires minimum 2 nodes per cluster for comparison
  - Standalone nodes (no Kubernetes workloads) are excluded from anomaly detection

### 2. **Per-Node Evaluation**
- **Golden Signals**: Queries Prometheus for each signal type, evaluates every instance
- **I/O Network**: Trains separate models for each node/signal combination
- **Disk**: Trains separate models for each node/mountpoint combination

### 3. **Time-Window Analysis**
- **Lookback Window**: Default 24 hours (configurable via `LOOKBACK_HOURS`)
- **Golden Signals**: Last 1 hour (configurable)
- **Forecast Horizon**: 7 days ahead

### 4. **Aggregation Strategy**
- **Per-cluster patterns**: Classification model trains separate IsolationForest per Kubernetes cluster, compares nodes within their cluster only
- **Per-node baselines**: I/O, Network, and Disk models learn individual node patterns
- **Threshold-based**: Golden signals use fixed thresholds across all nodes
- **Statistical deviation**: I/O/Network anomaly detection uses dual-threshold (absolute + percentage) with model confidence weighting

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
- `LOOKBACK_HOURS` (default: 24) - Window for classification model (temporal-aware if 3+ months data)
- `CONTAMINATION` (default: 0.12) - IsolationForest contamination rate (12% of nodes expected to be anomalous)
- `HORIZON_MIN` (default: 15) - Forecast horizon in minutes (can be overridden with `--forecast-horizon` flag)

**Temporal Awareness**:
- **Auto-enabled** when 3+ months (90 days) of historical data available
- **Classification Model**: Uses same-time historical patterns (hour, day-of-week) for feature extraction
- **I/O Network Anomaly**: Compares current values to same-time historical patterns instead of simple forecast baseline
- **Disk Models**: Prophet uses daily + weekly seasonality to learn batch job patterns
- **Benefits**: Reduces false positives from normal weekly/daily patterns (e.g., weekend backups, Monday morning spikes)

---

## Example: Complete Anomaly Scan

When `metrics.py --forecast` runs, it:

1. **Fetches data** for all nodes from Prometheus
2. **Cluster Identification**:
   - Identifies Kubernetes clusters via Prometheus labels or pod instance patterns
   - Classifies nodes as cluster members or standalone
3. **Classification Model**:
   - Extracts features for all nodes
   - Trains separate IsolationForest models per Kubernetes cluster
   - Compares nodes only within their cluster baseline
   - Flags anomalous nodes (e.g., "host02 has high host usage but low pod usage compared to cluster baseline")
   - Standalone nodes are excluded (require minimum 2 nodes per cluster)
4. **Host Pressure**:
   - Checks each node individually
   - Flags nodes with host pressure (e.g., "pi has 84% host memory but only 2% pod memory")
5. **Golden Signals**:
   - Queries 8 signal types across all nodes
   - Flags any node/signal combination exceeding thresholds
6. **I/O Network Crisis**:
   - For each node, for each signal (DISK_IO_WAIT, NET_TX_BW):
     - Loads or trains ensemble model
     - Forecasts 7 days ahead
     - Flags if threshold will be crossed within 30 days
7. **I/O Network Anomaly**:
   - For each node, for each signal (DISK_IO_WAIT, NET_TX_BW):
     - Compares recent actual values (24h) vs ensemble forecast
     - Calculates absolute and percentage deviations
     - Applies dual-threshold logic (absolute + percentage)
     - Considers model confidence (MAE)
     - Flags anomalies with severity (CRITICAL/WARNING/INFO) and human-readable descriptions
8. **Unified Correlated Anomaly**:
   - For each node:
     - Uses unified correlated model (all 6 metrics as features)
     - Compares recent composite values (24h) vs forecast baseline
     - Calculates deviations and identifies contributing metrics
     - Detects correlations between metrics
     - Flags anomalies with detailed metric breakdown (primary contributor, top 3, current values, correlations)
9. **Disk Crisis**:
   - For each node, for each mountpoint:
     - Loads or trains hybrid model
     - Forecasts 7 days ahead
     - Flags if 90% threshold will be crossed

**Result**: Comprehensive anomaly detection across the entire infrastructure at multiple levels.


