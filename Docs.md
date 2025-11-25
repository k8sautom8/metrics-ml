# User Requirements Document - Metrics AI

## Document Purpose

This document comprehensively describes all features, requirements, and outcomes of the Metrics AI system as discussed and implemented throughout the development process.

## Table of Contents

1. [System Overview](#system-overview)
2. [Core Features](#core-features)
3. [Execution Modes](#execution-modes)
4. [Model Types](#model-types)
5. [File Management](#file-management)
6. [Plot Generation](#plot-generation)
7. [Selective Retraining](#selective-retraining)
8. [Minimal Updates Strategy](#minimal-updates-strategy)
9. [Feature Outcomes](#feature-outcomes)
10. [Configuration Requirements](#configuration-requirements)

---

## 1. System Overview

### Purpose
Metrics AI is an intelligent forecasting and anomaly detection system for Kubernetes infrastructure that provides:
- Real-time predictions for infrastructure metrics
- Anomaly detection across multiple layers
- Efficient model updates without full retraining
- Comprehensive visualization and reporting

### Architecture
- **Dual-Layer Design**: Separate models for Host (full node) and Pod (Kubernetes workloads)
- **Ensemble Approach**: Combines Prophet, ARIMA, and LSTM models
- **Multi-Metric Support**: CPU, Memory, Disk Usage, Disk I/O, Network bandwidth
- **Manifest-Based Storage**: Efficient storage for multiple models per metric type

---

## 2. Core Features

### 2.1 Forecasting Models

#### Host Layer Forecasting
- **Input**: Host CPU and Memory metrics from Prometheus
- **Model**: Ensemble (Prophet + ARIMA + LSTM)
- **Output**: Forecasted CPU and Memory usage
- **Horizon**: Configurable (default: 7 days)
- **Files**: `host_forecast.pkl`, `host_arima.pkl`, `host_prophet_params.pkl`

#### Pod Layer Forecasting
- **Input**: Pod CPU and Memory metrics from Prometheus
- **Model**: Ensemble (Prophet + ARIMA + LSTM)
- **Output**: Forecasted Pod CPU and Memory usage
- **Horizon**: Configurable (default: 7 days)
- **Files**: `pod_forecast.pkl`, `pod_arima.pkl`, `pod_prophet_params.pkl`

#### Disk Full Prediction
- **Input**: Disk usage metrics from Prometheus
- **Model**: Hybrid (Linear trend + Prophet)
- **Output**: Days until disk reaches 90% capacity
- **Horizon**: 7 days
- **Storage**: Manifest-based (`disk_full_models.pkl`)
- **Features**:
  - Per-node, per-mountpoint predictions
  - Severity classification (CRITICAL, WARNING, SOON, OK)
  - CSV report generation

#### I/O and Network Crisis Prediction
- **Input**: Disk I/O wait and Network transmit bandwidth
- **Model**: Prophet (crisis detection)
- **Output**: Days until I/O or Network crisis threshold
- **Thresholds**: 
  - DISK_IO_WAIT: 30% (0.30 ratio)
  - NET_TX_BW: 120 MB/s (9.5e8 bytes/sec)
- **Storage**: Manifest-based (`io_net_models.pkl`)

#### I/O and Network Ensemble Forecast
- **Input**: Same as crisis prediction
- **Model**: Full ensemble (Prophet + ARIMA + LSTM)
- **Output**: Detailed forecasts with anomaly detection
- **Storage**: Manifest-based (`io_net_models.pkl`)

#### High-Level Forecast Workflow
- **Data ingestion**: Pull metrics from Prometheus/VictoriaMetrics (CPU, memory, disk, I/O, net) into timestamped dataframes.
- **Feature prep**: Engineer regressors (hour of day, weekend flag), align series per target (host, pod, per node/mount/signal).
- **Train/test split**: Time-ordered split using `TRAIN_FRACTION` (default 80%).
- **Model training**:
  - Prophet captures seasonality and changepoints.
  - ARIMA models autoregressive structure of the same target.
  - LSTM (optional) learns nonlinear dependencies over sliding windows.
- **Ensemble**: Combine Prophet, ARIMA, and LSTM outputs (uniform average) to produce the final forecast horizon.
- **Backtest**: Evaluate predictions on the held-out test set (MAE, RMSE) and log metrics/plots when applicable.
- **Persistence**: Save ensemble artifacts plus individual component metadata (Prophet params, ARIMA order, LSTM scaler) for reuse/minimal updates.
- **Forecast mode**: Reload cached artifacts, perform minimal updates (Prophet refit on 7d, ARIMA refit with cached order, LSTM fine-tune on last 2d), and regenerate the 7-day forecast + crisis overlays.

### 2.2 Anomaly Detection

#### Classification Model
- **Method**: Isolation Forest
- **Input**: Host CPU, Host Memory, Pod CPU, Pod Memory
- **Output**: Anomaly labels (-1 = anomalous, 1 = normal)
- **Features**:
  - Detects nodes with non-Kubernetes workloads
  - Identifies host pressure with minimal Kubernetes workload
  - Generates classification scatter plot

#### Golden Anomaly Detection
- **Method**: Multi-signal correlation analysis
- **Input**: Multiple infrastructure signals
- **Output**: Root-cause analysis of anomalies
- **Features**: Autonomous root-cause engine

---

## 3. Execution Modes

### 3.1 Training Mode (`--training`)

**Purpose**: Initial training or full retraining of all models

**Behavior**:
- Trains all models from scratch using full historical data
- Generates backtest plots for all models
- Displays backtest metrics (MAE, train/test split)
- Saves all models to disk
- Generates forecast plots

**When to Use**:
- First-time setup
- After significant infrastructure changes
- Periodic full retraining (e.g., weekly/monthly)

**Output**:
- All model files created/updated
- Backtest plots: `*_layer_backtest.png`
- Forecast plots: `*_layer_forecast.png`
- Backtest metrics displayed in console

**Example**:
```bash
python3 metrics.py --training
```

### 3.2 Forecast Mode (`--forecast`)

**Purpose**: Lightweight, frequent runs for real-time forecasting

**Behavior**:
- Uses cached models with minimal updates
- Applies minimal updates to all models (last 7 days for Prophet, latest data for ARIMA, 2 epochs for LSTM)
- Generates forecast plots only (no backtest plots)
- Saves updated models to disk
- Displays predictions and anomalies in tabular format:
  - Disk alerts table (CRITICAL/WARNING/SOON sections + full ETA table)
  - Classification anomalies (anomalous nodes)
  - Host pressure alerts (high host usage, low pod usage)
  - I/O+Network crisis predictions
  - Golden anomaly signals
- **No backtest metrics or plots**
- Respects global plot-window configuration. Supports per-run overrides via `--plot-history-hours` and `--plot-forecast-hours` for visual span control.
- Runs continuously at the cadence specified by `--interval` (default: 15 seconds). Use `--interval 0` for a single-shot execution.
- Optional alert sinks: `--alert-webhook <URL>` (POST JSON payload when actionable alerts fire) and `--pushgateway <URL>` (publish Prometheus gauges for Alertmanager).
- Optional dataset export: `--dump-csv <DIR>` writes the training data for host/pod/disk/I/O models to the specified directory for audit or offline experimentation.
- Alert dispatch includes all alert types: disk alerts, I/O+Network crisis/anomaly, golden signals, classification anomalies, and host pressure

**When to Use**:
- Scheduled frequent runs (every 15-60 seconds)
- Real-time monitoring with continuous alerting
- Production forecasting with SRE notifications
- Kubernetes Deployment (not CronJob) for continuous monitoring

**Output**:
- Updated model files (with latest timestamp)
- Forecast plots: `*_forecast.png`
- Tabular predictions and anomalies in console (disk alerts, classification anomalies, host pressure, I/O+Network crises, golden signals)
- Alert webhook payloads (when `--alert-webhook` specified and actionable alerts present)
- Pushgateway metrics (when `--pushgateway` specified)
- No backtest plots or metrics

**Example**:
```bash
python3 metrics.py --forecast
```

### 3.3 Show Backtest Mode (`--show-backtest`)

**Purpose**: View backtest performance without retraining

**Behavior**:
- Uses cached models
- Regenerates backtest plots
- Displays backtest metrics
- Does not update models (unless combined with retrain flags)

**When to Use**:
- Performance monitoring
- Model validation
- Debugging model accuracy

**Output**:
- Backtest plots: `*_layer_backtest.png`
- Backtest metrics displayed in console
- No forecast plots (unless combined with other modes)

**Example**:
```bash
python3 metrics.py --show-backtest
```

### 3.4 Normal Mode (No Flags)

**Purpose**: Use pre-trained models without updates

**Behavior**:
- Uses cached models as-is
- No plots generated
- No backtest metrics shown
- Fast execution

**When to Use**:
- Quick predictions without visualization
- Testing model loading
- Minimal resource usage

**Output**:
- Predictions in console
- No plots
- No metrics

**Example**:
```bash
python3 metrics.py
```

### 3.5 Lifecycle Summary (Training vs Forecast vs Backtest)

| Phase | Data Window | Processing Steps | Outputs | Interpretation |
|-------|-------------|------------------|---------|----------------|
| **Training** (`--training`) | Host/Pod: full `START_HOURS_AGO` lookback (default 360h ≈ 15 days). Disk/IO/Net: last 30–35 days per query. | - Time-ordered split (`TRAIN_FRACTION`, default 80% train / 20% test).<br>- Train Prophet, ARIMA, LSTM on train set.<br>- Run ensemble backtests on test set.<br>- Persist ensemble artifacts + component metadata (Prophet params, ARIMA order, LSTM scaler). | - Updated `*_forecast.pkl`, `*_arima.pkl`, `*_prophet_params.pkl`, `lstm_model.pkl`.<br>- Backtest metrics in console.<br>- Forecast + backtest plots in `FORECAST_PLOTS_DIR`.<br>- Manifests refreshed (`disk_full_models.pkl`, `io_net_models.pkl`). | Establishes baseline models. Backtest metrics quantify accuracy before deployment; plots show full history plus future projections for validation. |
| **Forecast** (`--forecast`) | Pulls fresh Prometheus samples for the last `START_HOURS_AGO` window but displays/plots only the configured `PLOT_HISTORY_HOURS` (default 7 days). Forecast visuals cover `PLOT_FORECAST_HOURS` (default 7 days). | - Load cached artifacts.<br>- Minimal updates: Prophet refit on last 7 days, ARIMA refit using cached order on latest series, LSTM fine-tune on last 2 days (2 epochs).<br>- Generate ensemble forecast (typically 7 days) and crisis overlays (disk/IO/Net).<br>- Save updated models + manifests (except when disabled via flags). | - Forecast plots for host/pod/disk/IO/Net (history + forecast windows labelled).<br>- Crisis plots per node/signal showing current value vs threshold.<br>- Console summary of divergences, crisis ETAs, anomalies. | “Live” run: answers operational questions (“Are we approaching a breach?” via crisis plots and “What is the full 7‑day outlook?” via ensemble layer plots). No backtest metrics shown to keep latency low. |
| **Backtest Display** (`--show-backtest` or training/retrain with plots enabled) | Uses cached train/test splits (same as training). | - Reload cached models.<br>- Regenerate backtest predictions for the stored test window.<br>- Render host/pod/disk/IO/Net backtest plots with train/test demarcation.<br>- Print MAE, RMSE, split statistics. | - Updated `*_layer_backtest.png`, crisis backtest plots, and console metrics. | Provides interpretability: how models performed historically, where train/test split lies, and what MAE values to expect. Forecast plots are suppressed unless combined with other modes. |

**Crisis vs Ensemble Plots (I/O + Network):**
- `{node}_{signal}_forecast.png` → Prophet + threshold “safety check” (ETA-to-breach).
- `{signal}_{node}_layer_forecast.png` → Full ensemble (Prophet + ARIMA + LSTM) 7‑day outlook.
Both are generated in `--forecast` mode, delivering complementary insights.

---

## 4. Model Types

### 4.1 Prophet Models

**Purpose**: Handle seasonality and trends

**Features**:
- Weekly seasonality detection
- Changepoint detection
- Uncertainty intervals

**Storage**:
- Hyperparameters saved separately for minimal updates
- Full models saved in ensemble files

**Minimal Update Strategy**:
- Loads saved hyperparameters
- Creates new model with same structure
- Fits on last 7 days of data
- Preserves learned seasonality patterns

### 4.2 ARIMA Models

**Purpose**: Capture autoregressive patterns

**Features**:
- Order: (2, 1, 0) - optimized for infrastructure metrics
- Handles trends and short-term dependencies

**Storage**:
- Model order and parameters saved separately
- Full models saved in ensemble files

**Minimal Update Strategy**:
- Loads saved model order
- Fits on latest data to incorporate recent trends
- Maintains model structure

### 4.3 LSTM Models

**Purpose**: Deep learning for complex patterns

**Features**:
- 2-layer LSTM architecture
- Sequence length: 60 (configurable)
- CPU-only implementation (no GPU required)

**Storage**:
- Model weights and scaler saved together
- Separate file: `lstm_model.pkl`

**Minimal Update Strategy**:
- Loads pre-trained model and scaler
- Fine-tunes for 2 epochs on last 2 days of data
- Preserves learned patterns while adapting to recent changes

**Requirement**: TensorFlow (optional)

---

## 5. File Management

### 5.1 Model Files Directory

**Environment Variable**: `MODEL_FILES_DIR`
**Default**: `./model_files`

**Behavior**:
- If absolute path provided, uses as-is
- If relative path provided, uses relative to current directory
- Directory created automatically if missing

**Files Stored**:
- Host/Pod ensemble models
- ARIMA models
- Prophet hyperparameters
- LSTM models
- Disk models manifest
- I/O and Network models manifest
- Anomaly detection models

### 5.2 Forecast Plots Directory

**Environment Variable**: `FORECAST_PLOTS_DIR`
**Default**: `./forecast_plots`

**Behavior**:
- If absolute path provided, uses as-is
- If relative path provided, uses relative to current directory
- Directory created automatically if missing

**Files Stored**:
- All forecast plots
- All backtest plots
- Classification plots
- CSV reports (e.g., `disk_full_prediction.csv`)

### 5.3 Model Persistence

**Requirement**: Models must be saved after updates

**Implementation**:
- **Training Mode**: Models saved after full training
- **Forecast Mode**: Models saved after minimal updates
- **Retrain Mode**: Only retrained models saved
- **Normal Mode**: No model updates, no saving

**Timestamp Behavior**:
- Model files updated with latest timestamp when saved
- Allows tracking of last update time
- Useful for monitoring model freshness

---

## 6. Plot Generation

### 6.1 Forecast Plots

**Generated When**:
- `--training` mode
- `--forecast` mode
- During retraining

**Not Generated When**:
- Normal mode (no flags)
- `--show-backtest` only (unless combined with training/forecast)

**Plot Types**:

#### Host/Pod Layer Forecasts
- **Format**: `{model_type}_layer_forecast.png`
- **Content**: 
  - Configurable historical window (default 7 days)
  - Configurable forecast window (default 7 days)
  - X-axis: Time (HH:MM format, 3-hour ticks)
  - Shows Prophet, ARIMA, LSTM, and Ensemble predictions

#### Disk Forecasts
- **Format**: `disk_{node}_{mountpoint}_forecast.png`
- **Content**:
  - Historical usage
  - Forecasted usage
  - Threshold line (90%)
  - ETA to threshold

#### I/O and Network Forecasts
- **Format**: `{node}_{signal}_forecast.png` or `{signal}_{node}_layer_forecast.png`
- **Content**:
  - Configurable historical window (default 7 days)
  - Configurable forecast window (default 7 days)
  - Threshold line
  - Current value and ETA
- **Interpretation**:
  - `{node}_{signal}_forecast.png` → “crisis/ETA” view produced by the dedicated safety check (single Prophet + threshold), answering *“Are we approaching a breach?”*
  - `{signal}_{node}_layer_forecast.png` → Full ensemble (Prophet + ARIMA + LSTM) forecast, answering *“What does the complete 7‑day outlook look like for this signal?”*
  - Both plots are generated during `--forecast`, but they serve complementary purposes: actionable threshold monitoring vs deep trend visualization.

### 6.2 Backtest Plots

**Generated When**:
- `--training` mode
- `--show-backtest` mode
- During retraining (with `--disk-retrain` or `--io-net-retrain`)

**Not Generated When**:
- `--forecast` mode (forecast plots only)
- Normal mode

**Plot Types**:

#### Host/Pod Layer Backtests
- **Format**: `{model_type}_layer_backtest.png`
- **Content**:
  - Train data
  - Test data (actual)
  - Prophet, ARIMA, LSTM, and Ensemble backtest predictions
  - Train/test split marker
  - MAE metrics in title

#### Disk Backtests
- **Format**: `disk_{node}_{mountpoint}_forecast.png` (same as forecast, but with train/test split)
- **Content**: Same as forecast plots with train/test visualization

#### I/O and Network Backtests
- **Format**: `{node}_{signal}_backtest.png`
- **Content**:
  - Train data
  - Test data
  - Forecast predictions
  - Threshold line
  - MAE and RMSE metrics

### 6.3 Classification Plots

**Generated When**:
- `--training` mode
- `--forecast` mode

**Not Generated When**:
- Normal mode
- `--show-backtest` only

**Plot Type**:
- **Format**: `classification_host_vs_pod.png`
- **Content**: Scatter plot of Host Memory vs Pod Memory with anomaly labels

### 6.4 Global Plot Window Configuration

- **Defaults**: `PLOT_HISTORY_HOURS=168`, `PLOT_FORECAST_HOURS=168` (7 days each).
- **Environment Overrides**: Set the env vars above to change defaults for all runs.
- **CLI Overrides**: `--plot-history-hours <int>` and `--plot-forecast-hours <int>` provide per-execution control.
- **Consistency**: All plot-generating routines (host, pod, disk, I/O, network, classification) read from the same globals so the visual spans stay aligned across artifacts.

---

## 7. Selective Retraining

### 7.1 Disk Model Retraining (`--disk-retrain`)

**Purpose**: Retrain specific disk models without full retraining

**Syntax**:
```bash
--disk-retrain <targets>
```

**Target Formats**:
- `all` - Retrain all disk models
- `host02` - Retrain all mounts for host02
- `host02:/` - Retrain specific mountpoint
- `host02:/,worker01:/home` - Multiple targets (comma-separated)

**Behavior**:
- Only specified models are retrained
- Other models use cached values
- Retrained models use minimal update (last 7 days) if model exists
- First-time training uses full data
- Generates backtest plots and metrics for retrained models
- Updates manifest with new predictions

**Outcome**:
- Faster execution (only retrains specified models)
- Updated predictions for retrained disks
- Backtest metrics for retrained models
- Manifest updated with latest timestamps

**Example**:
```bash
# Retrain all disks
python3 metrics.py --disk-retrain all

# Retrain specific disk
python3 metrics.py --disk-retrain host02:/
```

### 7.2 I/O and Network Model Retraining (`--io-net-retrain`)

**Purpose**: Retrain specific I/O and Network models

**Syntax**:
```bash
--io-net-retrain <targets>
```

**Target Formats**:
- `all` - Retrain all I/O and Network models
- `host02` - Retrain all signals for host02
- `host02:DISK_IO_WAIT` - Retrain specific signal for node
- `host02:DISK_IO_WAIT,worker01:NET_TX_BW` - Multiple targets

**Behavior**:
- Only specified models are retrained
- Other models use cached values
- Retrained models use minimal update (last 7 days) if model exists
- First-time training uses full data
- Generates backtest plots and metrics for retrained models
- Updates manifest with new models

**Outcome**:
- Faster execution
- Updated predictions for retrained signals
- Backtest metrics for retrained models
- Manifest updated

**Example**:
```bash
# Retrain all I/O and Network models
python3 metrics.py --io-net-retrain all

# Retrain specific signal
python3 metrics.py --io-net-retrain host02:DISK_IO_WAIT
```

### 7.3 Retraining with Backtest Display

**Combined Usage**:
```bash
# Retrain and show backtest metrics
python3 metrics.py --disk-retrain host02 --show-backtest
python3 metrics.py --io-net-retrain host02:DISK_IO_WAIT --show-backtest
```

**Outcome**:
- Retrained models
- Backtest plots generated
- Backtest metrics displayed
- Updated forecasts

---

## 8. Minimal Updates Strategy

### 8.1 Concept

**Requirement**: Models should receive minimal updates in forecast mode, not full retraining

**Rationale**:
- Pre-trained models know data seasonality and patterns
- Full retraining is time-consuming and unnecessary
- Recent data should be incorporated to adapt to trends
- Learned structure should be preserved

### 8.2 Implementation

#### Prophet Minimal Update
1. Load saved hyperparameters from disk
2. Create new Prophet model with same structure
3. Fit on last 7 days of data only
4. Preserves seasonality knowledge
5. Incorporates recent trends
6. Saves updated hyperparameters to disk (e.g., `host_prophet_params.pkl`, `pod_prophet_params.pkl`)

#### ARIMA Minimal Update
1. Load saved model order from disk
2. Fit ARIMA with same order on latest data
3. Incorporates recent trends
4. Maintains model structure
5. Saves updated ARIMA model to disk (e.g., `host_arima.pkl`, `pod_arima.pkl`)

#### LSTM Minimal Update
1. Load pre-trained model and scaler from disk
2. Fine-tune for 2 epochs on last 2 days of data
3. Preserves learned patterns
4. Adapts to recent changes
5. Saves updated model and scaler to disk (`lstm_model.pkl`)

### 8.3 When Applied

**Applied In**:
- `--forecast` mode (all models)
- Retraining existing models (uses minimal update instead of full training)

**Not Applied In**:
- Initial training (full training)
- First-time model creation (full training)
- Explicit retraining with `--training` flag (full training)

### 8.4 Outcome

- **Speed**: 10-30 seconds vs 5-15 minutes for full training
- **Accuracy**: Maintains learned patterns while adapting to trends
- **Efficiency**: Suitable for frequent runs (every 15-60 seconds)
- **Freshness**: Models updated with latest data

---

## 9. Feature Outcomes

### 9.1 Backtest Metrics Display

**Feature**: Control when backtest metrics are displayed

**Implementation**:
- **Training Mode**: Always shows backtest metrics
- **Show Backtest Mode**: Shows backtest metrics for cached models
- **Forecast Mode**: Does not show backtest metrics
- **Normal Mode**: Does not show backtest metrics

**Outcome**:
- Users can view model performance on demand
- Forecast mode remains lightweight
- Training mode provides full transparency

### 9.2 Plot Generation Control

**Feature**: Control when plots are generated

**Implementation**:
- **Training Mode**: Generates both forecast and backtest plots
- **Forecast Mode**: Generates forecast plots only
- **Show Backtest Mode**: Generates backtest plots only
- **Normal Mode**: No plots generated
- **Retrain Mode**: Generates plots for retrained models

**Outcome**:
- Efficient resource usage
- Plots generated only when needed
- Clear separation between forecast and backtest visualization

### 9.3 Model File Updates

**Feature**: Models updated and saved in forecast mode

**Implementation**:
- Minimal updates applied to all models
- Updated models saved to disk
- File timestamps reflect latest update
- Manifest files updated when changed

**Outcome**:
- Models stay current with latest data
- Timestamp tracking for model freshness
- Persistent updates across runs

### 9.4 Forecast Plot Consistency

**Feature**: Consistent forecast plot format across all models

**Implementation**:
- Host/Pod: 24h historical + 3h forecast
- Disk: Full historical + forecast to threshold
- I/O/Network: 24h historical + 3h forecast
- Consistent time formatting (HH:MM)
- Consistent styling and legends

**Outcome**:
- Uniform visualization experience
- Easy comparison across metrics
- Professional presentation

### 9.5 Selective Retraining

**Feature**: Retrain specific models without full retraining

**Implementation**:
- Granular control via CLI arguments
- Supports node names, mountpoints, signals
- Alias matching for flexible targeting
- Minimal updates for existing models
- Full training for new models

**Outcome**:
- Faster execution
- Targeted model updates
- Efficient resource usage
- Flexible retraining strategy

### 9.6 Alert Delivery

The system provides two alert delivery mechanisms for real-time notification of detected issues:

#### Webhook Alerts (`--alert-webhook`)

When `--alert-webhook <URL>` is supplied in `--forecast` mode, the system posts a JSON payload to the specified HTTP endpoint whenever actionable alerts are detected. The payload includes:

**Alert Categories:**
- **Disk Alerts**: CRITICAL (<3 days to 90%), WARNING (3-7 days), SOON (7-30 days). Only non-OK alerts are included (OK status alerts are filtered out).
- **I/O+Network Crisis**: Predicted I/O or network saturation within 30 days
- **I/O+Network Anomaly**: Anomalous I/O or network patterns detected by ensemble models
- **Golden Anomaly**: Root-cause signals detected via Prometheus queries (iowait high, inodes critical, network drops, OOM kills, fork bombs, FD leaks, TCP retransmissions, network saturation)
- **Classification Anomaly**: Nodes with anomalous host/pod usage patterns (detected by Isolation Forest). Indicates nodes where host and pod usage are misaligned (e.g., high host usage but low pod usage, suggesting non-Kubernetes workloads)
- **Host Pressure**: Nodes with high host CPU/memory usage but minimal Kubernetes workload (pod CPU/mem < 5%). Suggests OS-level processes (backups, cron jobs, daemons) consuming resources

**Payload Structure:**
- `timestamp`: ISO 8601 timestamp of alert generation
- `summary_text`: Human-readable summary of all alert counts
- `disk`: Counts (critical, warning, soon, total) and sample records (top 5) with full details
- `io_network_crisis`: Total count and sample records
- `io_network_anomaly`: Total count and sample records
- `golden_anomaly`: Total count and sample records
- `classification_anomaly`: Total count and sample records (includes instance, host_cpu, host_mem, pod_cpu, pod_mem, severity, signal, detected_at)
- `host_pressure`: Total count and sample records (includes instance, host_cpu, host_mem, severity, signal, detected_at)

**Sample Records:** Each alert category includes up to 5 sample records with full details to provide immediate context for SRE triage.

**Note:** Webhooks are only dispatched when there are actionable alerts (non-OK status). If all systems are healthy, no webhook is sent.

#### Pushgateway Metrics (`--pushgateway`)

When `--pushgateway <URL>` is supplied, the system publishes Prometheus-compatible metrics to the Pushgateway for integration with Alertmanager and Grafana dashboards.

**Published Metrics:**
- `metrics_ai_disk_alerts_critical` - Count of critical disk alerts
- `metrics_ai_disk_alerts_warning` - Count of warning disk alerts
- `metrics_ai_disk_alerts_soon` - Count of soon disk alerts
- `metrics_ai_disk_alerts_total` - Total non-OK disk alerts
- `metrics_ai_io_network_crisis_total` - I/O+Network crisis count
- `metrics_ai_io_network_anomaly_total` - I/O+Network anomaly count
- `metrics_ai_golden_anomaly_total` - Golden anomaly signal count
- `metrics_ai_classification_anomaly_total` - Classification anomaly count
- `metrics_ai_host_pressure_total` - Host pressure alert count

**Integration:** These metrics can be scraped by Prometheus and used in Alertmanager rules or Grafana dashboards for visualization and alerting.

#### Cadence Control

Both alert channels leverage the `--interval` loop (default 15 seconds) so a single container can continuously monitor and notify. Set `--interval 0` for ad‑hoc/manual runs without continuous monitoring.

**Production Deployment:**
- Deploy as a Kubernetes `Deployment` (not CronJob) with `--interval 15` for continuous monitoring
- Configure both `--alert-webhook` and `--pushgateway` for redundant alert delivery
- Webhook provides rich JSON payloads with sample records for immediate triage
- Pushgateway provides Prometheus metrics for integration with existing monitoring infrastructure

---

## 10. Configuration Requirements

### 10.1 Environment Variables

#### Required
- None (all have defaults)

#### Recommended
- `VM_URL`: Prometheus/VictoriaMetrics endpoint
- `MODEL_FILES_DIR`: Model storage location
- `FORECAST_PLOTS_DIR`: Plot storage location

#### Optional
- All other variables have sensible defaults
- Plot window overrides: `PLOT_HISTORY_HOURS`, `PLOT_FORECAST_HOURS`

#### CLI Overrides
- `--plot-history-hours <int>`: Override history window for the run
- `--plot-forecast-hours <int>`: Override forecast window for the run
- `--interval <int>`: Seconds between forecast iterations (default 15; set 0 to run once)
- `--alert-webhook <url>`: Notify an HTTP endpoint whenever alerts are present
- `--pushgateway <url>`: Push alert gauges to a Prometheus Pushgateway
- `--dump-csv <dir>`: Dump the training datasets for each model into the specified directory (useful for audits/offline analysis)

### 10.2 Data Requirements

#### Prometheus Queries
- Host CPU: `1 - avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) by (instance)`
- Host Memory: `(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes`
- Pod CPU: `sum(rate(container_cpu_usage_seconds_total{container!="POD",container!=""}[5m])) by (instance)`
- Pod Memory: `sum(container_memory_working_set_bytes{container!="POD",container!=""}[5m]) by (instance)`
- Disk Usage: `1 - (node_filesystem_free_bytes / node_filesystem_size_bytes)`
- Disk I/O: `avg by (instance) (rate(node_disk_io_time_seconds_total[5m]))`
- Network: `avg by (instance) (rate(node_network_transmit_bytes_total[5m]))`

#### Data Quality
- Minimum 100 data points for training
- Minimum 50 data points for backtesting
- Regular data collection (recommended: 1-minute intervals)

### 10.3 Storage Requirements

#### Model Files
- Host/Pod models: ~2-3 MB each
- LSTM model: ~60 MB (if used)
- Disk manifest: ~1-2 KB per disk
- I/O/Network manifest: ~500 KB per signal

#### Plot Files
- Forecast plots: ~100-300 KB each
- Backtest plots: ~100-300 KB each
- Classification plot: ~50-100 KB

#### Estimated Total
- Small cluster (6 nodes, 2 mounts): ~100 MB
- Medium cluster (20 nodes, 5 mounts): ~500 MB
- Large cluster (100 nodes, 10 mounts): ~2 GB

---

## Summary

### Key Features Delivered

1. ✅ **Dual-Layer Forecasting**: Host and Pod layers with separate models
2. ✅ **Ensemble Models**: Prophet + ARIMA + LSTM combination
3. ✅ **Multiple Metrics**: CPU, Memory, Disk, I/O, Network
4. ✅ **Anomaly Detection**: Classification and root-cause analysis
5. ✅ **Minimal Updates**: Efficient forecast mode with incremental updates
6. ✅ **Selective Retraining**: Granular control over model updates
7. ✅ **Comprehensive Plotting**: Forecast and backtest visualization
8. ✅ **Configurable Storage**: Environment variable-based paths
9. ✅ **Model Persistence**: Automatic saving after updates
10. ✅ **Performance Optimization**: Fast forecast mode for frequent runs

### Execution Modes

- **Training**: Full model training with backtest metrics
- **Forecast**: Lightweight forecasting with minimal updates
- **Show Backtest**: View performance without retraining
- **Normal**: Use cached models without updates

### File Management

- Configurable directories via environment variables
- Automatic directory creation
- Model files updated with latest timestamps
- Comprehensive plot generation

### Outcomes

- **Efficiency**: Forecast mode runs in 10-30 seconds
- **Accuracy**: Minimal updates preserve learned patterns
- **Flexibility**: Selective retraining for targeted updates
- **Transparency**: Backtest metrics available on demand
- **Usability**: Clear separation of modes and outputs


