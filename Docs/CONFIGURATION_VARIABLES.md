# Configuration Variables Explained

This document explains how each configuration variable is used in training and prediction across the Metrics AI system.

---

## 1. `HORIZON_MIN` (Default: 15 minutes)

**Purpose**: Forecast horizon - how far ahead the models predict.

### Usage in Training:
- **Prophet**: Creates future dataframe with `periods=HORIZON_MIN` for training
- **ARIMA**: Forecasts `steps=HORIZON_MIN` future values
- **LSTM**: 
  - Creates sequences where output length = `HORIZON_MIN`
  - Model architecture: `Dense(horizon_min)` output layer
  - Training data: `y.append(scaled[i:i+horizon_min])` - predicts next `HORIZON_MIN` values

### Usage in Prediction:
- All three models generate forecasts for exactly `HORIZON_MIN` minutes ahead
- Ensemble combines the first `HORIZON_MIN` values from each model
- Used to determine how many future data points to generate

### Override Options:
- **CLI Flag**: `--forecast-horizon <realtime|neartime|future>`
  - `realtime`: 15 minutes (short-term decisions)
  - `neartime`: 3 hours (180 minutes, near-term planning)
  - `future`: 7 days (10,080 minutes, long-term capacity planning)
- **Priority**: CLI flag > Environment variable > Default (15 minutes)
- **Display**: Horizon is printed in all outputs and included in plot titles for transparency

**Example**: `HORIZON_MIN=15` means models predict the next 15 minutes of resource usage.

**Code References**:
- `build_ensemble_forecast_model()`: `horizon_min=HORIZON_MIN` parameter
- `generate_forecast_from_cached_model()`: Uses `horizon_min` to generate forecasts
- LSTM training: `y.append(scaled[i:i+horizon_min])`
- ARIMA: `arima.forecast(steps=horizon_min)`
- Prophet: `future = prophet_model.make_future_dataframe(periods=HORIZON_MIN, freq='min')`
- CLI: `--forecast-horizon` flag overrides `HORIZON_MIN` in forecast mode

---

## 2. `LOOKBACK_HOURS` (Default: 24 hours)

**Purpose**: Time window for feature extraction in anomaly detection and cluster identification.

### Usage in Training:
- **Anomaly Detection (Classification Model)**:
  - Extracts average CPU/memory usage over last `LOOKBACK_HOURS`
  - **Temporal-Aware Features** (when 3+ months of data available):
    - Compares current values to same-time historical patterns (hour, day-of-week)
    - Features: `host_cpu_current_context`, `host_mem_current_context`, `pod_cpu_current_context`, `pod_mem_current_context`
    - Accounts for seasonality (e.g., Monday morning spikes, weekend batch jobs)
  - **Basic Features** (fallback):
    - Simple averages: `host_cpu`, `host_mem`, `pod_cpu`, `pod_mem` averaged over this window
  - Used to train IsolationForest models per cluster

### Usage in Prediction:
- **Cluster Identification**: 
  - Looks back `LOOKBACK_HOURS` to identify which nodes share pod workloads
  - Groups nodes into clusters based on pod instance patterns in this window
- **Anomaly Detection**:
  - **Temporal-Aware** (auto-enabled with 3+ months data):
    - Compares current values to same-time historical patterns
    - Same hour + same day-of-week (best match)
    - Same hour of day (daily pattern)
    - Same day of week (weekly pattern)
    - Weekend vs weekday (fallback)
  - **Basic Mode** (fallback):
    - Calculates average resource usage over last `LOOKBACK_HOURS` for each node
    - Compares current patterns against historical baseline from this window

**Example**: `LOOKBACK_HOURS=24` means the system analyzes the last 24 hours of data to:
- Identify cluster membership
- Extract features for anomaly detection (with temporal awareness if sufficient data)
- Establish baseline patterns

**Temporal Awareness**:
- **Auto-enabled** when 3+ months (90 days) of historical data available
- **Benefits**: Reduces false positives from normal weekly/daily patterns
- **Example**: Monday 9 AM high CPU compared to historical Monday 9 AM averages, not weekend averages

**Code References**:
- `identify_clusters()`: `lookback_hours=LOOKBACK_HOURS` - determines cluster membership
- `classification_model()`: `lookback_hours=LOOKBACK_HOURS` - feature extraction window with temporal awareness
- `extract_instance_features()`: Uses lookback window to calculate averages (temporal-aware when enabled)

---

## 3. `CONTAMINATION` (Default: 0.12 = 12%)

**Purpose**: Expected fraction of anomalous nodes in IsolationForest anomaly detection.

### Usage in Training:
- **IsolationForest Model**: 
  - `IsolationForest(contamination=CONTAMINATION, random_state=42)`
  - Trains separate models per Kubernetes cluster
  - Sets the expected proportion of outliers (anomalies) in the data

### Usage in Prediction:
- When predicting anomalies, IsolationForest flags approximately `CONTAMINATION * total_nodes` as anomalous
- Higher values = more nodes flagged as anomalous (more sensitive)
- Lower values = fewer nodes flagged (more conservative)

**Example**: `CONTAMINATION=0.12` means the model expects 12% of nodes to be anomalous.
- In a cluster of 10 nodes, ~1-2 nodes will typically be flagged
- In a cluster of 100 nodes, ~12 nodes will typically be flagged

**Code References**:
- `classification_model()`: `contamination=CONTAMINATION` parameter
- `IsolationForest(contamination=contamination, random_state=42)` - model initialization

---

## 4. `STEP` (Default: "60s")

**Purpose**: Query resolution/step size for Prometheus/VictoriaMetrics queries.

### Usage in Training:
- **Data Fetching**: 
  - `fetch_victoriametrics_metrics(query, start, end, step=STEP)`
  - Determines the time interval between data points in the query
  - Affects data density: smaller step = more data points, larger step = fewer points
  - **Automatic Optimization**: System attempts to use base `STEP` (60s) first to maximize data
  - **Adaptive Fallback**: If query fails (422 error), automatically retries with larger step sizes
  - **Query Chunking**: For very large time ranges, queries are split into 360-hour chunks to preserve maximum data resolution

### Usage in Prediction:
- Same as training - determines query resolution when fetching latest data
- Affects forecast accuracy: more frequent data (smaller step) = better predictions
- System automatically handles VictoriaMetrics query limits while maximizing data

### Adaptive Step Sizing:
The system automatically optimizes step size based on time range:
- **Small ranges (<360 hours)**: Uses base `STEP` (60s) - maximum data resolution
- **Large ranges (>360 hours)**: 
  - First attempts full range with base `STEP` (60s)
  - If query fails, splits into 360-hour chunks, each with 60s step
  - Combines chunks to preserve all data with maximum resolution
- **Very large ranges**: May use adaptive step sizing (300s, 600s, etc.) only if chunking fails

**Example**: 
- `STEP="60s"` = 1 data point per minute (60 data points per hour)
- `STEP="300s"` = 1 data point per 5 minutes (12 data points per hour)
- `STEP="3600s"` = 1 data point per hour (24 data points per day)

**Code References**:
- `fetch_and_preprocess_data()`: `step=STEP` parameter with automatic optimization
- `fetch_victoriametrics_metrics()`: Uses step size in query parameters
- `calculate_adaptive_step()`: Calculates optimal step size for large ranges

---

## 5. `START_HOURS_AGO` (Default: 360 hours = 15 days)

**Purpose**: How far back to fetch historical data from Prometheus/VictoriaMetrics.

### Usage in Training:
- **Data Fetching**:
  - `start = int((pd.Timestamp.now() - pd.Timedelta(hours=start_hours_ago)).timestamp())`
  - Determines the time range of historical data used for training
  - More data = better model training (up to a point)
  - **Query Optimization**: System automatically handles VictoriaMetrics query limits
  - **Data Preservation**: Uses query chunking to preserve maximum data resolution

### Usage in Prediction:
- Same as training - determines how much historical data to fetch for forecasts
- Used in both training mode and forecast mode
- System automatically optimizes queries to get maximum data

### Query Optimization Strategy:

The system uses a multi-tier approach to maximize data while avoiding query failures:

1. **Primary Attempt**: Tries full time range with base `STEP` (60s) to get maximum data
2. **Query Chunking**: If query fails (422 error), automatically splits into 360-hour chunks
   - Each chunk uses 60s step (preserves maximum resolution)
   - Chunks are combined and deduplicated
   - Result: All data preserved with maximum resolution
3. **Adaptive Step Fallback**: Only used if chunking fails (rare)
   - Calculates optimal step size based on time range
   - Targets ~15,000 data points per series to account for multiple instances

### Examples:
- `START_HOURS_AGO=360` = Fetch last 15 days of data (360 hours)
  - Single query with 60s step → ~21,600 points per series
- `START_HOURS_AGO=720` = Fetch last 30 days of data
  - Attempts single query with 60s step
  - If fails: Splits into 2 chunks of 360h each, both with 60s step
  - Combined result: ~43,200 points per series (all data preserved)
- `START_HOURS_AGO=1080` = Fetch last 45 days of data
  - Splits into 3 chunks of 360h each, all with 60s step
  - Combined result: ~64,800 points per series

### Trade-offs:
- **More data (larger value)**: 
  - Better for capturing long-term patterns (daily, weekly, monthly seasonality)
  - Enables monthly seasonality detection (requires ≥30 days)
  - Slightly slower processing (chunking overhead)
  - More memory usage
- **Less data (smaller value)**:
  - Faster queries and processing
  - May miss long-term patterns
  - Better for recent trend detection

### Performance Notes:
- **Chunking Overhead**: Minimal - chunks are processed sequentially but efficiently
- **Data Quality**: No data loss - chunking preserves all data with original resolution
- **Memory**: Linear scaling with time range (no exponential growth)

**Code References**:
- `fetch_and_preprocess_data()`: `start_hours_ago=START_HOURS_AGO` parameter with automatic optimization
- `calculate_adaptive_step()`: Calculates optimal step size for large ranges
- Used in all data fetching functions

---

## 6. `LSTM_SEQ_LEN` (Default: 60)

**Purpose**: Sequence length for LSTM model - how many past data points the LSTM looks at to predict the future.

### Usage in Training:
- **LSTM Model Architecture**:
  - Creates input sequences of length `LSTM_SEQ_LEN`
  - `X.append(scaled[i-LSTM_SEQ_LEN:i])` - sliding window of past `LSTM_SEQ_LEN` points
  - `y.append(scaled[i:i+horizon_min])` - predicts next `HORIZON_MIN` values
  - Model input shape: `(LSTM_SEQ_LEN, 1)`

### Usage in Prediction:
- Uses last `LSTM_SEQ_LEN` data points to generate forecast
- `last_seq = scaled[-LSTM_SEQ_LEN:].reshape(1, LSTM_SEQ_LEN, 1)`
- Minimum data requirement: `len(ts) >= LSTM_SEQ_LEN + horizon_min`

**Example**: `LSTM_SEQ_LEN=60` means:
- LSTM looks at the last 60 data points (60 minutes if data is 1-minute resolution)
- Uses these 60 points to predict the next `HORIZON_MIN` values

**Trade-offs**:
- **Larger value**: 
  - Captures longer-term patterns
  - Requires more training data
  - More memory usage
- **Smaller value**:
  - Focuses on recent patterns
  - Requires less data
  - May miss long-term trends

**Code References**:
- `build_ensemble_forecast_model()`: Used in LSTM sequence creation
- `generate_forecast_from_cached_model()`: Checks `len(ts) >= LSTM_SEQ_LEN + horizon_min`
- LSTM training: `for i in range(LSTM_SEQ_LEN, len(scaled) - horizon_min)`

---

## 7. `LSTM_EPOCHS` (Default: 10)

**Purpose**: Number of training epochs for LSTM model.

### Usage in Training:
- **LSTM Training**:
  - `model.fit(X, y, epochs=LSTM_EPOCHS, batch_size=32, verbose=0)`
  - Controls how many times the model sees the entire training dataset
  - More epochs = more training, potentially better accuracy (but risk of overfitting)

### Usage in Prediction:
- Not used directly in prediction, but affects model quality
- Well-trained models (from training) produce better forecasts

**Example**: `LSTM_EPOCHS=10` means:
- Model trains for 10 complete passes through the training data
- Early stopping may stop training earlier if no improvement (patience=3)

**Trade-offs**:
- **More epochs**: 
  - Better model fit (up to a point)
  - Risk of overfitting
  - Slower training
- **Fewer epochs**:
  - Faster training
  - May underfit (model not learning enough)
  - Early stopping helps prevent overfitting

**Code References**:
- `build_ensemble_forecast_model()`: `model.fit(X, y, epochs=LSTM_EPOCHS, ...)`
- Early stopping callback: `EarlyStopping(patience=3)` may stop before all epochs

---

## 8. `TRAIN_FRACTION` (Default: 0.8 = 80%)

**Purpose**: Fraction of data used for training vs testing in time-ordered split.

### Usage in Training:
- **Train/Test Split**:
  - `split_idx = max(1, int(len(ts) * TRAIN_FRACTION))`
  - `train_ts = ts.iloc[:split_idx]` - first 80% of data (chronologically)
  - `test_ts = ts.iloc[split_idx:]` - last 20% of data (chronologically)
  - Used for backtesting and model evaluation

### Usage in Prediction:
- Same split is used when retraining models
- Test set is used to calculate backtest metrics (MAE, RMSE)
- Ensures models are evaluated on unseen future data (realistic scenario)

**Example**: `TRAIN_FRACTION=0.8` means:
- 80% of historical data → Training set
- 20% of historical data → Test set (for backtesting)
- If you have 1000 data points:
  - Training: First 800 points (chronologically)
  - Testing: Last 200 points (chronologically)

**Why Time-Ordered Split?**
- Time series data must be split chronologically (not randomly)
- Training on future data to predict past would be unrealistic
- Ensures backtest metrics reflect real-world performance

**Code References**:
- `build_ensemble_forecast_model()`: `split_idx = max(1, int(len(pdf) * TRAIN_FRACTION))`
- `predict_disk_full_days()`: Uses `TRAIN_FRACTION` for train/test split
- All model training functions use this for splitting data

---

## Summary: How They Work Together

### Training Flow:
1. **Data Fetching**: `START_HOURS_AGO` + `STEP` → Fetch historical data
2. **Data Splitting**: `TRAIN_FRACTION` → Split into train/test sets
3. **Model Training**:
   - **Prophet**: Uses all training data, predicts `HORIZON_MIN` ahead
   - **ARIMA**: Uses all training data, predicts `HORIZON_MIN` ahead
   - **LSTM**: Uses `LSTM_SEQ_LEN` sequences, trains for `LSTM_EPOCHS`, predicts `HORIZON_MIN` ahead
4. **Anomaly Detection**: Uses `LOOKBACK_HOURS` for features, `CONTAMINATION` for threshold

### Prediction Flow:
1. **Data Fetching**: `START_HOURS_AGO` + `STEP` → Fetch latest data
2. **Forecast Generation**: All models predict `HORIZON_MIN` minutes ahead
3. **Anomaly Detection**: Uses `LOOKBACK_HOURS` window, `CONTAMINATION` rate
4. **Ensemble**: Combines Prophet + ARIMA + LSTM forecasts

### Key Relationships:
- **LSTM Requirements**: `len(data) >= LSTM_SEQ_LEN + HORIZON_MIN`
- **Training Data**: `START_HOURS_AGO` determines total data available
- **Data Density**: `STEP` affects how many data points in `START_HOURS_AGO` window
- **Train/Test Split**: `TRAIN_FRACTION` of `START_HOURS_AGO` data used for training

---

## Recommended Settings

### For Production (Balanced):
```bash
export HORIZON_MIN=15          # 15-minute forecasts
export LOOKBACK_HOURS=24       # 24-hour anomaly window
export CONTAMINATION=0.12      # 12% expected anomalies
export STEP="60s"              # 1-minute resolution
export START_HOURS_AGO=360     # 15 days of history
export LSTM_SEQ_LEN=60         # 60-point sequences
export LSTM_EPOCHS=10          # 10 training epochs
export TRAIN_FRACTION=0.8       # 80/20 train/test split
```

### For High-Frequency Monitoring:
```bash
export HORIZON_MIN=30          # 30-minute forecasts
export STEP="30s"              # 30-second resolution (more data)
export START_HOURS_AGO=168     # 7 days (faster queries)
```

### For Long-Term Analysis:
```bash
export HORIZON_MIN=60          # 1-hour forecasts
export START_HOURS_AGO=720     # 30 days of history
export STEP="60s"              # 1-minute resolution (system handles chunking automatically)
# Note: System automatically chunks 720h queries into 360h pieces
# Each chunk uses 60s step, preserving maximum data resolution
```

---

## See Also

- `../README.md` - General configuration documentation
- `SYSTEM_DOCUMENTATION.md` - Detailed system documentation
- `ANOMALY_DETECTION.md` - Anomaly detection specifics

