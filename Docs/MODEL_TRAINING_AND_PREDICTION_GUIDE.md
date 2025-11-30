# Complete Model Training and Prediction Guide

This document provides a comprehensive, step-by-step explanation of all models in the Metrics AI system, how they're trained, how they forecast, and how they're backtested.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Model Types](#model-types)
3. [Host/Pod Ensemble Models](#hostpod-ensemble-models)
4. [Disk Full Prediction Models](#disk-full-prediction-models)
5. [I/O Network Crisis Models](#io-network-crisis-models)
6. [I/O Network Ensemble Models](#io-network-ensemble-models)
7. [Classification/Anomaly Detection Models](#classificationanomaly-detection-models)
8. [Configuration Variables Impact](#configuration-variables-impact)
9. [Execution Flow Diagrams](#execution-flow-diagrams)

---

## System Overview

The Metrics AI system trains **5 main model types**:

1. **Host/Pod Ensemble Models** - CPU/Memory forecasting (Prophet + ARIMA + LSTM)
2. **Disk Full Prediction Models** - Disk usage forecasting (Linear + Prophet)
3. **I/O Network Crisis Models** - Crisis detection (Prophet-based)
4. **I/O Network Ensemble Models** - Full ensemble forecasting (Prophet + ARIMA + LSTM)
5. **Classification/Anomaly Models** - Anomaly detection (IsolationForest)

Each model type has distinct training, prediction, and backtesting flows.

---

## Model Types

### Model Type Summary

| Model Type | Purpose | Algorithms | Scope | Storage |
|------------|---------|------------|-------|---------|
| **Host/Pod Ensemble** | CPU/Memory forecasting | Prophet + ARIMA + LSTM | Per-cluster or standalone | `k8s_cluster_{id}_forecast.pkl` |
| **Disk Full** | Disk usage prediction | Linear trend + Prophet | Per node/mountpoint | Manifest: `disk_full_models.pkl` |
| **I/O Network Crisis** | Crisis detection | Prophet | Per node/signal | Manifest: `io_net_models.pkl` |
| **I/O Network Ensemble** | Full I/O/Net forecasting | Prophet + ARIMA + LSTM | Per node/signal | Manifest: `io_net_models.pkl` |
| **Classification** | Anomaly detection | IsolationForest | Per-cluster | `isolation_forest_anomaly.pkl` |

---

## Host/Pod Ensemble Models

**Function**: `build_ensemble_forecast_model()` (line 2956)  
**Wrapper**: `train_or_load_ensemble()` (line 1490)

### Purpose
Forecasts CPU and Memory usage for Kubernetes clusters and standalone nodes using an ensemble of Prophet, ARIMA, and LSTM models.

### Training Flow (Step-by-Step)

#### Step 1: Data Preparation
```python
# Input: df_cpu, df_mem (from Prometheus queries)
# Variables used: START_HOURS_AGO, STEP

1. Aggregate CPU data by timestamp (mean across all nodes in cluster)
   cpu_agg = df_cpu.groupby('timestamp')['value'].mean()

2. Aggregate Memory data by timestamp (if available)
   mem_agg = df_mem.groupby('timestamp')['value'].mean()

3. Combine CPU + Memory into single dataframe
   - If memory available: target = 'mem'
   - If no memory: target = 'cpu'

4. Create Prophet format: ['ds' (timestamp), 'y' (target value)]
   pdf = cpu_agg[['timestamp', target]].rename(columns={'timestamp':'ds', target:'y'})

5. Add regressors (hour of day, weekend flag)
   pdf['hour'] = pdf['ds'].dt.hour
   pdf['is_weekend'] = (pdf['ds'].dt.dayofweek >= 5).astype(int)
```

**Configuration Impact**:
- `START_HOURS_AGO`: Determines how much historical data is fetched
- `STEP`: Determines data point frequency (60s = 1 point per minute)

#### Step 2: Train/Test Split
```python
# Variable used: TRAIN_FRACTION (default: 0.8)

1. Calculate split index (time-ordered, not random!)
   split_idx = int(len(pdf) * TRAIN_FRACTION)
   # Example: 1000 points * 0.8 = 800 points for training

2. Split chronologically:
   train = pdf[pdf['ds'] <= test_cutoff]  # First 80% (chronologically)
   test_ts = pdf[pdf['ds'] > test_cutoff] # Last 20% (chronologically)

3. Store split metadata:
   split_info = {
       'train_fraction': TRAIN_FRACTION,
       'train_points': len(train),
       'test_points': len(test_ts),
       'train_start': train['ds'].min(),
       'train_end': train['ds'].max(),
       'test_start': test_ts.index.min(),
       'test_end': test_ts.index.max()
   }
```

**Configuration Impact**:
- `TRAIN_FRACTION`: Controls train/test split ratio
  - Higher (0.9) = More training data, less test data
  - Lower (0.7) = Less training data, more test data

#### Step 3: Prophet Training
```python
# Variables used: HORIZON_MIN

1. Initialize Prophet model with hyperparameters:
   m = Prophet(
       daily_seasonality=True,      # Daily patterns
       weekly_seasonality=True,     # Weekly patterns
       changepoint_prior_scale=0.05 # Trend flexibility
   )

2. Add regressors (external variables):
   m.add_regressor('hour')        # Hour of day
   m.add_regressor('is_weekend')  # Weekend flag

3. Fit on training data:
   m.fit(train)

4. Generate future dataframe:
   future = m.make_future_dataframe(periods=HORIZON_MIN, freq='min')
   # Creates HORIZON_MIN future data points

5. Add regressors to future:
   future['hour'] = future['ds'].dt.hour
   future['is_weekend'] = (future['ds'].dt.dayofweek >= 5).astype(int)

6. Generate forecast:
   f_prophet = m.predict(future)['yhat']
   # Returns forecast for training period + HORIZON_MIN future points

7. Save Prophet hyperparameters:
   prophet_params = {
       'daily_seasonality': True,
       'weekly_seasonality': True,
       'changepoint_prior_scale': 0.05
   }
   # Saved to: {model_path}_prophet_params.pkl
```

**Configuration Impact**:
- `HORIZON_MIN`: Determines how many future points Prophet generates
  - Default: 15 minutes ahead
  - Larger = longer forecast horizon

#### Step 4: ARIMA Training
```python
# Variables used: HORIZON_MIN

1. Convert to time series:
   ts = pd.Series(train.set_index('ds')['y'])

2. Set frequency (if not set):
   ts.index.freq = pd.infer_freq(ts.index)

3. Fit ARIMA model:
   arima = ARIMA(ts, order=(2,1,0)).fit()
   # order=(2,1,0) = AR(2), I(1), MA(0)
   # This is fixed - not configurable

4. Generate forecast:
   f_arima = arima.forecast(steps=HORIZON_MIN)
   # Forecasts HORIZON_MIN steps ahead

5. Create forecast index:
   f_arima.index = pd.date_range(
       start=ts.index[-1] + pd.Timedelta(minutes=1),
       periods=HORIZON_MIN,
       freq='min'
   )

6. Save ARIMA model:
   # Saved to: {model_path}_arima.pkl
```

**Configuration Impact**:
- `HORIZON_MIN`: Number of steps ARIMA forecasts ahead

#### Step 5: LSTM Training
```python
# Variables used: LSTM_SEQ_LEN, LSTM_EPOCHS, HORIZON_MIN

1. Check if LSTM is available and enough data:
   if LSTM_AVAILABLE and len(ts) > LSTM_SEQ_LEN + HORIZON_MIN:
       # Need at least LSTM_SEQ_LEN + HORIZON_MIN data points

2. Scale data (normalize to 0-1):
   scaler = MinMaxScaler()
   scaled = scaler.fit_transform(ts.values.reshape(-1, 1))

3. Create sequences (sliding window):
   X, y = [], []
   for i in range(LSTM_SEQ_LEN, len(scaled) - HORIZON_MIN):
       # Input: last LSTM_SEQ_LEN points
       X.append(scaled[i-LSTM_SEQ_LEN:i])
       # Output: next HORIZON_MIN points
       y.append(scaled[i:i+HORIZON_MIN])
   
   X, y = np.array(X), np.array(y)
   # X shape: (samples, LSTM_SEQ_LEN, 1)
   # y shape: (samples, HORIZON_MIN)

4. Build LSTM model:
   model = Sequential([
       LSTM(50, return_sequences=True, input_shape=(LSTM_SEQ_LEN, 1)),
       LSTM(50),
       Dense(HORIZON_MIN)  # Output layer = HORIZON_MIN neurons
   ])

5. Compile model:
   model.compile(optimizer='adam', loss='mse')

6. Train model:
   model.fit(
       X, y,
       epochs=LSTM_EPOCHS,      # Number of training passes
       batch_size=32,
       verbose=0,
       callbacks=[EarlyStopping(patience=3)]  # Stop if no improvement
   )

7. Generate forecast:
   last_seq = scaled[-LSTM_SEQ_LEN:].reshape(1, LSTM_SEQ_LEN, 1)
   lstm_pred = model.predict(last_seq, verbose=0)
   f_lstm = pd.Series(
       scaler.inverse_transform(lstm_pred)[0],
       index=f_arima.index
   )

8. Save LSTM model and scaler:
   # Saved to: lstm_model.pkl
   joblib.dump({'model': model, 'scaler': scaler}, LSTM_MODEL_PATH)
```

**Configuration Impact**:
- `LSTM_SEQ_LEN`: How many past points LSTM looks at (input sequence length)
  - Larger = longer memory, requires more data
- `LSTM_EPOCHS`: Number of training iterations
  - More = better fit (but risk of overfitting)
- `HORIZON_MIN`: Output layer size (how many future points to predict)

#### Step 6: Ensemble Creation
```python
# Combine all three models

1. Extract forecast values for horizon:
   prophet_tail = f_prophet.tail(HORIZON_MIN)
   f_arima = f_arima  # Already HORIZON_MIN length
   f_lstm = f_lstm    # Already HORIZON_MIN length

2. Average the three forecasts:
   ensemble = (prophet_tail + f_arima + f_lstm) / 3
   # Simple average of all three models

3. Save ensemble result:
   # Saved to: {model_path}_forecast.pkl
   # Contains: (prophet_model, arima_model, ensemble_forecast, metrics)
```

### Prediction Flow (Step-by-Step)

#### Forecast Mode (Minimal Updates)
```python
# Function: generate_forecast_from_cached_model() (line 1121)

1. Load cached models:
   cached = load_cached_ensemble(model_path)
   # Loads: Prophet model, ARIMA model, LSTM model, scaler

2. Fetch latest data:
   # Uses: START_HOURS_AGO, STEP
   df_cpu = fetch_and_preprocess_data(q_host_cpu)
   df_mem = fetch_and_preprocess_data(q_host_mem)

3. Prepare latest time series:
   ts = latest_data.set_index('ds')['y']

4. Prophet Minimal Update:
   # Load saved hyperparameters
   prophet_params = load_prophet_params()
   
   # Create new Prophet with same hyperparameters
   m = Prophet(**prophet_params)
   m.add_regressor('hour')
   m.add_regressor('is_weekend')
   
   # Fit on last 7 days only (minimal update)
   recent_data = ts.last('7D')  # Last 7 days
   m.fit(recent_data)
   
   # Generate forecast
   future = m.make_future_dataframe(periods=HORIZON_MIN, freq='min')
   f_prophet = m.predict(future)['yhat'].tail(HORIZON_MIN)

5. ARIMA Minimal Update:
   # Load saved ARIMA order
   arima_order = (2,1,0)  # Fixed
   
   # Fit on latest data
   arima = ARIMA(ts, order=arima_order).fit()
   f_arima = arima.forecast(steps=HORIZON_MIN)

6. LSTM Minimal Update:
   if len(ts) >= LSTM_SEQ_LEN + HORIZON_MIN:
       # Load cached model and scaler
       cached_lstm = load_lstm_model()
       
       # Fine-tune on last 2 days (2 epochs only)
       recent_scaled = scaler.transform(ts.last('2D').values.reshape(-1, 1))
       
       # Fine-tune model
       for epoch in range(2):  # Only 2 epochs
           model.fit(X_fine, y_fine, epochs=1, verbose=0)
       
       # Generate forecast
       last_seq = recent_scaled[-LSTM_SEQ_LEN:].reshape(1, LSTM_SEQ_LEN, 1)
       f_lstm = model.predict(last_seq)

7. Create ensemble:
   ensemble = (f_prophet + f_arima + f_lstm) / 3

8. Save updated models:
   # Models are updated and saved back to disk
```

**Configuration Impact**:
- `HORIZON_MIN`: Forecast length
- `START_HOURS_AGO`: How much data to fetch for updates
- `STEP`: Data resolution

### Backtesting Flow (Step-by-Step)

```python
# Performed during training or with --show-backtest flag

1. Use test set (last 20% of data):
   test_ts = pdf[pdf['ds'] > test_cutoff]['y']

2. Prophet Backtest:
   # Train Prophet on training set only
   mb = Prophet(...)
   mb.fit(train)
   
   # Predict on test period
   fut_b = mb.make_future_dataframe(periods=len(test_ts), freq='min')
   p_back = mb.predict(fut_b).reindex(test_ts.index)['yhat']

3. ARIMA Backtest:
   # Train ARIMA on training set only
   a_model = ARIMA(train_ts, order=(2,1,0)).fit()
   a_pred = a_model.forecast(steps=len(test_ts))

4. LSTM Backtest:
   # Train LSTM on training set only
   # (Same as training flow, but only on train data)
   # Predict on test set

5. Ensemble Backtest:
   ens_pred = (p_back + a_pred + l_back) / 3

6. Calculate Metrics:
   metrics = {
       'mae_prophet': mean_absolute_error(test_ts, p_back),
       'mae_arima': mean_absolute_error(test_ts, a_pred),
       'mae_lstm': mean_absolute_error(test_ts, l_back),
       'mae_ensemble': mean_absolute_error(test_ts, ens_pred),
       'rmse_prophet': sqrt(mean_squared_error(test_ts, p_back)),
       'rmse_arima': sqrt(mean_squared_error(test_ts, a_pred)),
       'rmse_lstm': sqrt(mean_squared_error(test_ts, l_back)),
       'rmse_ensemble': sqrt(mean_squared_error(test_ts, ens_pred))
   }

7. Generate backtest plot:
   # Shows: historical data, train/test split, predictions, actuals
   # Saved to: {model_path}_backtest.png
```

**Configuration Impact**:
- `TRAIN_FRACTION`: Determines test set size
- `HORIZON_MIN`: Not directly used (backtest uses full test period)

---

## Disk Full Prediction Models

**Function**: `predict_disk_full_days()` (line 2333)  
**Worker**: `_process_single_disk()` (line 2111)

### Purpose
Predicts when disk usage will reach 90% threshold for each node/mountpoint combination using hybrid linear trend + Prophet model.

### Training Flow (Step-by-Step)

#### Step 1: Data Preparation
```python
# Input: df_disk (from Prometheus disk usage query)
# Variables used: START_HOURS_AGO, STEP

1. Group by (entity, mountpoint):
   disk_groups = df_disk.groupby(['entity', mount_col])
   # Each group = one disk (one node + one mountpoint)

2. For each disk:
   ts = group.set_index('timestamp')['value'].sort_index()
   # Time series of disk usage percentage

3. Check minimum data requirement:
   if len(ts) < 50:
       skip this disk  # Need at least 50 data points
```

**Configuration Impact**:
- `START_HOURS_AGO`: How much historical data to fetch
- `STEP`: Data point frequency

#### Step 2: Train/Test Split
```python
# Variable used: TRAIN_FRACTION

1. Calculate split:
   split_idx = int(len(ts) * TRAIN_FRACTION)
   train_ts = ts.iloc[:split_idx]  # First 80%
   test_ts = ts.iloc[split_idx:]   # Last 20%
```

#### Step 3: Linear Trend Model
```python
# Simple linear regression to estimate trend

1. Fit linear model:
   from sklearn.linear_model import LinearRegression
   X = np.arange(len(train_ts)).reshape(-1, 1)
   y = train_ts.values
   linear_model = LinearRegression().fit(X, y)

2. Calculate trend:
   trend = linear_model.coef_[0]  # Slope (usage increase per time unit)

3. Predict days to 90%:
   current_usage = train_ts.iloc[-1]
   usage_to_90 = 90.0 - current_usage
   days_to_90_linear = (usage_to_90 / trend) / (24 * 60)  # Convert to days
   # If trend is negative or very small, set to 9999 (never)
```

#### Step 4: Prophet Model
```python
# Variable used: horizon_days (default: 7 days)

1. Prepare Prophet format:
   pdf = pd.DataFrame({
       'ds': train_ts.index,
       'y': train_ts.values
   })

2. Fit Prophet:
   m = Prophet(
       daily_seasonality=True,
       weekly_seasonality=True,
       changepoint_prior_scale=0.05
   )
   m.fit(pdf)

3. Generate forecast:
   future = m.make_future_dataframe(periods=horizon_days * 24 * 60, freq='min')
   # Forecast horizon_days ahead (in minutes)
   f_prophet = m.predict(future)['yhat']

4. Find when forecast reaches 90%:
   # Search through forecast to find first point >= 90%
   days_to_90_prophet = find_first_exceedance(f_prophet, threshold=90.0)
   # If never reaches 90%, set to 9999
```

**Configuration Impact**:
- `horizon_days`: How far ahead to forecast (default: 7 days)
- `TRAIN_FRACTION`: Train/test split

#### Step 5: Ensemble ETA
```python
# Combine linear and Prophet predictions

1. Calculate ensemble ETA:
   ensemble_eta = min(linear_eta, prophet_eta)
   # Use the more conservative (earlier) prediction

2. Determine alert severity:
   if ensemble_eta < 3:
       alert = 'CRITICAL'
   elif ensemble_eta < 7:
       alert = 'WARNING'
   elif ensemble_eta < 30:
       alert = 'SOON'
   else:
       alert = 'OK'
```

#### Step 6: Save Model
```python
# Saved to manifest: disk_full_models.pkl
manifest[key] = {
    'model': {
        'prophet': prophet_model,
        'linear_trend': trend,
        'last_training_time': timestamp,
        'current_usage': current_usage
    }
}
```

### Prediction Flow (Step-by-Step)

```python
# Forecast mode: Minimal updates

1. Load cached model from manifest:
   cached = manifest.get(key)

2. Fetch latest data:
   ts = latest_disk_data

3. Minimal Prophet update:
   # Fit on last 7 days only
   recent = ts.last('7D')
   m.fit(recent)
   f_prophet = m.predict(future)['yhat']

4. Update linear trend:
   # Recalculate trend from recent data
   trend = calculate_trend(recent)

5. Recalculate ETA:
   ensemble_eta = min(linear_eta, prophet_eta)

6. Update manifest:
   manifest[key] = updated_model
```

### Backtesting Flow (Step-by-Step)

```python
# Only performed during training (when retraining)

1. Use test set:
   test_ts = ts.iloc[split_idx:]

2. Train models on training set:
   # Same as training flow

3. Predict on test set:
   # Generate forecasts for test period
   # Find when predictions would have reached 90%

4. Calculate metrics:
   # Compare predicted vs actual usage in test period
   # Calculate MAE, RMSE for usage predictions
```

---

## I/O Network Crisis Models

**Function**: `predict_io_and_network_crisis_with_backtest()` (line 4845)  
**Worker**: `_process_single_node_io_crisis()` (line 4970)

### Purpose
Detects when I/O wait or network bandwidth will reach crisis thresholds using Prophet-based forecasting.

### Training Flow (Step-by-Step)

#### Step 1: Data Preparation
```python
# Input: df (I/O wait or network bandwidth data)
# Variables used: START_HOURS_AGO, STEP

1. Group by instance (node):
   node_groups = df.groupby('instance')
   # Each group = one node's signal data

2. For each node:
   ts = group.set_index('timestamp')['value'].sort_index()
   # Time series of signal values
```

#### Step 2: Train/Test Split
```python
# Variable used: TRAIN_FRACTION

split_idx = int(len(ts) * TRAIN_FRACTION)
train_ts = ts.iloc[:split_idx]
test_ts = ts.iloc[split_idx:]
```

#### Step 3: Prophet Training
```python
# Variable used: horizon_days (default: 7 days)

1. Prepare Prophet format:
   pdf = pd.DataFrame({
       'ds': train_ts.index,
       'y': train_ts.values
   })

2. Fit Prophet:
   m = Prophet(
       daily_seasonality=True,
       weekly_seasonality=True,
       changepoint_prior_scale=0.05,
       n_changepoints=10  # Optimized for speed
   )
   m.fit(pdf)

3. Generate forecast:
   future = m.make_future_dataframe(
       periods=horizon_days * 24 * 60,
       freq='min'
   )
   f_prophet = m.predict(future)['yhat']

4. Check for crisis (threshold crossing):
   # Find when forecast exceeds threshold
   crisis_detected = check_threshold_crossing(f_prophet, threshold)
   days_to_crisis = calculate_days_to_crisis(f_prophet, threshold)
```

#### Step 4: Save Model
```python
# Saved to manifest: io_net_models.pkl
manifest[key] = {
    'model': prophet_model,
    'threshold': threshold,
    'last_training_time': timestamp
}
```

### Prediction Flow (Step-by-Step)

```python
# Forecast mode: Minimal updates

1. Load cached model:
   cached = manifest.get(key)

2. Fetch latest data:
   ts = latest_signal_data

3. Minimal Prophet update:
   # Fit on last 7 days only
   recent = ts.last('7D')
   m.fit(recent)
   f_prophet = m.predict(future)['yhat']

4. Check for crisis:
   crisis_detected, days_to_crisis = check_crisis(f_prophet, threshold)

5. Update manifest:
   manifest[key] = updated_model
```

### Backtesting Flow (Step-by-Step)

```python
# Performed during training

1. Use test set:
   test_ts = ts.iloc[split_idx:]

2. Train on training set:
   # Same as training flow

3. Predict on test set:
   # Generate forecasts for test period
   # Check if crisis would have been detected

4. Calculate metrics:
   # True positives, false positives, etc.
```

---

## I/O Network Ensemble Models

**Function**: `predict_io_and_network_ensemble()` (line 5629)  
**Worker**: `_process_single_node_io_ensemble()` (line 5480)

### Purpose
Full ensemble forecasting (Prophet + ARIMA + LSTM) for I/O wait and network bandwidth, with anomaly detection.

### Training Flow (Step-by-Step)

#### Step 1-2: Same as I/O Network Crisis
```python
# Data preparation and train/test split
# Uses: START_HOURS_AGO, STEP, TRAIN_FRACTION
```

#### Step 3: Ensemble Training
```python
# Uses: HORIZON_MIN, LSTM_SEQ_LEN, LSTM_EPOCHS

1. Prophet Training:
   # Same as Host/Pod ensemble (Step 3)

2. ARIMA Training:
   # Same as Host/Pod ensemble (Step 4)

3. LSTM Training:
   # Same as Host/Pod ensemble (Step 5)

4. Create Ensemble:
   ensemble = (prophet + arima + lstm) / 3
```

#### Step 4: Anomaly Detection
```python
# Detects anomalies in current values

1. Calculate current value:
   current = ts.iloc[-1]

2. Calculate forecast value:
   forecast = ensemble.iloc[0]  # Next time step

3. Calculate deviation:
   deviation_pct = abs(current - forecast) / forecast * 100

4. Apply dual threshold:
   if deviation_pct > absolute_threshold OR deviation_pct > percentage_threshold:
       anomaly_detected = True
       severity = determine_severity(deviation_pct)
```

#### Step 5: Crisis Detection
```python
# Check if forecast indicates crisis

1. Check forecast against threshold:
   crisis_detected = any(ensemble > threshold)

2. Calculate days to crisis:
   days_to_crisis = find_first_exceedance(ensemble, threshold)
```

#### Step 6: Save Model
```python
# Saved to manifest: io_net_models.pkl
manifest[key] = {
    'model': {
        'prophet': prophet_model,
        'arima': arima_model,
        'lstm': lstm_model,
        'scaler': scaler
    },
    'threshold': threshold,
    'last_training_time': timestamp
}
```

### Prediction Flow (Step-by-Step)

```python
# Forecast mode: Minimal updates

1. Load cached models:
   cached = manifest.get(key)

2. Fetch latest data:
   ts = latest_signal_data

3. Minimal updates (same as Host/Pod ensemble):
   # Prophet: last 7 days
   # ARIMA: latest data
   # LSTM: last 2 days, 2 epochs

4. Generate ensemble forecast:
   ensemble = (prophet + arima + lstm) / 3

5. Detect anomalies:
   # Compare current vs forecast

6. Detect crisis:
   # Check forecast against threshold

7. Update manifest:
   manifest[key] = updated_models
```

### Backtesting Flow (Step-by-Step)

```python
# Same as Host/Pod ensemble backtesting
# Calculates MAE, RMSE for Prophet, ARIMA, LSTM, and ensemble
```

---

## Classification/Anomaly Detection Models

**Function**: `classification_model()` (line 4398)

### Purpose
Detects anomalous nodes using IsolationForest, comparing nodes within the same Kubernetes cluster.

### Training Flow (Step-by-Step)

#### Step 1: Feature Extraction
```python
# Variables used: LOOKBACK_HOURS

1. Calculate lookback window:
   now = pd.Timestamp.now()
   start = now - pd.Timedelta(hours=LOOKBACK_HOURS)

2. Extract features for each node:
   for each node:
       host_cpu = mean(df_host_cpu[node][start:now])
       host_mem = mean(df_host_mem[node][start:now])
       pod_cpu = mean(df_pod_cpu[node][start:now])
       pod_mem = mean(df_pod_mem[node][start:now])
   
   features = [host_cpu, host_mem, pod_cpu, pod_mem]

3. Create feature matrix:
   feats = pd.DataFrame({
       'entity': nodes,
       'host_cpu': host_cpu_values,
       'host_mem': host_mem_values,
       'pod_cpu': pod_cpu_values,
       'pod_mem': pod_mem_values,
       'cluster_id': cluster_ids
   })
```

**Configuration Impact**:
- `LOOKBACK_HOURS`: Time window for feature extraction
  - Larger = longer-term patterns
  - Smaller = recent patterns only

#### Step 2: Cluster Grouping
```python
# Group nodes by Kubernetes cluster

1. For each cluster:
   cluster_feats = feats[feats['cluster_id'] == cluster_id]

2. Skip if:
   - Standalone nodes (no cluster)
   - Unknown cluster (legacy)
   - Single node cluster (need at least 2 for comparison)
```

#### Step 3: IsolationForest Training
```python
# Variable used: CONTAMINATION

1. For each cluster:
   # Normalize features
   scaler = StandardScaler()
   X = scaler.fit_transform(cluster_feats[['host_cpu','host_mem','pod_cpu','pod_mem']])

2. Train IsolationForest:
   iso = IsolationForest(
       contamination=CONTAMINATION,  # Expected fraction of anomalies
       random_state=42
   )
   cluster_labels = iso.fit_predict(X)
   # Returns: -1 (anomalous) or 1 (normal)

3. Store results:
   feats.loc[cluster_feats.index, 'anomaly'] = cluster_labels
```

**Configuration Impact**:
- `CONTAMINATION`: Expected fraction of anomalous nodes
  - 0.12 = 12% expected to be anomalous
  - Higher = more nodes flagged
  - Lower = fewer nodes flagged

#### Step 4: Save Model
```python
# Saved to: isolation_forest_anomaly.pkl
# Per-cluster models are stored
```

### Prediction Flow (Step-by-Step)

```python
# Same as training (no separate prediction mode)

1. Extract features (last LOOKBACK_HOURS):
   # Same as training Step 1

2. Group by cluster:
   # Same as training Step 2

3. Predict anomalies:
   # Use trained IsolationForest models
   labels = iso.predict(X)

4. Return anomalous nodes:
   anomalous = feats[feats['anomaly'] == -1]
```

### Backtesting Flow

```python
# No explicit backtesting for anomaly detection
# Model is trained and used immediately
# Performance evaluated by reviewing flagged nodes
```

---

## Configuration Variables Impact

### Summary Table

| Variable | Host/Pod Ensemble | Disk Models | I/O Network Crisis | I/O Network Ensemble | Classification |
|----------|------------------|-------------|-------------------|---------------------|---------------|
| `HORIZON_MIN` | ✅ Forecast length | ❌ | ❌ | ✅ Forecast length | ❌ |
| `LOOKBACK_HOURS` | ❌ | ❌ | ❌ | ❌ | ✅ Feature window |
| `CONTAMINATION` | ❌ | ❌ | ❌ | ❌ | ✅ Anomaly rate |
| `STEP` | ✅ Data resolution | ✅ Data resolution | ✅ Data resolution | ✅ Data resolution | ✅ Data resolution |
| `START_HOURS_AGO` | ✅ Data range | ✅ Data range | ✅ Data range | ✅ Data range | ✅ Data range |
| `LSTM_SEQ_LEN` | ✅ Sequence length | ❌ | ❌ | ✅ Sequence length | ❌ |
| `LSTM_EPOCHS` | ✅ Training epochs | ❌ | ❌ | ✅ Training epochs | ❌ |
| `TRAIN_FRACTION` | ✅ Train/test split | ✅ Train/test split | ✅ Train/test split | ✅ Train/test split | ❌ |

### Detailed Impact

#### `HORIZON_MIN` (15 minutes)
- **Host/Pod Ensemble**: Forecast length for all three models
- **I/O Network Ensemble**: Forecast length for all three models
- **Disk Models**: Not used (uses `horizon_days` instead)
- **I/O Network Crisis**: Not used (uses `horizon_days` instead)

#### `LOOKBACK_HOURS` (24 hours)
- **Classification**: Time window for feature extraction
- **Other models**: Not used

#### `CONTAMINATION` (0.12)
- **Classification**: Expected fraction of anomalous nodes
- **Other models**: Not used

#### `STEP` ("60s")
- **All models**: Query resolution for Prometheus data
- Affects data density and query performance

#### `START_HOURS_AGO` (360 hours)
- **All models**: Historical data range
- More data = better training (but slower)

#### `LSTM_SEQ_LEN` (60)
- **Host/Pod Ensemble**: Input sequence length
- **I/O Network Ensemble**: Input sequence length
- **Other models**: Not used

#### `LSTM_EPOCHS` (10)
- **Host/Pod Ensemble**: Training iterations
- **I/O Network Ensemble**: Training iterations
- **Other models**: Not used

#### `TRAIN_FRACTION` (0.8)
- **All forecasting models**: Train/test split ratio
- **Classification**: Not used (no explicit train/test split)

---

## Execution Flow Diagrams

### Training Mode (`--training`)

```
1. Fetch Data
   ├─ START_HOURS_AGO: Determine time range
   ├─ STEP: Determine resolution
   └─ Queries: Host CPU, Host Mem, Pod CPU, Pod Mem, Disk, I/O, Network

2. Cluster Identification
   ├─ LOOKBACK_HOURS: Feature extraction window
   └─ Group nodes by cluster

3. Host/Pod Ensemble Training
   ├─ TRAIN_FRACTION: Split data
   ├─ HORIZON_MIN: Forecast length
   ├─ LSTM_SEQ_LEN: Sequence length
   ├─ LSTM_EPOCHS: Training epochs
   └─ Train: Prophet, ARIMA, LSTM → Ensemble

4. Disk Models Training
   ├─ TRAIN_FRACTION: Split data
   └─ Train: Linear + Prophet per disk

5. I/O Network Crisis Training
   ├─ TRAIN_FRACTION: Split data
   └─ Train: Prophet per node/signal

6. I/O Network Ensemble Training
   ├─ TRAIN_FRACTION: Split data
   ├─ HORIZON_MIN: Forecast length
   ├─ LSTM_SEQ_LEN: Sequence length
   ├─ LSTM_EPOCHS: Training epochs
   └─ Train: Prophet, ARIMA, LSTM → Ensemble

7. Classification Training
   ├─ LOOKBACK_HOURS: Feature window
   ├─ CONTAMINATION: Anomaly rate
   └─ Train: IsolationForest per cluster

8. Backtesting
   ├─ Use test set (20% of data)
   └─ Calculate: MAE, RMSE for all models

9. Save Models
   └─ Save all models and manifests
```

### Forecast Mode (`--forecast`)

```
1. Fetch Latest Data
   ├─ START_HOURS_AGO: Determine time range
   └─ STEP: Determine resolution

2. Load Cached Models
   └─ Load all saved models

3. Minimal Updates
   ├─ Host/Pod: Prophet (7 days), ARIMA (latest), LSTM (2 days, 2 epochs)
   ├─ Disk: Prophet (7 days), Linear (recalculate)
   ├─ I/O Crisis: Prophet (7 days)
   └─ I/O Ensemble: Prophet (7 days), ARIMA (latest), LSTM (2 days, 2 epochs)

4. Generate Forecasts
   ├─ HORIZON_MIN: Forecast length (Host/Pod, I/O Ensemble)
   └─ horizon_days: Forecast length (Disk, I/O Crisis)

5. Detect Anomalies
   ├─ Classification: LOOKBACK_HOURS, CONTAMINATION
   └─ I/O Ensemble: Statistical deviation

6. Detect Crises
   └─ Check forecasts against thresholds

7. Save Updated Models
   └─ Update manifests and model files
```

---

## Key Takeaways

1. **All models use time-ordered train/test splits** (not random) - critical for time series
2. **Minimal updates in forecast mode** - only recent data used to update models
3. **Ensemble models** combine multiple algorithms for robustness
4. **Configuration variables** have specific impacts on each model type
5. **Manifest-based storage** allows efficient management of many models
6. **Parallel processing** speeds up training/prediction for large deployments

---

## See Also

- `CONFIGURATION_VARIABLES.md` - Detailed variable explanations
- `../README.md` - General documentation
- `SYSTEM_DOCUMENTATION.md` - System architecture
- `ANOMALY_DETECTION.md` - Anomaly detection specifics

