# HORIZON_MIN vs horizon_days: Forecast Length Explained

## Overview

The system uses two different parameters to control forecast length, depending on the model type:
- **`HORIZON_MIN`**: Forecast length in **minutes** (default: 15 minutes)
- **`horizon_days`**: Forecast length in **days** (default: 7 days)

---

## Usage by Model Type

### 1. **Host/Pod Ensemble Models**

**Parameter Used**: Hardcoded to **7 days** (not `HORIZON_MIN`)

**Code Reference**:
```python
# Line 7275, 6767: Training and forecast modes
horizon_min=7*24*60  # 7 days = 10,080 minutes
```

**Why?** Host/Pod models predict CPU/Memory usage for the next 7 days, regardless of `HORIZON_MIN` setting.

**Forecast Output**: Predicts next 7 days (10,080 minutes) of CPU/Memory usage.

---

### 2. **Disk Full Models**

**Parameter Used**: `horizon_days` (default: 7 days)

**Code Reference**:
```python
# Line 2333, 6942, 7494
predict_disk_full_days(df_disk, horizon_days=7, ...)
```

**How It's Used**:
- Prophet forecast: `future = m.make_future_dataframe(periods=horizon_days*24*10, freq='6H')`
  - Creates forecast for `horizon_days * 24 * 10` periods at 6-hour intervals
  - For 7 days: `7 * 24 * 10 = 1,680` periods = 7 days of 6-hour intervals
- Linear trend: Calculates days until 90% threshold is reached
- Result: Predicts when disk will reach 90% capacity within `horizon_days`

**Forecast Output**: Days until disk reaches 90% capacity (up to `horizon_days` ahead).

---

### 3. **I/O Network Crisis Models**

**Parameter Used**: `horizon_days` (default: 7 days)

**Code Reference**:
```python
# Line 4845
predict_io_and_network_crisis_with_backtest(horizon_days=7, ...)
```

**How It's Used**:
- Prophet forecast: Similar to Disk models
- Crisis detection: Checks if I/O wait or network bandwidth will exceed thresholds within `horizon_days`
- Result: Predicts crisis events (threshold breaches) within `horizon_days`

**Forecast Output**: Crisis predictions (threshold breaches) within `horizon_days` ahead.

---

### 4. **I/O Network Ensemble Models**

**Parameter Used**: `horizon_days` (default: 7 days), **converted to minutes**

**Code Reference**:
```python
# Line 5629, 5920
predict_io_and_network_ensemble(horizon_days=7, ...)
# Inside function:
horizon_min=horizon_days * 24 * 60  # Convert days to minutes
# Then calls:
build_ensemble_forecast_model(..., horizon_min=horizon_days * 24 * 60)
```

**How It's Used**:
- Converts `horizon_days` to minutes: `horizon_min = horizon_days * 24 * 60`
- For 7 days: `7 * 24 * 60 = 10,080` minutes
- Uses same logic as Host/Pod Ensemble (Prophet + ARIMA + LSTM)
- Result: Predicts I/O wait and network bandwidth for next `horizon_days`

**Forecast Output**: Predicts next `horizon_days` (in minutes) of I/O wait and network bandwidth.

---

## Summary Table

| Model Type | Parameter | Default | Unit | Actual Forecast Length |
|------------|-----------|---------|------|----------------------|
| **Host/Pod Ensemble** | Hardcoded | `7*24*60` | Minutes | 7 days (10,080 minutes) |
| **Disk Full** | `horizon_days` | 7 | Days | 7 days |
| **I/O Network Crisis** | `horizon_days` | 7 | Days | 7 days |
| **I/O Network Ensemble** | `horizon_days` → converted | 7 | Days → Minutes | 7 days (10,080 minutes) |

---

## Why Two Different Parameters?

### **`HORIZON_MIN` (Minutes)**
- **Purpose**: Short-term forecasting (originally designed for 15-minute predictions)
- **Used For**: Fine-grained time-series forecasting
- **Default**: 15 minutes
- **Note**: Currently **NOT used** in main execution (Host/Pod models use hardcoded 7 days)

### **`horizon_days` (Days)**
- **Purpose**: Long-term forecasting (days/weeks ahead)
- **Used For**: Capacity planning, crisis detection, disk full prediction
- **Default**: 7 days
- **Why Days?**: 
  - Disk capacity planning needs day-level granularity
  - Crisis detection needs to look days ahead
  - More intuitive for operational planning

---

## How They Control Forecasts

### **`HORIZON_MIN` Usage** (when used):

```python
# Prophet
future = m.make_future_dataframe(periods=horizon_min, freq='min')
# Creates forecast for next 'horizon_min' minutes

# ARIMA
f_arima = arima.forecast(steps=horizon_min)
# Forecasts next 'horizon_min' data points

# LSTM
Dense(horizon_min)  # Output layer predicts 'horizon_min' values
# Model architecture predicts next 'horizon_min' time steps
```

### **`horizon_days` Usage**:

```python
# Disk Models - Prophet
future = m.make_future_dataframe(periods=horizon_days*24*10, freq='6H')
# Creates forecast for 'horizon_days' at 6-hour intervals

# I/O Network Ensemble - Converted to minutes
horizon_min = horizon_days * 24 * 60  # 7 days = 10,080 minutes
# Then uses same logic as HORIZON_MIN
```

---

## Configuration Impact

### **Changing `HORIZON_MIN`**:
- **Currently**: Has **NO effect** on Host/Pod models (they use hardcoded 7 days)
- **Would affect**: Only if code is modified to use `HORIZON_MIN` instead of hardcoded value
- **Default**: 15 minutes (unused in current implementation)

### **Changing `horizon_days`**:
- **Disk Models**: Changes forecast window for disk capacity predictions
- **I/O Crisis**: Changes how far ahead crisis detection looks
- **I/O Ensemble**: Changes forecast length (converted to minutes)
- **Default**: 7 days

---

## Code References

### Host/Pod Models:
- **Training**: `metrics.py:7275` - `horizon_min=7*24*60`
- **Forecast**: `metrics.py:6767` - `horizon_min=7*24*60`
- **Function**: `build_ensemble_forecast_model()` - `horizon_min` parameter

### Disk Models:
- **Training**: `metrics.py:7494` - `horizon_days=7`
- **Forecast**: `metrics.py:6942` - `horizon_days=7`
- **Function**: `predict_disk_full_days()` - `horizon_days` parameter
- **Prophet**: `metrics.py:2267` - `periods=horizon_days*24*10, freq='6H'`

### I/O Network Crisis:
- **Training**: `metrics.py:7659` - `horizon_days=7`
- **Function**: `predict_io_and_network_crisis_with_backtest()` - `horizon_days` parameter

### I/O Network Ensemble:
- **Training**: `metrics.py:7688` - `horizon_days=7`
- **Function**: `predict_io_and_network_ensemble()` - `horizon_days` parameter
- **Conversion**: `metrics.py:5920` - `horizon_min=horizon_days * 24 * 60`

---

## Key Takeaways

1. **`HORIZON_MIN`** (15 minutes) is **currently unused** in main execution
2. **Host/Pod models** use **hardcoded 7 days** (not configurable via `HORIZON_MIN`)
3. **`horizon_days`** (7 days) is used for:
   - Disk capacity planning
   - I/O crisis detection
   - I/O network ensemble forecasting
4. **I/O Network Ensemble** converts `horizon_days` to minutes internally
5. **All models** predict **7 days ahead** by default (except `HORIZON_MIN` which is unused)

---

## Future Considerations

If you want to make Host/Pod models use `HORIZON_MIN`:
- Change `horizon_min=7*24*60` to `horizon_min=HORIZON_MIN` in:
  - Line 7275 (training mode)
  - Line 6767 (forecast mode)
  - Line 7373 (standalone models)
  - Line 7312 (K8s host model)

This would allow configuring Host/Pod forecast length via `HORIZON_MIN` environment variable.


