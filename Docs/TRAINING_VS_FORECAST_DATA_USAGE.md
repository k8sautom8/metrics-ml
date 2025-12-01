# Training vs Forecast Mode: Data Usage Explained

## Overview

This document explains how `START_HOURS_AGO` is used in training vs forecast modes, and how models leverage historical data for predictions.

---

## Key Points

### 1. **All Models Use `START_HOURS_AGO` for Data Fetching**

Both training and forecast modes fetch data using `START_HOURS_AGO` (default: 360 hours = 15 days):

```python
# Both modes use the same data fetching function
df_cpu = fetch_and_preprocess_data(q_host_cpu)  # Uses START_HOURS_AGO
```

**Why?** Even in forecast mode, we need historical context for:
- Cluster identification
- Entity canonicalization
- Feature extraction
- Model updates

### 2. **Automatic Query Optimization**

The system automatically optimizes queries to maximize data while avoiding VictoriaMetrics limits:

- **Primary Strategy**: Attempts full time range with base `STEP` (60s) for maximum data resolution
- **Query Chunking**: If query fails (422 error), automatically splits into 360-hour chunks
  - Each chunk uses 60s step (preserves maximum resolution)
  - Chunks are combined and deduplicated
  - Result: All data preserved with maximum resolution
- **Adaptive Step Fallback**: Only used if chunking fails (rare)
  - Calculates optimal step size based on time range
  - Targets ~15,000 data points per series

**Example with `START_HOURS_AGO=720` (30 days)**:
1. Attempts single query: 720h with 60s step → fails (422 error)
2. Automatically splits into 2 chunks: 360h each, both with 60s step
3. Combines chunks: ~43,200 points per series (all data preserved)
4. Result: Same data quality as 360h, but for 720h range

---

## Training Mode (`--training`)

### Data Flow:
1. **Fetch Data**: Uses `START_HOURS_AGO` to fetch full historical data (360 hours)
2. **Train Models**: Trains on **full** `START_HOURS_AGO` data
3. **Save Models**: Saves trained models with learned patterns

### Example:
- `START_HOURS_AGO = 360` hours (15 days)
- Fetches 15 days of historical data with 60s step
- Single query succeeds → ~21,600 points per series
- Trains Prophet, ARIMA, LSTM on all 15 days
- Models learn daily and weekly patterns
- Saves models to disk

### Example with Large Time Range:
- `START_HOURS_AGO = 720` hours (30 days)
- Attempts single query with 60s step → fails (422 error)
- Automatically splits into 2 chunks of 360h each
- Each chunk uses 60s step → ~21,600 points per series per chunk
- Combines chunks → ~43,200 points per series total
- Trains Prophet, ARIMA, LSTM on all 30 days
- Models learn daily, weekly, and monthly patterns (monthly seasonality enabled)
- Saves models to disk

---

## Forecast Mode (`--forecast`)

### Data Flow:
1. **Fetch Data**: Uses `START_HOURS_AGO` to fetch full historical data (360 hours)
2. **Load Cached Models**: Loads pre-trained models (trained on full `START_HOURS_AGO` data)
3. **Minimal Updates**: Updates models with recent data only (not full retraining)
4. **Generate Forecasts**: Uses updated models to predict future

### Model-Specific Minimal Updates:

#### **Host/Pod Ensemble Models**:

| Model | Data Used for Minimal Update | Why |
|-------|------------------------------|-----|
| **Prophet** | Last 7 days only | Preserves learned seasonality/patterns, incorporates recent trends |
| **ARIMA** | Full `START_HOURS_AGO` data | ARIMA needs full series to maintain stationarity and patterns |
| **LSTM** | Last 2 days, 2 epochs | Fine-tunes learned patterns with recent changes |

**Code Reference**:
- Prophet: `recent_data = pdf.tail(min(len(pdf), 7*24*60))` (line 1231)
- ARIMA: `arima = ARIMA(ts, order=cached_order).fit()` (line 1274) - uses full `ts`
- LSTM: Fine-tunes on last 2 days (line 1320-1340)

#### **Disk Models**:

| Model | Data Used for Minimal Update |
|-------|------------------------------|
| **Prophet** | Last 7 days only |
| **Linear Trend** | Recalculated from full data |

**Code Reference**: `recent_pdf = pdf.tail(min(len(pdf), 7*24*6))` (line 2612)

#### **I/O Network Models**:

| Model | Data Used for Minimal Update |
|-------|------------------------------|
| **Crisis (Prophet)** | Last 7 days only |
| **Ensemble (Prophet)** | Last 7 days only |
| **Ensemble (ARIMA)** | Full `START_HOURS_AGO` data |
| **Ensemble (LSTM)** | Last 2 days, 2 epochs |

---

## Answer to Your Questions

### Q1: Do all models make use of `START_HOURS_AGO` for training?

**Yes**. All models use `START_HOURS_AGO` to fetch historical data during training. The models are trained on the full `START_HOURS_AGO` window (default: 360 hours = 15 days).

### Q2: Are trained/saved models used for future forecasting?

**Yes**. In forecast mode (`--forecast`), the system:
1. Loads cached/pre-trained models from disk
2. Applies minimal updates using recent data
3. Generates forecasts from the updated models

### Q3: When forecasting, do we forecast based on data trained with `START_HOURS_AGO` + minimal updates?

**Yes, but with nuances**:

1. **The cached model** contains knowledge learned from the full `START_HOURS_AGO` training period (15 days of patterns, seasonality, trends)

2. **Minimal updates** incorporate recent trends:
   - **Prophet**: Last 7 days only (preserves learned structure, adds recent trends)
   - **ARIMA**: Full `START_HOURS_AGO` data (needs full series for stationarity)
   - **LSTM**: Last 2 days, 2 epochs (fine-tunes learned patterns)

3. **Forecasts are generated** from the updated models, which combine:
   - Long-term patterns (from original training on full `START_HOURS_AGO` data)
   - Recent trends (from minimal updates)

---

## Visual Flow

### Training Mode:
```
START_HOURS_AGO (360h) → Fetch Data → Train on Full Data → Save Models
```

### Forecast Mode:
```
START_HOURS_AGO (360h) → Fetch Data → Load Cached Models → Minimal Updates → Forecast
                         ↓
                    Full data fetched
                         ↓
                    But updates use:
                    - Prophet: Last 7 days
                    - ARIMA: Full data
                    - LSTM: Last 2 days
```

---

## Why This Design?

### Benefits:
1. **Speed**: Minimal updates are much faster than full retraining
2. **Accuracy**: Preserves long-term patterns while incorporating recent trends
3. **Efficiency**: Can run frequently (every 15 seconds) without heavy computation

### Trade-offs:
- Minimal updates may miss major pattern shifts (requires full retraining)
- ARIMA uses full data (slower but more accurate)
- Prophet/LSTM use recent data only (faster but may miss long-term changes)

---

## When to Use Full Retraining

Use `--training` flag when:
- Initial setup
- Major infrastructure changes
- Periodic full retraining (weekly/monthly)
- After significant pattern shifts

Use `--forecast` flag when:
- Frequent monitoring (every 15 seconds)
- Quick predictions
- Minimal resource usage
- Recent trends are most important

---

## Code References

- `fetch_and_preprocess_data()`: Uses `START_HOURS_AGO` for data fetching
- `train_or_load_ensemble()`: Handles model loading and minimal updates
- `generate_forecast_from_cached_model()`: Implements minimal updates for Host/Pod models
- `predict_disk_full_days()`: Implements minimal updates for Disk models
- `predict_io_and_network_ensemble()`: Implements minimal updates for I/O Network models


