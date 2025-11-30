# Visual Architecture and Flow Guide

This document provides visual representations of the Metrics AI system architecture, model flows, and comparisons.

## Color Coding Legend

All diagrams use a consistent color scheme for easy identification:

| Color | Usage | Hex Code |
|-------|-------|----------|
| 🔵 **Navy Blue** | Data Sources, Prophet Models, Start/End nodes | `#2C3E50` |
| 🟢 **Forest Green** | Data Fetching, Data Processing, OK status | `#27AE60` |
| 🟡 **Amber** | Model Training, Updates, Warnings, Aggregation | `#E67E22` |
| 🟣 **Slate Purple** | Storage, Ensemble Models, Forecasts, Output | `#6C5CE7` |
| 🔴 **Burgundy** | Critical alerts, Anomalies, Save operations | `#C0392B` |
| 🟦 **Teal** | ARIMA Models, Data Processing, Load operations | `#16A085` |
| 🟪 **Indigo** | LSTM Models, External Systems, Save operations | `#5B4B8A` |
| ⚫ **Charcoal** | Decision points, Process steps | `#34495E` |

**Note**: Mermaid diagrams support color coding and styling. Some viewers (like GitHub, GitLab, VS Code with Mermaid extensions) will render these colors automatically. For enhanced interactivity and animations, consider using tools like [Mermaid Live Editor](https://mermaid.live/) or embedding in web pages with custom CSS animations.

---

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Model Comparison Tables](#model-comparison-tables)
3. [Training Flow Diagrams](#training-flow-diagrams)
4. [Prediction Flow Diagrams](#prediction-flow-diagrams)
5. [Data Flow Diagrams](#data-flow-diagrams)
6. [Configuration Impact Matrix](#configuration-impact-matrix)

---

## System Architecture Overview

### High-Level Architecture

```mermaid
graph TB
    subgraph "Data Sources"
        P[Prometheus/VictoriaMetrics]
    end
    
    subgraph "Metrics AI System"
        subgraph "Data Layer"
            DF[Data Fetching<br/>START_HOURS_AGO, STEP]
            CI[Cluster Identification<br/>LOOKBACK_HOURS]
        end
        
        subgraph "Model Layer"
            HPE[Host/Pod Ensemble<br/>Prophet + ARIMA + LSTM]
            DFM[Disk Full Models<br/>Linear + Prophet]
            IOC[I/O Network Crisis<br/>Prophet]
            IOE[I/O Network Ensemble<br/>Prophet + ARIMA + LSTM]
            CLS[Classification/Anomaly<br/>IsolationForest]
        end
        
        subgraph "Storage Layer"
            MF[Model Files<br/>.pkl files]
            MN[Manifests<br/>disk_full_models.pkl<br/>io_net_models.pkl]
            PL[Plots<br/>.png files]
        end
        
        subgraph "Output Layer"
            FC[Forecasts]
            AN[Anomalies]
            CR[Crises]
            AL[Alerts]
        end
    end
    
    subgraph "External Systems"
        WH[Webhooks]
        PG[Pushgateway]
    end
    
    P --> DF
    DF --> CI
    CI --> HPE
    CI --> DFM
    CI --> IOC
    CI --> IOE
    CI --> CLS
    
    HPE --> MF
    DFM --> MN
    IOC --> MN
    IOE --> MN
    CLS --> MF
    
    HPE --> FC
    DFM --> CR
    IOC --> CR
    IOE --> FC
    IOE --> AN
    CLS --> AN
    
    FC --> PL
    CR --> AL
    AN --> AL
    AL --> WH
    AL --> PG
    
    classDef dataSource fill:#2C3E50,stroke:#1A252F,stroke-width:3px,color:#fff
    classDef dataLayer fill:#27AE60,stroke:#1E8449,stroke-width:2px,color:#fff
    classDef modelLayer fill:#E67E22,stroke:#D35400,stroke-width:2px,color:#fff
    classDef storageLayer fill:#6C5CE7,stroke:#5A4FCF,stroke-width:2px,color:#fff
    classDef outputLayer fill:#16A085,stroke:#138D75,stroke-width:2px,color:#fff
    classDef externalSystem fill:#5B4B8A,stroke:#4A3D73,stroke-width:2px,color:#fff
    
    class P dataSource
    class DF,CI dataLayer
    class HPE,DFM,IOC,IOE,CLS modelLayer
    class MF,MN,PL storageLayer
    class FC,AN,CR,AL outputLayer
    class WH,PG externalSystem
```

### Model Types Architecture

```mermaid
graph LR
    subgraph "Ensemble Models"
        HPE[Host/Pod Ensemble<br/>CPU + Memory]
        IOE[I/O Network Ensemble<br/>I/O Wait + Network BW]
    end
    
    subgraph "Single Algorithm Models"
        DFM[Disk Full<br/>Linear + Prophet]
        IOC[I/O Crisis<br/>Prophet Only]
    end
    
    subgraph "Anomaly Detection"
        CLS[Classification<br/>IsolationForest]
    end
    
    HPE --> |Prophet| P1[Prophet Model]
    HPE --> |ARIMA| A1[ARIMA Model]
    HPE --> |LSTM| L1[LSTM Model]
    HPE --> |Combine| E1[Ensemble Forecast]
    
    IOE --> |Prophet| P2[Prophet Model]
    IOE --> |ARIMA| A2[ARIMA Model]
    IOE --> |LSTM| L2[LSTM Model]
    IOE --> |Combine| E2[Ensemble Forecast]
    
    DFM --> |Linear| LT[Linear Trend]
    DFM --> |Prophet| P3[Prophet Model]
    DFM --> |Combine| E3[ETA to 90%]
    
    IOC --> |Prophet| P4[Prophet Model]
    IOC --> |Threshold| CR[Crisis Detection]
    
    CLS --> |IsolationForest| IF[Anomaly Labels]
    
    classDef ensemble fill:#E67E22,stroke:#D35400,stroke-width:3px,color:#fff
    classDef singleAlgo fill:#27AE60,stroke:#1E8449,stroke-width:3px,color:#fff
    classDef anomaly fill:#C0392B,stroke:#A93226,stroke-width:3px,color:#fff
    classDef prophet fill:#2C3E50,stroke:#1A252F,stroke-width:2px,color:#fff
    classDef arima fill:#16A085,stroke:#138D75,stroke-width:2px,color:#fff
    classDef lstm fill:#5B4B8A,stroke:#4A3D73,stroke-width:2px,color:#fff
    classDef output fill:#6C5CE7,stroke:#5A4FCF,stroke-width:2px,color:#fff
    
    class HPE,IOE ensemble
    class DFM,IOC singleAlgo
    class CLS anomaly
    class P1,P2,P3,P4 prophet
    class A1,A2 arima
    class L1,L2 lstm
    class E1,E2,E3,CR,LT,IF output
```

---

## Model Comparison Tables

### Model Types Overview

| Model Type | Purpose | Algorithms | Scope | Storage | Forecast Horizon |
|------------|---------|------------|-------|---------|------------------|
| **Host/Pod Ensemble** | CPU/Memory forecasting | Prophet + ARIMA + LSTM | Per-cluster or standalone | `k8s_cluster_{id}_forecast.pkl` | `HORIZON_MIN` (default: 15 min) |
| **Disk Full** | Disk usage prediction | Linear trend + Prophet | Per node/mountpoint | Manifest: `disk_full_models.pkl` | `horizon_days` (default: 7 days) |
| **I/O Network Crisis** | Crisis detection | Prophet | Per node/signal | Manifest: `io_net_models.pkl` | `horizon_days` (default: 7 days) |
| **I/O Network Ensemble** | Full I/O/Net forecasting | Prophet + ARIMA + LSTM | Per node/signal | Manifest: `io_net_models.pkl` | `HORIZON_MIN` (default: 15 min) |
| **Classification** | Anomaly detection | IsolationForest | Per-cluster | `isolation_forest_anomaly.pkl` | N/A (real-time) |

### Training vs Prediction Comparison

| Aspect | Training Mode (`--training`) | Forecast Mode (`--forecast`) | Normal Mode (default) |
|--------|----------------------------|------------------------------|----------------------|
| **Data Usage** | Full `START_HOURS_AGO` window | Full `START_HOURS_AGO` window | Full `START_HOURS_AGO` window |
| **Model Updates** | Full retraining | Minimal updates (recent data only) | No updates (use cached) |
| **Prophet Update** | Full training on all data | Last 7 days only | No update |
| **ARIMA Update** | Full training | Latest data refit | No update |
| **LSTM Update** | Full training (`LSTM_EPOCHS` epochs) | Fine-tune last 2 days (2 epochs) | No update |
| **Backtesting** | ✅ Yes (calculates MAE, RMSE) | ❌ No | ❌ No (unless `--show-backtest`) |
| **Plots Generated** | Forecast + Backtest | Forecast only | None (unless `--plot`) |
| **Execution Time** | ~5-15 minutes | ~10-30 seconds | ~5-10 seconds |
| **Use Case** | Initial setup, periodic retraining | Frequent monitoring | Quick status check |

### Algorithm Comparison

| Algorithm | Strengths | Weaknesses | Used In | Configuration Variables |
|-----------|-----------|------------|---------|------------------------|
| **Prophet** | Handles seasonality, trends, holidays | Slower training | All models | `HORIZON_MIN`, `horizon_days` |
| **ARIMA** | Fast, good for stationary data | Limited to linear patterns | Host/Pod, I/O Ensemble | `HORIZON_MIN` |
| **LSTM** | Captures complex patterns, non-linear | Requires more data, slower | Host/Pod, I/O Ensemble | `LSTM_SEQ_LEN`, `LSTM_EPOCHS`, `HORIZON_MIN` |
| **Linear Trend** | Very fast, simple | Only captures linear trends | Disk models | None (fixed) |
| **IsolationForest** | Unsupervised, no labels needed | Sensitive to contamination rate | Classification | `LOOKBACK_HOURS`, `CONTAMINATION` |

### Storage Comparison

| Model Type | Storage Method | File Pattern | Manifest | Update Frequency |
|------------|---------------|--------------|----------|------------------|
| **Host/Pod Ensemble** | Individual files | `k8s_cluster_{id}_forecast.pkl` | No | Per-cluster |
| **Disk Full** | Manifest-based | `disk_full_models.pkl` | Yes | Per disk (node+mountpoint) |
| **I/O Network Crisis** | Manifest-based | `io_net_models.pkl` | Yes | Per node+signal |
| **I/O Network Ensemble** | Manifest-based | `io_net_models.pkl` | Yes | Per node+signal |
| **Classification** | Individual files | `isolation_forest_anomaly.pkl` | No | Per-cluster |

---

## Training Flow Diagrams

### Complete Training Flow

```mermaid
flowchart TD
    Start([Start: --training]) --> Fetch[Fetch Data from Prometheus<br/>START_HOURS_AGO, STEP]
    
    Fetch --> HostData[Host CPU/Memory Data]
    Fetch --> PodData[Pod CPU/Memory Data]
    Fetch --> DiskData[Disk Usage Data]
    Fetch --> IOData[I/O Wait Data]
    Fetch --> NetData[Network Bandwidth Data]
    
    HostData --> ClusterID[Identify Clusters<br/>LOOKBACK_HOURS]
    PodData --> ClusterID
    
    ClusterID --> GroupNodes[Group Nodes by Cluster]
    
    GroupNodes --> HostPodTrain[Train Host/Pod Ensemble Models]
    GroupNodes --> DiskTrain[Train Disk Models]
    GroupNodes --> IOTrain[Train I/O Network Models]
    GroupNodes --> ClassTrain[Train Classification Models]
    
    HostPodTrain --> HPSplit[Train/Test Split<br/>TRAIN_FRACTION]
    HPSplit --> HPProphet[Train Prophet<br/>HORIZON_MIN]
    HPSplit --> HPARIMA[Train ARIMA<br/>HORIZON_MIN]
    HPSplit --> HPLSTM[Train LSTM<br/>LSTM_SEQ_LEN, LSTM_EPOCHS, HORIZON_MIN]
    HPProphet --> HPEnsemble[Create Ensemble]
    HPARIMA --> HPEnsemble
    HPLSTM --> HPEnsemble
    HPEnsemble --> HPBacktest[Backtest on Test Set]
    HPBacktest --> HPSave[Save Models]
    
    DiskTrain --> DiskSplit[Train/Test Split<br/>TRAIN_FRACTION]
    DiskSplit --> DiskLinear[Train Linear Trend]
    DiskSplit --> DiskProphet[Train Prophet<br/>horizon_days]
    DiskLinear --> DiskETA[Calculate ETA to 90%]
    DiskProphet --> DiskETA
    DiskETA --> DiskSave[Save to Manifest]
    
    IOTrain --> IOSplit[Train/Test Split<br/>TRAIN_FRACTION]
    IOSplit --> IOCrisis[Train Crisis Models<br/>Prophet, horizon_days]
    IOSplit --> IOEnsemble[Train Ensemble Models<br/>Prophet + ARIMA + LSTM, HORIZON_MIN]
    IOCrisis --> IOSave[Save to Manifest]
    IOEnsemble --> IOSave
    
    ClassTrain --> ClassFeat[Extract Features<br/>LOOKBACK_HOURS]
    ClassFeat --> ClassISO[Train IsolationForest<br/>CONTAMINATION]
    ClassISO --> ClassSave[Save Models]
    
    HPSave --> AllDone[All Models Trained]
    DiskSave --> AllDone
    IOSave --> AllDone
    ClassSave --> AllDone
    
    AllDone --> GeneratePlots[Generate Plots]
    GeneratePlots --> End([End: Models Saved])
    
    classDef startEnd fill:#2C3E50,stroke:#1A252F,stroke-width:4px,color:#fff
    classDef dataFetch fill:#27AE60,stroke:#1E8449,stroke-width:3px,color:#fff
    classDef dataProcess fill:#16A085,stroke:#138D75,stroke-width:2px,color:#fff
    classDef modelTrain fill:#E67E22,stroke:#D35400,stroke-width:3px,color:#fff
    classDef prophet fill:#2C3E50,stroke:#1A252F,stroke-width:2px,color:#fff
    classDef arima fill:#16A085,stroke:#138D75,stroke-width:2px,color:#fff
    classDef lstm fill:#5B4B8A,stroke:#4A3D73,stroke-width:2px,color:#fff
    classDef ensemble fill:#6C5CE7,stroke:#5A4FCF,stroke-width:3px,color:#fff
    classDef save fill:#C0392B,stroke:#A93226,stroke-width:3px,color:#fff
    classDef output fill:#E67E22,stroke:#D35400,stroke-width:3px,color:#fff
    
    class Start,End startEnd
    class Fetch,HostData,PodData,DiskData,IOData,NetData dataFetch
    class ClusterID,GroupNodes dataProcess
    class HostPodTrain,DiskTrain,IOTrain,ClassTrain,HPSplit,DiskSplit,IOSplit modelTrain
    class HPProphet,DiskProphet,IOCrisis prophet
    class HPARIMA arima
    class HPLSTM,IOEnsemble lstm
    class HPEnsemble,DiskETA,AllDone ensemble
    class HPSave,DiskSave,IOSave,ClassSave save
    class HPBacktest,GeneratePlots output
```

### Host/Pod Ensemble Training Detail

```mermaid
flowchart TD
    Start([Start Training]) --> DataPrep[Prepare Data<br/>Aggregate CPU + Memory]
    
    DataPrep --> Split[Train/Test Split<br/>TRAIN_FRACTION = 0.8]
    
    Split --> TrainSet[Training Set: 80%]
    Split --> TestSet[Test Set: 20%]
    
    TrainSet --> Prophet[Train Prophet<br/>daily_seasonality=True<br/>weekly_seasonality=True]
    TrainSet --> ARIMA[Train ARIMA<br/>order=2,1,0]
    TrainSet --> LSTM[Train LSTM<br/>LSTM_SEQ_LEN=60<br/>LSTM_EPOCHS=10]
    
    Prophet --> PForecast[Prophet Forecast<br/>HORIZON_MIN steps]
    ARIMA --> AForecast[ARIMA Forecast<br/>HORIZON_MIN steps]
    LSTM --> LForecast[LSTM Forecast<br/>HORIZON_MIN steps]
    
    PForecast --> Ensemble[Create Ensemble<br/>Average of 3 forecasts]
    AForecast --> Ensemble
    LForecast --> Ensemble
    
    Ensemble --> Save[Save Models<br/>prophet, arima, lstm, ensemble]
    
    TestSet --> Backtest[Backtest on Test Set]
    Backtest --> Metrics[Calculate MAE, RMSE]
    Metrics --> Save
    
    Save --> End([End])
    
    classDef startEnd fill:#2C3E50,stroke:#1A252F,stroke-width:4px,color:#fff
    classDef dataPrep fill:#27AE60,stroke:#1E8449,stroke-width:2px,color:#fff
    classDef split fill:#16A085,stroke:#138D75,stroke-width:2px,color:#fff
    classDef trainSet fill:#E67E22,stroke:#D35400,stroke-width:2px,color:#fff
    classDef testSet fill:#6C5CE7,stroke:#5A4FCF,stroke-width:2px,color:#fff
    classDef prophet fill:#2C3E50,stroke:#1A252F,stroke-width:2px,color:#fff
    classDef arima fill:#16A085,stroke:#138D75,stroke-width:2px,color:#fff
    classDef lstm fill:#5B4B8A,stroke:#4A3D73,stroke-width:2px,color:#fff
    classDef ensemble fill:#6C5CE7,stroke:#5A4FCF,stroke-width:3px,color:#fff
    classDef save fill:#C0392B,stroke:#A93226,stroke-width:3px,color:#fff
    classDef backtest fill:#E67E22,stroke:#D35400,stroke-width:2px,color:#fff
    
    class Start,End startEnd
    class DataPrep dataPrep
    class Split split
    class TrainSet trainSet
    class TestSet,Backtest,Metrics testSet
    class Prophet,PForecast prophet
    class ARIMA,AForecast arima
    class LSTM,LForecast lstm
    class Ensemble ensemble
    class Save save
```

### Disk Model Training Detail

```mermaid
flowchart TD
    Start([Start Training]) --> Group[Group by Node + Mountpoint]
    
    Group --> ForEach{For Each Disk}
    
    ForEach --> Data[Get Time Series<br/>Disk Usage %]
    
    Data --> Check{Enough Data?<br/>>= 50 points}
    Check -->|No| Skip[Skip Disk]
    Check -->|Yes| Split[Train/Test Split<br/>TRAIN_FRACTION]
    
    Split --> Linear[Train Linear Trend<br/>Calculate slope]
    Split --> Prophet[Train Prophet<br/>horizon_days=7]
    
    Linear --> LinearETA[Calculate Days to 90%<br/>Linear Model]
    Prophet --> ProphetETA[Calculate Days to 90%<br/>Prophet Forecast]
    
    LinearETA --> EnsembleETA[Ensemble ETA<br/>min of both]
    ProphetETA --> EnsembleETA
    
    EnsembleETA --> Severity{Determine Severity}
    Severity -->|ETA < 3 days| Critical[CRITICAL]
    Severity -->|ETA < 7 days| Warning[WARNING]
    Severity -->|ETA < 30 days| Soon[SOON]
    Severity -->|ETA >= 30 days| OK[OK]
    
    Critical --> Save[Save to Manifest]
    Warning --> Save
    Soon --> Save
    OK --> Save
    Skip --> Next{More Disks?}
    Save --> Next
    
    Next -->|Yes| ForEach
    Next -->|No| End([End])
    
    classDef startEnd fill:#2C3E50,stroke:#1A252F,stroke-width:4px,color:#fff
    classDef process fill:#27AE60,stroke:#1E8449,stroke-width:2px,color:#fff
    classDef decision fill:#34495E,stroke:#2C3E50,stroke-width:3px,color:#fff
    classDef prophet fill:#2C3E50,stroke:#1A252F,stroke-width:2px,color:#fff
    classDef linear fill:#16A085,stroke:#138D75,stroke-width:2px,color:#fff
    classDef ensemble fill:#6C5CE7,stroke:#5A4FCF,stroke-width:3px,color:#fff
    classDef critical fill:#C0392B,stroke:#A93226,stroke-width:3px,color:#fff
    classDef warning fill:#E67E22,stroke:#D35400,stroke-width:2px,color:#fff
    classDef soon fill:#27AE60,stroke:#1E8449,stroke-width:2px,color:#fff
    classDef ok fill:#16A085,stroke:#138D75,stroke-width:2px,color:#fff
    classDef save fill:#5B4B8A,stroke:#4A3D73,stroke-width:3px,color:#fff
    
    class Start,End startEnd
    class Group,Data,Split process
    class ForEach,Check,Severity,Next decision
    class Prophet,ProphetETA prophet
    class Linear,LinearETA linear
    class EnsembleETA ensemble
    class Critical critical
    class Warning warning
    class Soon soon
    class OK ok
    class Save save
```

---

## Prediction Flow Diagrams

### Forecast Mode Flow

```mermaid
flowchart TD
    Start([Start: --forecast]) --> Fetch[Fetch Latest Data<br/>START_HOURS_AGO, STEP]
    
    Fetch --> Load[Load Cached Models]
    
    Load --> UpdateHP[Update Host/Pod Models]
    Load --> UpdateDisk[Update Disk Models]
    Load --> UpdateIO[Update I/O Network Models]
    Load --> UpdateClass[Update Classification]
    
    UpdateHP --> HPProphet[Prophet: Last 7 days]
    UpdateHP --> HPARIMA[ARIMA: Latest data]
    UpdateHP --> HPLSTM[LSTM: Last 2 days, 2 epochs]
    HPProphet --> HPForecast[Generate Forecast<br/>HORIZON_MIN]
    HPARIMA --> HPForecast
    HPLSTM --> HPForecast
    
    UpdateDisk --> DiskProphet[Prophet: Last 7 days]
    UpdateDisk --> DiskLinear[Recalculate Linear Trend]
    DiskProphet --> DiskForecast[Calculate ETA to 90%]
    DiskLinear --> DiskForecast
    
    UpdateIO --> IOCrisis[I/O Crisis: Prophet Last 7 days]
    UpdateIO --> IOEnsemble[I/O Ensemble: All 3 models]
    IOCrisis --> IOCheck[Check Crisis Threshold]
    IOEnsemble --> IOForecast[Generate Forecast<br/>HORIZON_MIN]
    IOEnsemble --> IOAnomaly[Detect Anomalies]
    
    UpdateClass --> ClassFeat[Extract Features<br/>LOOKBACK_HOURS]
    ClassFeat --> ClassPredict[Predict Anomalies<br/>CONTAMINATION]
    
    HPForecast --> Aggregate[Aggregate Results]
    DiskForecast --> Aggregate
    IOCheck --> Aggregate
    IOForecast --> Aggregate
    IOAnomaly --> Aggregate
    ClassPredict --> Aggregate
    
    Aggregate --> Alerts[Generate Alerts]
    Alerts --> Save[Save Updated Models]
    Save --> Plots[Generate Plots if --plot]
    Plots --> End([End])
    
    classDef startEnd fill:#2C3E50,stroke:#1A252F,stroke-width:4px,color:#fff
    classDef dataFetch fill:#27AE60,stroke:#1E8449,stroke-width:3px,color:#fff
    classDef load fill:#16A085,stroke:#138D75,stroke-width:3px,color:#fff
    classDef update fill:#E67E22,stroke:#D35400,stroke-width:2px,color:#fff
    classDef prophet fill:#2C3E50,stroke:#1A252F,stroke-width:2px,color:#fff
    classDef arima fill:#16A085,stroke:#138D75,stroke-width:2px,color:#fff
    classDef lstm fill:#5B4B8A,stroke:#4A3D73,stroke-width:2px,color:#fff
    classDef forecast fill:#6C5CE7,stroke:#5A4FCF,stroke-width:3px,color:#fff
    classDef anomaly fill:#C0392B,stroke:#A93226,stroke-width:2px,color:#fff
    classDef aggregate fill:#E67E22,stroke:#D35400,stroke-width:3px,color:#fff
    classDef output fill:#27AE60,stroke:#1E8449,stroke-width:3px,color:#fff
    
    class Start,End startEnd
    class Fetch dataFetch
    class Load load
    class UpdateHP,UpdateDisk,UpdateIO,UpdateClass update
    class HPProphet,DiskProphet,IOCrisis prophet
    class HPARIMA arima
    class HPLSTM,IOEnsemble lstm
    class HPForecast,DiskForecast,IOForecast forecast
    class IOAnomaly,ClassPredict anomaly
    class Aggregate aggregate
    class Alerts,Save,Plots output
```

### Minimal Update Flow (Forecast Mode)

```mermaid
flowchart LR
    subgraph "Prophet Minimal Update"
        P1[Load Saved<br/>Hyperparameters] --> P2[Create Prophet<br/>with same params]
        P2 --> P3[Fit on Last 7 Days<br/>Only]
        P3 --> P4[Generate Forecast<br/>HORIZON_MIN]
    end
    
    subgraph "ARIMA Minimal Update"
        A1[Load Saved<br/>ARIMA Order] --> A2[Fit on Latest Data<br/>Same order 2,1,0]
        A2 --> A3[Generate Forecast<br/>HORIZON_MIN]
    end
    
    subgraph "LSTM Minimal Update"
        L1[Load Saved<br/>Model + Scaler] --> L2[Fine-tune Last 2 Days<br/>2 Epochs Only]
        L2 --> L3[Generate Forecast<br/>HORIZON_MIN]
    end
    
    P4 --> Ensemble[Combine Forecasts]
    A3 --> Ensemble
    L3 --> Ensemble
    
    Ensemble --> Save[Save Updated Models]
    
    classDef load fill:#16A085,stroke:#138D75,stroke-width:2px,color:#fff
    classDef process fill:#E67E22,stroke:#D35400,stroke-width:2px,color:#fff
    classDef prophet fill:#2C3E50,stroke:#1A252F,stroke-width:2px,color:#fff
    classDef arima fill:#16A085,stroke:#138D75,stroke-width:2px,color:#fff
    classDef lstm fill:#5B4B8A,stroke:#4A3D73,stroke-width:2px,color:#fff
    classDef forecast fill:#6C5CE7,stroke:#5A4FCF,stroke-width:2px,color:#fff
    classDef ensemble fill:#6C5CE7,stroke:#5A4FCF,stroke-width:3px,color:#fff
    classDef save fill:#C0392B,stroke:#A93226,stroke-width:3px,color:#fff
    
    class P1,A1,L1 load
    class P2,P3,A2,L2 process
    class P1,P2,P3,P4 prophet
    class A1,A2,A3 arima
    class L1,L2,L3 lstm
    class P4,A3,L3 forecast
    class Ensemble ensemble
    class Save save
```

---

## Data Flow Diagrams

### Data Flow: Training to Prediction

```mermaid
graph LR
    subgraph "Training Phase"
        T1[Prometheus Data<br/>START_HOURS_AGO hours] --> T2[Train/Test Split<br/>TRAIN_FRACTION]
        T2 --> T3[Train Models]
        T3 --> T4[Backtest on Test Set]
        T4 --> T5[Save Models]
    end
    
    subgraph "Prediction Phase"
        P1[Prometheus Data<br/>Latest START_HOURS_AGO] --> P2[Load Cached Models]
        P2 --> P3[Minimal Updates]
        P3 --> P4[Generate Forecasts]
        P4 --> P5[Detect Anomalies/Crises]
        P5 --> P6[Save Updated Models]
    end
    
    T5 -->|Models Saved| P2
    P6 -->|Updated Models| P2
    
    classDef trainingData fill:#27AE60,stroke:#1E8449,stroke-width:2px,color:#fff
    classDef trainingProcess fill:#E67E22,stroke:#D35400,stroke-width:2px,color:#fff
    classDef trainingSave fill:#C0392B,stroke:#A93226,stroke-width:3px,color:#fff
    classDef predictionData fill:#2C3E50,stroke:#1A252F,stroke-width:2px,color:#fff
    classDef predictionProcess fill:#16A085,stroke:#138D75,stroke-width:2px,color:#fff
    classDef predictionOutput fill:#6C5CE7,stroke:#5A4FCF,stroke-width:2px,color:#fff
    classDef predictionSave fill:#5B4B8A,stroke:#4A3D73,stroke-width:3px,color:#fff
    
    class T1,P1 trainingData
    class T2,T3,T4 trainingProcess
    class T5 trainingSave
    class P2,P3 predictionProcess
    class P4,P5 predictionOutput
    class P6 predictionSave
```

### Configuration Variables Flow

```mermaid
graph TD
    subgraph "Data Fetching Variables"
        START[START_HOURS_AGO<br/>360 hours] --> Fetch[Data Fetching]
        STEP[STEP<br/>60s] --> Fetch
    end
    
    subgraph "Training Variables"
        TRAIN[TRAIN_FRACTION<br/>0.8] --> Split[Train/Test Split]
        HORIZON[HORIZON_MIN<br/>15 min] --> Models[Model Training]
        LSTM_SEQ[LSTM_SEQ_LEN<br/>60] --> Models
        LSTM_EPOCH[LSTM_EPOCHS<br/>10] --> Models
    end
    
    subgraph "Anomaly Detection Variables"
        LOOKBACK[LOOKBACK_HOURS<br/>24 hours] --> Anomaly[Feature Extraction]
        CONTAM[CONTAMINATION<br/>0.12] --> Anomaly
    end
    
    Fetch --> Split
    Split --> Models
    Models --> Forecast[Forecast Generation]
    Anomaly --> AnomalyDet[Anomaly Detection]
    
    Forecast --> Output[Output: Forecasts, Crises, Anomalies]
    AnomalyDet --> Output
    
    classDef dataVar fill:#27AE60,stroke:#1E8449,stroke-width:2px,color:#fff
    classDef trainVar fill:#E67E22,stroke:#D35400,stroke-width:2px,color:#fff
    classDef anomalyVar fill:#C0392B,stroke:#A93226,stroke-width:2px,color:#fff
    classDef process fill:#16A085,stroke:#138D75,stroke-width:3px,color:#fff
    classDef output fill:#6C5CE7,stroke:#5A4FCF,stroke-width:3px,color:#fff
    
    class START,STEP,Fetch dataVar
    class TRAIN,HORIZON,LSTM_SEQ,LSTM_EPOCH,Split,Models trainVar
    class LOOKBACK,CONTAM,Anomaly anomalyVar
    class Forecast,AnomalyDet process
    class Output output
```

---

## Configuration Impact Matrix

### Detailed Configuration Impact

| Variable | Default | Host/Pod Ensemble | Disk Models | I/O Crisis | I/O Ensemble | Classification | Impact Description |
|----------|---------|-------------------|-------------|------------|--------------|----------------|-------------------|
| **START_HOURS_AGO** | 360 | ✅ High | ✅ High | ✅ High | ✅ High | ✅ High | More data = better training, but slower |
| **STEP** | "60s" | ✅ Medium | ✅ Medium | ✅ Medium | ✅ Medium | ✅ Medium | Smaller = more data points, better accuracy |
| **TRAIN_FRACTION** | 0.8 | ✅ High | ✅ High | ✅ High | ✅ High | ❌ N/A | Controls train/test split ratio |
| **HORIZON_MIN** | 15 | ✅ Critical | ❌ N/A | ❌ N/A | ✅ Critical | ❌ N/A | Forecast length in minutes |
| **LSTM_SEQ_LEN** | 60 | ✅ High | ❌ N/A | ❌ N/A | ✅ High | ❌ N/A | Input sequence length for LSTM |
| **LSTM_EPOCHS** | 10 | ✅ Medium | ❌ N/A | ❌ N/A | ✅ Medium | ❌ N/A | Training iterations for LSTM |
| **LOOKBACK_HOURS** | 24 | ❌ N/A | ❌ N/A | ❌ N/A | ❌ N/A | ✅ Critical | Feature extraction window |
| **CONTAMINATION** | 0.12 | ❌ N/A | ❌ N/A | ❌ N/A | ❌ N/A | ✅ Critical | Expected anomaly rate |

### Impact Legend
- ✅ **Critical**: Directly affects model behavior/accuracy
- ✅ **High**: Significant impact on results
- ✅ **Medium**: Moderate impact
- ❌ **N/A**: Not used by this model

### Variable Priority by Model

```mermaid
graph TD
    subgraph "Host/Pod Ensemble"
        HP1[HORIZON_MIN<br/>Critical]
        HP2[TRAIN_FRACTION<br/>High]
        HP3[LSTM_SEQ_LEN<br/>High]
        HP4[START_HOURS_AGO<br/>High]
        HP5[LSTM_EPOCHS<br/>Medium]
        HP6[STEP<br/>Medium]
    end
    
    subgraph "Disk Models"
        D1[TRAIN_FRACTION<br/>High]
        D2[START_HOURS_AGO<br/>High]
        D3[STEP<br/>Medium]
    end
    
    subgraph "I/O Network"
        IO1[HORIZON_MIN<br/>Critical - Ensemble]
        IO2[TRAIN_FRACTION<br/>High]
        IO3[START_HOURS_AGO<br/>High]
        IO4[LSTM_SEQ_LEN<br/>High - Ensemble]
        IO5[STEP<br/>Medium]
    end
    
    subgraph "Classification"
        C1[LOOKBACK_HOURS<br/>Critical]
        C2[CONTAMINATION<br/>Critical]
        C3[START_HOURS_AGO<br/>High]
        C4[STEP<br/>Medium]
    end
    
    classDef critical fill:#C0392B,stroke:#A93226,stroke-width:3px,color:#fff
    classDef high fill:#E67E22,stroke:#D35400,stroke-width:2px,color:#fff
    classDef medium fill:#27AE60,stroke:#1E8449,stroke-width:2px,color:#fff
    
    class HP1,IO1,C1,C2 critical
    class HP2,HP3,HP4,D1,D2,IO2,IO3,IO4,C3 high
    class HP5,HP6,D3,IO5,C4 medium
```

---

## Model Execution Order

### Normal Execution Flow

```mermaid
sequenceDiagram
    participant User
    participant System
    participant Prometheus
    participant Models
    participant Storage
    
    User->>+System: Run metrics.py
    System->>+Prometheus: Fetch Data (START_HOURS_AGO, STEP)
    Prometheus-->>-System: Return Data
    
    System->>System: Identify Clusters (LOOKBACK_HOURS)
    
    System->>+Models: Train/Load Host/Pod Models
    Models->>Storage: Save/Load Models
    Models-->>-System: Models Ready
    
    System->>+Models: Train/Load Disk Models
    Models->>Storage: Save/Load to Manifest
    Models-->>-System: Models Ready
    
    System->>+Models: Train/Load I/O Crisis Models
    Models->>Storage: Save/Load to Manifest
    Models-->>-System: Models Ready
    
    System->>+Models: Train/Load I/O Ensemble Models
    Models->>Storage: Save/Load to Manifest
    Models-->>-System: Models Ready
    
    System->>+Models: Train/Load Classification Models
    Models->>Storage: Save/Load Models
    Models-->>-System: Models Ready
    
    System->>System: Generate Forecasts
    System->>System: Detect Anomalies
    System->>System: Detect Crises
    
    System->>User: Display Results
    System->>Storage: Save Plots (if enabled)
    System-->>-User: Complete
```

---

## Quick Reference: Model Selection Guide

### When to Use Each Model

| Use Case | Model Type | Why |
|----------|------------|-----|
| **CPU/Memory forecasting** | Host/Pod Ensemble | Combines 3 algorithms for robust predictions |
| **Disk capacity planning** | Disk Full | Predicts when disk will reach 90% |
| **I/O performance crisis** | I/O Network Crisis | Fast detection of I/O bottlenecks |
| **I/O/Network forecasting** | I/O Network Ensemble | Full ensemble for accurate predictions |
| **Anomaly detection** | Classification | Identifies unusual resource patterns |

### Performance Characteristics

| Model Type | Training Time | Prediction Time | Accuracy | Use Case |
|------------|--------------|-----------------|----------|----------|
| **Host/Pod Ensemble** | ~2-5 min | ~1-2 sec | High | Production forecasting |
| **Disk Full** | ~30 sec | ~0.5 sec | Medium-High | Capacity planning |
| **I/O Network Crisis** | ~20 sec | ~0.3 sec | Medium | Quick crisis detection |
| **I/O Network Ensemble** | ~1-3 min | ~1 sec | High | Detailed I/O analysis |
| **Classification** | ~10 sec | ~0.1 sec | Medium | Anomaly detection |

---

## See Also

- `MODEL_TRAINING_AND_PREDICTION_GUIDE.md` - Detailed step-by-step guide
- `CONFIGURATION_VARIABLES.md` - Variable explanations
- `../README.md` - General documentation
- `SYSTEM_DOCUMENTATION.md` - System architecture

