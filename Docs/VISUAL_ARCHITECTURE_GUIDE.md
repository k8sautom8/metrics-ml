# Visual Architecture and Flow Guide

This document provides visual representations of the Metrics AI system architecture, model flows, and comparisons.

## Color Coding Legend

All diagrams use a consistent color scheme for easy identification:

| Color | Usage | Hex Code |
|-------|-------|----------|
| 🔵 **Light Blue** | Data Sources, Prophet Models, Start/End nodes | `#0078D4` |
| 🟢 **Light Green** | Data Fetching, Data Processing, OK status | `#107C10` |
| 🟡 **Light Orange** | Model Training, Updates, Warnings, Aggregation | `#FF8C00` |
| 🟣 **Light Purple** | Storage, Ensemble Models, Forecasts, Output | `#8764B8` |
| 🔴 **Light Red** | Critical alerts, Anomalies, Save operations | `#E81123` |
| 🟦 **Light Teal** | ARIMA Models, Data Processing, Load operations | `#00B7C3` |
| 🟪 **Light Indigo** | LSTM Models, External Systems, Save operations | `#5C2D91` |
| 🟩 **Light Green** | LightGBM Models, Gradient Boosting | `#28A745` |
| ⚫ **Light Gray** | Decision points, Process steps, Skipped models | `#6B6B6B` |

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
            HPE[Host/Pod Ensemble<br/>Prophet + ARIMA + LSTM + LightGBM]
            DFM[Disk Full Models<br/>Linear + Prophet]
            IOC[I/O Network Crisis<br/>Prophet]
            IOE[I/O Network Ensemble<br/>Prophet + ARIMA + LSTM + LightGBM]
            UCM[Unified Correlated Models<br/>Prophet + LSTM + LightGBM<br/>ARIMA skipped]
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
    CI --> UCM
    CI --> CLS
    
    HPE --> MF
    DFM --> MN
    IOC --> MN
    IOE --> MN
    UCM --> MF
    CLS --> MF
    
    HPE --> FC
    DFM --> CR
    IOC --> CR
    IOE --> FC
    IOE --> AN
    UCM --> FC
    UCM --> CR
    UCM --> AN
    CLS --> AN
    
    FC --> PL
    CR --> AL
    AN --> AL
    AL --> WH
    AL --> PG
    
    classDef dataSource fill:#0078D4,stroke:#005A9E,stroke-width:3px,color:#fff
    classDef dataLayer fill:#107C10,stroke:#0B5A0B,stroke-width:2px,color:#fff
    classDef modelLayer fill:#FF8C00,stroke:#CC7000,stroke-width:2px,color:#fff
    classDef storageLayer fill:#8764B8,stroke:#6B4F93,stroke-width:2px,color:#fff
    classDef outputLayer fill:#00B7C3,stroke:#00929A,stroke-width:2px,color:#fff
    classDef externalSystem fill:#5C2D91,stroke:#4A2473,stroke-width:2px,color:#fff
    
    class P dataSource
    class DF,CI dataLayer
    class HPE,DFM,IOC,IOE,UCM,CLS modelLayer
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
        UCM[Unified Correlated<br/>All 6 Metrics]
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
    HPE --> |LightGBM| LG1[LightGBM Model]
    HPE --> |Combine| E1[Ensemble Forecast]
    
    IOE --> |Prophet| P2[Prophet Model]
    IOE --> |ARIMA| A2[ARIMA Model]
    IOE --> |LSTM| L2[LSTM Model]
    IOE --> |LightGBM| LG2[LightGBM Model]
    IOE --> |Combine| E2[Ensemble Forecast]
    
    UCM --> |Prophet| P3[Prophet Model]
    UCM --> |LSTM| L3[LSTM Model]
    UCM --> |LightGBM| LG3[LightGBM Model]
    UCM --> |ARIMA| AS[ARIMA Skipped]
    UCM --> |Combine| E3[Ensemble Forecast]
    
    DFM --> |Linear| LT[Linear Trend]
    DFM --> |Prophet| P4[Prophet Model]
    DFM --> |Combine| E4[ETA to 90%]
    
    IOC --> |Prophet| P5[Prophet Model]
    IOC --> |Threshold| CR[Crisis Detection]
    
    CLS --> |IsolationForest| IF[Anomaly Labels]
    
    classDef ensemble fill:#FF8C00,stroke:#CC7000,stroke-width:3px,color:#fff
    classDef singleAlgo fill:#107C10,stroke:#0B5A0B,stroke-width:3px,color:#fff
    classDef anomaly fill:#E81123,stroke:#BA0E1C,stroke-width:3px,color:#fff
    classDef prophet fill:#0078D4,stroke:#005A9E,stroke-width:2px,color:#fff
    classDef arima fill:#00B7C3,stroke:#00929A,stroke-width:2px,color:#fff
    classDef lstm fill:#5C2D91,stroke:#4A2473,stroke-width:2px,color:#fff
    classDef lightgbm fill:#28A745,stroke:#1E7E34,stroke-width:2px,color:#fff
    classDef skipped fill:#6B6B6B,stroke:#555555,stroke-width:2px,color:#fff
    classDef output fill:#8764B8,stroke:#6B4F93,stroke-width:2px,color:#fff
    
    class HPE,IOE,UCM ensemble
    class DFM,IOC singleAlgo
    class CLS anomaly
    class P1,P2,P3,P4 prophet
    class A1,A2 arima
    class L1,L2,L3 lstm
    class LG1,LG2,LG3 lightgbm
    class AS skipped
    class E1,E2,E3,E4,CR,LT,IF output
```

---

## Model Comparison Tables

### Model Types Overview

| Model Type | Purpose | Algorithms | Scope | Storage | Forecast Horizon |
|------------|---------|------------|-------|---------|------------------|
| **Host/Pod Ensemble** | CPU/Memory forecasting | Prophet + ARIMA + LSTM + LightGBM | Per-cluster or standalone | `k8s_cluster_{id}_forecast.pkl` | `HORIZON_MIN` (default: 15 min) |
| **Disk Full** | Disk usage prediction | Linear trend + Prophet (daily + weekly seasonality) | Per node/mountpoint | Manifest: `disk_full_models.pkl` | `horizon_days` (default: 7 days) |
| **I/O Network Crisis** | Crisis detection | Prophet (daily + weekly seasonality) | Per node/signal | Manifest: `io_net_models.pkl` | `horizon_days` (default: 7 days) |
| **I/O Network Ensemble** | Full I/O/Net forecasting | Prophet + ARIMA + LSTM + LightGBM (temporal-aware anomaly detection) | Per node/signal | Manifest: `io_net_models.pkl` | `HORIZON_MIN` (default: 15 min) |
| **Unified Correlated** | Multi-metric ensemble | Prophet + LSTM + LightGBM (ARIMA skipped) | Per node (all 6 metrics) | `unified_correlated_{node}_forecast.pkl` | `HORIZON_MIN` (default: 15 min) |
| **Classification** | Anomaly detection | IsolationForest (temporal-aware if 3+ months data) | Per-cluster | `isolation_forest_anomaly.pkl` | N/A (real-time) |

### Training vs Prediction Comparison

| Aspect | Training Mode (`--training`) | Forecast Mode (`--forecast`) | Normal Mode (default) |
|--------|----------------------------|------------------------------|----------------------|
| **Data Usage** | Full `START_HOURS_AGO` window | Full `START_HOURS_AGO` window | Full `START_HOURS_AGO` window |
| **Model Updates** | Full retraining | Minimal updates (recent data only) | No updates (use cached) |
| **Prophet Update** | Full training on all data | Last 7 days only | No update |
| **ARIMA Update** | Full training | Latest data refit | No update (skipped for unified correlated models) |
| **LSTM Update** | Full training (`LSTM_EPOCHS` epochs) | Fine-tune last 2 days (2 epochs) | No update |
| **LightGBM Update** | Full training (with early stopping) | Latest data refit (with early stopping) | No update |
| **Backtesting** | ✅ Yes (calculates MAE, RMSE) | ❌ No | ❌ No (unless `--show-backtest`) |
| **Plots Generated** | Forecast + Backtest | Forecast only | None (unless `--plot`) |
| **Execution Time** | ~5-15 minutes | ~10-30 seconds | ~5-10 seconds |
| **Use Case** | Initial setup, periodic retraining | Frequent monitoring | Quick status check |

### Algorithm Comparison

| Algorithm | Strengths | Weaknesses | Used In | Configuration Variables |
|-----------|-----------|------------|---------|------------------------|
| **Prophet** | Handles seasonality (daily + weekly), trends, holidays | Slower training | All models | `HORIZON_MIN`, `horizon_days` |
| **ARIMA** | Fast, good for stationary data | Limited to linear patterns, poor on normalized data | Host/Pod, I/O Ensemble (skipped for unified correlated) | `HORIZON_MIN` |
| **LSTM** | Captures complex patterns, non-linear | Requires more data, slower | Host/Pod, I/O Ensemble, Unified Correlated | `LSTM_SEQ_LEN`, `LSTM_EPOCHS`, `HORIZON_MIN` |
| **LightGBM** | High accuracy, fast training, early stopping | Requires minimum 4 features | Host/Pod, I/O Ensemble, Unified Correlated | `LIGHTGBM_ENABLED`, `HORIZON_MIN` |
| **Linear Trend** | Very fast, simple | Only captures linear trends | Disk models | None (fixed) |
| **IsolationForest** | Unsupervised, no labels needed, temporal-aware | Sensitive to contamination rate | Classification | `LOOKBACK_HOURS`, `CONTAMINATION` (temporal features if 3+ months data) |

### Storage Comparison

| Model Type | Storage Method | File Pattern | Manifest | Update Frequency |
|------------|---------------|--------------|----------|------------------|
| **Host/Pod Ensemble** | Individual files | `k8s_cluster_{id}_forecast.pkl` | No | Per-cluster |
| **Disk Full** | Manifest-based | `disk_full_models.pkl` | Yes | Per disk (node+mountpoint) |
| **I/O Network Crisis** | Manifest-based | `io_net_models.pkl` | Yes | Per node+signal |
| **I/O Network Ensemble** | Manifest-based | `io_net_models.pkl` | Yes | Per node+signal |
| **Unified Correlated** | Individual files | `unified_correlated_{node}_forecast.pkl` | No | Per node (all 6 metrics) |
| **Classification** | Individual files | `isolation_forest_anomaly.pkl` | No | Per-cluster |

---

## Training Flow Diagrams

### Complete Training Flow

```mermaid
flowchart TD
    Start([Start: --training]) --> Fetch[Fetch Host/Pod Data<br/>START_HOURS_AGO, STEP]
    
    Fetch --> Alias[Alias Resolution<br/>& Entity Canonicalization]
    Alias --> ClusterID[Identify Clusters<br/>LOOKBACK_HOURS]
    ClusterID --> GroupNodes[Group Nodes by Cluster<br/>K8s / Standalone / Unknown]
    
    GroupNodes --> K8sTrain[Train K8s Cluster Models<br/>Per Cluster: Host+Pod Ensemble]
    K8sTrain --> K8sHost[Train K8s Host Model<br/>For Divergence Calculation]
    K8sHost --> UnknownTrain{Unknown Cluster<br/>Nodes?}
    UnknownTrain -->|Yes| UnknownModel[Train Unknown Cluster Model]
    UnknownTrain -->|No| StandaloneCheck{Standalone<br/>Nodes?}
    UnknownModel --> StandaloneCheck
    StandaloneCheck -->|Yes| StandaloneModel[Train Standalone Model<br/>Host Only]
    StandaloneCheck -->|No| Divergence[Calculate Divergence<br/>K8s Host vs Combined]
    StandaloneModel --> Divergence
    
    Divergence --> ClassTrain[Train Classification Model<br/>IsolationForest]
    ClassTrain --> FetchDisk[Fetch Disk Data<br/>Last 30 days]
    FetchDisk --> DiskTrain[Train Disk Models<br/>Per Disk: Linear + Prophet]
    DiskTrain --> GoldenAnomaly[Golden Anomaly Detection<br/>Root-Cause Analysis]
    GoldenAnomaly --> FetchIO[Fetch I/O & Network Data<br/>Last 30-35 days]
    FetchIO --> IOCrisis[Train I/O Network Crisis Models<br/>Prophet per Node/Signal]
    IOCrisis --> IOEnsemble[Train I/O Network Ensemble<br/>Prophet + ARIMA + LSTM + LightGBM]
    
    K8sTrain --> K8sSplit[Train/Test Split<br/>TRAIN_FRACTION]
    K8sSplit --> K8sProphet[Train Prophet<br/>HORIZON_MIN]
    K8sSplit --> K8sARIMA[Train ARIMA<br/>HORIZON_MIN]
    K8sSplit --> K8sLSTM[Train LSTM<br/>LSTM_SEQ_LEN, LSTM_EPOCHS]
    K8sSplit --> K8sLightGBM[Train LightGBM<br/>Early Stopping]
    K8sProphet --> K8sEnsemble[Create Ensemble]
    K8sARIMA --> K8sEnsemble
    K8sLSTM --> K8sEnsemble
    K8sLightGBM --> K8sEnsemble
    K8sEnsemble --> K8sBacktest[Backtest on Test Set]
    K8sBacktest --> K8sSave[Save Models]
    
    DiskTrain --> DiskSplit[Train/Test Split<br/>TRAIN_FRACTION]
    DiskSplit --> DiskLinear[Train Linear Trend]
    DiskSplit --> DiskProphet[Train Prophet<br/>horizon_days]
    DiskLinear --> DiskETA[Calculate ETA to 90%]
    DiskProphet --> DiskETA
    DiskETA --> DiskSave[Save to Manifest]
    
    IOCrisis --> IOSplit[Train/Test Split<br/>TRAIN_FRACTION]
    IOSplit --> IOCrisisProphet[Train Crisis Prophet]
    IOCrisisProphet --> IOCrisisSave[Save to Manifest]
    
    IOEnsemble --> IOESplit[Train/Test Split<br/>TRAIN_FRACTION]
    IOESplit --> IOEProphet[Train Prophet<br/>HORIZON_MIN]
    IOESplit --> IOEARIMA[Train ARIMA<br/>HORIZON_MIN<br/>Skipped for Unified]
    IOESplit --> IOELSTM[Train LSTM<br/>LSTM_SEQ_LEN, LSTM_EPOCHS]
    IOESplit --> IOELightGBM[Train LightGBM<br/>Early Stopping]
    IOEProphet --> IOEEnsemble[Create Ensemble]
    IOEARIMA --> IOEEnsemble
    IOELSTM --> IOEEnsemble
    IOELightGBM --> IOEEnsemble
    IOEEnsemble --> IOEBacktest[Backtest on Test Set]
    IOEBacktest --> IOESave[Save to Manifest]
    
    ClassTrain --> ClassFeat[Extract Features<br/>LOOKBACK_HOURS<br/>Temporal-Aware if 3+ months data]
    ClassFeat --> ClassISO[Train IsolationForest<br/>CONTAMINATION<br/>Uses temporal features if available]
    ClassISO --> ClassSave[Save Models]
    
    K8sSave --> AllDone[All Models Trained]
    DiskSave --> AllDone
    IOCrisisSave --> AllDone
    IOESave --> AllDone
    ClassSave --> AllDone
    
    AllDone --> GeneratePlots[Generate Plots]
    GeneratePlots --> End([End: Models Saved])
    
    classDef startEnd fill:#0078D4,stroke:#005A9E,stroke-width:4px,color:#fff
    classDef dataFetch fill:#107C10,stroke:#0B5A0B,stroke-width:3px,color:#fff
    classDef dataProcess fill:#00B7C3,stroke:#00929A,stroke-width:2px,color:#fff
    classDef modelTrain fill:#FF8C00,stroke:#CC7000,stroke-width:3px,color:#fff
    classDef decision fill:#6B6B6B,stroke:#555555,stroke-width:3px,color:#fff
    classDef prophet fill:#0078D4,stroke:#005A9E,stroke-width:2px,color:#fff
    classDef arima fill:#00B7C3,stroke:#00929A,stroke-width:2px,color:#fff
    classDef lstm fill:#5C2D91,stroke:#4A2473,stroke-width:2px,color:#fff
    classDef ensemble fill:#8764B8,stroke:#6B4F93,stroke-width:3px,color:#fff
    classDef save fill:#E81123,stroke:#BA0E1C,stroke-width:3px,color:#fff
    classDef output fill:#FF8C00,stroke:#CC7000,stroke-width:3px,color:#fff
    
    class Start,End startEnd
    class Fetch,FetchDisk,FetchIO dataFetch
    class Alias,ClusterID,GroupNodes,Divergence,GoldenAnomaly dataProcess
    class K8sTrain,K8sHost,UnknownModel,StandaloneModel,ClassTrain,DiskTrain,IOCrisis,IOEnsemble,K8sSplit,DiskSplit,IOSplit,IOESplit modelTrain
    class UnknownTrain,StandaloneCheck decision
    class K8sProphet,DiskProphet,IOCrisisProphet,IOEProphet prophet
    class K8sARIMA,IOEARIMA arima
    class K8sLSTM,IOELSTM lstm
    class K8sLightGBM,IOELightGBM lightgbm
    class K8sEnsemble,DiskETA,IOEEnsemble ensemble
    class K8sSave,DiskSave,IOCrisisSave,IOESave,ClassSave save
    class K8sBacktest,IOEBacktest,GeneratePlots output
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
    TrainSet --> LightGBM[Train LightGBM<br/>Early Stopping<br/>LIGHTGBM_ENABLED]
    
    Prophet --> PForecast[Prophet Forecast<br/>HORIZON_MIN steps]
    ARIMA --> AForecast[ARIMA Forecast<br/>HORIZON_MIN steps]
    LSTM --> LForecast[LSTM Forecast<br/>HORIZON_MIN steps]
    LightGBM --> LGForecast[LightGBM Forecast<br/>HORIZON_MIN steps]
    
    PForecast --> Ensemble[Create Ensemble<br/>Average of active models]
    AForecast --> Ensemble
    LForecast --> Ensemble
    LGForecast --> Ensemble
    
    Ensemble --> Save[Save Models<br/>prophet, arima, lstm, lightgbm, ensemble]
    
    TestSet --> Backtest[Backtest on Test Set]
    Backtest --> Metrics[Calculate MAE, RMSE]
    Metrics --> Save
    
    Save --> End([End])
    
    classDef startEnd fill:#0078D4,stroke:#005A9E,stroke-width:4px,color:#fff
    classDef dataPrep fill:#107C10,stroke:#0B5A0B,stroke-width:2px,color:#fff
    classDef split fill:#00B7C3,stroke:#00929A,stroke-width:2px,color:#fff
    classDef trainSet fill:#FF8C00,stroke:#CC7000,stroke-width:2px,color:#fff
    classDef testSet fill:#8764B8,stroke:#6B4F93,stroke-width:2px,color:#fff
    classDef prophet fill:#0078D4,stroke:#005A9E,stroke-width:2px,color:#fff
    classDef arima fill:#00B7C3,stroke:#00929A,stroke-width:2px,color:#fff
    classDef lstm fill:#5C2D91,stroke:#4A2473,stroke-width:2px,color:#fff
    classDef lightgbm fill:#28A745,stroke:#1E7E34,stroke-width:2px,color:#fff
    classDef ensemble fill:#8764B8,stroke:#6B4F93,stroke-width:3px,color:#fff
    classDef save fill:#E81123,stroke:#BA0E1C,stroke-width:3px,color:#fff
    classDef backtest fill:#FF8C00,stroke:#CC7000,stroke-width:2px,color:#fff
    
    class Start,End startEnd
    class DataPrep dataPrep
    class Split split
    class TrainSet trainSet
    class TestSet,Backtest,Metrics testSet
    class Prophet,PForecast prophet
    class ARIMA,AForecast arima
    class LSTM,LForecast lstm
    class LightGBM,LGForecast lightgbm
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
    
    Linear --> LinearETA[Calculate Days to 90%<br/>Linear Trend]
    Prophet --> ProphetETA[Calculate Days to 90%<br/>Prophet Forecast]
    
    LinearETA --> EnsembleETA[Ensemble ETA<br/>min of both]
    ProphetETA --> EnsembleETA
    
    EnsembleETA --> Severity{Determine Severity}
    Severity -->|ETA <= 0 days| Critical[CRITICAL]
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
    
    classDef startEnd fill:#0078D4,stroke:#005A9E,stroke-width:4px,color:#fff
    classDef process fill:#107C10,stroke:#0B5A0B,stroke-width:2px,color:#fff
    classDef decision fill:#6B6B6B,stroke:#555555,stroke-width:3px,color:#fff
    classDef prophet fill:#0078D4,stroke:#005A9E,stroke-width:2px,color:#fff
    classDef linear fill:#00B7C3,stroke:#00929A,stroke-width:2px,color:#fff
    classDef ensemble fill:#8764B8,stroke:#6B4F93,stroke-width:3px,color:#fff
    classDef critical fill:#E81123,stroke:#BA0E1C,stroke-width:3px,color:#fff
    classDef warning fill:#FF8C00,stroke:#CC7000,stroke-width:2px,color:#fff
    classDef soon fill:#107C10,stroke:#0B5A0B,stroke-width:2px,color:#fff
    classDef ok fill:#00B7C3,stroke:#00929A,stroke-width:2px,color:#fff
    classDef save fill:#5C2D91,stroke:#4A2473,stroke-width:3px,color:#fff
    
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
    Start([Start: --forecast]) --> Fetch[Fetch Host/Pod Data<br/>START_HOURS_AGO, STEP]
    
    Fetch --> Alias[Alias Resolution<br/>& Entity Canonicalization]
    Alias --> ClusterID[Identify Clusters<br/>LOOKBACK_HOURS]
    ClusterID --> GroupNodes[Group Nodes by Cluster]
    
    GroupNodes --> LoadHP[Load Cached Host/Pod Models]
    LoadHP --> UpdateHP[Update Host/Pod Models<br/>Minimal Updates]
    UpdateHP --> HPProphet[Prophet: Last 7 days]
    UpdateHP --> HPARIMA[ARIMA: Latest data]
    UpdateHP --> HPLSTM[LSTM: Last 2 days, 2 epochs]
    UpdateHP --> HPLightGBM[LightGBM: Latest data<br/>Early Stopping]
    HPProphet --> HPForecast[Generate Forecast<br/>HORIZON_MIN]
    HPARIMA --> HPForecast
    HPLSTM --> HPForecast
    HPLightGBM --> HPForecast
    
    HPForecast --> UpdateClass[Update Classification Model]
    UpdateClass --> ClassFeat[Extract Features<br/>LOOKBACK_HOURS<br/>Temporal-Aware if 3+ months]
    ClassFeat --> ClassPredict[Predict Anomalies<br/>CONTAMINATION<br/>Uses temporal features]
    
    ClassPredict --> FetchDisk[Fetch Disk Data<br/>Last 30 days]
    FetchDisk --> LoadDisk[Load Disk Manifest]
    LoadDisk --> UpdateDisk[Update Disk Models<br/>Per Disk: Minimal Updates]
    UpdateDisk --> DiskProphet[Prophet: Last 7 days]
    UpdateDisk --> DiskLinear[Recalculate Linear Trend]
    DiskProphet --> DiskForecast[Calculate ETA to 90%]
    DiskLinear --> DiskForecast
    
    DiskForecast --> GoldenAnomaly[Golden Anomaly Detection<br/>Root-Cause Analysis]
    
    GoldenAnomaly --> FetchIO[Fetch I/O & Network Data<br/>Last 30-35 days]
    FetchIO --> LoadIO[Load I/O Network Manifest]
    LoadIO --> UpdateIOCrisis[Update I/O Crisis Models<br/>Prophet: Last 7 days]
    UpdateIOCrisis --> IOCheck[Check Crisis Threshold]
    
    IOCheck --> UpdateIOEnsemble[Update I/O Ensemble Models<br/>Prophet + ARIMA + LSTM + LightGBM]
    UpdateIOEnsemble --> IOEProphet[Prophet: Last 7 days]
    UpdateIOEnsemble --> IOEARIMA[ARIMA: Latest data<br/>Skipped for Unified]
    UpdateIOEnsemble --> IOELSTM[LSTM: Last 2 days, 2 epochs]
    UpdateIOEnsemble --> IOELightGBM[LightGBM: Latest data<br/>Early Stopping]
    IOEProphet --> IOForecast[Generate Forecast<br/>HORIZON_MIN]
    IOEARIMA --> IOForecast
    IOELSTM --> IOForecast
    IOELightGBM --> IOForecast
    IOForecast --> IOAnomaly[Detect Anomalies<br/>Temporal-Aware Baseline<br/>Same-time patterns]
    
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
    
    classDef startEnd fill:#0078D4,stroke:#005A9E,stroke-width:4px,color:#fff
    classDef dataFetch fill:#107C10,stroke:#0B5A0B,stroke-width:3px,color:#fff
    classDef dataProcess fill:#00B7C3,stroke:#00929A,stroke-width:2px,color:#fff
    classDef load fill:#00B7C3,stroke:#00929A,stroke-width:3px,color:#fff
    classDef update fill:#FF8C00,stroke:#CC7000,stroke-width:2px,color:#fff
    classDef prophet fill:#0078D4,stroke:#005A9E,stroke-width:2px,color:#fff
    classDef arima fill:#00B7C3,stroke:#00929A,stroke-width:2px,color:#fff
    classDef lstm fill:#5C2D91,stroke:#4A2473,stroke-width:2px,color:#fff
    classDef forecast fill:#8764B8,stroke:#6B4F93,stroke-width:3px,color:#fff
    classDef anomaly fill:#E81123,stroke:#BA0E1C,stroke-width:2px,color:#fff
    classDef aggregate fill:#FF8C00,stroke:#CC7000,stroke-width:3px,color:#fff
    classDef output fill:#107C10,stroke:#0B5A0B,stroke-width:3px,color:#fff
    
    class Start,End startEnd
    class Fetch,FetchDisk,FetchIO dataFetch
    class Alias,ClusterID,GroupNodes,GoldenAnomaly dataProcess
    class LoadHP,LoadDisk,LoadIO load
    class UpdateHP,UpdateDisk,UpdateIOCrisis,UpdateIOEnsemble,UpdateClass update
    class HPProphet,DiskProphet,IOCrisis,IOEProphet prophet
    class HPARIMA,IOEARIMA arima
    class HPLSTM,IOELSTM lstm
    class HPLightGBM,IOELightGBM lightgbm
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
    
    subgraph "LightGBM Minimal Update"
        LG1[Load Saved<br/>Model] --> LG2[Refit on Latest Data<br/>Early Stopping]
        LG2 --> LG3[Generate Forecast<br/>HORIZON_MIN]
    end
    
    P4 --> Ensemble[Combine Forecasts<br/>Only Active Models]
    A3 --> Ensemble
    L3 --> Ensemble
    LG3 --> Ensemble
    
    Ensemble --> Save[Save Updated Models]
    
    classDef load fill:#00B7C3,stroke:#00929A,stroke-width:2px,color:#fff
    classDef process fill:#FF8C00,stroke:#CC7000,stroke-width:2px,color:#fff
    classDef prophet fill:#0078D4,stroke:#005A9E,stroke-width:2px,color:#fff
    classDef arima fill:#00B7C3,stroke:#00929A,stroke-width:2px,color:#fff
    classDef lstm fill:#5C2D91,stroke:#4A2473,stroke-width:2px,color:#fff
    classDef forecast fill:#8764B8,stroke:#6B4F93,stroke-width:2px,color:#fff
    classDef ensemble fill:#8764B8,stroke:#6B4F93,stroke-width:3px,color:#fff
    classDef save fill:#E81123,stroke:#BA0E1C,stroke-width:3px,color:#fff
    
    class P1,A1,L1,LG1 load
    class P2,P3,A2,L2,LG2 process
    class P1,P2,P3,P4 prophet
    class A1,A2,A3 arima
    class L1,L2,L3 lstm
    class LG1,LG2,LG3 lightgbm
    class P4,A3,L3,LG3 forecast
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
    
    classDef trainingData fill:#107C10,stroke:#0B5A0B,stroke-width:2px,color:#fff
    classDef trainingProcess fill:#FF8C00,stroke:#CC7000,stroke-width:2px,color:#fff
    classDef trainingSave fill:#E81123,stroke:#BA0E1C,stroke-width:3px,color:#fff
    classDef predictionData fill:#0078D4,stroke:#005A9E,stroke-width:2px,color:#fff
    classDef predictionProcess fill:#00B7C3,stroke:#00929A,stroke-width:2px,color:#fff
    classDef predictionOutput fill:#8764B8,stroke:#6B4F93,stroke-width:2px,color:#fff
    classDef predictionSave fill:#5C2D91,stroke:#4A2473,stroke-width:3px,color:#fff
    
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
        LIGHTGBM[LIGHTGBM_ENABLED<br/>True] --> Models
    end
    
    subgraph "Anomaly Detection Variables"
        LOOKBACK[LOOKBACK_HOURS<br/>24 hours] --> Anomaly[Feature Extraction<br/>Temporal-Aware if 3+ months]
        CONTAM[CONTAMINATION<br/>0.12] --> Anomaly
    end
    
    subgraph "CLI Overrides"
        CLI_HORIZON[--forecast-horizon<br/>realtime|neartime|future] --> Models
        CLI_PARALLEL[--parallel N] --> Models
    end
    
    Fetch --> Split
    Split --> Models
    Models --> Forecast[Forecast Generation]
    Anomaly --> AnomalyDet[Anomaly Detection]
    
    Forecast --> Output[Output: Forecasts, Crises, Anomalies]
    AnomalyDet --> Output
    
    classDef dataVar fill:#107C10,stroke:#0B5A0B,stroke-width:2px,color:#fff
    classDef trainVar fill:#FF8C00,stroke:#CC7000,stroke-width:2px,color:#fff
    classDef anomalyVar fill:#E81123,stroke:#BA0E1C,stroke-width:2px,color:#fff
    classDef process fill:#00B7C3,stroke:#00929A,stroke-width:3px,color:#fff
    classDef output fill:#8764B8,stroke:#6B4F93,stroke-width:3px,color:#fff
    
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
| **HORIZON_MIN** | 15 | ✅ Critical | ❌ N/A | ❌ N/A | ✅ Critical | ❌ N/A | Forecast length in minutes (can be overridden with `--forecast-horizon`) |
| **LSTM_SEQ_LEN** | 60 | ✅ High | ❌ N/A | ❌ N/A | ✅ High | ❌ N/A | Input sequence length for LSTM |
| **LSTM_EPOCHS** | 10 | ✅ Medium | ❌ N/A | ❌ N/A | ✅ Medium | ❌ N/A | Training iterations for LSTM |
| **LIGHTGBM_ENABLED** | True | ✅ High | ❌ N/A | ❌ N/A | ✅ High | ❌ N/A | Enable/disable LightGBM (enabled by default) |
| **LOOKBACK_HOURS** | 24 | ❌ N/A | ❌ N/A | ❌ N/A | ❌ N/A | ✅ Critical | Feature extraction window (temporal-aware if 3+ months data) |
| **CONTAMINATION** | 0.12 | ❌ N/A | ❌ N/A | ❌ N/A | ❌ N/A | ✅ Critical | Expected anomaly rate |
| **--forecast-horizon** | N/A | ✅ Override | ❌ N/A | ❌ N/A | ✅ Override | ❌ N/A | CLI flag: realtime=15min, neartime=3h, future=7d |
| **--parallel** | N/A | ✅ Override | ✅ Override | ✅ Override | ✅ Override | ❌ N/A | CLI flag: Override CPU detection, bypasses thresholds |

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
    
    classDef critical fill:#E81123,stroke:#BA0E1C,stroke-width:3px,color:#fff
    classDef high fill:#FF8C00,stroke:#CC7000,stroke-width:2px,color:#fff
    classDef medium fill:#107C10,stroke:#0B5A0B,stroke-width:2px,color:#fff
    
    class HP1,IO1,C1,C2 critical
    class HP2,HP3,HP4,D1,D2,IO2,IO3,IO4,C3 high
    class HP5,HP6,D3,IO5,C4 medium
```

---

## Model Execution Order

### Normal Execution Flow (Training Mode)

```mermaid
sequenceDiagram
    participant User
    participant System
    participant Prometheus
    participant Models
    participant Storage
    
    User->>+System: Run metrics.py --training
    System->>+Prometheus: Fetch Host/Pod Data (START_HOURS_AGO, STEP)
    Prometheus-->>-System: Return Data
    
    System->>System: Alias Resolution & Canonicalization
    System->>System: Identify Clusters (LOOKBACK_HOURS)
    System->>System: Group Nodes by Cluster
    
    System->>+Models: Train K8s Cluster Models (per cluster)
    Models->>Storage: Save Models
    Models-->>-System: Models Ready
    
    System->>+Models: Train K8s Host Model (for divergence)
    Models->>Storage: Save Models
    Models-->>-System: Model Ready
    
    System->>+Models: Train Unknown Cluster Model (if needed)
    Models->>Storage: Save Models
    Models-->>-System: Model Ready
    
    System->>+Models: Train Standalone Model (if needed)
    Models->>Storage: Save Models
    Models-->>-System: Model Ready
    
    System->>System: Calculate Divergence
    
    System->>+Models: Train Classification Model
    Models->>Storage: Save Models
    Models-->>-System: Model Ready
    
    System->>+Prometheus: Fetch Disk Data (Last 30 days)
    Prometheus-->>-System: Return Data
    System->>+Models: Train Disk Models (per disk)
    Models->>Storage: Save to Manifest
    Models-->>-System: Models Ready
    
    System->>System: Golden Anomaly Detection
    
    System->>+Prometheus: Fetch I/O & Network Data (Last 30-35 days)
    Prometheus-->>-System: Return Data
    System->>+Models: Train I/O Crisis Models
    Models->>Storage: Save to Manifest
    Models-->>-System: Models Ready
    
    System->>+Models: Train I/O Ensemble Models
    Models->>Storage: Save to Manifest
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
| **CPU/Memory forecasting** | Host/Pod Ensemble | Combines 4 algorithms (Prophet, ARIMA, LSTM, LightGBM) for robust predictions |
| **Disk capacity planning** | Disk Full | Predicts when disk will reach 90% |
| **I/O performance crisis** | I/O Network Crisis | Fast detection of I/O bottlenecks |
| **I/O/Network forecasting** | I/O Network Ensemble | Full ensemble for accurate predictions |
| **Multi-metric system health** | Unified Correlated | Single model using all 6 metrics with cross-correlations |
| **Anomaly detection** | Classification | Identifies unusual resource patterns |

### Performance Characteristics

| Model Type | Training Time | Prediction Time | Accuracy | Use Case |
|------------|--------------|-----------------|----------|----------|
| **Host/Pod Ensemble** | ~2-5 min | ~1-2 sec | High | Production forecasting |
| **Disk Full** | ~30 sec | ~0.5 sec | Medium-High | Capacity planning |
| **I/O Network Crisis** | ~20 sec | ~0.3 sec | Medium | Quick crisis detection |
| **I/O Network Ensemble** | ~1-3 min | ~1 sec | High | Detailed I/O analysis |
| **Unified Correlated** | ~2-4 min | ~1-2 sec | High | Multi-metric system health |
| **Classification** | ~10 sec | ~0.1 sec | Medium | Anomaly detection |

---

## See Also

- `MODEL_TRAINING_AND_PREDICTION_GUIDE.md` - Detailed step-by-step guide
- `CONFIGURATION_VARIABLES.md` - Variable explanations
- `../README.md` - General documentation
- `SYSTEM_DOCUMENTATION.md` - System architecture

