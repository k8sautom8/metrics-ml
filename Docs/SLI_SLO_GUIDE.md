# SLI/SLO Framework Guide

## Overview

The SLI/SLO (Service Level Indicator/Service Level Objective) framework tracks system reliability metrics and calculates error budgets based on Google SRE best practices.

## Configuration

### YAML Config File

Create a `sli_slo_config.yaml` file (or specify a custom path with `--sli-slo-config`):

```yaml
settings:
  evaluation_window_days: 30
  error_budget_alert_threshold: 20  # Alert when < 20% budget remaining
  sli_calculation_window_minutes: 5

slis:
  - name: disk_availability
    description: "Percentage of time disk usage < 90%"
    type: "availability"
    query_type: "internal"
    aggregation: "mean"
    good_threshold: 0.90
    slo_target: 99.9
    error_budget_percent: 0.1
    alert_severity: "P1"
```

### SLI Types

1. **Internal SLIs** (`query_type: "internal"`):
   - Calculated from anomaly detection results
   - Examples: `node_health`, `disk_availability`
   - No Prometheus query needed

2. **Prometheus SLIs** (`query_type: "prometheus"`):
   - Calculated from Prometheus queries
   - Requires Prometheus query definition
   - Currently not fully implemented (returns None)

## Usage

### Basic Usage

```bash
# Use default config (sli_slo_config.yaml in current directory)
python3 metrics.py --forecast --sli-slo-config

# Use custom config path
python3 metrics.py --forecast --sli-slo-config /path/to/config.yaml

# With alerts
python3 metrics.py --forecast --interval 15 \
  --alert-webhook http://localhost:8080/alerts \
  --sli-slo-config
```

### Output

The SLI/SLO framework outputs:

1. **Console Report**: Shows current SLI values, SLO compliance, and error budget status
2. **Alert Integration**: Error budgets at risk are included in alert summaries
3. **Webhook Payload**: SLI/SLO results included in alert webhook (if configured)

### Example Output

```
================================================================================
SLI/SLO STATUS
================================================================================

DISK AVAILABILITY
  Description: Percentage of time disk usage < 90% across all nodes
  Current SLI: 95.00%
  SLO Target: 99.9%
  Compliance: 0.00% ✗ NON-COMPLIANT
  Error Budget Remaining: 0.10% ⚠️  AT RISK
  ⚠️  ALERT: Error budget below threshold (P1)

NODE HEALTH
  Description: Percentage of nodes without critical anomalies
  Current SLI: 98.50%
  SLO Target: 99.95%
  Compliance: 0.00% ✗ NON-COMPLIANT
  Error Budget Remaining: 1.45% ⚠️  AT RISK
  ⚠️  ALERT: Error budget below threshold (P0)
================================================================================
```

## SLI Definitions

### Pre-configured SLIs

1. **disk_availability**: Percentage of disks below 90% usage
2. **io_performance**: Percentage of time CPU iowait < 30%
3. **network_performance**: Percentage of time network bandwidth < 80%
4. **node_health**: Percentage of nodes without critical anomalies
5. **disk_forecast_accuracy**: Accuracy of disk capacity forecasts
6. **alert_accuracy**: Percentage of alerts that required action

### Adding Custom SLIs

Add new SLIs to `sli_slo_config.yaml`:

```yaml
slis:
  - name: my_custom_sli
    description: "My custom SLI description"
    type: "availability"  # availability, latency, throughput, error_rate
    query_type: "internal"  # or "prometheus"
    aggregation: "mean"
    good_threshold: 0.95
    slo_target: 99.5
    error_budget_percent: 0.5
    alert_severity: "P1"  # P0, P1, P2, P3
```

## Error Budget

### How It Works

- **Error Budget** = 100% - SLO Target
- **Budget Consumed** = 100% - Compliance %
- **Budget Remaining** = Total Budget - Budget Consumed

### Alert Threshold

When error budget remaining falls below the threshold (default: 20%), an alert is generated with the specified severity.

## Integration with Alerts

SLI/SLO results are automatically included in:

1. **Alert Summary**: Shows budget at risk count
2. **Webhook Payload**: Includes `sli_slo` section with results
3. **Console Output**: Full SLI/SLO status report

## Dependencies

- **PyYAML**: Required for YAML parsing
  ```bash
  pip install pyyaml
  ```

If PyYAML is not installed, SLI/SLO tracking is disabled with a warning message.

## Future Enhancements

- [ ] Prometheus query execution for Prometheus-based SLIs
- [ ] Historical SLI tracking and trend analysis
- [ ] SLO compliance reports (daily/weekly/monthly)
- [ ] Error budget burn rate alerts
- [ ] Integration with incident management systems
- [ ] SLI/SLO dashboards

## Troubleshooting

### Config Not Found

```
SLI/SLO config not found at default locations. SLI/SLO tracking disabled.
```

**Solution**: Create `sli_slo_config.yaml` in the current directory or specify path with `--sli-slo-config`.

### PyYAML Not Installed

```
PyYAML not found. SLI/SLO config disabled. Install with: pip install pyyaml
```

**Solution**: Install PyYAML: `pip install pyyaml`

### SLI Returns None

Some SLIs may return `None` if:
- Required data is not available
- Prometheus queries are not yet implemented
- Configuration is incorrect

Check the console output for specific error messages.


