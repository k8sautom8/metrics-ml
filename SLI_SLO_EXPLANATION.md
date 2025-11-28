# SLI/SLO Output Explanation

## Understanding the SLI/SLO Status Report

### What You're Seeing

The SLI/SLO framework now tracks **8 different Service Level Indicators**:

1. **NODE HEALTH** - Percentage of nodes without critical anomalies
2. **DISK AVAILABILITY** - Percentage of disks with usage < 90%
3. **HOST CPU PERFORMANCE** - Percentage of nodes with host CPU < 80%
4. **HOST MEMORY PERFORMANCE** - Percentage of nodes with host memory < 85%
5. **POD CPU PERFORMANCE** - Percentage of nodes with pod CPU < 80%
6. **POD MEMORY PERFORMANCE** - Percentage of nodes with pod memory < 85%
7. **I/O PERFORMANCE** - Percentage of nodes without I/O crisis/anomalies
8. **NETWORK PERFORMANCE** - Percentage of nodes without network crisis/anomalies

### Example Output Breakdown

```
NODE HEALTH
  Description: Percentage of nodes without critical anomalies
  Current SLI: 66.67%
  SLO Target: 99.95%
  Compliance: 0.00% ✗ NON-COMPLIANT
  Error Budget Remaining: 0.00% ⚠️  AT RISK
  ⚠️  ALERT: Error budget below threshold (P0)
```

**What This Means:**
- **Current SLI: 66.67%** = 4 out of 6 nodes are healthy (2 nodes have anomalies)
- **SLO Target: 99.95%** = We want 99.95% of nodes to be healthy
- **Compliance: 0.00%** = Current measurement (66.67%) is below target (99.95%), so compliance is 0%
- **Error Budget Remaining: 0.00%** = All error budget consumed (100% - 99.95% = 0.05% budget, all used)
- **AT RISK** = Error budget is below 20% threshold, triggering P0 alert

### Why Some SLIs Show 100%

**DISK FORECAST ACCURACY** and **ALERT ACCURACY** currently show 100% because:
- They require historical data tracking (not yet implemented)
- The default fallback returns 100% when data is unavailable
- These will be properly implemented in a future update

### Missing SLIs - Now Implemented!

The following SLIs are now calculated from your actual data:

✅ **Host CPU Performance** - From `df_hcpu` (host CPU metrics)
✅ **Host Memory Performance** - From `df_hmem` (host memory metrics)  
✅ **Pod CPU Performance** - From `df_pcpu` (pod CPU metrics)
✅ **Pod Memory Performance** - From `df_pmem` (pod memory metrics)
✅ **I/O Performance** - From `crisis_df` and `anomaly_df` (I/O issues)
✅ **Network Performance** - From `crisis_df` and `anomaly_df` (network issues)
✅ **Disk Availability** - From `disk_alerts` (disk usage data)

### How SLI Values Are Calculated

1. **Node Health**: `healthy_nodes / total_nodes`
   - Healthy = nodes without classification anomalies, host pressure, or golden signal issues

2. **Disk Availability**: `disks_below_90% / total_disks`
   - Good = disks with `current_% < 90`

3. **Host/Pod CPU/Memory**: `nodes_below_threshold / total_nodes`
   - Uses latest values from time series data
   - Thresholds: CPU < 80%, Memory < 85%

4. **I/O/Network Performance**: `nodes_without_issues / total_nodes`
   - Calculated from crisis and anomaly detection results

### Error Budget Calculation

```
Error Budget = 100% - SLO Target
Budget Consumed = 100% - Compliance %
Budget Remaining = Total Budget - Budget Consumed
```

**Example:**
- SLO Target: 99.95%
- Error Budget: 0.05% (100% - 99.95%)
- Current Compliance: 0% (SLI below target)
- Budget Consumed: 100% (100% - 0%)
- Budget Remaining: -0.05% → Clamped to 0.00%

### Alert Severity Levels

- **P0** = Critical - Immediate action required (Node Health)
- **P1** = High - Action needed soon (Disk, CPU, Memory, I/O, Network)
- **P2** = Medium - Monitor and investigate (Forecast accuracy, Alert accuracy)

### Next Steps

To see all SLIs in action, run:

```bash
python3 metrics.py --forecast --sli-slo-config sli_slo_config.yaml --interval 0
```

You should now see all 8 SLIs calculated from your actual infrastructure data!


