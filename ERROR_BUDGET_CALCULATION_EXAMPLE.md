# Error Budget Calculation: Step-by-Step Example

## Real Example from Your Output: NODE HEALTH

Let's walk through the exact calculation for your NODE HEALTH SLI.

---

## Step 1: Gather the Data

From your output:
- **Current SLI**: 66.67% (4 healthy nodes out of 6)
- **SLO Target**: 99.95%
- **Total Nodes**: 6
- **Unhealthy Nodes**: 2 (host02 and pi)

---

## Step 2: Calculate Compliance

### Code Logic:
```python
def calculate_slo_compliance(sli_value, slo_target):
    target_ratio = slo_target / 100.0  # 99.95 / 100 = 0.9995
    return 100.0 if sli_value >= target_ratio else 0.0
```

### Your Calculation:
```
SLI Value = 0.6667 (66.67% as decimal)
SLO Target Ratio = 99.95 / 100 = 0.9995

Is 0.6667 >= 0.9995?
No → Compliance = 0.0%
```

**Result**: Compliance = **0.0%** (not meeting target)

---

## Step 3: Calculate Total Error Budget

### Formula:
```
Total Budget = 100% - SLO Target
```

### Your Calculation:
```
Total Budget = 100% - 99.95% = 0.05%
```

**Result**: Total Budget = **0.05%**

### What This Means in Real Time:
```
0.05% = 0.0005 (as decimal)
0.0005 × 30 days × 24 hours × 60 minutes
= 0.0005 × 43,200 minutes
= 21.6 minutes
```

You have **21.6 minutes** of "unhealthy time" allowed in a 30-day window.

---

## Step 4: Calculate Budget Consumed

### Formula:
```
Budget Consumed = 100% - Compliance %
```

### Your Calculation:
```
Budget Consumed = 100% - 0% = 100%
```

**Result**: Budget Consumed = **100%** (all budget used)

---

## Step 5: Calculate Budget Remaining

### Formula:
```
Budget Remaining = Total Budget - Budget Consumed
```

### Your Calculation:
```
Budget Remaining = 0.05% - 100% = -99.95%
```

Since we can't have negative budget, it's clamped:
```
Budget Remaining = max(0.0, -99.95%) = 0.00%
```

**Result**: Budget Remaining = **0.00%** (exhausted)

---

## Step 6: Check Alert Threshold

### Code Logic:
```python
error_budget_threshold = 20  # From config
budget_percent_remaining = (budget_remaining / total_budget) * 100
budget_at_risk = budget_percent_remaining < error_budget_threshold
```

### Your Calculation:
```
Total Budget = 0.05%
Budget Remaining = 0.00%

Budget Percent Remaining = (0.00% / 0.05%) × 100 = 0%

Is 0% < 20%?
Yes → Budget at Risk = True
```

**Result**: **⚠️ AT RISK** (below 20% threshold)

---

## Complete Calculation Summary

```
┌─────────────────────────────────────────────────┐
│ NODE HEALTH SLI Calculation                     │
├─────────────────────────────────────────────────┤
│ Input:                                           │
│   • Current SLI: 66.67% (4/6 nodes healthy)   │
│   • SLO Target: 99.95%                          │
│   • Alert Threshold: 20%                        │
├─────────────────────────────────────────────────┤
│ Step 1: Compliance                              │
│   66.67% < 99.95%? → Compliance = 0.0%         │
├─────────────────────────────────────────────────┤
│ Step 2: Total Budget                            │
│   100% - 99.95% = 0.05%                         │
│   (21.6 minutes in 30 days)                     │
├─────────────────────────────────────────────────┤
│ Step 3: Budget Consumed                         │
│   100% - 0% = 100%                              │
├─────────────────────────────────────────────────┤
│ Step 4: Budget Remaining                        │
│   0.05% - 100% = -99.95% → Clamped to 0.00%    │
├─────────────────────────────────────────────────┤
│ Step 5: Alert Check                             │
│   (0.00% / 0.05%) × 100 = 0%                    │
│   0% < 20%? → AT RISK = True                    │
├─────────────────────────────────────────────────┤
│ Output:                                          │
│   • Compliance: 0.00% ✗ NON-COMPLIANT          │
│   • Budget Remaining: 0.00% ⚠️ AT RISK          │
│   • Alert: P0 (Critical)                        │
└─────────────────────────────────────────────────┘
```

---

## Comparison: DISK AVAILABILITY (Healthy Example)

Let's compare with a healthy SLI:

### Input:
- Current SLI: 100.00% (all disks < 90% usage)
- SLO Target: 99.9%

### Calculation:
```
Step 1: Compliance
  100% >= 99.9%? → Compliance = 100.0% ✓

Step 2: Total Budget
  100% - 99.9% = 0.1%

Step 3: Budget Consumed
  100% - 100% = 0%

Step 4: Budget Remaining
  0.1% - 0% = 0.1% ✓

Step 5: Alert Check
  (0.1% / 0.1%) × 100 = 100%
  100% < 20%? → AT RISK = False ✓
```

**Result**: **✓ OK** (all budget intact)

---

## Why the Numbers Seem Small

You might wonder: "Why is 0.05% so small?"

### The Math:
- **99.95% SLO** means you must be healthy **99.95% of the time**
- That leaves only **0.05%** for failures
- In a 30-day window: **0.05% = 21.6 minutes**

### Real-World Context:
- **99.95% uptime** = "Five 9s" = Very strict SLO
- **99.9% uptime** = "Three 9s" = Still strict, but more lenient
- **99.5% uptime** = "Two 9s" = More reasonable for many services

### Your SLOs:
| SLI | SLO Target | Error Budget | Time in 30 Days |
|-----|------------|--------------|-----------------|
| Node Health | 99.95% | 0.05% | 21.6 minutes |
| Disk Availability | 99.9% | 0.1% | 43.2 minutes |
| I/O Performance | 99.5% | 0.5% | 3.6 hours |
| Network Performance | 99.9% | 0.1% | 43.2 minutes |

---

## Recovery Scenario

### Current State:
- SLI: 66.67% (4/6 healthy)
- Budget: 0.00% (exhausted)

### After Fixing 2 Nodes:
- SLI: 100% (6/6 healthy)
- Compliance: 100% (meets 99.95% target)
- Budget Consumed: 0%
- Budget Remaining: 0.05% ✓

**Note**: In a production system with time-windowed tracking, budget recovery would be gradual over the 30-day window, not instantaneous.

---

## Key Takeaways

1. **Error Budget = Safety Margin**: The amount of "failure" you can tolerate
2. **Small Numbers = Strict SLOs**: 99.95% SLO = only 0.05% error budget
3. **Binary Compliance**: Current implementation is all-or-nothing (100% or 0%)
4. **Budget Exhaustion**: When budget = 0%, you're violating your SLO
5. **Alert Threshold**: System warns when budget drops below 20%

The error budget is your **guardrail** - when it's gone, you need to stop making changes and fix reliability issues.

