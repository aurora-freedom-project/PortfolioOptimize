# Portfolio Optimization Investment Analysis Report

## Executive Summary

This report presents a comprehensive analysis of the current investment portfolio and its optimized alternative based on modern portfolio theory. The optimization model has identified a significantly improved asset allocation that increases expected annual returns from 13.43% to 16.77% while achieving better risk-adjusted performance metrics. This optimization comes with a moderate increase in portfolio volatility (from 14.81% to 18.55%) but delivers superior Sharpe and Sortino ratios, indicating more efficient risk utilization. Major recommendations include substantial reallocation away from banking stocks toward technology, real estate, and entertainment sectors to maximize portfolio efficiency.

## Current Portfolio Assessment

### Key Performance Metrics

| Metric | Current Portfolio | Optimized Portfolio | Difference |
|--------|------------------|---------------------|------------|
| Expected Annual Return | 13.43% | 16.77% | +3.34% |
| Standard Deviation (Risk) | 14.81% | 18.55% | +3.74% |
| Sharpe Ratio | 0.648 | 0.697 | +0.049 |
| Sortino Ratio | 0.833 | 0.972 | +0.139 |
| Risk-Free Rate | 3.83% | 3.83% | 0.00% |

### Current Portfolio Allocation

The current portfolio consists of 20 Australian Securities Exchange (ASX) stocks with the following allocation:

| Stock | Current Weight | Sector |
|-------|---------------|--------|
| BHP.AX | 11.00% | Mining |
| CBA.AX | 9.50% | Banking |
| CSL.AX | 8.50% | Healthcare |
| WBC.AX | 6.50% | Banking |
| NAB.AX | 6.50% | Banking |
| ANZ.AX | 6.00% | Banking |
| FMG.AX | 5.50% | Mining |
| RIO.AX | 5.00% | Mining |
| MQG.AX | 4.50% | Financial Services |
| WES.AX | 4.00% | Retail |
| WOW.AX | 3.50% | Retail |
| TLS.AX | 3.00% | Telecommunications |
| GMG.AX | 2.50% | Real Estate |
| XRO.AX | 2.50% | Technology |
| ALL.AX | 2.00% | Entertainment |
| COL.AX | 2.00% | Retail |
| MIN.AX | 1.50% | Mining |
| RMD.AX | 1.50% | Healthcare |
| SHL.AX | 1.00% | Healthcare |
| REA.AX | 1.00% | Real Estate |

The current allocation shows significant concentration in banking (28.5%) and mining (22.5%) sectors, creating potential sectoral risk exposure.

## Optimized Portfolio Analysis

### Optimized Portfolio Allocation

The mean-variance optimization algorithm proposes the following improved allocation:

| Stock | Current Weight | Optimized Weight | Change | 
|-------|---------------|------------------|--------|
| ALL.AX | 2.00% | 16.00% | +14.00% |
| FMG.AX | 5.50% | 11.00% | +5.50% |
| CSL.AX | 8.50% | 11.00% | +2.50% |
| GMG.AX | 2.50% | 10.00% | +7.50% |
| MQG.AX | 4.50% | 9.00% | +4.50% |
| REA.AX | 1.00% | 8.00% | +7.00% |
| RMD.AX | 1.50% | 8.00% | +6.50% |
| WES.AX | 4.00% | 7.00% | +3.00% |
| XRO.AX | 2.50% | 6.00% | +3.50% |
| ANZ.AX | 6.00% | 4.00% | -2.00% |
| BHP.AX | 11.00% | 4.00% | -7.00% |
| COL.AX | 2.00% | 3.78% | +1.78% |
| RIO.AX | 5.00% | 0.64% | -4.36% |
| MIN.AX | 1.50% | 0.57% | -0.93% |
| CBA.AX | 9.50% | 1.00% | -8.50% |
| WBC.AX | 6.50% | 0.00% | -6.50% |
| NAB.AX | 6.50% | 0.00% | -6.50% |
| WOW.AX | 3.50% | 0.00% | -3.50% |
| TLS.AX | 3.00% | 0.00% | -3.00% |
| SHL.AX | 1.00% | 0.00% | -1.00% |

### Sectoral Shift Analysis

The optimization model recommends significant reallocation from traditional banking and basic materials sectors toward growth-oriented sectors:

- **Major Increases:**
  - Entertainment (ALL.AX): +14.00%
  - Real Estate (GMG.AX, REA.AX): +14.50% combined
  - Healthcare (CSL.AX, RMD.AX): +9.00% combined
  - Technology (XRO.AX): +3.50%

- **Major Decreases:**
  - Banking sector (CBA.AX, WBC.AX, NAB.AX, ANZ.AX): -23.50% combined
  - Mining sector (BHP.AX, RIO.AX, MIN.AX): -12.29% combined
  - Telecommunications (TLS.AX): -3.00%

This sectoral shift aligns with global trends favoring technology, healthcare, and specialized real estate over traditional banking and commodity stocks.

## Individual Asset Performance Analysis

### Best Performers

| Stock | Annual Return | Volatility | Sharpe Ratio | Sortino Ratio | Max Drawdown |
|-------|--------------|------------|--------------|---------------|--------------|
| FMG.AX | 31.71% | 44.60% | 0.666 | 1.043 | -47.75% |
| ALL.AX | 28.17% | 29.81% | 0.878 | 1.219 | -59.03% |
| XRO.AX | 26.75% | 37.01% | 0.669 | 0.959 | -58.43% |
| GMG.AX | 22.47% | 25.84% | 0.792 | 1.072 | -41.64% |
| REA.AX | 19.17% | 30.04% | 0.571 | 0.796 | -45.80% |

### Weakest Performers

| Stock | Annual Return | Volatility | Sharpe Ratio | Sortino Ratio | Max Drawdown |
|-------|--------------|------------|--------------|---------------|--------------|
| TLS.AX | 0.12% | 19.46% | -0.097 | -0.129 | -52.39% |
| ANZ.AX | 4.56% | 23.53% | 0.109 | 0.141 | -49.64% |
| WBC.AX | 5.33% | 23.28% | 0.143 | 0.186 | -51.74% |
| NAB.AX | 7.40% | 22.33% | 0.242 | 0.312 | -52.23% |
| SHL.AX | 7.14% | 23.08% | 0.223 | 0.305 | -44.51% |

The optimization model has correctly identified and eliminated or reduced exposure to the weakest performers while increasing allocation to assets with superior risk-adjusted returns.

## Correlation Analysis Insights

The asset correlation matrix reveals important portfolio diversification dynamics:

1. **High Correlation Cluster:** Most Australian equity assets show high positive correlations (0.7-0.9) with each other, limiting diversification benefits within the portfolio.

2. **Lower Correlation Assets:**
   - TLS.AX (Telstra) exhibits low or negative correlations with many other assets, providing unique diversification benefits despite its poor standalone performance.
   - WBC.AX (Westpac) demonstrates lower correlations compared to other banking stocks.

3. **Sector Correlations:**
   - Banking stocks (CBA.AX, WBC.AX, NAB.AX, ANZ.AX) are highly correlated with each other (>0.8).
   - Mining stocks (BHP.AX, RIO.AX, FMG.AX) show strong internal correlations (>0.9).
   - Healthcare stocks (CSL.AX, RMD.AX, SHL.AX) exhibit moderate to high correlations (0.8-0.9).

## Strategic Recommendations

Based on the comprehensive portfolio analysis, we recommend the following implementation strategy:

### 1. Core Position Adjustments

- **Significantly Increase (>5%):**
  - ALL.AX (Aristocrat Leisure): Increase from 2.00% to 16.00%
  - GMG.AX (Goodman Group): Increase from 2.50% to 10.00%
  - FMG.AX (Fortescue Metals): Increase from 5.50% to 11.00%
  - REA.AX (REA Group): Increase from 1.00% to 8.00%
  - RMD.AX (ResMed): Increase from 1.50% to 8.00%

- **Significantly Decrease (>5%):**
  - CBA.AX (Commonwealth Bank): Reduce from 9.50% to 1.00%
  - BHP.AX (BHP Group): Reduce from 11.00% to 4.00%
  - WBC.AX (Westpac Banking): Eliminate position (from 6.50% to 0%)
  - NAB.AX (National Australia Bank): Eliminate position (from 6.50% to 0%)

### 2. Sector Rebalancing Strategy

- **Financial Services Transformation:** Shift from traditional banking (CBA, WBC, NAB, ANZ) to diversified financial services (MQG.AX).
- **Resources Consolidation:** Concentrate mining exposure in FMG.AX while reducing BHP.AX and RIO.AX positions.
- **Growth Sector Expansion:** Substantially increase exposure to entertainment (ALL.AX), real estate (GMG.AX, REA.AX), and healthcare (CSL.AX, RMD.AX).

### 3. Implementation Timeline

To minimize market impact and trading costs, we recommend executing this portfolio transformation in three phases over 30-45 days:

- **Phase 1 (Days 1-15):** Complete exit from eliminated positions (WBC.AX, NAB.AX, WOW.AX, TLS.AX, SHL.AX).
- **Phase 2 (Days 16-30):** Reduce overweight positions (BHP.AX, CBA.AX, RIO.AX).
- **Phase 3 (Days 31-45):** Establish target weights in growth positions (ALL.AX, GMG.AX, REA.AX, RMD.AX).

### 4. Risk Management Considerations

The optimized portfolio increases volatility by 3.74 percentage points (from 14.81% to 18.55%). This higher risk profile should be monitored through:

- Setting trailing stop-loss orders on volatile positions (e.g., FMG.AX, XRO.AX)
- Implementing quarterly rebalancing to maintain target weights
- Considering partial hedging through options on the ASX 200 index

## Conclusion

The optimized portfolio represents a significant improvement over the current allocation, with expected annual returns increasing by 3.34 percentage points while maintaining a better risk-adjusted performance profile. The transformation involves a strategic shift from traditional banking and basic materials toward growth-oriented sectors including entertainment, healthcare, and specialized real estate.

While the optimized allocation increases absolute volatility, both the Sharpe ratio (0.697 vs. 0.648) and Sortino ratio (0.972 vs. 0.833) show meaningful improvements, indicating more efficient utilization of risk. 

The implementation of this optimized allocation should be executed methodically over 30-45 days to minimize market impact and transaction costs. Regular monitoring and quarterly rebalancing are recommended to maintain the portfolio's improved risk-return characteristics.

---

*Disclaimer: This analysis is based on historical data and optimization models. Past performance is not indicative of future results. All investments involve risk, including possible loss of principal. This report should not be construed as investment advice but rather as an analytical framework for portfolio consideration.*
