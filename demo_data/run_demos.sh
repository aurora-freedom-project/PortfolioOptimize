#!/bin/bash
# Portfolio Optimizer Demo Script
# Generated automatically

echo 'Running Demo Scenario 1...'
python -m shell --data demo_data/merged_stock_prices.csv --tickers ANZ.AX,CBA.AX,NAB.AX --method mean_variance --start-date 2023-01-01 --end-date 2023-12-31 --allocations ANZ.AX:0.3,CBA.AX:0.4,NAB.AX:0.3 --show-charts

echo 'Running Demo Scenario 2...'
echo 'Risk Parity:'
python -m shell --data demo_data/merged_stock_prices.csv --tickers ANZ.AX,CBA.AX,RIO.AX,WOW.AX,MQG.AX --advanced-method risk_parity --start-date 2023-01-01 --show-charts

echo 'CVaR Optimization:'
python -m shell --data demo_data/merged_stock_prices.csv --tickers ANZ.AX,CBA.AX,NAB.AX,WBC.AX --advanced-method min_cvar --confidence-level 0.05 --start-date 2023-01-01

echo 'Market Neutral:'
python -m shell --data demo_data/merged_stock_prices.csv --tickers CBA.AX,RIO.AX,WOW.AX,BHP.AX,CSL.AX,MQG.AX --advanced-method market_neutral --target-volatility 0.15 --start-date 2023-01-01

echo 'Running Demo Scenario 3...'
python -m shell --data demo_data/merged_stock_prices.csv --run-backtest --tickers ANZ.AX,CBA.AX,RIO.AX,WOW.AX,BHP.AX,MQG.AX --rebalance-frequency M --lookback-window 252 --start-date 2022-06-01 --allocations ANZ.AX:0.15,CBA.AX:0.25,RIO.AX:0.2,WOW.AX:0.15,BHP.AX:0.15,MQG.AX:0.1 --output backtest_results.json

echo 'Running Demo Scenario 4...'
python -m shell --data demo_data/merged_stock_prices.csv --generate-report --client-name ASX Investment Fund --report-type QUARTERLY --tickers ANZ.AX,CBA.AX,RIO.AX,WOW.AX,BHP.AX,CSL.AX,MQG.AX,NAB.AX --start-date 2023-01-01 --allocations ANZ.AX:0.125,CBA.AX:0.2,RIO.AX:0.15,WOW.AX:0.1,BHP.AX:0.15,CSL.AX:0.125,MQG.AX:0.1,NAB.AX:0.125

echo 'Running Demo Scenario 5...'
echo 'Load CSV to Database:'
python -m shell --data demo_data/merged_stock_prices.csv --database-url sqlite:///demo_portfolio.db --update-cache --tickers ANZ.AX,CBA.AX,MQG.AX,NAB.AX,RIO.AX,WOW.AX,BHP.AX,CSL.AX,TLS.AX,WBC.AX --cache-days 730

echo 'Fast Database Retrieval:'
python -m shell --database-url sqlite:///demo_portfolio.db --tickers ANZ.AX,CBA.AX,RIO.AX,WOW.AX,MQG.AX --method mean_variance --start-date 2023-01-01 --show-charts

echo 'Hybrid Database + Real-time:'
python -m shell --data-source database --fallback-sources yahoo,file --database-url sqlite:///demo_portfolio.db --data demo_data/merged_stock_prices.csv --tickers ANZ.AX,CBA.AX,RIO.AX,WOW.AX --method mean_variance

echo 'Running Demo Scenario 6...'
echo 'Real-time Optimization:'
python -m shell --data-source yahoo --real-time --tickers ANZ.AX,CBA.AX,NAB.AX --method mean_variance --show-charts

echo 'Real-time Price Streaming:'
python -m shell --stream-prices --tickers ANZ.AX,CBA.AX,RIO.AX,WOW.AX --stream-interval 30

echo 'Hybrid Historical + Real-time:'
python -m shell --data-source file --data demo_data/merged_stock_prices.csv --fallback-sources yahoo --real-time --tickers ANZ.AX,CBA.AX,NAB.AX --method mean_variance

echo 'Running Demo Scenario 7...'
echo 'Excel File:'
python -m shell --data demo_data/prices.xlsx --tickers ANZ.AX,CBA.AX,NAB.AX --method mean_variance

echo 'Parquet File (Fast):'
python -m shell --data demo_data/prices.parquet --tickers ANZ.AX,CBA.AX,RIO.AX,WOW.AX,MQG.AX --advanced-method risk_parity

