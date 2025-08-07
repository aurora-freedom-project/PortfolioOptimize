#!/usr/bin/env python3
"""
Legacy Migration Script
=======================

Migrate from the old streamlined_optimizer.py to the new institutional platform.
This script helps existing users transition their data, configurations, and workflows.
"""

import json
import pandas as pd
import shutil
from pathlib import Path
from typing import Dict, Any, List
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class LegacyMigrator:
    """Migration tool for transitioning from legacy system"""
    
    def __init__(self, legacy_dir: str = ".", target_dir: str = "portfolio_optimizer_migrated"):
        self.legacy_dir = Path(legacy_dir)
        self.target_dir = Path(target_dir)
        self.migration_report = {
            "files_migrated": [],
            "configurations_created": [],
            "warnings": [],
            "errors": []
        }
    
    def migrate_all(self):
        """Run complete migration process"""
        logger.info("Starting migration from legacy system...")
        
        # Create target directory
        self.target_dir.mkdir(exist_ok=True)
        
        # Migrate data files
        self.migrate_data_files()
        
        # Migrate configurations
        self.migrate_configurations()
        
        # Create new workflow examples
        self.create_workflow_examples()
        
        # Generate migration report
        self.generate_migration_report()
        
        logger.info(f"Migration completed. Files migrated to: {self.target_dir}")
    
    def migrate_data_files(self):
        """Migrate data files to new structure"""
        logger.info("Migrating data files...")
        
        # Create data directory
        data_dir = self.target_dir / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Migrate CSV files
        csv_files = list(self.legacy_dir.glob("*.csv"))
        for csv_file in csv_files:
            target_file = data_dir / csv_file.name
            shutil.copy2(csv_file, target_file)
            self.migration_report["files_migrated"].append({
                "source": str(csv_file),
                "target": str(target_file),
                "type": "data"
            })
            logger.info(f"Migrated: {csv_file.name}")
        
        # Migrate results and charts
        if (self.legacy_dir / "charts").exists():
            charts_dir = self.target_dir / "legacy_charts"
            shutil.copytree(self.legacy_dir / "charts", charts_dir, dirs_exist_ok=True)
            self.migration_report["files_migrated"].append({
                "source": str(self.legacy_dir / "charts"),
                "target": str(charts_dir),
                "type": "charts"
            })
    
    def migrate_configurations(self):
        """Create new configuration files based on legacy usage"""
        logger.info("Creating new configuration files...")
        
        config_dir = self.target_dir / "config"
        config_dir.mkdir(exist_ok=True)
        
        # Analyze legacy usage patterns
        legacy_config = self.analyze_legacy_usage()
        
        # Create main configuration
        main_config = self.create_main_config(legacy_config)
        config_file = config_dir / "main_config.json"
        with open(config_file, 'w') as f:
            json.dump(main_config, f, indent=2)
        
        self.migration_report["configurations_created"].append({
            "file": str(config_file),
            "description": "Main configuration with inferred settings"
        })
        
        # Create constraints configuration if needed
        if legacy_config.get("has_constraints"):
            constraints_config = self.create_constraints_config(legacy_config)
            constraints_file = config_dir / "constraints.json"
            with open(constraints_file, 'w') as f:
                json.dump(constraints_config, f, indent=2)
            
            self.migration_report["configurations_created"].append({
                "file": str(constraints_file),
                "description": "Portfolio constraints configuration"
            })
        
        # Create investor views template if Black-Litterman was used
        if legacy_config.get("used_black_litterman"):
            views_template = self.create_investor_views_template(legacy_config)
            views_file = config_dir / "investor_views_template.json"
            with open(views_file, 'w') as f:
                json.dump(views_template, f, indent=2)
            
            self.migration_report["configurations_created"].append({
                "file": str(views_file),
                "description": "Black-Litterman investor views template"
            })
    
    def analyze_legacy_usage(self) -> Dict[str, Any]:
        """Analyze legacy files to understand usage patterns"""
        analysis = {
            "tickers": [],
            "used_black_litterman": False,
            "used_hrp": False,
            "has_constraints": False,
            "data_files": [],
            "risk_free_rate": 0.02
        }
        
        # Analyze CSV files for tickers
        csv_files = list(self.legacy_dir.glob("*.csv"))
        for csv_file in csv_files:
            if "price" in csv_file.name.lower() or "stock" in csv_file.name.lower():
                try:
                    df = pd.read_csv(csv_file, nrows=1)
                    # Skip date column, get asset tickers
                    tickers = [col for col in df.columns if col.lower() not in ['date', 'datetime', 'timestamp']]
                    analysis["tickers"].extend(tickers)
                    analysis["data_files"].append(csv_file.name)
                except Exception as e:
                    logger.warning(f"Could not analyze {csv_file.name}: {e}")
        
        # Remove duplicates
        analysis["tickers"] = list(set(analysis["tickers"]))
        
        # Check for legacy result files to understand methods used
        result_files = list(self.legacy_dir.glob("*result*.json"))
        for result_file in result_files:
            try:
                with open(result_file, 'r') as f:
                    results = json.load(f)
                
                if "black_litterman" in results:
                    analysis["used_black_litterman"] = True
                
                if "hrp" in results:
                    analysis["used_hrp"] = True
                    
            except Exception as e:
                logger.warning(f"Could not analyze {result_file.name}: {e}")
        
        return analysis
    
    def create_main_config(self, legacy_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create main configuration based on legacy analysis"""
        config = {
            "data_sources": {
                "csv": {
                    "enabled": True,
                    "file_path": f"data/{legacy_analysis['data_files'][0]}" if legacy_analysis['data_files'] else "data/prices.csv",
                    "date_column": "date",
                    "parse_dates": True
                },
                "yahoo_finance": {
                    "enabled": True,
                    "cache_duration_hours": 24,
                    "fallback_sources": ["alpha_vantage"]
                }
            },
            "optimization": {
                "default_method": "mean_variance",
                "risk_free_rate": legacy_analysis.get("risk_free_rate", 0.02),
                "l2_regularization": 0.01,
                "confidence_level": 0.95
            },
            "backtesting": {
                "default_strategies": ["mean_variance"],
                "rebalance_frequency": "M",
                "transaction_costs": 0.001,
                "lookback_window": 252
            },
            "reporting": {
                "default_format": "pdf",
                "include_charts": True,
                "template": "institutional",
                "output_directory": "reports"
            },
            "charts": {
                "default_format": "html",
                "theme": "plotly_white",
                "width": 1200,
                "height": 800
            }
        }
        
        # Add Black-Litterman to strategies if it was used
        if legacy_analysis.get("used_black_litterman"):
            config["backtesting"]["default_strategies"].append("black_litterman")
        
        # Add HRP if it was used
        if legacy_analysis.get("used_hrp"):
            config["backtesting"]["default_strategies"].append("hrp")
        
        return config
    
    def create_constraints_config(self, legacy_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create constraints configuration template"""
        tickers = legacy_analysis.get("tickers", [])
        
        config = {
            "description": "Portfolio constraints configuration",
            "weight_constraints": {},
            "sector_constraints": {},
            "risk_constraints": {
                "max_portfolio_volatility": 0.20,
                "max_tracking_error": 0.05
            }
        }
        
        # Add basic weight constraints for each ticker
        for ticker in tickers[:10]:  # Limit to first 10 tickers
            config["weight_constraints"][ticker] = {
                "min": 0.0,
                "max": 0.3
            }
        
        # Add example sector constraint
        if len(tickers) >= 3:
            config["sector_constraints"]["Technology"] = {
                "min": 0.0,
                "max": 0.6,
                "assets": tickers[:3]
            }
        
        return config
    
    def create_investor_views_template(self, legacy_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create investor views template for Black-Litterman"""
        tickers = legacy_analysis.get("tickers", [])
        
        template = {
            "description": "Investor views template for Black-Litterman optimization",
            "views": {},
            "market_outlook": {
                "equity_risk_premium": 0.06,
                "interest_rate_environment": "neutral",
                "volatility_regime": "moderate"
            }
        }
        
        # Add example views for first few tickers
        example_returns = [0.10, 0.08, 0.12, 0.09, 0.11]
        example_confidences = [0.7, 0.6, 0.8, 0.5, 0.7]
        
        for i, ticker in enumerate(tickers[:5]):
            template["views"][ticker] = {
                "expected_return": example_returns[i % len(example_returns)],
                "confidence": example_confidences[i % len(example_confidences)],
                "rationale": f"Example view for {ticker} - update with your analysis"
            }
        
        return template
    
    def create_workflow_examples(self):
        """Create example workflows and scripts"""
        logger.info("Creating workflow examples...")
        
        examples_dir = self.target_dir / "examples"
        examples_dir.mkdir(exist_ok=True)
        
        # Basic optimization example
        basic_example = self.create_basic_example()
        with open(examples_dir / "basic_optimization.py", 'w') as f:
            f.write(basic_example)
        
        # CLI usage examples
        cli_examples = self.create_cli_examples()
        with open(examples_dir / "cli_usage.md", 'w') as f:
            f.write(cli_examples)
        
        # Migration instructions
        migration_guide = self.create_migration_guide()
        with open(examples_dir / "migration_guide.md", 'w') as f:
            f.write(migration_guide)
        
        self.migration_report["configurations_created"].extend([
            {"file": str(examples_dir / "basic_optimization.py"), "description": "Basic optimization example"},
            {"file": str(examples_dir / "cli_usage.md"), "description": "CLI usage examples"},
            {"file": str(examples_dir / "migration_guide.md"), "description": "Migration guide"}
        ])
    
    def create_basic_example(self) -> str:
        """Create basic optimization example script"""
        return '''#!/usr/bin/env python3
"""
Basic Portfolio Optimization Example
===================================

This example shows how to migrate from the legacy streamlined_optimizer.py
to the new institutional platform.
"""

from portfolio_optimizer import OptimizationEngine, OptimizationConfig, OptimizationMethod
from portfolio_optimizer.core.data_sources import DataSourceManager, DataSourceType
from portfolio_optimizer.core.optimization import ConstraintManager
import pandas as pd
from datetime import datetime, timedelta

def main():
    """Run basic portfolio optimization"""
    print("🚀 Portfolio Optimizer - New Platform Example")
    
    # Setup data manager
    data_manager = DataSourceManager()
    
    # Configure CSV data source (migrated from legacy)
    csv_config = {
        "file_path": "data/merged_stock_prices.csv",  # Your migrated data file
        "date_column": "date",
        "parse_dates": True
    }
    data_manager.configure_source(DataSourceType.CSV, csv_config)
    
    # Define portfolio
    tickers = ['ANZ.AX', 'CBA.AX', 'MQG.AX', 'NAB.AX', 'RIO.AX', 'WOW.AX']
    start_date = datetime.now() - timedelta(days=1000)  # ~3 years
    end_date = datetime.now()
    
    # Load data
    try:
        prices = data_manager.get_market_data(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            source=DataSourceType.CSV
        )
        print(f"✅ Loaded data: {len(prices)} observations for {len(prices.columns)} assets")
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        print("💡 Tip: Make sure your CSV file is in the data/ directory")
        return
    
    # Setup constraints (optional)
    constraints = ConstraintManager()
    # Add maximum 30% weight per asset
    for ticker in tickers:
        constraints.add_weight_constraint(ticker, min_weight=0.0, max_weight=0.3)
    
    # Configure optimization
    config = OptimizationConfig(
        method=OptimizationMethod.MEAN_VARIANCE,
        risk_free_rate=0.02,
        l2_regularization=0.01,  # Institutional-grade regularization
        max_weight=0.3
    )
    
    # Run optimization
    engine = OptimizationEngine(constraint_manager=constraints)
    result = engine.optimize(prices, config)
    
    if result.success:
        print("\\n✅ Optimization completed successfully!")
        print("\\n📊 Optimal Portfolio Weights:")
        for asset, weight in result.weights.items():
            if weight > 0.001:  # Only show significant weights
                print(f"  {asset}: {weight:.1%}")
        
        print("\\n📈 Portfolio Performance:")
        perf = result.performance
        print(f"  Expected Return: {perf['expected_return']:.1%}")
        print(f"  Volatility: {perf['volatility']:.1%}")
        print(f"  Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
        
        print("\\n💡 Next Steps:")
        print("  1. Try different optimization methods: black_litterman, hrp, risk_parity")
        print("  2. Add investor views for Black-Litterman optimization")
        print("  3. Run backtesting: portfolio-optimizer --run-backtest")
        print("  4. Generate reports: portfolio-optimizer --generate-report")
        
    else:
        print(f"❌ Optimization failed: {result.message}")

if __name__ == "__main__":
    main()
'''
    
    def create_cli_examples(self) -> str:
        """Create CLI usage examples"""
        return '''# CLI Usage Examples

## Basic Commands

### Mean-Variance Optimization
```bash
# Using CSV data (migrated from legacy)
portfolio-optimizer --data data/merged_stock_prices.csv \\
  --tickers ANZ.AX,CBA.AX,MQG.AX,NAB.AX,RIO.AX,WOW.AX \\
  --method mean_variance \\
  --show-charts

# Using real-time data
portfolio-optimizer --tickers AAPL,GOOGL,MSFT \\
  --method mean_variance \\
  --real-time \\
  --max-weight 0.3 \\
  --generate-report
```

### Black-Litterman Optimization
```bash
# With investor views
portfolio-optimizer --data data/merged_stock_prices.csv \\
  --tickers ANZ.AX,CBA.AX,MQG.AX \\
  --method black_litterman \\
  --investor-views config/investor_views_template.json \\
  --market-caps data/market_caps.csv \\
  --generate-report --report-format pdf
```

### Risk Parity
```bash
portfolio-optimizer --tickers AAPL,GOOGL,MSFT,TSLA \\
  --method risk_parity \\
  --max-weight 0.4 \\
  --show-charts
```

### Hierarchical Risk Parity
```bash
portfolio-optimizer --data data/merged_stock_prices.csv \\
  --tickers ANZ.AX,CBA.AX,MQG.AX,NAB.AX,RIO.AX,WOW.AX \\
  --method hrp \\
  --generate-report
```

## Advanced Commands

### With Constraints
```bash
portfolio-optimizer --data data/merged_stock_prices.csv \\
  --tickers ANZ.AX,CBA.AX,MQG.AX,NAB.AX \\
  --method mean_variance \\
  --constraints config/constraints.json \\
  --l2-regularization 0.01 \\
  --show-charts
```

### Backtesting Multiple Strategies
```bash
portfolio-optimizer --run-backtest \\
  --data data/merged_stock_prices.csv \\
  --tickers ANZ.AX,CBA.AX,MQG.AX,NAB.AX,RIO.AX,WOW.AX \\
  --strategies mean_variance,black_litterman,hrp \\
  --rebalance-frequency M \\
  --transaction-costs 0.001 \\
  --output backtest_results.json
```

### Configuration File Usage
```bash
# Use main configuration
portfolio-optimizer --config config/main_config.json \\
  --tickers ANZ.AX,CBA.AX,MQG.AX \\
  --generate-report

# Save current settings
portfolio-optimizer --tickers AAPL,GOOGL,MSFT \\
  --method mean_variance \\
  --risk-free-rate 0.025 \\
  --save-config my_settings.json
```

## Utility Commands

### List Available Methods
```bash
portfolio-optimizer --list-methods
```

### Check Data Source Health
```bash
portfolio-optimizer --health-check
```

### Help
```bash
portfolio-optimizer --help
```

## Migration from Legacy

### Old Way (streamlined_optimizer.py)
```python
from streamlined_optimizer import create_optimizer

portfolio_data = {
    "tickers": ["ANZ.AX", "CBA.AX", "MQG.AX"],
    "start_date": "2022-01-01",
    "end_date": "2024-12-31"
}

optimizer = create_optimizer()
result = optimizer.optimize_portfolio(portfolio_data)
```

### New Way (CLI)
```bash
portfolio-optimizer --data data/merged_stock_prices.csv \\
  --tickers ANZ.AX,CBA.AX,MQG.AX \\
  --start-date 2022-01-01 \\
  --end-date 2024-12-31 \\
  --method mean_variance \\
  --show-charts \\
  --output results.json
```

### New Way (Python API)
```python
from portfolio_optimizer import OptimizationEngine, OptimizationConfig, OptimizationMethod
from portfolio_optimizer.core.data_sources import DataSourceManager

data_manager = DataSourceManager()
prices = data_manager.get_market_data(['ANZ.AX', 'CBA.AX', 'MQG.AX'], '2022-01-01', '2024-12-31')

config = OptimizationConfig(method=OptimizationMethod.MEAN_VARIANCE)
engine = OptimizationEngine()
result = engine.optimize(prices, config)
```
'''
    
    def create_migration_guide(self) -> str:
        """Create comprehensive migration guide"""
        return '''# Migration Guide: Legacy to New Platform

## Overview

This guide helps you migrate from the legacy `streamlined_optimizer.py` to the new institutional-grade platform.

## Key Differences

### Old System
- Single script (`streamlined_optimizer.py`)
- Limited to Mean-Variance and Black-Litterman
- Basic constraint handling
- Fixed chart generation

### New System
- Modular architecture with multiple optimization methods
- Comprehensive CLI and Python API
- Advanced constraint management
- Professional reporting and backtesting
- Real-time data integration

## Migration Steps

### 1. Data Migration
Your existing data files have been automatically migrated:
- `merged_stock_prices.csv` → `data/merged_stock_prices.csv`
- `market_caps.csv` → `data/market_caps.csv`
- Legacy charts → `legacy_charts/`

### 2. Configuration Migration
New configuration files have been created based on your usage:
- `config/main_config.json` - Main settings
- `config/constraints.json` - Portfolio constraints
- `config/investor_views_template.json` - Black-Litterman views

### 3. Code Migration

#### Old Code Pattern
```python
from streamlined_optimizer import create_optimizer

portfolio_data = {
    "tickers": ["ANZ.AX", "CBA.AX"],
    "allocations": {"ANZ.AX": 0.5, "CBA.AX": 0.5},
    "start_date": "2022-01-01",
    "end_date": "2024-12-31"
}

optimizer = create_optimizer()
result = optimizer.optimize_portfolio(portfolio_data)
```

#### New Code Pattern
```python
from portfolio_optimizer import OptimizationEngine, OptimizationConfig, OptimizationMethod
from portfolio_optimizer.core.data_sources import DataSourceManager

# Load data
data_manager = DataSourceManager()
prices = data_manager.get_market_data(
    tickers=['ANZ.AX', 'CBA.AX'],
    start_date='2022-01-01',
    end_date='2024-12-31'
)

# Configure optimization
config = OptimizationConfig(
    method=OptimizationMethod.MEAN_VARIANCE,
    risk_free_rate=0.02
)

# Run optimization
engine = OptimizationEngine()
result = engine.optimize(prices, config)
```

### 4. CLI Migration

Instead of running Python scripts, you can now use the CLI:

```bash
# Replace your old script execution
portfolio-optimizer --data data/merged_stock_prices.csv \\
  --tickers ANZ.AX,CBA.AX \\
  --method mean_variance \\
  --show-charts \\
  --generate-report
```

## New Features Available

### 1. Additional Optimization Methods
```bash
# Risk Parity
portfolio-optimizer --method risk_parity --tickers AAPL,GOOGL,MSFT

# Hierarchical Risk Parity
portfolio-optimizer --method hrp --tickers AAPL,GOOGL,MSFT

# CVaR Optimization
portfolio-optimizer --method cvar --tickers AAPL,GOOGL,MSFT
```

### 2. Advanced Constraints
```python
from portfolio_optimizer.core.optimization import ConstraintManager

constraints = ConstraintManager()
constraints.add_weight_constraint("AAPL", min_weight=0.05, max_weight=0.25)
constraints.add_sector_constraint("Technology", ["AAPL", "GOOGL"], max_weight=0.6)
```

### 3. Backtesting
```bash
portfolio-optimizer --run-backtest \\
  --strategies mean_variance,black_litterman,hrp \\
  --rebalance-frequency M
```

### 4. Real-time Data
```bash
portfolio-optimizer --real-time --tickers AAPL,GOOGL,MSFT
```

## Troubleshooting

### Common Issues

1. **Data not found**: Ensure CSV files are in the `data/` directory
2. **Import errors**: Install new requirements: `pip install -r requirements.txt`
3. **Configuration issues**: Check configuration files in `config/` directory

### Getting Help

```bash
# List available methods
portfolio-optimizer --list-methods

# Check data sources
portfolio-optimizer --health-check

# Full help
portfolio-optimizer --help
```

## Performance Improvements

The new platform includes:
- L2 regularization for better optimization stability
- Intelligent data caching
- Vectorized operations
- Memory-efficient algorithms
- Parallel processing support

## Next Steps

1. **Test Migration**: Run the basic example to ensure everything works
2. **Explore New Methods**: Try HRP, Risk Parity, and CVaR optimization
3. **Setup Backtesting**: Compare different strategies
4. **Configure Constraints**: Add institutional-grade constraints
5. **Generate Reports**: Create professional PDF reports

## Support

If you encounter issues during migration:
1. Check the examples in the `examples/` directory
2. Review configuration files in `config/`
3. Run health checks: `portfolio-optimizer --health-check`
4. Refer to the main documentation in README.md
'''
    
    def generate_migration_report(self):
        """Generate comprehensive migration report"""
        report_file = self.target_dir / "migration_report.json"
        
        # Add summary statistics
        self.migration_report["summary"] = {
            "total_files_migrated": len(self.migration_report["files_migrated"]),
            "configurations_created": len(self.migration_report["configurations_created"]),
            "warnings_count": len(self.migration_report["warnings"]),
            "errors_count": len(self.migration_report["errors"]),
            "migration_successful": len(self.migration_report["errors"]) == 0
        }
        
        with open(report_file, 'w') as f:
            json.dump(self.migration_report, f, indent=2)
        
        # Print summary
        print("\\n" + "="*60)
        print("MIGRATION REPORT")
        print("="*60)
        print(f"Files migrated: {self.migration_report['summary']['total_files_migrated']}")
        print(f"Configurations created: {self.migration_report['summary']['configurations_created']}")
        print(f"Warnings: {self.migration_report['summary']['warnings_count']}")
        print(f"Errors: {self.migration_report['summary']['errors_count']}")
        
        if self.migration_report["summary"]["migration_successful"]:
            print("\\n✅ Migration completed successfully!")
            print(f"\\n📁 New files location: {self.target_dir}")
            print("\\n🚀 Next steps:")
            print("  1. cd portfolio_optimizer_migrated")
            print("  2. python examples/basic_optimization.py")
            print("  3. Read examples/migration_guide.md for detailed instructions")
        else:
            print("\\n❌ Migration completed with errors. Check migration_report.json")
        
        print("="*60)


def main():
    """Main CLI for migration tool"""
    parser = argparse.ArgumentParser(
        description="Migrate from legacy streamlined_optimizer.py to new platform"
    )
    parser.add_argument("--legacy-dir", type=str, default=".", 
                       help="Directory containing legacy files")
    parser.add_argument("--target-dir", type=str, default="portfolio_optimizer_migrated",
                       help="Target directory for migrated files")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be migrated without doing it")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
        print(f"Would migrate from: {args.legacy_dir}")
        print(f"Would migrate to: {args.target_dir}")
        
        # Analyze what would be migrated
        legacy_dir = Path(args.legacy_dir)
        csv_files = list(legacy_dir.glob("*.csv"))
        json_files = list(legacy_dir.glob("*.json"))
        
        print(f"\\nWould migrate {len(csv_files)} CSV files:")
        for csv_file in csv_files:
            print(f"  - {csv_file.name}")
        
        print(f"\\nWould analyze {len(json_files)} JSON files for configuration:")
        for json_file in json_files:
            print(f"  - {json_file.name}")
        
        return
    
    # Run migration
    migrator = LegacyMigrator(args.legacy_dir, args.target_dir)
    migrator.migrate_all()


if __name__ == "__main__":
    main()