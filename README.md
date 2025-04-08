# Secure Finance Report Readme

A comprehensive simulation framework for analyzing a CDP-style (Collateralized Debt Position) stablecoin protocol—**USDFC**—backed by Filecoin (FIL). This project provides multiple simulation approaches to evaluate protocol stability, user behavior, and economic outcomes under various market conditions.

## Overview

The USDsf simulator is modeled after the Liquity protocol (originally deployed on Ethereum) and tests its viability when ported to the Filecoin network. The framework focuses on stress-testing economic conditions and evaluating protocol behavior from the perspective of key participants:

- **Liquidity Providers**
- **Market Makers**
- **Stablecoin Holders**

## Key Features

- Simulates minting and redemption of FIL-backed stablecoins
- Evaluates protocol health metrics and user positions over time
- Analyzes user behaviors and their financial outcomes
- Supports multiple economic scenarios (bullish, bearish, sideways)
- Enables parameter variation and stress testing
- Tracks collateral ratios, redemptions, liquidations, and protocol risk
- Advanced simulation options including SPV models and tranche-based debt allocation

## Dependencies

```bash
pip install numpy pandas matplotlib tqdm yfinance concurrent.futures
```

## Simulation Modules

### 1. USDsf_Simulator.py

**Basic simulation environment** for CDP-style stablecoin protocol backed by Filecoin.

#### Parameters
| Parameter               | Description                                                    |
|-------------------------|----------------------------------------------------------------|
| `starting_price`        | Initial price of FIL in USD                                    |
| `num_users`             | Number of simulated protocol users                             |
| `initial_collateral`    | Total initial FIL collateral deposited                         |
| `collateral_ratio`      | Initial collateral ratio set by users (e.g., 1.5 for 150%)     |
| `min_collateral_ratio`  | Minimum ratio required to avoid liquidation                    |
| `liquidation_penalty`   | Penalty rate for liquidated positions                          |
| `redemption_fee`        | Fee percentage during redemption of USDFC for FIL              |
| `scenario`              | Market trend: "bull", "bear", or "sideways"                    |
| `timesteps`             | Number of time periods to simulate                             |

#### Output Visualizations
- Collateral ratios over time
- Total redemption value
- Liquidation events
- User net positions

### 2. MC_base.py

**Monte Carlo simulation** of the USDsf stability mechanism using Geometric Brownian Motion.

#### Key Features
- Parallel execution for multiple scenarios
- Behavioral modeling of troves (open, close, adjust, liquidate)
- Price stabilization through redemption and issuance logic
- Export of simulation results to CSV

### 3. SPV_MC.py

**Ownership model simulation** comparing Private Loan (PL) and Special Purpose Vehicle (SPV) models.

#### Key Features
- Configurable ownership model (PL or SPV troves)
- Reacquisition logic for post-redemption behavior
- Detailed tracking of redemptions, liquidity, and collateral health
- Scenario comparison support

### 4. scenarios.py

**Real-world scenario simulations** with 13 predefined events to stress-test the protocol.

#### Available Scenarios
1. Storage Provider Workflow (minting/redemption)
2. Cross-Ecosystem Payment
3. Cross-Chain Swap
4. Lending Market Utilization
5. Borrowing Market
6. Low Collateral Edge Case
7. Market Crash (50% FIL price drop)
8. Empty Stability Pool
9. Depeg (price deviation from peg)
10. Low Exchange Liquidity
11. Increasing Redemption Fees
12. Oracle Exploit
13. Mass Adoption

#### Outputs
- Aggregated metrics in DataFrame
- LaTeX summary tables
- Optional visualization plots

### 5. USDsf_Simulator_tranches.py

**Advanced simulation with debt tranching** to test seniority and redemption priorities.

#### Key Features
- Two debt categories: normal and preferential (pref) tranches
- Priority-based redemption logic (normal first, then preferential)
- Comprehensive tracking of tranche-specific redemption probabilities

## Usage Examples

### Basic Simulation
```python
python scenarios.py
```

### Monte Carlo Analysis
```python
python MC_base.py
```

### Ownership Model Comparison
```python
python SPV_MC.py
```

### Tranche-Based Simulation
```python
python USDsf_Simulator_tranches.py
```

## Use Cases

1. **Protocol Stability Assessment**  
   Evaluate USDsf behavior under different market conditions and volatility regimes

2. **Tokenomics Design**  
   Test impact of fee adjustments, airdrops, and collateralization rules

3. **Stress Testing**  
   Analyze protocol resilience during extreme price movements or liquidity shortages

4. **Ownership Structure Analysis**  
   Compare PL vs SPV models for redemption pressure and liquidity needs

5. **Debt Seniority Impact**  
   Assess how priority rules between tranches affect systemic stability

6. **Governance Support**  
   Provide data-driven analysis for parameter adjustment proposals
