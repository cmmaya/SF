import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from datetime import datetime, timedelta

# ------------------------- Configuration & Parameters -------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Policy parameters
rate_issuance = 0.01
rate_redemption = 0.01

# Airdrop (kept for consistency)
quantity_LQTY_airdrop = 1902.6

# LQTY token simulation parameters
price_LQTY_initial = 0.4    # initial price
LQTY_sigma = 0.8            # annualized volatility
LQTY_mu = 0.2               # annualized drift
n_steps = 8760              # simulation steps (hours)
granularity = "1h"
LQTY_total_supply = 100000000

# Stability pool parameters
initial_return = 0.2
stability_initial = 0
sd_stability = 0.001
drift_stability = 1.002
theta = 0.001

# Natural rate parameters
natural_rate_initial = 0.2
sd_natural_rate = 0.002

# Trove parameters
sd_opentroves = 0.5
n_steady = 0.5
collateral_gamma_k = 10
collateral_gamma_theta = 500
target_cr_a = 1.1
target_cr_b = 0.1
target_cr_chi_square_df = 16
rational_inattention_gamma_k = 4
rational_inattention_gamma_theta = 0.08
alpha = 0.3
sd_closetroves = 0.5
beta = 0.2

# Liquidity pool parameters (not used explicitly)
sd_liquidity = 0.001
drift_liquidity = 1.0003
delta = -20
f = 1

# Global time periods (in hours)
day = 24
month = 24 * 30
year = 24 * 365
period = year

# GBM parameters for FIL price simulation
drift_GBM = 0.1
vol_GBM = 0.5
S0 = 3

# ------------------------- Simulation Class for Scenarios -------------------------
class LUSDStabilitySimulationScenario:
    def __init__(self, simulation_id=0, scenario=1):
        """
        Initialize the simulation for a given test scenario.
        scenario: integer 1 to 13 describing the test case.
        For scenario 13 (mass adoption), the initial number of troves is set to 500.
        """
        self.simulation_id = simulation_id
        self.scenario = scenario
        random.seed(simulation_id)
        np.random.seed(simulation_id)
        self.day = day
        self.month = month
        self.year = year
        self.period = period
        self.drift_GBM = drift_GBM
        self.vol_GBM = vol_GBM
        self.initial_open = 500 if scenario == 13 else 10
        self.events = []  # list of (time_index, message) tuples

    def fil_gbm_simulation(self, granularity="1h", periods=365):
        """Simulate a FIL price path using GBM."""
        n_steps_sim = periods
        dt = 1 / n_steps_sim
        mu = drift_GBM
        sigma = vol_GBM
        t = np.linspace(0, 1, n_steps_sim)
        W = np.cumsum(np.random.standard_normal(n_steps_sim)) * np.sqrt(dt)
        S = S0 * np.exp((mu - 0.5 * sigma**2)*t + sigma * W)
        return S, mu, sigma

    def simulate_lqty_token(self, sigma, mu, S0, n_steps, granularity="1h"):
        """Simulate LQTY token prices using GBM."""
        dt = 1 / (365 * 24) if granularity == "1h" else 1 / 365
        t = np.linspace(0, n_steps * dt, n_steps)
        W = np.cumsum(np.random.standard_normal(n_steps)) * np.sqrt(dt)
        prices = S0 * np.exp((mu - 0.5 * sigma**2)*t + sigma * W)
        return prices

    def inject_scenario_events(self, idx, current_price_fil, troves, data):
        """
        Inject events into the simulation based on the selected scenario.
        Modify troves or current_price_fil as needed.
        """
        # Scenario 1: Storage Provider Workflow
        if self.scenario == 1:
            if idx == 100:
                if not troves.empty:
                    troves.loc[0, 'Supply'] += 1000
                    self.events.append((idx, "Minted 1000 USDFC"))
            if idx == 500:
                if not troves.empty and troves.loc[0, 'Supply'] >= 1000:
                    troves.loc[0, 'Supply'] -= 1000
                    self.events.append((idx, "Redeemed 1000 USDFC"))
        # Scenario 2: Cross-Ecosystem Payment
        elif self.scenario == 2:
            if idx == 200:
                if not troves.empty:
                    troves.loc[0, 'Supply'] -= 200
                    self.events.append((idx, "Processed 200 USDFC payment"))
        # Scenario 3: Cross-Chain Swap
        elif self.scenario == 3:
            if idx == 300:
                if not troves.empty:
                    troves.loc[0, 'Supply'] -= 150
                    self.events.append((idx, "Cross-chain swap: -150 USDFC"))
        # Scenario 4: Lending Market Utilization
        elif self.scenario == 4:
            if idx == 400:
                if not troves.empty:
                    troves.loc[0, 'Supply'] += 500
                    self.events.append((idx, "Lending deposit: +500 USDFC"))
            if idx == 800:
                if not troves.empty and troves.loc[0, 'Supply'] >= 500:
                    troves.loc[0, 'Supply'] -= 500
                    self.events.append((idx, "Lending withdrawal: -500 USDFC"))
        # Scenario 5: Borrowing Market Utilization
        elif self.scenario == 5:
            if idx == 450:
                if not troves.empty:
                    troves.loc[0, 'Supply'] += 300
                    self.events.append((idx, "Borrowed 300 USDFC"))
            if idx == 900:
                if not troves.empty and troves.loc[0, 'Supply'] >= 300:
                    troves.loc[0, 'Supply'] -= 300
                    self.events.append((idx, "Repaid 300 USDFC"))
        # Scenario 6: Low Collateral Edge Case
        elif self.scenario == 6:
            if idx == 150:
                if not troves.empty:
                    troves.loc[0, 'Supply'] = current_price_fil * troves.loc[0, 'fil_Quantity'] / 1.1
                    self.events.append((idx, "Set trove to 110% collateral"))
            if idx == 160:
                current_price_fil *= 0.98
                self.events.append((idx, "2% price drop triggering liquidation"))
                return current_price_fil, troves
        # Scenario 7: Market Crash Simulation
        elif self.scenario == 7:
            if idx == 300:
                current_price_fil *= 0.5
                self.events.append((idx, "50% market crash"))
                return current_price_fil, troves
        # Scenario 8: Empty Stability Pool
        elif self.scenario == 8:
            if idx == 250:
                current_price_fil *= 0.1
                self.events.append((idx, "90% price crash"))
                return current_price_fil, troves
        # Scenario 9: Depeg Scenario
        elif self.scenario == 9:
            if idx == 350:
                self.events.append((idx, "USDFC depeg event"))
            return current_price_fil, troves
        # Scenario 10: Low/No Exchange Liquidity
        elif self.scenario == 10:
            if idx == 400:
                if not troves.empty:
                    troves.loc[0, 'Supply'] = max(troves.loc[0, 'Supply'] - 300, 0)
                    self.events.append((idx, "Forced redemption of 300 USDFC"))
                return current_price_fil, troves
        # Scenario 11: Increasing Redemption Fees Scenario
        elif self.scenario == 11:
            if idx == 500:
                if not troves.empty:
                    troves.loc[0, 'Supply'] -= 100
                    self.events.append((idx, "Redemption: -100 USDFC"))
            if idx == 501:
                if not troves.empty:
                    troves.loc[0, 'Supply'] -= 120
                    self.events.append((idx, "Consecutive redemption: -120 USDFC"))
                return current_price_fil, troves
        # Scenario 12: Oracle Exploit Scenario
        elif self.scenario == 12:
            if idx == 550:
                current_price_fil *= 0.8
                self.events.append((idx, "Oracle exploit: price dropped to 80%"))
                return current_price_fil, troves
        # Scenario 13: Mass Adoption Simulation
        elif self.scenario == 13:
            if idx % 100 == 0:
                if not troves.empty:
                    troves['Supply'] += 50
                    self.events.append((idx, "Mass minting: +50 USDFC each"))
                return current_price_fil, troves
        
        return current_price_fil, troves

    def run(self):
        """
        Run the simulation over T time steps.
        Returns a DataFrame of results and the events list.
        """
        price_fil, mu, sigma = self.fil_gbm_simulation(granularity="1h", periods=365)
        _ = self.simulate_lqty_token(LQTY_sigma, LQTY_mu, price_LQTY_initial, n_steps, granularity="1h")
        T = len(price_fil)
        
        # (Optional) Precompute a natural rate series; not used here.
        natural_rate_series = np.empty(T)
        natural_rate_series[0] = natural_rate_initial
        for i in range(1, T):
            shock = np.random.normal(0, sd_natural_rate)
            natural_rate_series[i] = natural_rate_series[i-1] * (1 + shock)
        
        results = []
        # Initialize troves (non-buffer: only PL troves)
        troves_list = []
        issuance_LUSD_open = 0.0
        for _ in range(self.initial_open):
            CR_ratio = target_cr_a + target_cr_b * np.random.chisquare(df=target_cr_chi_square_df)
            quantity_fil = np.random.gamma(collateral_gamma_k, scale=collateral_gamma_theta)
            rational_inattention = np.random.gamma(rational_inattention_gamma_k, scale=rational_inattention_gamma_theta)
            supply_trove = price_fil[0] * quantity_fil / CR_ratio
            issuance_LUSD_open += rate_issuance * supply_trove
            troves_list.append({
                "fil_Price": price_fil[0],
                "fil_Quantity": quantity_fil,
                "CR_initial": CR_ratio,
                "Supply": supply_trove,
                "Rational_inattention": rational_inattention,
                "CR_current": CR_ratio,
                "type": "PL"
            })
        troves = pd.DataFrame(troves_list)
        
        # Record initial simulation state
        initial_data = {
            "Price_LUSD": 1.00,
            "Price_fil": price_fil[0],
            "n_open": self.initial_open,
            "n_close": 0,
            "n_liquidate": 0,
            "n_redempt": 0,
            "n_troves": self.initial_open,
            "stability": 0.5 * troves["Supply"].sum(),
            "liquidity": 0.5 * troves["Supply"].sum(),
            "redemption_pool": 0,
            "supply_LUSD": troves["Supply"].sum(),
            "issuance_fee": issuance_LUSD_open,
            "redemption_fee": 0,
            "airdrop_gain": 0,
            "liquidation_gain": 0,
            "return_stability": initial_return,
            "annualized_earning": 0,
            "MC_LQTY": 0,
            "price_LQTY": price_LQTY_initial
        }
        results.append(initial_data)
        
        # Main simulation loop
        for idx in range(1, T):
            current_price_fil = price_fil[idx]
            # Update FIL price for all troves
            troves['fil_Price'] = current_price_fil
            
            # Inject scenario events
            current_price_fil, troves = self.inject_scenario_events(idx, current_price_fil, troves, pd.DataFrame(results))
            
            # For simplicity, simulate Price_LUSD near 1.0 with a small noise.
            price_LUSD_current = 1.0 + 0.01 * np.random.randn()
            if self.scenario == 9 and 350 <= idx < 360:
                price_LUSD_current = 0.95  # simulate depeg during a window
            if price_LUSD_current < 0:
                break
            
            supply_LUSD = troves["Supply"].sum()
            row = {
                "Price_LUSD": price_LUSD_current,
                "Price_fil": current_price_fil,
                "n_open": self.initial_open,
                "n_close": 0,
                "n_liquidate": 0,
                "n_redempt": 0,
                "n_troves": troves.shape[0],
                "stability": 0.5 * supply_LUSD,
                "liquidity": 0.5 * supply_LUSD,
                "redemption_pool": 0,
                "supply_LUSD": supply_LUSD,
                "issuance_fee": issuance_LUSD_open,
                "redemption_fee": 0,
                "airdrop_gain": 0,
                "liquidation_gain": 0,
                "return_stability": initial_return,
                "annualized_earning": 0,
                "MC_LQTY": 0,
                "price_LQTY": price_LQTY_initial
            }
            results.append(row)
        
        data = pd.DataFrame(results)
        return data, self.events

# ------------------------- Monte Carlo Simulator -------------------------
def monte_carlo_simulation(scenario, n_runs=100):
    metrics = []
    for i in tqdm(range(n_runs), desc=f"MC for Scenario {scenario}"):
        sim = LUSDStabilitySimulationScenario(simulation_id=i, scenario=scenario)
        data, events = sim.run()
        # Record summary metrics:
        final_supply = data["supply_LUSD"].iloc[-1]
        min_fil = data["Price_fil"].min()
        max_fil = data["Price_fil"].max()
        num_events = len(events)
        # Count event types by scanning event messages:
        event_counts = {"mint": 0, "redeem": 0, "payment": 0, "swap": 0,
                        "lending_deposit": 0, "lending_withdrawal": 0,
                        "borrow": 0, "repay": 0, "liquidation": 0,
                        "market_crash": 0, "oracle": 0, "mass_minting": 0}
        for ev in events:
            msg = ev[1].lower()
            if "minted" in msg:
                event_counts["mint"] += 1
            if "redeemed" in msg:
                event_counts["redeem"] += 1
            if "payment" in msg:
                event_counts["payment"] += 1
            if "swap" in msg:
                event_counts["swap"] += 1
            if "lending deposit" in msg:
                event_counts["lending_deposit"] += 1
            if "lending withdrawal" in msg:
                event_counts["lending_withdrawal"] += 1
            if "borrowed" in msg:
                event_counts["borrow"] += 1
            if "repaid" in msg:
                event_counts["repay"] += 1
            if "liquidation" in msg:
                event_counts["liquidation"] += 1
            if "market crash" in msg:
                event_counts["market_crash"] += 1
            if "oracle" in msg:
                event_counts["oracle"] += 1
            if "mass minting" in msg:
                event_counts["mass_minting"] += 1

        metric = {
            "final_supply": final_supply,
            "min_FIL_price": min_fil,
            "max_FIL_price": max_fil,
            "num_events": num_events
        }
        metric.update(event_counts)
        metrics.append(metric)
    return pd.DataFrame(metrics)

# ------------------------- LaTeX Table Generator -------------------------
def generate_latex_table(df, scenario):
    # Compute averages and standard deviations
    avg = df.mean()
    std = df.std()
    table = "\\begin{table}[H]\n\\centering\n"
    table += f"\\caption{{Monte Carlo Results for Scenario {scenario}}}\n"
    table += "\\begin{tabular}{lrr}\n\\toprule\nMetric & Mean & Std \\\\\n\\midrule\n"
    for col in df.columns:
        table += f"{col} & {avg[col]:.2f} & {std[col]:.2f} \\\\\n"
    table += "\\bottomrule\n\\end{tabular}\n\\label{tab:scenario" + str(scenario) + "}\n\\end{table}\n"
    return table

# ------------------------- Main Monte Carlo Execution -------------------------
def run_monte_carlo_for_all_scenarios(scenarios=range(1,14), n_runs=100):
    summary_tables = {}
    all_results = {}
    for s in scenarios:
        print(f"Running Monte Carlo for Scenario {s} ...")
        df = monte_carlo_simulation(scenario=s, n_runs=n_runs)
        all_results[s] = df
        latex_table = generate_latex_table(df, scenario=s)
        summary_tables[s] = latex_table
        print(latex_table)
    return all_results, summary_tables

# ------------------------- Visualization for One Scenario -------------------------
def plot_scenario(sim, data, events, scenario):
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(data.index, data['Price_fil'], label='FIL Price', color='blue')
    ax[0].set_ylabel('FIL Price')
    ax[0].set_title(f"Scenario {scenario}: FIL Price Over Time")
    ax[0].legend()
    
    ax[1].plot(data.index, data['supply_LUSD'], label='USDFC Supply', color='orange')
    ax[1].set_ylabel('USDFC Supply')
    ax[1].set_title(f"Scenario {scenario}: USDFC Supply Over Time")
    for ev in events:
        ev_idx, ev_msg = ev
        ax[1].axvline(x=ev_idx, color='red', linestyle='--', alpha=0.7)
        ax[1].text(ev_idx, data['supply_LUSD'].min()*1.05, ev_msg, rotation=90, fontsize=8, color='red')
    ax[1].set_xlabel('Time Step')
    plt.tight_layout()
    plt.show()

# ------------------------- Main Execution -------------------------
if __name__ == '__main__':
    # Run Monte Carlo simulations for all 13 scenarios (adjust n_runs as needed)
    all_results, summary_tables = run_monte_carlo_for_all_scenarios(scenarios=range(1,14), n_runs=100)
    
    # Optionally, visualize one simulation run for a selected scenario (e.g., Scenario 1)
    sim_example = LUSDStabilitySimulationScenario(simulation_id=0, scenario=1)
    data_example, events_example = sim_example.run()
    plot_scenario(sim_example, data_example, events_example, scenario=1)
