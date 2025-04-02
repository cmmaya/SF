#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 09:45:00 2025

@author: juanpablomadrigalcianci
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import numpy as np
import pandas as pd
import random
import concurrent.futures
from datetime import datetime
# from tqdm import tqdm  # optional, if you want progress bars

###############################################################################
# Global Config
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Example parameters (same as your base code) ---------------------------------
base_rate_initial = 0
base_rate_beta = 2
base_rate_decay = 0.917
rate_issuance = 0.01
rate_redemption = 0.01
rate_issuance_min = 0.005
rate_redemption_min = 0.005

quantity_LQTY_airdrop = 1902.6

price_LQTY_initial = 0.4
LQTY_sigma = 0.8
LQTY_mu = 0.2
n_steps = 8760
granularity = "1h"
LQTY_total_supply = 100000000

initial_return = 0.2
sd_return = 0.001
stability_initial = 0
sd_stability = 0.001
drift_stability = 1.002
theta = 0.001

natural_rate_initial = 0.2
sd_natural_rate = 0.002

initial_open = 10
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

PE_ratio = 50

liquidity_initial = 0
sd_liquidity = 0.001
drift_liquidity = 1.0003
delta = -20
f = 1

day = 24
month = 24 * 30
year = 24 * 365
period = year

drift_GBM = 0.1
vol_GBM = 0.5
S0 = 3

name = 'base'


###############################################################################
# Class: USDsfStabilitySimulation
###############################################################################
class USDsfStabilitySimulation:
    """
    A single-run simulation of the USDsf protocol with either PL or SPV troves.
    """
    def __init__(
        self,
        simulation_id=0,
        use_spv=False,
        reacquire_probability=0.0
    ):
        """
        Args:
            simulation_id (int): RNG seed offset + ID for logs.
            use_spv (bool): If True, troves are labeled "SPV". If False, "PL".
            reacquire_probability (float): Probability PL reacquires FIL if SPV trove is redeemed.
        """
        self.simulation_id = simulation_id
        random.seed(simulation_id)
        np.random.seed(simulation_id)

        # Setup basic references
        self.day = day
        self.month = month
        self.year = year
        self.period = period
        self.drift_GBM = drift_GBM
        self.vol_GBM = vol_GBM

        # Scenario flags
        self.use_spv = use_spv
        self.reacquire_probability = reacquire_probability

        # Track redemption from SPV
        self.spv_fil_redeemed_total = 0.0
        self.spv_redemption_events = 0
        self.spv_reacquire_count = 0

        # Track redemption from PL
        self.pl_fil_redeemed_total = 0.0
        self.pl_redemption_events = 0


    ############################################################################
    # Fil Price Simulation
    ############################################################################
    def fil_gbm_simulation(self, granularity="1h", periods=365):
        # Very simplified version
        n_steps_sim = periods
        dt = 1 / n_steps_sim
        t = np.linspace(0, 1, n_steps_sim)
        W = np.cumsum(np.random.standard_normal(n_steps_sim)) * np.sqrt(dt)
        S = S0 * np.exp((self.drift_GBM - 0.5 * self.vol_GBM**2) * t + self.vol_GBM * W)
        return S, self.drift_GBM, self.vol_GBM

    ############################################################################
    # LQTY Price Simulation
    ############################################################################
    def simulate_lqty_token(self, sigma, mu, S0, n_steps, granularity="1h"):
        dt = 1 / (365 * 24) if granularity == "1h" else 1 / 365
        t = np.linspace(0, n_steps * dt, n_steps)
        W = np.cumsum(np.random.standard_normal(n_steps)) * np.sqrt(dt)
        prices = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)
        return prices

    ############################################################################
    # run() -> (data, total_liquidations, final_liquidity, final_price)
    ############################################################################
    def run(self):
        # Price arrays
        price_fil, _, _ = self.fil_gbm_simulation(granularity="1h", periods=365)
        price_LQTY_array = self.simulate_lqty_token(
            LQTY_sigma, LQTY_mu, price_LQTY_initial, n_steps, granularity="1h"
        )
        T = len(price_fil)

        # Precompute "natural rate"
        natural_rate_series = np.empty(T)
        natural_rate_series[0] = natural_rate_initial
        for i in range(1, T):
            shock = np.random.normal(0, sd_natural_rate)
            natural_rate_series[i] = natural_rate_series[i - 1] * (1 + shock)

        # Setup results, troves, initial conditions
        results = []
        initial_row = {
            "Price_USDsf": 1.0,
            "Price_fil": price_fil[0],
            "n_open": initial_open,
            "n_close": 0,
            "n_liquidate": 0,
            "n_redempt": 0,
            "n_troves": initial_open,
            "stability": float(stability_initial),
            "liquidity": 0.0,
            "redemption_pool": 0.0,
            "supply_USDsf": 0.0,
            "issuance_fee": 0.0,
            "redemption_fee": 0.0,
            "airdrop_gain": 0.0,
            "liquidation_gain": 0.0,
            "return_stability": initial_return,
            "annualized_earning": 0.0,
            "MC_LQTY": 0.0,
            "price_LQTY": price_LQTY_initial,

            # Our redemption counters
            "spv_fil_redeemed_total": 0.0,
            "spv_redemption_events": 0,
            "spv_reacquire_count": 0,
            "pl_fil_redeemed_total": 0.0,
            "pl_redemption_events": 0
        }
        results.append(initial_row)

        troves = pd.DataFrame(columns=[
            "owner", "fil_Price", "fil_Quantity", "CR_initial",
            "Supply", "Rational_inattention", "CR_current"
        ])

        # ========== HELPER FUNCTIONS ==========

        def open_troves(troves, idx, price_USDsf_previous, price_fil_current):
            issuance_open = 0.0
            shock_open = random.normalvariate(0, sd_opentroves)
            n_current = troves.shape[0]
            if idx <= 0:
                n_new = initial_open
            elif price_USDsf_previous <= 1 + rate_issuance:
                n_new = max(0, n_steady * (1 + shock_open))
            else:
                n_new = max(0, n_steady * (1 + shock_open)) + alpha * (price_USDsf_previous - rate_issuance - 1) * n_current
            n_new = int(round(n_new))

            new_list = []
            for _ in range(n_new):
                CR_ratio = target_cr_a + target_cr_b * np.random.chisquare(df=target_cr_chi_square_df)
                qty_fil = np.random.gamma(collateral_gamma_k, scale=collateral_gamma_theta)
                r_inattention = np.random.gamma(rational_inattention_gamma_k, scale=rational_inattention_gamma_theta)
                supply_trove = price_fil_current * qty_fil / CR_ratio
                issuance_open += rate_issuance * supply_trove

                # Owner depends on scenario
                owner_label = "SPV" if self.use_spv else "PL"
                new_list.append({
                    "owner": owner_label,
                    "fil_Price": price_fil_current,
                    "fil_Quantity": qty_fil,
                    "CR_initial": CR_ratio,
                    "Supply": supply_trove,
                    "Rational_inattention": r_inattention,
                    "CR_current": CR_ratio
                })
            if new_list:
                troves = pd.concat([troves, pd.DataFrame(new_list)], ignore_index=True)
            return troves, n_new, issuance_open

        def close_troves(troves, idx, price_USDsf_prev):
            n_troves = troves.shape[0]
            shock_close = np.random.normal(0, sd_closetroves)
            if idx <= 240:
                num_close = int(round(np.random.uniform(0, 1)))
            elif price_USDsf_prev >= 1:
                num_close = int(round(max(0, n_steady * (1 + shock_close))))
            else:
                num_close = int(round(
                    max(0, n_steady * (1 + shock_close))
                    + beta * (1 - price_USDsf_prev) * n_troves
                ))
            num_close = min(num_close, n_troves)
            if num_close > 0:
                drop_idx = np.random.choice(troves.index, size=num_close, replace=False)
                troves = troves.drop(drop_idx).reset_index(drop=True)
            return troves, num_close

        def liquidate_troves(troves, idx, data, price_fil_current):
            troves = troves.copy()
            troves['CR_current'] = troves['fil_Price'] * troves['fil_Quantity'] / troves['Supply']
            if idx - 1 < 0:
                price_USDsf_prev = 1.0
                price_LQTY_prev = price_LQTY_initial
                stability_prev = 0
            else:
                price_USDsf_prev = data.loc[idx - 1, 'Price_USDsf']
                price_LQTY_prev = data.loc[idx - 1, 'price_LQTY']
                stability_prev = data.loc[idx - 1, 'stability']

            mask_under = troves['CR_current'] < 1.1
            troves_liquidated = troves[mask_under]
            troves_remaining = troves[~mask_under].reset_index(drop=True)
            debt_liquidated = troves_liquidated['Supply'].sum()
            fil_liquidated = troves_liquidated['fil_Quantity'].sum()
            n_liquidated = troves_liquidated.shape[0]

            liquidation_gain = fil_liquidated * price_fil_current - debt_liquidated * price_USDsf_prev
            if debt_liquidated > stability_prev and stability_prev > 0:
                liquidation_gain *= stability_prev / debt_liquidated
            airdrop_gain = price_LQTY_prev * quantity_LQTY_airdrop

            if idx == 0:
                return_stab = initial_return
            elif idx < month:
                # ...
                return_stab = 0.0  # keep it simple
            else:
                # ...
                return_stab = 0.0

            return (troves_remaining, return_stab,
                    debt_liquidated, fil_liquidated,
                    liquidation_gain, airdrop_gain, n_liquidated)

        def adjust_troves(troves, idx):
            issuance_adjust = 0.0
            troves = troves.copy()
            ratio = np.random.uniform(0, 1)
            p = np.random.uniform(0, 1, size=len(troves))
            troves['CR_current'] = troves['fil_Price'] * troves['fil_Quantity'] / troves['Supply']
            check = ((troves['CR_current'] - troves['CR_initial']) /
                     (troves['CR_initial'] * troves['Rational_inattention']))
            mask_low = (p >= ratio) & (check < -1)
            troves.loc[mask_low, 'Supply'] = (
                troves.loc[mask_low, 'fil_Price'] * troves.loc[mask_low, 'fil_Quantity'] /
                troves.loc[mask_low, 'CR_initial']
            )

            mask_high = (p >= ratio) & (check > 2)
            supply_new = (
                troves.loc[mask_high, 'fil_Price']
                * troves.loc[mask_high, 'fil_Quantity']
                / troves.loc[mask_high, 'CR_initial']
            )
            issuance_adjust += (rate_issuance * (supply_new - troves.loc[mask_high, 'Supply'])).sum()
            troves.loc[mask_high, 'Supply'] = supply_new

            mask_coll = (p < ratio) & ((check < -1) | (check > 2))
            troves.loc[mask_coll, 'fil_Quantity'] = (
                troves.loc[mask_coll, 'CR_initial']
                * troves.loc[mask_coll, 'Supply']
                / troves.loc[mask_coll, 'fil_Price']
            )
            return troves, issuance_adjust

        def stability_update(stability_prev, return_prev, idx, total_supply):
            shock_stab = np.random.normal(0, sd_stability)
            nat_rate = natural_rate_series[idx]
            if idx <= month:
                stability_pool = (
                    stability_prev
                    * drift_stability
                    * (1 + shock_stab)
                    * (1 + return_prev - nat_rate)**theta
                )
            else:
                stability_pool = (
                    stability_prev
                    * (1 + shock_stab)
                    * (1 + return_prev - nat_rate)**theta
                )
            return min(stability_pool, total_supply)

        def calculate_price(price_prev, liquidity_pool, liquidity_pool_next):
            return price_prev * (liquidity_pool / liquidity_pool_next)**(1 / delta) if liquidity_pool_next != 0 else price_prev

        def price_stabilizer(troves, idx, data, stability_pool, n_open, price_fil_current):
            issuance_stabilizer = 0.0
            redemption_fee = 0.0
            n_redempt = 0
            redemption_pool = 0.0

            if idx - 1 < 0:
                liquidity_pool_prev = 0
                price_USDsf_prev = 1.0
            else:
                liquidity_pool_prev = data.loc[idx - 1, 'liquidity']
                price_USDsf_prev = data.loc[idx - 1, 'Price_USDsf']

            supply = troves['Supply'].sum()
            shock_liq = np.random.normal(0, sd_liquidity)
            liquidity_pool_next = liquidity_pool_prev * drift_liquidity * (1 + shock_liq)
            liquidity_pool = supply - stability_pool
            price_USDsf_current = calculate_price(price_USDsf_prev, liquidity_pool, liquidity_pool_next)

            # (A) Ceiling
            if price_USDsf_current > 1.1 + rate_issuance:
                supply_wanted = stability_pool + liquidity_pool_next * ((1.1 + rate_issuance) / price_USDsf_prev)**delta
                supply_trove = supply_wanted - supply
                CR_ratio = 1.1
                rational_inattention = 0.1
                qty_fil = supply_trove * CR_ratio / price_fil_current
                issuance_stabilizer = rate_issuance * supply_trove
                owner_label = "SPV" if self.use_spv else "PL"

                new_row = {
                    "owner": owner_label,
                    "fil_Price": price_fil_current,
                    "fil_Quantity": qty_fil,
                    "CR_initial": CR_ratio,
                    "Supply": supply_trove,
                    "Rational_inattention": rational_inattention,
                    "CR_current": CR_ratio
                }
                troves = pd.concat([troves, pd.DataFrame([new_row])], ignore_index=True)
                price_USDsf_current = 1.1 + rate_issuance
                liquidity_pool = supply_wanted - stability_pool
                n_open += 1

            # (B) Floor
            if price_USDsf_current < 1 - rate_redemption:
                shock_red = np.random.normal(0, 0.001)
                redemption_ratio = max(1, 0.8 * (1 + shock_red))
                supply_target = stability_pool + liquidity_pool_next * ((1 - rate_redemption) / price_USDsf_prev)**delta
                supply_diff = supply - supply_target
                if supply_diff < redemption_ratio * liquidity_pool:
                    redemption_pool = supply_diff
                    price_USDsf_current = 1 - rate_redemption
                else:
                    redemption_pool = redemption_ratio * liquidity_pool
                    price_USDsf_current = calculate_price(price_USDsf_prev, liquidity_pool, liquidity_pool_next)
                redemption_fee = redemption_pool * (rate_redemption + redemption_pool / supply if supply else 0)

                # Simplified
                troves = troves.sort_values(by='CR_current', ascending=True).reset_index(drop=True)
                redeemed_so_far = 0.0
                i = 0
                while redeemed_so_far < redemption_pool and i < len(troves):
                    trove_supply = troves.at[i, 'Supply']
                    if trove_supply <= 0:
                        i += 1
                        continue

                    if redeemed_so_far + trove_supply <= redemption_pool:
                        # Full
                        redeemed_amount = trove_supply
                        redeemed_so_far += redeemed_amount
                        owner = troves.at[i, 'owner']
                        if owner == "SPV":
                            self.spv_fil_redeemed_total += redeemed_amount
                            self.spv_redemption_events += 1
                            if random.random() < self.reacquire_probability:
                                self.spv_reacquire_count += 1
                        elif owner == "PL":
                            self.pl_fil_redeemed_total += redeemed_amount
                            self.pl_redemption_events += 1

                        troves.at[i, 'Supply'] = 0
                        old_qty = troves.at[i, 'fil_Quantity']
                        fil_lost = old_qty * (redeemed_amount / trove_supply) if trove_supply > 0 else 0
                        troves.at[i, 'fil_Quantity'] = old_qty - fil_lost
                        troves.at[i, 'CR_current'] = 9999 if troves.at[i, 'Supply'] <= 0 else (
                            price_fil_current * troves.at[i, 'fil_Quantity'] / troves.at[i, 'Supply']
                        )
                        n_redempt += 1
                    else:
                        # Partial
                        needed = redemption_pool - redeemed_so_far
                        troves.at[i, 'Supply'] = trove_supply - needed
                        old_qty = troves.at[i, 'fil_Quantity']
                        fil_lost = old_qty * (needed / trove_supply)
                        troves.at[i, 'fil_Quantity'] = old_qty - fil_lost
                        troves.at[i, 'CR_current'] = (
                            price_fil_current * troves.at[i, 'fil_Quantity']
                            / troves.at[i, 'Supply']
                        )
                        redeemed_so_far += needed

                        owner = troves.at[i, 'owner']
                        if owner == "SPV":
                            self.spv_fil_redeemed_total += needed
                            self.spv_redemption_events += 1
                            if random.random() < self.reacquire_probability:
                                self.spv_reacquire_count += 1
                        elif owner == "PL":
                            self.pl_fil_redeemed_total += needed
                            self.pl_redemption_events += 1

                        n_redempt += 1
                    i += 1

            return (
                price_USDsf_current,
                liquidity_pool,
                troves,
                issuance_stabilizer,
                redemption_fee,
                n_redempt,
                redemption_pool,
                n_open
            )


        def LQTY_market(idx, data, price_LQTY_prev):
            if idx <= month:
                price_LQTY_current = price_LQTY_prev
                ann_earn = (idx / month)**0.5 * np.random.normal(2, 0.5)
            else:
                rev_iss = data.loc[max(0, idx - month):idx, 'issuance_fee'].sum()
                rev_red = data.loc[max(0, idx - month):idx, 'redemption_fee'].sum()
                ann_earn = 365 * (rev_iss + rev_red) / 30
                discount = idx / self.year
                price_LQTY_current = discount * PE_ratio * ann_earn / LQTY_total_supply
            quantity_LQTY = (LQTY_total_supply / 3) * (1 - 0.5**(idx / self.period))
            MC_LQTY = price_LQTY_current * quantity_LQTY
            return price_LQTY_current, ann_earn, MC_LQTY

        # ========== INITIAL TROVES OPENING ==========
        troves, n_open, issuance_open = open_troves(
            troves, 0, initial_row["Price_USDsf"], price_fil[0]
        )
        initial_row["issuance_fee"] = issuance_open * initial_row["Price_USDsf"]
        initial_row["supply_USDsf"] = troves["Supply"].sum()
        initial_row["liquidity"] = 0.5 * troves["Supply"].sum()
        initial_row["stability"] = 0.5 * troves["Supply"].sum()

        # main loop
        for idx in range(1, T):
            current_price_fil = price_fil[idx]
            troves['fil_Price'] = current_price_fil

            # Get the last row's data
            last_row = results[-1].copy()
            price_USDsf_prev = last_row["Price_USDsf"]
            price_LQTY_prev = last_row["price_LQTY"]

            df_results = pd.DataFrame(results)  # for reference

            # 1) Liquidations
            (troves, return_stab, debt_liq, fil_liq, liq_gain, airdrop_gain, n_liq) = liquidate_troves(
                troves, idx, df_results, current_price_fil
            )

            # 2) close troves
            troves, n_close = close_troves(troves, idx, price_USDsf_prev)

            # 3) adjust troves
            troves, issuance_adjust = adjust_troves(troves, idx)

            # 4) open troves
            troves, n_open_new, issuance_open2 = open_troves(
                troves, idx, price_USDsf_prev, current_price_fil
            )
            n_open = n_open_new

            # 5) stability
            tot_supply = troves["Supply"].sum()
            stability_pool = stability_update(last_row["stability"], return_stab, idx, tot_supply)

            # 6) price stabilizer
            (price_USDsf_cur,
             liquidity_pool,
             troves,
             issuance_stabilizer,
             redemption_fee,
             n_redempt,
             redemption_pool,
             n_open) = price_stabilizer(
                troves, idx, df_results, stability_pool, n_open, current_price_fil
            )

            if liquidity_pool < 0:
                break

            # 7) LQTY market
            price_LQTY_cur, ann_earn, MC_LQTY_cur = LQTY_market(idx, df_results, price_LQTY_prev)

            # issuance fee
            issuance_fee = price_USDsf_cur * (issuance_adjust + issuance_open2 + issuance_stabilizer)

            row = {
                "Price_USDsf": price_USDsf_cur,
                "Price_fil": current_price_fil,
                "n_open": n_open,
                "n_close": n_close,
                "n_liquidate": n_liq,
                "n_redempt": n_redempt,
                "n_troves": troves.shape[0],
                "stability": stability_pool,
                "liquidity": liquidity_pool,
                "redemption_pool": redemption_pool,
                "supply_USDsf": tot_supply,
                "issuance_fee": issuance_fee,
                "redemption_fee": redemption_fee,
                "airdrop_gain": airdrop_gain,
                "liquidation_gain": liq_gain,
                "return_stability": return_stab,
                "annualized_earning": ann_earn,
                "MC_LQTY": MC_LQTY_cur,
                "price_LQTY": price_LQTY_cur,

                # redemption counters so far
                "spv_fil_redeemed_total": self.spv_fil_redeemed_total,
                "spv_redemption_events": self.spv_redemption_events,
                "spv_reacquire_count": self.spv_reacquire_count,
                "pl_fil_redeemed_total": self.pl_fil_redeemed_total,
                "pl_redemption_events": self.pl_redemption_events
            }
            results.append(row)

            if price_USDsf_cur < 0:
                break

        data = pd.DataFrame(results)
        total_liquidations = data["n_liquidate"].sum()
        final_liquidity = data["liquidity"].iloc[-1]
        final_price = data["Price_USDsf"].iloc[-1]

        return data, total_liquidations, final_liquidity, final_price


###############################################################################
# Monte Carlo Runner
###############################################################################
def run_scenario(
    n_sims=10,
    use_spv=False,
    reacquire_probability=0.0,
    parallel=True
):
    """
    Runs multiple simulations (Monte Carlo) for the chosen scenario.

    Args:
        n_sims (int): Number of runs
        use_spv (bool): If True, troves are labeled "SPV"; else "PL".
        reacquire_probability (float): Probability PL reacquires FIL after SPV redemption.
        parallel (bool): Run in parallel or serially.

    Returns:
        pd.DataFrame: Aggregated results with columns for final redemption counters, etc.
    """
    sims_data = []

    def worker(sim_id):
        sim = USDsfStabilitySimulation(
            simulation_id=sim_id,
            use_spv=use_spv,
            reacquire_probability=reacquire_probability
        )
        data, tot_liq, fin_liq, fin_price = sim.run()

        # For the final result, we might only store summary stats (or all rows).
        # Example: just store final row data
        final_row = data.iloc[-1].copy()
        final_row["run_id"] = sim_id
        final_row["use_spv"] = use_spv
        return final_row

    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(worker, i) for i in range(n_sims)]
            for f in concurrent.futures.as_completed(futures):
                try:
                    sims_data.append(f.result())
                except Exception as e:
                    logger.error(f"Simulation error: {e}")
    else:
        for i in range(n_sims):
            sims_data.append(worker(i))

    df_all = pd.DataFrame(sims_data).reset_index(drop=True)
    return df_all


def main():
    """
    Example usage: 
      - Scenario A: PL troves
      - Scenario B: SPV troves
    Then compare how much FIL is redeemed from PL in each scenario.
    """

    # 1) Scenario A: PL Troves
    logger.info("Running Monte Carlo for PL scenario (A) ...")
    df_pl = run_scenario(n_sims=50, use_spv=False, reacquire_probability=0.0, parallel=False)
    # 2) Scenario B: SPV Troves
    logger.info("Running Monte Carlo for SPV scenario (B) ...")
    df_spv = run_scenario(n_sims=50, use_spv=True, reacquire_probability=0.5, parallel=False)

    # Show some summary of forced redemption for PL
    # In scenario B, we expect pl_fil_redeemed_total ~ 0
    # In scenario A, we might see a non-zero average
    avg_pl_redemption_A = df_pl["pl_fil_redeemed_total"].mean()
    avg_pl_redemption_B = df_spv["pl_fil_redeemed_total"].mean()

    logger.info(f"Scenario A (PL) - Average PL forced redemption: {avg_pl_redemption_A:.4f}")
    logger.info(f"Scenario B (SPV) - Average PL forced redemption: {avg_pl_redemption_B:.4f}")

    # Save or analyze further
    df_pl.to_csv(f"monte_carlo_pl_{name}.csv", index=False)
    df_spv.to_csv(f"monte_carlo_spv_{name}.csv", index=False)
    logger.info("Saved scenario results to CSV.")


if __name__ == "__main__":
    main()
