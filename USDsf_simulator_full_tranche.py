#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 18:30:53 2025

@author: juanpablomadrigalcianci
"""



import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import concurrent.futures
import random
from datetime import datetime, timedelta
import yfinance as yf

# ------------------------- Configuration & Parameters -------------------------

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Policy & fee parameters
base_rate_initial = 0
base_rate_beta = 2
base_rate_decay = 0.917
rate_issuance = 0.01
rate_redemption = 0.01
rate_issuance_min = 0.005
rate_redemption_min = 0.005

# Airdrop
quantity_LQTY_airdrop = 1902.6

# LQTY token simulation parameters
price_LQTY_initial = 0.4  # initial price
LQTY_sigma = 0.8        # annualized volatility (80%)
LQTY_mu = 0.2           # annualized drift (20%)
n_steps = 8760          # number of simulation steps (hours)
granularity = "1h"      # hourly simulation
LQTY_total_supply = 100000000

# Stability pool parameters
initial_return = 0.2
sd_return = 0.001
stability_initial = 0
sd_stability = 0.001
drift_stability = 1.002
theta = 0.001

# Natural rate parameters
natural_rate_initial = 0.2
sd_natural_rate = 0.002

# Troves parameters
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
alpha = 0.3  # sensitivity to USDsf price & issuance fee
sd_closetroves = 0.5
beta = 0.2   # sensitivity to USDsf price

# PE ratio for LQTY market
PE_ratio = 50

# Liquidity pool parameters
liquidity_initial = 0
sd_liquidity = 0.001
drift_liquidity = 1.0003
delta = -20
f=1
# Global time periods (in hours)
day = 24
month = 24 * 30
year = 24 * 365
period = year


drift_GBM=0.1
vol_GBM=0.5
S0=3
pref_ratio=0.2
sim_name='base'

# ------------------------- Simulation Class Definition -------------------------

class USDsfStabilitySimulation:
    def __init__(self, simulation_id=0):
        """
        Initialize the simulation.
        Optionally seeds the RNGs for reproducibility.
        """
        self.simulation_id = simulation_id
        random.seed(simulation_id)
        np.random.seed(simulation_id)
        self.day = day
        self.month = month
        self.year = year
        self.period = period
        self.drift_GBM=drift_GBM
        self.vol_GBM=vol_GBM
        self.cp=0
        self.cn=0
        self.ce=0

    def fil_gbm_simulation(self, granularity="1h", periods=365):
        """
        Fetch FIL historical data from Yahoo Finance, compute drift/volatility,
        simulate a GBM price path, and return the simulated prices.
        """
     
        end_date = datetime.now()
        start_date = end_date - timedelta(days=periods)
        # try:
        #     fil_data = yf.download("FIL-USD", start=start_date, end=end_date, interval=granularity)
        #     if fil_data.empty:
        #         raise ValueError("No data fetched for FIL-USD")
        # except Exception as e:
        #     logger.error("Error fetching FIL data: %s", e)
        #     raise
        
        # historical_prices = fil_data['Close'].ffill()
        # if self.drift_GBM == None or self.vol_GBM==None:
        #     returns = np.log(historical_prices / historical_prices.shift(1)).dropna()
        #     mu = returns.mean() * len(returns)
        #     sigma = returns.std() * np.sqrt(len(returns))
        # else:
        #     mu=self.drift_GBM
        #     sigma=self.vol_GBM
        
        # logger.info(f"Drift (mu): {mu:.4f}, Volatility (sigma): {sigma:.4f}")
        
        #S0 = historical_prices.iloc[-1]
        n_steps_sim = periods
        dt = 1 / n_steps_sim
        mu=drift_GBM
        sigma=vol_GBM
        t = np.linspace(0, 1, n_steps_sim)
        W = np.cumsum(np.random.standard_normal(n_steps_sim)) * np.sqrt(dt)
        S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)
        return S, mu, sigma

    def simulate_lqty_token(self, sigma, mu, S0, n_steps, granularity="1h"):
        """
        Simulate LQTY token prices using Geometric Brownian Motion.
        """
        dt = 1 / (365 * 24) if granularity == "1h" else 1 / 365
        t = np.linspace(0, n_steps * dt, n_steps)
        W = np.cumsum(np.random.standard_normal(n_steps)) * np.sqrt(dt)
        prices = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)
        return prices

    def run(self):
        """
        Runs a single simulation and returns key outcome metrics.
        
        Returns:
            data (pd.DataFrame): Time-series outcomes.
            total_liquidations (float): Total liquidations during the run.
            final_liquidity (float): Final liquidity level.
            final_Price_USDsf (float): Final USDsf price.
        """
        # --- Run FIL & LQTY price simulations ---
        price_fil, _, _ = self.fil_gbm_simulation(granularity="1h", periods=365)
        price_LQTY_array = self.simulate_lqty_token(LQTY_sigma, LQTY_mu, price_LQTY_initial, n_steps, granularity="1h")
        T = len(price_fil)
        
        # Precompute natural rate series
        natural_rate_series = np.empty(T)
        natural_rate_series[0] = natural_rate_initial
        for i in range(1, T):
            shock = np.random.normal(0, sd_natural_rate)
            natural_rate_series[i] = natural_rate_series[i-1] * (1 + shock)
        
        # ------------- Internal Simulation Functions -------------
        def liquidate_troves(troves, idx, data, price_fil_current):
            troves = troves.copy()
            troves['CR_current'] = troves['fil_Price'] * troves['fil_Quantity'] / troves['Supply']
            if idx - 1 < 0:
                price_USDsf_previous = 1.0
                price_LQTY_previous = price_LQTY_initial
                stability_pool_previous = 0
            else:
                price_USDsf_previous = data.loc[idx-1, 'Price_USDsf']
                price_LQTY_previous = data.loc[idx-1, 'price_LQTY']
                stability_pool_previous = data.loc[idx-1, 'stability']
            mask = troves['CR_current'] < 1.1
            troves_liquidated = troves[mask]
            troves_remaining = troves[~mask].reset_index(drop=True)
            debt_liquidated = troves_liquidated['Supply'].sum()
            fil_liquidated = troves_liquidated['fil_Quantity'].sum()
            n_liquidated = troves_liquidated.shape[0]
            
            liquidation_gain = fil_liquidated * price_fil_current - debt_liquidated * price_USDsf_previous
            if debt_liquidated > stability_pool_previous and stability_pool_previous > 0:
                liquidation_gain *= stability_pool_previous / debt_liquidated
            airdrop_gain = price_LQTY_previous * quantity_LQTY_airdrop

            if idx == 0:
                return_stability = initial_return
            elif idx < self.month:
                return_stability = (self.year / idx) * (
                    data.loc[0:idx, 'liquidation_gain'].sum() + data.loc[0:idx, 'airdrop_gain'].sum()
                ) / (price_USDsf_previous * stability_pool_previous) if stability_pool_previous else 0
            else:
                return_stability = (self.year / self.month) * (
                    data.loc[idx - self.month:idx, 'liquidation_gain'].sum() + data.loc[idx - self.month:idx, 'airdrop_gain'].sum()
                ) / (price_USDsf_previous * stability_pool_previous) if stability_pool_previous else 0

            return troves_remaining, return_stability, debt_liquidated, fil_liquidated, liquidation_gain, airdrop_gain, n_liquidated

        def close_troves(troves, idx, price_USDsf_previous):
            n_troves = troves.shape[0]
            shock = np.random.normal(0, sd_closetroves)
            if idx <= 240:
                number_closetroves = int(round(np.random.uniform(0, 1)))
            elif price_USDsf_previous >= 1:
                number_closetroves = int(round(max(0, n_steady * (1 + shock))))
            else:
                number_closetroves = int(round(max(0, n_steady * (1 + shock)) + beta * (1 - price_USDsf_previous) * n_troves))
            number_closetroves = min(number_closetroves, n_troves)
            if number_closetroves > 0:
                drop_indices = np.random.choice(troves.index, size=number_closetroves, replace=False)
                troves = troves.drop(drop_indices).reset_index(drop=True)
            return troves, number_closetroves

        def adjust_troves(troves, idx):
            issuance_USDsf_adjust = 0.0
            troves = troves.copy()
            ratio = np.random.uniform(0, 1)
            p = np.random.uniform(0, 1, size=len(troves))
            check = (troves['CR_current'] - troves['CR_initial']) / (troves['CR_initial'] * troves['Rational_inattention'])
            mask_low = (p >= ratio) & (check < -1)
            troves.loc[mask_low, 'Supply'] = (
                troves.loc[mask_low, 'fil_Price'] * troves.loc[mask_low, 'fil_Quantity'] /
                troves.loc[mask_low, 'CR_initial']
            )
            mask_high = (p >= ratio) & (check > 2)
            supply_new = (
                troves.loc[mask_high, 'fil_Price'] * troves.loc[mask_high, 'fil_Quantity'] /
                troves.loc[mask_high, 'CR_initial']
            )
            issuance_USDsf_adjust += (rate_issuance * (supply_new - troves.loc[mask_high, 'Supply'])).sum()
            troves.loc[mask_high, 'Supply'] = supply_new
            mask_collat = (p < ratio) & ((check < -1) | (check > 2))
            troves.loc[mask_collat, 'fil_Quantity'] = (
                troves.loc[mask_collat, 'CR_initial'] * troves.loc[mask_collat, 'Supply'] /
                troves.loc[mask_collat, 'fil_Price']
            )
            return troves, issuance_USDsf_adjust

        def open_troves(troves, idx, price_USDsf_previous, price_fil_current):
            issuance_USDsf_open = 0.0
            shock = random.normalvariate(0, sd_opentroves)
            n_troves = troves.shape[0]
            if idx <= 0:
                number_opentroves = initial_open
            elif price_USDsf_previous <= 1 + rate_issuance:
                number_opentroves = max(0, n_steady * (1 + shock))
            else:
                number_opentroves = max(0, n_steady * (1 + shock)) + alpha * (price_USDsf_previous - rate_issuance - 1) * n_troves
            number_opentroves = int(round(number_opentroves))
            
            new_troves = []
            for _ in range(number_opentroves):
                CR_ratio = target_cr_a + target_cr_b * np.random.chisquare(df=target_cr_chi_square_df)
                quantity_fil = np.random.gamma(collateral_gamma_k, scale=collateral_gamma_theta)
                rational_inattention = np.random.gamma(rational_inattention_gamma_k, scale=rational_inattention_gamma_theta)
                supply_trove = price_fil_current * quantity_fil / CR_ratio
                issuance_USDsf_open += rate_issuance * supply_trove
                new_troves.append({
                    "fil_Price": price_fil_current,
                    "fil_Quantity": quantity_fil,
                    "CR_initial": CR_ratio,
                    "Supply": supply_trove,
                    "Rational_inattention": rational_inattention,
                    "CR_current": CR_ratio,
                    "preferential":(np.random.random()<pref_ratio)
                })
            if new_troves:
                troves = pd.concat([troves, pd.DataFrame(new_troves)], ignore_index=True)
            return troves, number_opentroves, issuance_USDsf_open

        def stability_update(stability_pool_previous, return_previous, idx, total_supply):
            shock = np.random.normal(0, sd_stability)
            natural_rate_current = natural_rate_series[idx]
            if idx <= self.month:
                stability_pool = stability_pool_previous * drift_stability * (1 + shock) * (1 + return_previous - natural_rate_current)**theta
            else:
                stability_pool = stability_pool_previous * (1 + shock) * (1 + return_previous - natural_rate_current)**theta
            return min(stability_pool, total_supply)

        def calculate_price(price_USDsf_previous, liquidity_pool, liquidity_pool_next):
            return price_USDsf_previous * (liquidity_pool / liquidity_pool_next)**(1 / delta)

        def price_stabilizer(troves, idx, data, stability_pool, n_open, price_fil_current):
            issuance_USDsf_stabilizer = 0.0
            redemption_fee = 0.0
            n_redempt = 0
            redempted = 0.0
            redemption_pool = 0.0
            
            liquidity_pool_previous = data.loc[idx-1, 'liquidity'] if idx-1 >= 0 else 0
            price_USDsf_previous = data.loc[idx-1, 'Price_USDsf'] if idx-1 >= 0 else 1.0
            supply = troves['Supply'].sum()
            
            shock_liquidity = np.random.normal(0, sd_liquidity)
            liquidity_pool_next = liquidity_pool_previous * drift_liquidity * (1 + shock_liquidity)
            liquidity_pool = supply - stability_pool
            price_USDsf_current = calculate_price(price_USDsf_previous, liquidity_pool, liquidity_pool_next)
            
            # Ceiling arbitrage adjustment
            if price_USDsf_current > 1.1 + rate_issuance:
                supply_wanted = stability_pool + liquidity_pool_next * ((1.1 + rate_issuance) / price_USDsf_previous)**delta
                supply_trove = supply_wanted - supply
                CR_ratio = 1.1
                rational_inattention = 0.1
                quantity_fil = supply_trove * CR_ratio / price_fil_current
                issuance_USDsf_stabilizer = rate_issuance * supply_trove
                new_trove = {
                    "fil_Price": price_fil_current,
                    "fil_Quantity": quantity_fil,
                    "CR_initial": CR_ratio,
                    "Supply": supply_trove,
                    "Rational_inattention": rational_inattention,
                    "CR_current": CR_ratio,
                    "preferential":(np.random.random()<pref_ratio)

                }
                troves = pd.concat([troves, pd.DataFrame([new_trove])], ignore_index=True)
                price_USDsf_current = 1.1 + rate_issuance
                liquidity_pool = supply_wanted - stability_pool
                n_open += 1
            
            # Floor arbitrage adjustment
            if price_USDsf_current < 1 - rate_redemption:
                shock_redemption = np.random.normal(0, 0.01)
                redemption_ratio = max(1, 0.8 * (1 + shock_redemption))
                supply_target = stability_pool + liquidity_pool_next * ((1 - rate_redemption) / price_USDsf_previous)**delta
                supply_diff = supply - supply_target
                if supply_diff < redemption_ratio * liquidity_pool:
                    redemption_pool = supply_diff
                    price_USDsf_current = 1 - rate_redemption
                else:
                    redemption_pool = redemption_ratio * liquidity_pool
                    price_USDsf_current = calculate_price(price_USDsf_previous, liquidity_pool, liquidity_pool_next)
                redemption_fee = redemption_pool * (rate_redemption + redemption_pool / supply if supply else 0)
            
                redemption_pool = redemption_pool*np.exp(-redemption_fee**f)
                troves = troves.sort_values(by=['preferential','CR_current'], ascending=True).reset_index(drop=True)
                if not troves.empty:
                    redempted = troves.at[0, 'Supply']
                    n_redempt += 1
                    i = 0
                    self.ce+=1
                    if troves.at[i, 'preferential']:
                            self.cp+=1
                    else:
                            self.cn+=1
                    
  
                    
                    while redempted < redemption_pool and i < len(troves):
                        i += 1
                        if i < len(troves):
                            redempted += troves.at[i, 'Supply']
                            n_redempt += 1
                            self.ce+=1
       
                        if troves.at[i, 'preferential']:
                                self.cp+=1
                        else:
                                self.cn+=1
                            
                            
                    if i < len(troves):
                        residual = redemption_pool - (redempted - troves.at[i, 'Supply'])
                        troves.at[i, 'Supply'] -= residual
                        troves.at[i, 'fil_Quantity'] -= residual / price_fil_current
                        troves.at[i, 'CR_current'] = price_fil_current * troves.at[i, 'fil_Quantity'] / troves.at[i, 'Supply']
            
            return price_USDsf_current, liquidity_pool, troves, issuance_USDsf_stabilizer, redemption_fee, n_redempt, redemption_pool, n_open,self.ce,self.cp,self.cn

        def LQTY_market(idx, data, price_LQTY_prev):
            if idx <= self.month:
                price_LQTY_current = price_LQTY_prev
                annualized_earning = (idx / self.month)**0.5 * np.random.normal(200000000, 500000)
            else:
                revenue_issuance = data.loc[idx - self.month:idx, 'issuance_fee'].sum()
                revenue_redemption = data.loc[idx - self.month:idx, 'redemption_fee'].sum()
                annualized_earning = 365 * (revenue_issuance + revenue_redemption) / 30
                discount = idx / self.period
                price_LQTY_current = discount * PE_ratio * annualized_earning / LQTY_total_supply
            quantity_LQTY = (LQTY_total_supply / 3) * (1 - 0.5**(idx / self.period))
            MC_LQTY_current = price_LQTY_current * quantity_LQTY
            return price_LQTY_current, annualized_earning, MC_LQTY_current

        # ------------- Main Simulation Loop -------------
        results = []
        initial_data = {
            "Price_USDsf": 1.00,
            "Price_fil": price_fil[0],
            "n_open": initial_open,
            "n_close": 0,
            "n_liquidate": 0,
            "n_redempt": 0,
            "n_troves": initial_open,
            "stability": float(stability_initial),
            "liquidity": 0,
            "redemption_pool": 0,
            "supply_USDsf": 0,
            "issuance_fee": 0,
            "redemption_fee": 0,
            "airdrop_gain": 0,
            "liquidation_gain": 0,
            "return_stability": initial_return,
            "annualized_earning": 0,
            "MC_LQTY": 0,
            "price_LQTY": price_LQTY_initial,
            "N_redemptions":0,
            "R_norm":0,
            "R_pref":0
        }
        results.append(initial_data)
        
        # Open initial troves
        troves = pd.DataFrame(columns=["fil_Price", "fil_Quantity", "CR_initial", "Supply", 
                                       "Rational_inattention", "CR_current"])
        troves, n_open, issuance_USDsf_open = open_troves(troves, 0, initial_data["Price_USDsf"], price_fil[0])
        initial_data["issuance_fee"] = issuance_USDsf_open * initial_data["Price_USDsf"]
        initial_data["supply_USDsf"] = troves["Supply"].sum()
        initial_data["liquidity"] = 0.5 * troves["Supply"].sum()
        initial_data["stability"] = 0.5 * troves["Supply"].sum()
        
        for idx in range(1, T):
            current_price_fil = price_fil[idx]
            troves['fil_Price'] = current_price_fil
            price_USDsf_previous = results[-1]["Price_USDsf"]
            price_LQTY_previous = results[-1]["price_LQTY"]
            
            df_results = pd.DataFrame(results)
            (troves, return_stability, debt_liquidated, fil_liquidated, 
             liquidation_gain, airdrop_gain, n_liquidate) = liquidate_troves(troves, idx, df_results, current_price_fil)
            troves, n_close = close_troves(troves, idx, price_USDsf_previous)
            troves, issuance_USDsf_adjust = adjust_troves(troves, idx)
            troves, n_open_new, issuance_USDsf_open = open_troves(troves, idx, price_USDsf_previous, current_price_fil)
            n_open = n_open_new
            total_supply = troves['Supply'].sum()
            stability_pool = stability_update(results[-1]["stability"], return_stability, idx, total_supply)
            (price_USDsf_current, liquidity_pool, troves, issuance_USDsf_stabilizer,
             redemption_fee, n_redempt, redemption_pool, n_open,ce,cp,cn) = price_stabilizer(troves, idx, df_results, stability_pool, n_open, current_price_fil)
            if liquidity_pool < 0:
                break
            price_LQTY_current, annualized_earning, MC_LQTY_current = LQTY_market(idx, df_results, price_LQTY_previous)
            
            issuance_fee = price_USDsf_current * (issuance_USDsf_adjust + issuance_USDsf_open + issuance_USDsf_stabilizer)
            n_troves = troves.shape[0]
            supply_USDsf = troves['Supply'].sum()
            
            row = {
                "Price_USDsf": float(price_USDsf_current),
                "Price_fil": float(current_price_fil),
                "n_open": float(n_open),
                "n_close": float(n_close),
                "n_liquidate": float(n_liquidate),
                "n_redempt": float(n_redempt),
                "n_troves": float(n_troves),
                "stability": float(stability_pool),
                "liquidity": float(liquidity_pool),
                "redemption_pool": float(redemption_pool),
                "supply_USDsf": float(supply_USDsf),
                "issuance_fee": float(issuance_fee),
                "redemption_fee": float(redemption_fee),
                "airdrop_gain": float(airdrop_gain),
                "liquidation_gain": float(liquidation_gain),
                "return_stability": float(return_stability),
                "annualized_earning": float(annualized_earning),
                "MC_LQTY": float(MC_LQTY_current),
                "price_LQTY": float(price_LQTY_current),
                "N_redemptions":ce,
                "R_norm":cn,
                "R_pref":cp
            }
            results.append(row)
            if price_USDsf_current < 0:
                break
        
        data = pd.DataFrame(results)
        total_liquidations = data["n_liquidate"].sum()
        final_liquidity = data["liquidity"].iloc[-1]
        final_Price_USDsf = data["Price_USDsf"].iloc[-1]
        return data, total_liquidations, final_liquidity, final_Price_USDsf

# ------------------------- Monte Carlo Runner -------------------------

def run_monte_carlo(n_simulations=100, parallel=True):
    """
    Run multiple Monte Carlo simulations.
    
    Args:
        n_simulations (int): Number of simulation runs.
        parallel (bool): Whether to run simulations in parallel.
    
    Returns:
        mc_df (pd.DataFrame): Combined results from all simulations.
    """
    simulation_results = []
    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(USDsfStabilitySimulation(sim_id).run): sim_id for sim_id in range(n_simulations)}
            for future in tqdm(concurrent.futures.as_completed(futures), total=n_simulations, desc="Monte Carlo Simulations"):
                sim_id = futures[future]
                try:
                    data, total_liq, final_liq, final_price = future.result()
                    data['run'] = sim_id
                    data['day'] = np.arange(len(data))
                    simulation_results.append(data)
                except Exception as e:
                    logger.error(f"Simulation {sim_id} encountered an error: {e}")
    else:
        for sim_id in tqdm(range(n_simulations), desc="Monte Carlo Simulations"):
            try:
                data, total_liq, final_liq, final_price = USDsfStabilitySimulation(sim_id).run()
                data['run'] = sim_id
                data['day'] = np.arange(len(data))
                simulation_results.append(data)
            except Exception as e:
                logger.error(f"Simulation {sim_id} encountered an error: {e}")
    
    if simulation_results:
        mc_df = pd.concat(simulation_results, ignore_index=True)
        return mc_df
    else:
        logger.error("No simulation results available.")
        return pd.DataFrame()

sim=USDsfStabilitySimulation(simulation_id=1)

data, total_liquidations, final_liquidity, final_Price_USDsf=sim.run()
