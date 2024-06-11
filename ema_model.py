# !/usr/bin/env python3.11
# -*- coding: utf-8 -*-

"""

@author: ghsher

Script to run any number of experiments on the CRAB model
using the EMA Workbench. 

Information about the EMA Workbench is available at
 https://emaworkbench.readthedocs.io/en/latest/

"""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ema_workbench import (
    Model,
    ReplicatorModel,
    RealParameter,
    IntegerParameter,
    Constant,
    TimeSeriesOutcome,
)
from ema_workbench.util import CaseError

from model import CRAB_Model
from CRAB_agents import *
from ema_data_collection import *

# Constants
HH_ATTRIBUTES = pd.read_csv("data/HH_attributes.csv", index_col=0)
FIRM_ATTRIBUTES = pd.read_csv("data/Firm_attributes.csv", index_col=0)
PMT_WEIGHTS = pd.read_csv("data/PMT_weights.csv", index_col=0)

MIGRATION = {"Regional": False, "RoW": True}
CCA = {"Households": True, "Firms": True}
SOCIAL_NET = True
FIRMS_RD = True

FIRM_TYPES = [
    CapitalFirm,
    ConsumptionGoodFirm,
    ServiceFirm
]

def CRAB_model_wrapper(
        debt_sales_ratio: float=2.0, wage_sensitivity_prod: float=0.2,
        init_markup: float=0.25, capital_firm_cap_out_ratio: float=0.4,
        min_unempl_emigration: float=0.04, migration_unempl_bounds_diff: float=0.15,
        deu_discount_factor: float=1.0, flood_intensity: int=3000, flood_timing: int=40,
        seed=0, steps: int=120, outcomes: list=[]) -> None:

    model = CRAB_Model(
        # Standard parameters
        # TODO: If including HH/Firm attributes as a factor, rework this.
        # (Suggestion: move the read_csv into CRAB_Model.__init__(),
        #  and control it with a CategoricalParameter that indicates
        #  the file name).
        HH_attributes=HH_ATTRIBUTES,
        firm_flood_depths=FIRM_ATTRIBUTES,
        PMT_weights=PMT_WEIGHTS,
        firms_RD=FIRMS_RD,
        social_net=SOCIAL_NET,
        migration=MIGRATION, CCA=CCA,
        # Controllable parameters
        debt_sales_ratio=debt_sales_ratio,
        wage_sensitivity_prod=wage_sensitivity_prod,
        init_markup=init_markup,
        capital_firm_cap_out_ratio=capital_firm_cap_out_ratio,
        min_unempl_emigration=min_unempl_emigration,
        migration_unempl_bounds_diff=migration_unempl_bounds_diff,
        deu_discount_factor=deu_discount_factor,
        flood_timing=flood_timing,
        # Experiment parameters
        flood_intensity=flood_intensity,
        random_seed=seed)
    
    for _ in tqdm(range(steps), total=steps, leave=False,
                  desc=f"RUN (SEED={seed}, FLD={flood_intensity}@{flood_timing})"):
        model.step()

    model_df = model.datacollector.get_model_vars_dataframe()
    agent_df = model.datacollector.get_agent_vars_dataframe()

    # Separate and group agent data for aggregation
    # TODO: Handle disaggregation for household metrics here.
    agent_dfs = {}
    for firm in FIRM_TYPES:
        agent_dfs[firm] = agent_df[agent_df['Type'] == firm.__name__].groupby('Step')
    agent_dfs[Household] = agent_df[agent_df['Type'] == 'Household'].groupby('Step')
    agent_dfs['Firms'] = agent_df[(agent_df['Type'] != 'Household') &
                                  (agent_df['Type'] != 'Government')].groupby('Step')
    del agent_df

    out = {}

    # 0. Populations
    out['Household Population'] = get_population(agent_dfs[Household], aslist=True)
    out['Firm Population'] = get_population(agent_dfs['Firms'], aslist=True)
    for firm in FIRM_TYPES:
        name = firm.__name__
        out[f'{name} Population'] = get_population(agent_dfs[firm], aslist=True)

    # 1. Economic performance
    for firm in FIRM_TYPES:
        name = firm.__name__
        out[f'{name} Production Made'] = get_production(name, model_df)
    out['GDP'] = get_GDP(model_df)
    out['Unemployment Rate'] = get_unemployment(agent_dfs[Household])

    # 2. Wealth
    out['Median Net Worth'] = get_median_net_worth(agent_dfs[Household])
    out['Total Firm Resources'] = get_total_net_worth(agent_dfs['Firms'])
    out['Median House Value'] = get_median_house_value(agent_dfs[Household])
    out['Median Wage'] = get_median_wage(agent_dfs[Household])
    out['Minimum Wage'] = get_minimum_wage(model_df)

    # 3. Firm competition
    # TODO: Wait on understanding firm size distrs. better before capturing firm size shares.
    # for firm in FIRM_TYPES:
    #     name = firm.__name__
    #     out[f'Share of Large Firms ({name})'] = get_share_large_firms(agent_dfs[firm])
    # out['Share of Large Firms (All)'] = get_share_large_firms(agent_dfs['Firms'])
    # for firm in FIRM_TYPES:
    #     name = firm.__name__
    #     out[f'10th Percentile Firm Size ({name})'] = get_10th_p_firm_size(agent_dfs[firm])
    #     out[f'90th Percentile Firm Size ({name})'] = get_90th_p_firm_size(agent_dfs[firm])
    #     out[f'Median Firm Size ({name})'] = get_median_firm_size(agent_dfs[firm])

    # 4. Inequality & Gini
    out['Gini Coefficient'] = get_gini(agent_dfs[Household])

    # 5. Impact
    out['Total Household Damages'] = get_total_damage(agent_dfs[Household])
    out['Average Income-Weighted Damages'] = get_average_damage_income_ratio(agent_dfs[Household])
    # TODO: Recovery

    # 6. Debt
    out['Total Firm Debt'] = get_total_firm_debt(agent_dfs['Firms'])
    # TODO: Government debt/deficit

    for key, val in out.items():
        if np.isnan(val).any():
            experiment_summary = {
                'debt_sales_ratio' : debt_sales_ratio,
                'wage_sensitivity_prod' : wage_sensitivity_prod,
                'init_markup' : init_markup,
                'capital_firm_cap_out_ratio' : capital_firm_cap_out_ratio,
                'min_unempl_emigration' : min_unempl_emigration,
                'migration_unempl_bounds_diff' : migration_unempl_bounds_diff,
                'deu_discount_factor' : deu_discount_factor,
                'flood_timing' : flood_timing,
            }
            raise CaseError(f"Have NaN in {key}", experiment_summary)
    
    return out

def get_EMA_CRAB_model(flood_intensities=None, seeds=None, steps=None):
    if seeds is None:
        model = Model("CRAB", function=CRAB_model_wrapper)
    else:
        model = ReplicatorModel("CRAB", function=CRAB_model_wrapper)
        model.replications = [dict(seed=s) for s in seeds]

    # 1. Define uncertainties & constant parameters:
    model.uncertainties = [
        RealParameter("debt_sales_ratio", 0.8, 5),
        RealParameter("wage_sensitivity_prod", 0.0, 1.0),
        RealParameter("init_markup", 0.05, 0.5),
        RealParameter("capital_firm_cap_out_ratio", 0.2, 0.6),
        RealParameter("min_unempl_emigration", 0.02, 0.08),
        RealParameter("migration_unempl_bounds_diff", 0.10, 0.25),
        RealParameter("deu_discount_factor", 0.8, 1.0),
        IntegerParameter("flood_timing", 30, 80),
    ]

    constants = []
    if steps is not None:
        constants.append(
            Constant('steps', steps)
        )
    if flood_intensities is not None:
        constants.append(
            Constant("flood_intensity", flood_intensities[0][0]),
        )
    model.constants = constants

    # 2. Define outcomes of interest to track
    outcomes = [
        TimeSeriesOutcome('Household Population'),
        TimeSeriesOutcome('Unemployment Rate'),
        TimeSeriesOutcome('Gini Coefficient'),
        TimeSeriesOutcome('Median Net Worth'),
        TimeSeriesOutcome('Median House Value'),
        TimeSeriesOutcome('Median Wage'),
        TimeSeriesOutcome('Minimum Wage'),
        TimeSeriesOutcome('Total Household Damages'),
        TimeSeriesOutcome('Average Income-Weighted Damages'),

        TimeSeriesOutcome('Firm Population'),
        TimeSeriesOutcome('GDP'),
        TimeSeriesOutcome('Total Firm Resources'),
        TimeSeriesOutcome('Total Firm Debt'),
        # TimeSeriesOutcome('Share of Large Firms (All)'),
    ]
    for firm in FIRM_TYPES:
        name = firm.__name__
        outcomes.append(TimeSeriesOutcome(f'{name} Population'))
        outcomes.append(TimeSeriesOutcome(f'{name} Production Made'))
        # outcomes.append(TimeSeriesOutcome(f'Share of Large Firms ({name})'))
        # outcomes.append(TimeSeriesOutcome(f'10th Percentile Firm Size ({name})'))
        # outcomes.append(TimeSeriesOutcome(f'90th Percentile Firm Size ({name})'))
        # outcomes.append(TimeSeriesOutcome(f'Median Firm Size ({name})'))
    model.outcomes = outcomes

    return model