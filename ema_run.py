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
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from ema_workbench import (
    ema_logging,
    ReplicatorModel,
    RealParameter,
    CategoricalParameter,
    # Constant,
    ArrayOutcome,
    # TimeSeriesOutcome,
    SequentialEvaluator,
    MultiprocessingEvaluator,
    perform_experiments,
    Samplers,
    save_results,
)

from model import CRAB_Model
from CRAB_agents import *
from ema_data_collection import *

print(":: Reading Adaptation & Flood Depth attribute CSVs")
HH_ATTRIBUTES = pd.read_csv("Input/HH_attributes.csv", index_col=0)
FIRM_ATTRIBUTES = pd.read_csv("Input/Firm_attributes.csv", index_col=0)
PMT_WEIGHTS = pd.read_csv("Input/PMT_weights.csv", index_col=0)
print(":::: Done reading")

CCA_ENABLED = True
SOCIAL_NET_ENABLED = True
FIRMS_RD_ENABLED = True

N_REPLICATIONS = 1
RANDOM_SEEDS = np.arange(0, 999999, int(999999/N_REPLICATIONS))
STEPS = 200 # 5 year burn-in + 25 years model time

FLOOD_NARRATIVES = [
        {75: 1000},
        {40: 100, 60: 100, 80: 100, 100: 100},
        {q:r for q,r in list(zip(range(30, 130, 5), [10 for _ in range(20)]))},
        {60: 1000, 100: 100},
        {60: 1000, 120: 1000},
]

FIRM_TYPES = [
	Agriculture,
	Industry,
	Construction,
	Transport,
	Utilities,
	Private_services,
	Public_services,
	Wholesale_Retail,
    C26,
]


def CRAB_model_wrapper(
        debt_sales_ratio: float=2.0, wage_sensitivity_prod: float=0.2,
        init_mkup: float=0.25, flood_narrative: dict={}, #str='A',
        seed=0, steps: int=200, outcomes: list=[]) -> None:

    model = CRAB_Model(
        # Standard parameters
        # TODO: If including HH/Firm attributes as a factor, rework this.
        # (Suggestion: move the read_csv into CRAB_Model.__init__(),
        #  and control it with a CategoricalParameter that indicates
        #  the file name).
        HH_attributes=HH_ATTRIBUTES,
        firm_flood_depths=FIRM_ATTRIBUTES, PMT_weights=PMT_WEIGHTS,
        firms_RD=FIRMS_RD_ENABLED,
        CCA=CCA_ENABLED, social_net=SOCIAL_NET_ENABLED,
        # Controllable parameters
        debt_sales_ratio=debt_sales_ratio,
        wage_sensitivity_prod=wage_sensitivity_prod,
        init_mkup=init_mkup,
        flood_when=flood_narrative,
        random_seed=seed)
    
    for _ in tqdm(range(STEPS), total=STEPS, leave=False,
                  desc=f"MODEL RUN: DSR={debt_sales_ratio:.2}, WSP={wage_sensitivity_prod:.2}, {flood_narrative}"):
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
    out['Firm Population'] = get_population(agent_dfs['Firms'])
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
    for firm in FIRM_TYPES:
        name = firm.__name__
        out[f'Share of Large Firms ({name})'] = get_share_large_firms(agent_dfs[firm])
    out['Share of Large Firms (All)'] = get_share_large_firms(agent_dfs['Firms'])

    # 4. Gini (TODO)

    # 5. Impact
    out['Total Household Damages'] = get_total_damage(agent_dfs[Household])
    out['Average Income-Weighted Damages'] = get_average_damage_income_ratio(agent_dfs[Household])
    # TODO: Recovery

    # 6. Debt
    out['Total Household Debt'] = get_total_household_debt(agent_dfs[Household])
    out['Average Household Debt'] = get_average_household_debt(agent_dfs[Household])
    out['Total Firm Debt'] = get_total_firm_debt(agent_dfs['Firms'])
    # TODO: The latter by different industries
    # TODO: Government debt/deficit

    return out

# Runtime output settings
# ema_logging.LOG_FORMAT = "[%(name)s/%(levelname)s/%(processName)s] %(message)s"
ema_logging.LOG_FORMAT = "[EMA] %(message)s"
ema_logging.log_to_stderr(ema_logging.INFO) #, pass_root_logger_level=True) # Uncomment for MPI

# Build up the EMA_workbench Model object
model = ReplicatorModel("CRAB", function=CRAB_model_wrapper)

# 1. Assign number of replications:
model.replications = N_REPLICATIONS

# 2. Define uncertainties & constant parameters:
model.uncertainties = [
    RealParameter("debt_sales_ratio", 0.8, 5),
    RealParameter("wage_sensitivity_prod", 0.0, 1.0),
    RealParameter("init_markup", 0.05, 0.5),
    CategoricalParameter("flood_narrative", FLOOD_NARRATIVES, pff=True),
]

model.constants = []

# 3. Define outcomes of interest to track
outcomes = [
    ArrayOutcome('Household Population'),
    ArrayOutcome('Unemployment Rate'),
    ArrayOutcome('Median Net Worth'),
    ArrayOutcome('Median House Value'),
    ArrayOutcome('Median Wage'),
    ArrayOutcome('Minimum Wage'),
    ArrayOutcome('Total Household Damages'),
    ArrayOutcome('Average Income-Weighted Damages'),
    ArrayOutcome('Total Household Debt'),
    ArrayOutcome('Average Household Debt'),

    ArrayOutcome('Firm Population'),
    ArrayOutcome('GDP'),
    ArrayOutcome('Total Firm Resources'),
    ArrayOutcome('Total Firm Debt'),
    ArrayOutcome('Share of Large Firms (All)'),
]
for firm in FIRM_TYPES:
    name = firm.__name__
    outcomes.append(ArrayOutcome(f'{name} Population'))
    outcomes.append(ArrayOutcome(f'{name} Production Made'))
    outcomes.append(ArrayOutcome(f'Share of Large Firms ({name})'))
model.outcomes = outcomes

# Run experiments!!!
# NOTE: Change to MultiprocessingEvaluator when on Linux
# NOTE: Best for now to manually define Scenarios instead of sampling
#        until you fix the PFF problem
with SequentialEvaluator(model) as evaluator:
    results = evaluator.perform_experiments(
        scenarios=1,
        uncertainty_sampling=Samplers.LHS
    )
    
save_results(results, "results/0408_EMA_test_run.tar.gz")