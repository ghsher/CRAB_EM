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
    save_results
)

from model import CRAB_Model

HH_ATTRIBUTES = pd.read_csv("Input/HH_attributes.csv", index_col=0)
FIRM_ATTRIBUTES = pd.read_csv("Input/Firm_attributes.csv", index_col=0)
PMT_WEIGHTS = pd.read_csv("Input/PMT_weights.csv", index_col=0)
CCA_ENABLED = True
SOCIAL_NET_ENABLED = True

N_REPLICATIONS = 2
RANDOM_SEEDS = np.arange(0, 999999, int(999999/N_REPLICATIONS))
STEPS = 260 # 5 year burn-in + 60 years model time

FLOOD_NARRATIVES = {
        'A' : {40: 1000},
        'B' : {20: 10, 40: 10, 60: 10, 80: 10},
}

# Model KPIs are just CRAB_Model Datacollector column names
MODEL_KPIs = ['HH consumption', 'Unemployment rate', 'Avg wage']

# Agent KPIs are given as column names, agent type, aggregation scheme,
# and disaggregation (i.e. agent grouping) scheme
# TODO: After replacing the DataCollector, implement disagg
AGENT_KPIs = [
    {'name': 'Production made', 'type': 'Firm',
     'agg': 'sum', 'disagg': ['None']},
    {'name': 'Net worth', 'type': 'Household',
     'agg': 'mean', 'disagg': ['None']},
]

def CRAB_model_wrapper(
        debt_sales_ratio: float=2.0, wage_sensitivity_prod: float=0.2,
        flood_narrative: str='A',
        seed=0, steps: int=200, outcomes: list=[]) -> None:
    
    model = CRAB_Model(
        # Standard parameters
        # TODO: If including HH/Firm attributes as a factor, rework this.
        # (Suggestion: move the read_csv into CRAB_Model.__init__(),
        #  and control it with a CategoricalParameter that indicates
        #  the file name).
        HH_attributes=HH_ATTRIBUTES,
        firm_flood_depths=FIRM_ATTRIBUTES, PMT_weights=PMT_WEIGHTS,
        CCA=CCA_ENABLED, social_net=SOCIAL_NET_ENABLED,
        # Controllable parameters
        debt_sales_ratio=debt_sales_ratio,
        wage_sensitivity_prod=wage_sensitivity_prod,
        flood_times=FLOOD_NARRATIVES[flood_narrative],
        random_seed=seed)
    for _ in range(steps):
        model.step()
    
    # TODO: Replace all of the below when you've replaced the Model's DC
    #       (should probably be a priority right now)
    # TODO: Also, handle disagg.
    model_df = model.datacollector.get_model_vars_dataframe()
    agent_df = model.datacollector.get_agent_vars_dataframe()

    out = {}
    for KPI in MODEL_KPIs:
        out[KPI] = model_df[KPI].tolist()
    for KPI in AGENT_KPIs:
        sub_df = None
        if KPI['type'] == 'Household':
            sub_df = agent_df[agent_df['Type'] == "<class 'CRAB_agents.Household'>"]
        elif KPI['type'] == 'Firm':
            sub_df = agent_df[(agent_df['Type'] != "<class 'CRAB_agents.Household'>") & 
                              (agent_df['Type'] != "<class 'government.Government'>")]
        out[KPI['name']] = sub_df.groupby('Step').agg({KPI['name']: KPI['agg']})

    return out


# Runtime output settings
ema_logging.LOG_FORMAT = "[%(name)s/%(levelname)s/%(processName)s] %(message)s"
ema_logging.log_to_stderr(ema_logging.INFO) #, pass_root_logger_level=True) # Uncomment for MPI

# Build up the EMA_workbench Model object
model = ReplicatorModel("CRAB", function=CRAB_model_wrapper)

# 1. Assign number of replications:
model.replications = N_REPLICATIONS

# 2. Define uncertainties & constant parameters:
model.uncertainties = [
    RealParameter("debt_sales_ratio", 0.8, 5),
    RealParameter("wage_sensitivity_prod", 0.0, 1.0),
    CategoricalParameter("flood_narrative", ['A', 'B'], pff=True),
]

model.constants = []

# 3. Define outcomes of interest to track
outcomes = []
for KPI in MODEL_KPIs:
    outcomes.append(ArrayOutcome(KPI))
for KPI in AGENT_KPIs:
    out_name = f"{KPI['agg'].capitalize()} {KPI['name']}"
    outcomes.append(ArrayOutcome(out_name, KPI['name']))
model.outcomes = outcomes

# Run experiments!!!
# NOTE: Change to MultiprocessingEvaluator when on Linux
# NOTE: Best for now to manually define Scenarios instead of sampling
#        until you fix the PFF problem
with SequentialEvaluator(model) as evaluator:
    results = evaluator.perform_experiments(
        scenarios=20,
        uncertainty_sampling=Samplers.LHS
    )
    
save_results(results, "results/0328_EMA_test_run.tar.gz")