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
from datetime import datetime
from tqdm.auto import tqdm

from ema_workbench import (
    ema_logging,
    SequentialEvaluator,
    MultiprocessingEvaluator,
    MPIEvaluator,
    Scenario,
    save_results,
)
from ema_workbench.util import CaseError

from model import CRAB_Model
from CRAB_agents import *
from ema_data_collection import *
from ema_model import get_EMA_CRAB_model

N_REPLICATIONS = 1
RANDOM_SEEDS = np.arange(0, 1000000, int(1000000/N_REPLICATIONS))

STEPS = 120 # 5 year burn-in + 25 years model time

FLOOD_INTENSITIES = [
    [3000]
]

if __name__ == "__main__":
    # Runtime output settings
    # ema_logging.LOG_FORMAT = "[%(name)s/%(levelname)s/%(processName)s] %(message)s"
    ema_logging.LOG_FORMAT = "[EMA] %(message)s"
    ema_logging.log_to_stderr(ema_logging.INFO, pass_root_logger_level=True) # Uncomment for MPI
    
    # Fetch an EMA_workbench Model object
    model = get_EMA_CRAB_model(
        flood_intensities=FLOOD_INTENSITIES,
        seeds=RANDOM_SEEDS,
        steps=STEPS
    )

    # Read Scenarios from CSV
    scenarios = pd.read_csv('data/2_scenarios_0611.csv', index_col=False)
    N_SCENARIOS = len(scenarios)

    scen_list = []
    for idx, scen in scenarios.iterrows():
        scen_list.append(Scenario(idx, **scen))

    # Run experiments !!!
    with MPIEvaluator(model) as evaluator:
    # with SequentialEvaluator(model) as evaluator: # left in for local testing
        results = evaluator.perform_experiments(
            scenarios=scen_list,
        )
        
    # Process replications: for now, save each separately
    #  Have to extend the experiments df to match sure index still matches
    #  with the "tall" outcomes arrays
    experiments, outcomes = results
    new_exps = {col:[] for col in experiments}
    new_outs = {out:[] for out in outcomes}

    # Create new dicts as baseline for "tall" dataset
    for rep in range(N_REPLICATIONS):
        for scen in range(N_SCENARIOS):
            for col in experiments:
                new_exps[col].append(experiments.loc[scen, col])
            for out in outcomes:
                new_outs[out].append(list(outcomes[out][scen][rep]))
    # Make sure outcomes are saved as np arrays
    for out in new_outs:
        new_outs[out] = np.array(new_outs[out])

    # Cast experiments to Pandas DF
    experiments = pd.DataFrame(new_exps)
    outcomes = new_outs
    del new_exps # delete to save memory for disk write
    del new_outs #  (not sure this is necessary)

    # Save results !!
    results = experiments, outcomes

    DATE = datetime.now().strftime("%m%d")
    filename = f"results/{N_SCENARIOS}_scen__{N_REPLICATIONS}_reps__{DATE}.tar.gz"
    save_results(results, filename)
