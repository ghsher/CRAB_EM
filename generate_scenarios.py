# !/usr/bin/env python3.11
# -*- coding: utf-8 -*-

"""

@author: ghsher

Script to run any number of experiments on the CRAB model
using the EMA Workbench. 

Information about the EMA Workbench is available at
 https://emaworkbench.readthedocs.io/en/latest/

"""

import pandas as pd
from datetime import datetime

from ema_workbench import (
    Scenario
)

from ema_workbench.em_framework import (
    sample_uncertainties, 
)

from ema_model import get_EMA_CRAB_model

if __name__ == "__main__":
    model = get_EMA_CRAB_model()

    N_SCENARIOS = 1 # XXX: Edit here

    scenarios = sample_uncertainties(
        model,
        n_samples=N_SCENARIOS,
    )
    scenarios.kind = Scenario

    scenarios_dict = {unc.name:[] for unc in model.uncertainties}
    for i, scen in enumerate(scenarios):
        for unc,lvl in scen.items():
            scenarios_dict[unc].append(lvl)

    scenarios_df = pd.DataFrame(scenarios_dict)

    DATE = datetime.now().strftime("%m%d")
    filename = f'data/{N_SCENARIOS}_scenarios__{DATE}.csv'
    scenarios_df.to_csv(filename, index=False)
    print(f'saved Scenarios to {filename}')
                        