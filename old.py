# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

@author: TabernaA

Script to run the model for one single run and save macro (model level) and
micro (agent level) outputs.

"""

import time

import pickle

import random
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import additional_functions as af
# import seaborn as sns
import argparse
from model import CRAB_Model

seed_value = 12345678
random.seed(seed_value)
np.random.seed(seed=seed_value)

import dill as pickle

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model


# Setup argument parser to accept t from command line
parser = argparse.ArgumentParser(description='Run CRAB_Model with command line arguments.')
parser.add_argument('--t', type=int, help='Time step to be used in the model', required=True)
args = parser.parse_args()

# Use the value of t provided from the command line
t = args.t
# --------
# COMMENT:
# Make hyperparameters user input? Put all model parameters here?
# --------
runs = 1
steps = 4
seeds = [0] #, 40816326, 61224489, 81632652, 102040816, 122448979,
        # 142857142, 163265305, 183673469, 204081632, 224489795, 244897958,
         #265306122, 285714285, 306122448, 326530611, 346938775, 367346938,
         #387755101, 408163264, 428571428, 448979591, 469387754, 489795917,
         #510204081, 530612244, 551020407, 571428570, 591836734, 612244897,
         #632653060, 653061223, 673469387, 693877550, 714285713, 734693876,
         #755102040, 775510203, 795918366, 816326529, 836734693, 857142856,
         #877551019, 897959182, 918367346, 938775509, 959183672, 979591835,
         #999999999]

micro_variables = []
macro_variables = []

if t == 0:

    for i, seed in enumerate(seeds):
        # print(("Model initialized with " + str(H) + " Households"))
        print("Run num", i, "with random seed", seed)

        tic = time.time()
        model = CRAB_Model(F1=125, F2=200, F3=300, H=5000, Exp=500, T=0.03,
                    RCP =2.5, floods = False,
                    cca_eff={"Elevation": 1, "Wet_proof": 0.4, "Dry_proof": 0.85},
                    av_network=7, n_regions=1, hh_cca=True,insurance_hh = True,
                    insurance_firms = False, collect_each = 1, seed = seed_value,
                        o = 0.3, max_u_entry = 0.2, min_u_exit = 0.05)
        model.reset_randomizer(seed)  
        for j in range(steps):
            print("#------------ step", j+1, "------------#")
            model.step()
        toc = time.time()
        runtime = toc - tic
        # runtimes.append(runtime)
        print("MODEL runtime: " + str(runtime))
        print()
        save_model(model, 'saved_model.pkl')
        tac = time.time()
        runtime = tac - toc
        print("save model " + str(runtime))

else:
    tic = time.time()
    model = load_model('saved_model.pkl')
    toc = time.time()
    runtime = toc - tic
    print("load model " + str(runtime))
    
    for j in range(steps):
            print("#------------ step", j+1, "------------#")
            model.step()
    tac = time.time()
    runtime = tac - toc
    # runtimes.append(runtime)
    print("MODEL runtime: " + str(runtime))
    print()
    save_model(model, 'saved_model.pkl')




    # # --------
    # # COMMENT:
    # # Look into efficient output saving later
    # # --------
    macro_variable = model.datacollector.get_model_vars_dataframe()
    # # Iteratively add dataframe to list
    macro_variables.append(macro_variable)
    micro_variable = model.datacollector.get_agent_vars_dataframe()
    micro_variable = micro_variable.dropna()
    # # micro_variable[["Demand coastal", "Demand Inland"]] = pd.DataFrame(micro_variable.Real_demand_cap.to_list(), index=micro_variable.index)
    # # micro_variable[["Competitiveness region 0", "Competitiveness region 1", "Competitiveness export"]] = pd.DataFrame(micro_variable.Competitiveness.to_list(), index=micro_variable.index)
    # # micro_variable[["MS region 0", "MS region 1", "MS export"]] = pd.DataFrame(micro_variable.Ms.to_list(), index=micro_variable.index)
    # # micro_variable[["Prod A", "Prod B"]] = pd.DataFrame(micro_variable.Prod.to_list(), index=micro_variable.index)
    micro_variables.append(micro_variable)
# runtime_dict[H] = runtimes

# # SAVE
# with open("test1_4000Households_20runs_400steps.pkl", "wb") as f:
#     pickle.dump(runtime_dict, f)


# # ----------------------------------------------------------------------------
# #                              Output manipulation


# macro_variable[["Av comp 0", "Av comp 1", "Av comp exp"]] = pd.DataFrame(macro_variable.Competitiveness_Regional.to_list(), index=macro_variable.index)
# # macro_variable[["LD cons 0", "LD cons 1"]] = pd.DataFrame(macro_variable.LD_cons.to_list(), index=macro_variable.index)
# macro_variable[["Cons region 0", "Cons region 1"]] = pd.DataFrame(macro_variable.Population_Regional_Cons_Firms.to_list(), index=macro_variable.index)
# # macro_variable[["Cons orders region 0", "Cons orders region 1"]] = pd.DataFrame(macro_variable.orders_cons.to_list(), index=macro_variable.index)
# # macro_variable[["Serv orders region 0", "Serv orders region 1"]] = pd.DataFrame(macro_variable.orders_serv.to_list(), index=macro_variable.index)
# # macro_variable[["Orders received region 0", "Orders received region 1"]] = pd.DataFrame(macro_variable.orders_received.to_list(), index=macro_variable.index)
# macro_variable[["Cap region 0", "Cap region 1"]] = pd.DataFrame(macro_variable.Population_Regional_Cap_Firms.to_list(), index=macro_variable.index)
# macro_variable[["Households region 0", "Households region 1"]] = pd.DataFrame(macro_variable.Population_Regional_Households.to_list(), index=macro_variable.index)
# # macro_variable[["Cons price region 0", "Cons price region 1"]] = pd.DataFrame(macro_variable.Cosumption_price_average.to_list(), index=macro_variable.index)
# # macro_variable[["CCA coeff 0", "CCA coeff 1"]] =pd.DataFrame(macro_variable.Average_CCA_coeff.to_list(), index=macro_variable.index)
# # macro_variable[["Prod cons region 0", "Prod cons region 1", "Delta prod cons region 0", "Delta prod cons region 1"]] = pd.DataFrame(macro_variable.Consumption_firms_av_prod.to_list(), index=macro_variable.index)
# macro_variable[["Prod region 0", "Prod region 1", "Delta prod region 0", "Delta prod region 1"]] = pd.DataFrame(macro_variable.Regional_average_productivity.to_list(), index=macro_variable.index)
# macro_variable[["GDP region 0", "GDP region 1", "GDP total"]] = pd.DataFrame(macro_variable.GDP.to_list(), index=macro_variable.index)
# macro_variable[["Unemployment region 0", "Unemployment region 1", "Unemployment diff 0", "Unemployment diff 1", "Unemployment total"]] = pd.DataFrame(macro_variable.Unemployment_Regional.to_list(), index=macro_variable.index)
# # macro_variable[["MS track 0", "MS track 1"]] = pd.DataFrame(macro_variable.MS_track.to_list(), index=macro_variable.index)
# macro_variable[["CONS 0", "CONS 1", "CONS Total", "Export"]] = pd.DataFrame(macro_variable.CONSUMPTION.to_list(), index=macro_variable.index)
# macro_variable[["INV 0", "INV 1", "INV Total"]] = pd.DataFrame(macro_variable.INVESTMENT.to_list(), index=macro_variable.index)
# # macro_variable[["GDP cons region 0", "GDP cons region 1", "GDP cons total"]] = pd.DataFrame(macro_variable.GDP_cons.to_list(), index=macro_variable.index)
# macro_variable_csv_data = macro_variable.to_csv("data_model_.csv", index=True)
# macro_variable[["Aggr unemployment region 0", "Aggr unemployment region 1"]] = pd.DataFrame(macro_variable.Aggregate_Unemployment.to_list(), index=macro_variable.index)
# macro_variable["Aggr unemployment"] = macro_variable["Aggr unemployment region 0"] + macro_variable["Aggr unemployment region 1"]
# macro_variable[["Aggr employment region 0", "Aggr employment region 1"]] = pd.DataFrame(macro_variable.Aggregate_Employment.to_list(), index=macro_variable.index)
# # macro_variable[["LD cons region 0", "LD cons region 1"]] = pd.DataFrame(macro_variable.LD_cons.to_list(), index=macro_variable.index)
# macro_variable[["Feas prod cons region 0", "Feas prod cons region 1", "Feas prod serv region 0", "Feas prod serv region 1"]] = pd.DataFrame(macro_variable["Feas prod"].to_list(), index=macro_variable.index)
# macro_variable["Aggr employment"] = macro_variable["Aggr employment region 0"] + macro_variable["Aggr employment region 1"]

# macro_variable[["Wages region 0", "Wages region 1", "Wage diff 0", "Wage diff 1"]] = pd.DataFrame(macro_variable.Average_Salary.to_list(), index=macro_variable.index)
# # macro_variable["GDP_cap"] = macro_variable["Real_GDP_cap_coastal"] +  macro_variable["Real_GDP_cap_internal"]
# # macro_variable["GDP_cons"] = macro_variable["Real_GDP_cons_coastal"] +  macro_variable["Real_GDP_cons_internal"]
# # gdp_tot = macro_variable["Real GDP coastal"] + macro_variable["Real GDP internal"]
# macro_variable.to_csv("macro_variables_last_run.csv")
# # gdp_tot.iloc[5:].pct_change().mean()
