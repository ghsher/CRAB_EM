# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

@author: TabernaA

Script to run the model for one single run and save macro (model level) and
micro (agent level) outputs.

"""

import os
import time
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from gams import GamsWorkspace
from multiprocessing import Pool
# import seaborn as sns
import argparse
from model import CRAB_Model
import sys

# Increase recursion limit
sys.setrecursionlimit(3000) 

# Set working directory
script_directory = os.path.dirname(os.path.abspath(__file__))
#os.chdir(script_directory)
print("Current Working Directory:", os.getcwd())



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
parser.add_argument('--t', type=str, help='Time step to be used in the model', required=True)
parser.add_argument('--input', type=str, help='Path to the GDX file', required=True)
parser.add_argument('--output', type=str, help='Path to the output file', required=True)
args = parser.parse_args()

# Use the value of t provided from the command line
with open(args.t, 'r') as f:
    file_content = f.read().strip() # Read and clean up the file content
    print(file_content) # Print the content

    # First convert to float, then to integer
    t = int(float(file_content))

print(t)
input_filename = args.input
#gdx_file_path = r"C:\Users\ataberna\Documents\GitHub\CRAB_EU_EMS_coupling\SimulationResults.gdx"
gdx_file_path = os.path.join(script_directory, input_filename)
print( gdx_file_path)

# create GAMs WORKSPACE

WS = GamsWorkspace()
db = WS.add_database_from_gdx(gdx_file_path)

# -- READ FILES -- #
HH_attributes = pd.read_csv("Input/HH_attributes.csv", index_col=0)
firm_flood_depths = pd.read_csv("Input/Firm_attributes.csv", index_col=0)
PMT_weights = pd.read_csv("Input/PMT_weights.csv", index_col=0)


all_data = []
sectors = ['Agriculture', 'Industry_capital', 'Industry_rest', 'Construction',
           'Utilities', 'Logistics', 'Transport', 'Private_Services', 'Public_Services']
current_year = str(t)
regions = ['NL11']

# Assuming 'db' is a list of symbol objects with a 'name' attribute and iterates to give records
for i, symbol in enumerate(db):
    filtered_data = []
    #print(symbol.name)
    for record in symbol:
        region, sector, year = record.keys
        if region in regions and sector in sectors and year == current_year:
            #print(region, sector, year, record.value)
            # Create a dictionary for each record and append to filtered_data
            filtered_data.append({'region': region, 'sector': sector, f'{symbol.name}': record.value})

    # If filtered_data is not empty, create a DataFrame and append it to all_data
    if filtered_data:
        all_data.append(pd.DataFrame(filtered_data))

# Combine all DataFrames into one DataFrame using reduce and merge
from functools import reduce

# Use functools.reduce to perform a cumulative merge operation
# Use outer join to ensure all data is kept and aligned without repetition
combined_data = reduce(lambda left, right: pd.merge(left, right, on=['region', 'sector'], how='outer'), all_data)
# in combined_data, if sector == 'Industry_capital', set 'Consumption' to 0
combined_data.loc[combined_data['sector'] == 'Industry_capital', 'Consumption'] = 0
# in combined_data. normalize the columns Consumption so that it sums to 1
combined_data['Consumption'] = combined_data['Consumption'] / combined_data['Consumption'].sum()

combined_data = combined_data.drop(columns=['region'])
# save as cge_input.csv
combined_data.to_csv('cge_input.csv', index=False)

## TODO: Discuss disaggregation of this data -
# - most data are disaggregated in ABM i.e. \ demand ,
# - some  are equal for all agents (i.e. K  output ratio)
# wages might be a bit more difficult to disaggregate (but
# but we can try to get a similar distirbution and mean by education by changing things in the ABM)
# you send us a reserve wage by education and people can debate


# print  head of the dataframe
print('I am the head of the dataframe, I can be used as an ABM input ', combined_data.head())


# --------
# COMMENT:
# Make hyperparameters user input? Put all model parameters here?
# --------
STEPS = 4
N_RUNS = 8
#RANDOM_SEEDS =  [0, 10, 20, 30, 40, 50, 60, 70, 80, 90] #np.arange(0, 100, int(100/N_RUNS))
SEED = 0

#def run_model(seed):
if t == 2020:
    #print("Run num", i, "with random seed", seed)

    tic = time.time()
    model = CRAB_Model(SEED ,HH_attributes, firm_flood_depths, PMT_weights,
                    CCA=True, social_net=True)
    STEPS = 20

else:
    tic = time.time()
    model = load_model(f'saved_model_{SEED}.pkl')
    #model.reset_randomizer(seed)  
    toc = time.time()
    runtime = toc - tic
    print("load model " + str(runtime))

#model.reset_randomizer(seed)  
'''
for j in range(STEPS):
    print("#------------ step", j+1, "------------#")
    model.step()
'''
model.year = t
for _ in tqdm(range(STEPS)):
    model.step()

toc = time.time()
runtime = toc - tic
print("MODEL runtime: " + str(runtime))
print()
save_model(model, f'saved_model_{SEED}.pkl')
tac = time.time()
runtime = tac - toc
print("save model " + str(runtime))
# # --------
# # COMMENT:
# # Look into efficient output saving later
# # --------

# get 
# # ----------------------------------------------------------------------------
# #                              Output manipulation

macro_variables = model.datacollector.get_model_vars_dataframe()
micro_variables = model.datacollector.get_agent_vars_dataframe()

sectors = ['Agriculture', 'Industry_capital', 'Industry_rest', 'Construction',
           'Utilities', 'Logistics', 'Transport', 'Private_Services', 'Public_Services']

pattern = '(' + '|'.join(sectors) + ')'  # create a pattern that matches any of the sectors

# use the str.extract method to extract the matching part of the string
micro_variables['Type'] = micro_variables['Type'].str.extract(pattern, expand=False)

macro_variables['Step'] = macro_variables.index + 1
micro_variables.reset_index(inplace=True)
# save the macro and micro variables to csv
macro_variables.to_csv('macro_variables.csv', index=False)
# get the Step where the year is t
first_step = macro_variables[macro_variables['Year'] == t]['Step'].iloc[0]
last_step = macro_variables[macro_variables['Year'] == t]['Step'].iloc[-1]


output_filename = args.output
#gdx_file_path = r"C:\Users\ataberna\Documents\GitHub\CRAB_EU_EMS_coupling\SimulationResults.gdx"
gdx_file_path_out = os.path.join(script_directory, output_filename)
print(gdx_file_path_out)

# create GAMs WORKSPACE output
db_out = WS.add_database_from_gdx(gdx_file_path_out)

columns_out = ['Capital amount']
columns_names = ['K_firm_loss']
data_out = {}
n = 0
# groupby by micro_variables
micro_variables_grouped = micro_variables.groupby(['Step', 'Type'])[columns_out].mean().reset_index()
# for each column in micro_variables, get the % change between first step and last step , for each sector in the column 'Type'
for column in micro_variables_grouped.columns:
    if column in columns_out:
        data_out[columns_names[n]] = {}
        # get the % change between first step and last step, for each sector in the column 'Type'
        for sector in sectors:
            df_sector = micro_variables_grouped[(micro_variables_grouped['Type'] == sector)]
            first_value = df_sector[df_sector['Step'] == first_step][column].values[0]
            last_value = df_sector[df_sector['Step'] == last_step][column].values[0]
            # perc change
            perc_change = (last_value - first_value) / first_value * 100
            data_out[columns_names[n]][sector] = perc_change
        n += 1


for symbol in db_out:
    print(symbol.name)
    if symbol.name in data_out:
        data_to_add = data_out[symbol.name]
        for record in symbol:
            if record.keys[0] == 'NL11':
                if record.keys[2] == str(t):
                    # record_keys[1] is the sector, that is the same name in data_out
                    # I need to match them to update the value
                    sector = record.keys[1]
                    if t > 2020:
                        record.value = data_to_add[sector]
                    else:
                        record.value = 0
                    
                


# save the updated database to a new GDX file, using the output file name 
output_filnename = args.output
gdx_file_path_out = os.path.join(script_directory, output_filnename)
# save a sa gdx
db_out.export(gdx_file_path_out)
