from CRAB_agents import *

###################################
########## GENERAL RULES ##########
###################################

# 1. Functions assume data are already filtered (e.g. by firm/household)
# 2. Functions assume Agent data are already grouped by 'Step', unless otherwise noted

###############
### HELPERS ###
###############

def get_model_data(col, model_data):
    return model_data[col].tolist()

def get_aggd_agent_data(col, agg, agent_data, quant=0.5):
    if agg == 'quantile':
        def q(x):
            return x.quantile(quant)
        return agent_data.agg({col:q})[col].tolist()
    return agent_data.agg({col:agg})[col].tolist()

######################################################
### 0. POPULATION (outcome & endogenous predictor) ###
######################################################

#   a. Populations
def get_population(agent_data, aslist=False):
    if aslist:
        return agent_data.size().tolist()
    else:
        return agent_data.size()

#########################################
### 1. ECONOMIC PERFORMANCE (outcome) ###
#########################################

#	a(i). Production (per type of firm)
def get_production(industry_name, model_data):
    col = f'{industry_name} Production Made'
    return get_model_data(col, model_data)

#	a(ii). GDP
def get_GDP(model_data):
    return model_data.loc[
                : ,
                model_data.columns.str.contains('.+Production Made')
            ].sum(axis=1).tolist()

#   b(i). Unemployment rate
def get_unemployment(hh_data, hh_pop=None):
    # If not given population, get it
    if hh_pop is None:
        hh_pop = get_population(hh_data)
    
    # Sum employment numbers and subtract from population to find unemployment
    employed_hhs = hh_data.agg({'Employed' : 'sum'})['Employed']
    unemployment_rate = (hh_pop - employed_hhs) / hh_pop

    return unemployment_rate.tolist()

#   c(i). Gini index
def get_gini(hh_data, hh_pop=None):
    # If not given population, get it
    if hh_pop is None:
        hh_pop = get_population(hh_data)#, aslist=True)

    ginis = []
    for step, row in hh_data:
        # Get agent wealths at current timestep
        wealths = row['Net Worth'] + row['Wage'] # + row['House Value'] ?
        wealths = sorted(wealths.tolist())

        # Get agent population at current timestep
        pop = hh_pop[step]

        # Calculate Gini for current timestep
        rel_wealths = [w * (pop - i) for i, w in enumerate(wealths)]
        B = sum(rel_wealths) / (pop * sum(wealths))
        ginis.append(1 + (1/pop) - 2*B)
    
    return ginis

##################################################
### 2. WEALTH (outcome & endogenous predictor) ###
##################################################

#   a. Net worth (a(i): Households (median), a(ii): Firms (sum))
def get_median_net_worth(agent_data):
    return get_aggd_agent_data('Net Worth', 'median', agent_data)
def get_total_net_worth(agent_data):
    return get_aggd_agent_data('Net Worth', 'sum', agent_data)

#   b. Median house value
def get_median_house_value(agent_data):
    return get_aggd_agent_data('House Value', 'median', agent_data)

#   c. Median income
def get_median_wage(agent_data):
    return get_aggd_agent_data('Wage', 'median', agent_data)

#   d. Minimum wage
def get_minimum_wage(model_data):
    return get_model_data('Minimum Wage', model_data)

##################################################
### 3. FIRM CRITICALITY (endogenous predictor) ###
##################################################

#   a. % of companies in industry that are "large" firms
def get_share_large_firms(agent_data, industry_pop=None):
    if industry_pop is None:
        industry_pop = get_population(agent_data)
    num_large_firms = agent_data.obj[agent_data.obj['Firm Size'] >= 50].groupby('Step').size()
    share_large_firms = num_large_firms / industry_pop

    return share_large_firms.tolist()

#   b. Quantiles firm size
def get_10th_p_firm_size(agent_data):
    return get_aggd_agent_data('Firm Size', 'quantile', agent_data, quant=0.1)
def get_90th_p_firm_size(agent_data):
    return get_aggd_agent_data('Firm Size', 'quantile', agent_data, quant=0.9)
def get_median_firm_size(agent_data):
    return get_aggd_agent_data('Firm Size', 'median', agent_data)

###########################
### 5. IMPACT (outcome) ###
###########################

#   a(i). Total household damages
def get_total_damage(hh_data):
    return get_aggd_agent_data('Damages', 'sum', hh_data)

#   a(ii). Average household damages as a proportion of income
def get_average_damage_income_ratio(hh_data):
    hh_data = hh_data.obj
    hh_data['Damages/Income'] = hh_data['Damages'] / hh_data['Wage']
    return get_aggd_agent_data('Damages/Income', 'mean', hh_data.groupby('Step'))

#   b. Household wealth recovery
# def recovery(hh_data):
#       FML HOW TODO THIS 
#       just measure in mesa hahaha

#########################
### 6. DEBT (outcome) ###
#########################

#   a(i). Total firm  debt
def get_total_firm_debt(firm_data):
    return get_aggd_agent_data('Debt', 'sum', firm_data)

#   b. Government debt/deficit
#       TODO