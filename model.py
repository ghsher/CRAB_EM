# !/usr/bin/env python3.11
# -*- coding: utf-8 -*-

"""

@author: lizverbeek, TabernaA

The model class for the Climate-economy Regional Agent-Based (CRAB) model).
This class is based on the MESA Model class.

"""

from __future__ import annotations  # Used for type checking

import itertools
import numpy as np
import pandas as pd
import networkx as nx

from collections import defaultdict

from mesa import Model
from mesa.time import BaseScheduler
from mesa import DataCollector

from CRAB_agents import *
from government import Government
from schedule import StagedActivationByType
from datacollection import model_vars, agent_vars


# -- INITIALIZE REGION SIZES -- #
N_REGIONS = 1                                       # Number of regions
REGIONS = range(N_REGIONS)
N_HOUSEHOLDS = {REGIONS[0]: 10000,                  # Number of households per region
                }
N_FIRMS = {REGIONS[0]: {CapitalFirm: 50,            # Number of firms per type per region
                        ConsumptionGoodFirm: 100,
                        ServiceFirm: 300},
           }
N_NEW_FIRMS = {REGIONS[0]: {CapitalFirm: 1,       # Avg number of new firms per timestep
                            ConsumptionGoodFirm: 1,     
                            ServiceFirm: 1},
            }

# -- FIRM INITIALIZATION ATTRIBUTES -- #
INIT_NET_WORTH = {CapitalFirm: 50,              # Initial net worth
                  ConsumptionGoodFirm: 50,  
                  ServiceFirm: 50}
INIT_CAP_AMOUNT = {CapitalFirm: 3,              # Initial capital per machine
                   ConsumptionGoodFirm: 2,
                   ServiceFirm: 2}
INIT_N_MACHINES = {CapitalFirm: 20,             # Initial number of machines
                   ConsumptionGoodFirm: 20,
                   ServiceFirm: 15}

# -- ADAPTATION ATTRIBUTES -- #
AVG_HH_CONNECTIONS = 7


class CRAB_Model(Model):
    """Model class for the CRAB model. """

    def __init__(self, random_seed: int, HH_attributes: pd.DataFrame,
                 firm_flood_depths: pd.DataFrame, PMT_weights: pd.DataFrame,
                 firms_RD: bool=True, social_net: bool=True,
                 migration: dict={"Regional": False, "RoW": False},
                 CCA: dict={"Households": False, "Firms": False},
                 debt_sales_ratio: float=2.0,
                 wage_sensitivity_prod: float=0.2,
                 init_markup: float=0.25,
                 capital_firm_cap_out_ratio: float=0.4,
                 min_unempl_emigration: float=0.04,
                 migration_unempl_bounds_diff: float=0.15,
                 deu_discount_factor: float=1.0,
                 flood_timing: int=40,
                 flood_intensity: int=3000) -> None:
        """Initialization of the CRAB model.

        Args:
            random_seed         : Random seed for model 
            HH_attributes       : Household attributes from synthetic population file
            firm_flood_depths   : Firm buildings flood depths per return period
            PMT_weights         : Weights for household CCA decision-making
            firms_RD            : Boolean (firms research and development on/off)
            migration           : Dict (migration_type: True/False) indicating
                                    types of migration
            flood_when          : Dict (year: return period) of flood occurrence(s)
            CCA                 : Dict (agent_type: True/False) indicating whether
                                    households and or firms do CCA
            social_net          : Boolean (social network on/off)
        """
        super().__init__()

        # -- SAVE NUMBER OF REGIONS -- #
        self.n_regions = N_REGIONS

        # -- INITIALIZE SCHEDULER -- #
        # Initialize StagedActivation scheduler
        stages = ["stage1", "stage2", "stage3", "stage4",
                  "stage5", "stage6", "stage7", "stage8"]
        self.schedule = StagedActivationByType(self, stage_list=stages)

        # --- REGULATE STOCHASTICITY --- #
        # Regulate stochasticity with numpy random generator for each agent type and region
        self.RNGs = {}  
        # Create random generator per region per agent type
        agent_types = list(N_FIRMS[0].keys()) + [Household] + [Government]
        for i, agent_class in enumerate(agent_types):
            self.RNGs[agent_class] = np.random.default_rng(random_seed + i)
        # Create separate random generator for household adaptation process
        self.RNGs["Adaptation"] = np.random.default_rng(random_seed + i + 1)
        self.RNGs["Firms_RD"] = np.random.default_rng(random_seed + i + 2)
        self.RNGs["HH_migration_RoW"] = np.random.default_rng(random_seed + i + 3)
        self.RNGs["HH_migration_regional"] = np.random.default_rng(random_seed + i + 4)
        # ------------------------------

        # -- MIGRATION ATTRIBUTES -- #
        self.migration = migration
        self.min_unempl_emigration = min_unempl_emigration
        self.max_unempl_immigration = self.min_unempl_emigration * migration_unempl_bounds_diff

        # -- FLOOD and ADAPTATION ATTRIBUTES -- #
        self.flood_intensity = flood_intensity
        self.flood_timing = flood_timing
        self.firm_flood_depths = firm_flood_depths
        self.HH_attributes = HH_attributes
        self.CCA = CCA
        if self.CCA["Households"]:
            self.social_net = social_net
            self.PMT_weights = PMT_weights

        # -- SAVE INPUT FACTORS AS CONSTANTS -- #
        # Firms' allowed debt as a ratio of sales
        self.DEBT_SALES_RATIO = debt_sales_ratio
        # Sensitivity of wage changes to firm productivity
        self.WAGE_SENSITIVITY_PROD = wage_sensitivity_prod
        # Capital output ratio per firm type
        self.CAP_OUT_RATIO = {                        
            CapitalFirm: capital_firm_cap_out_ratio,             
            ConsumptionGoodFirm: 1,
            ServiceFirm: 1
        }
        # Discount factor for firms weighing adaptation decisions
        self.DEU_DISCOUNT_FACTOR = deu_discount_factor

        # -- INITIALIZE AGENTS -- #
        # Agent control parameters
        self.INIT_MARKUP = {}
        for k in N_FIRMS[REGIONS[0]]: # proxy for firm types
            self.INIT_MARKUP[k] = init_markup # TODO: diff between capital and cons?

        # Add households and firms per region
        self.governments = defaultdict(list)
        self.firms = defaultdict(list)
        self.households = defaultdict(list)
        self.social_networks = {}

        # Add households and firms per region
        for region in REGIONS:
            # -- CREATE FIRMS -- #
            self.firms[region] = {}
            for firm_type, N in N_FIRMS[region].items():
                self.firms[region][firm_type] = []
                # Take random subsample from synthetic population
                idx = self.RNGs[firm_type].choice(firm_flood_depths.index, N)
                for _, f_attributes in firm_flood_depths.loc[idx].iterrows():
                    flood_depths = f_attributes.filter(regex="Flood depth")
                    flood_depths = {int(RP.lstrip("Flood depth RP")): depth
                                    for RP, depth in flood_depths.items()}
                    self.add_firm(firm_type, region=region,
                                  flood_depths=flood_depths,
                                  area=f_attributes["area"],
                                  property_value=f_attributes["Property_value:income"],
                                  market_share=1/N_FIRMS[region][firm_type],
                                  net_worth=INIT_NET_WORTH[firm_type],
                                  init_n_machines=INIT_N_MACHINES[firm_type],
                                  init_cap_amount=INIT_CAP_AMOUNT[firm_type],
                                  cap_out_ratio=self.CAP_OUT_RATIO[firm_type],
                                  markup=self.INIT_MARKUP[firm_type],
                                  )
            
            # -- CREATE HOUSEHOLDS -- #
            self.households[region] = []
            # Take random subsample from synthetic population
            idx = self.RNGs[Household].choice(self.HH_attributes.index,
                                              N_HOUSEHOLDS[region])
            for _, attributes in self.HH_attributes.loc[idx].iterrows():
                hh = self.add_household(region, attributes)

            # -- SOCIAL NETWORK -- #
            G = nx.watts_strogatz_graph(n=N_HOUSEHOLDS[region],
                                        k=AVG_HH_CONNECTIONS, p=0)
            # Relabel nodes for consistency with agent IDs
            start_id = self.households[region][0].unique_id
            G = nx.convert_node_labels_to_integers(G, first_label=start_id)
            self.social_networks[region] = G

            # -- CREATE GOVERNMENT -- #
            self.add_government(region)

        # -- CONNECT SUPPLIERS TO CAPITAL FIRMS -- #
        # NOTE: cannot be done during initialization of firms, since
        #       CapitalFirms have to be connected to themselves.
        for region in REGIONS:
            # NOTE: assumes suppliers within same region.
            cap_firms = self.get_firms_by_type(CapitalFirm, region)
            for firm in cap_firms:
                # Get supplier
                firm.supplier = self.RNGs[type(firm)].choice(cap_firms)
                # Append brochure to offers
                firm.offers = {firm.supplier: firm.supplier.brochure}
                # Also keep track of clients on supplier side
                firm.supplier.clients.append(firm)

        # -- FIRM ATTRIBUTES (per region) -- #
        self.firms_RD = firms_RD
        # Keep track of firms leaving and entering
        self.firm_subsidiaries = defaultdict(list)
        self.firms_to_remove = defaultdict(list)

        # -- DATACOLLECTION -- #
        self.datacollector = DataCollector(model_reporters=model_vars,
                                           agent_reporters=agent_vars)
        # ----------------------------

    def add_government(self, region: int) -> None:
        """Add government to the specified region of the CRAB model. """
        gov = Government(self, region)
        self.governments[region] = gov
        self.schedule.add(gov)

    def add_household(self, region: int, attributes: pd.DataFrame) -> Type[Household]:
        """Add new household to the CRAB model and scheduler. """
        hh = Household(self, region, attributes)
        self.households[region].append(hh)
        self.schedule.add(hh)
        return hh

    def add_firm(self, firm_class: type, region: int, **kwargs) -> Type[Firm]:
        """Add a firm to the CRAB model and scheduler.
        
        Args:
            firm_class          : Type of firm
            region              : Region to add firm to
            **market_share      : 
        """
        firm = firm_class(model=self, region=region, **kwargs)
        self.firms[region][firm_class].append(firm)
        self.schedule.add(firm)
        return firm

    def create_new_firm(self, firm_type: type, region: int) -> None:
        """Create subsidiary of given firm.

        Args:
            firm_type       : Firm class
            region          : Region to create new firm in
        """
        gov = self.governments[region]
        cap_firms = self.get_firms_by_type(CapitalFirm, region)

        # Initialize net worth at (bounded) average net worth
        net_worth = (max(50, round(gov.avg_net_worth_per_sector[firm_type], 4)))
        # Get capital amount for new firms from government
        capital_amount = round(gov.capital_new_firm[firm_type] *
                               self.CAP_OUT_RATIO[firm_type])

        # Initialize random flood depth (from properties file)
        idx = self.RNGs[firm_type].choice(self.firm_flood_depths.index)
        f_attributes = self.firm_flood_depths.loc[idx]
        flood_depths = {int(RP.lstrip("Flood depth RP")): depth
                        for RP, depth in f_attributes.filter(regex="Flood depth").items()}
        area = f_attributes["area"]
        property_value = f_attributes["Property_value:income"]
        
        # Choose random supplier from capital firms
        supplier = self.RNGs[firm_type].choice(cap_firms)
        prod = supplier.brochure["prod"]
        # Set wage to sectoral average (+ noise)
        noise = self.RNGs[firm_type].normal(0, 0.02)
        wage = round(gov.avg_wage_per_sector[firm_type] + noise, 3)

        if firm_type == CapitalFirm:
            # Initialize (sold) machines prod as current regional best
            machine_prod = gov.top_prod
            # Draw a change in productivity from a beta distribution
            bounds = (-0.1, 0.05)
            prod_change = (1 + bounds[0] +
                           self.RNGs["Firms_RD"].beta(2, 4) * (bounds[1]-bounds[0]))
           
            machine_prod *= prod_change
            market_share = 1/N_FIRMS[region][firm_type]
            # Create new firm
            new_firm = firm_type(model=self, region=region, flood_depths=flood_depths,
                                 area=area, property_value=property_value,
                                 market_share=market_share, net_worth=net_worth,
                                 init_n_machines=1, init_cap_amount=capital_amount,
                                 cap_out_ratio=self.CAP_OUT_RATIO[firm_type],
                                 markup=self.INIT_MARKUP[firm_type],
                                 supplier=supplier, sales=capital_amount, prod=prod,
                                 machine_prod=machine_prod, wage=wage, lifetime=0)
        elif firm_type == ConsumptionGoodFirm or ServiceFirm:
            # Create new firm
            new_firm = firm_type(model=self, region=region, flood_depths=flood_depths,
                                 area=area, property_value=property_value,
                                 market_share=0, init_n_machines=1, init_cap_amount=capital_amount,
                                 cap_out_ratio=self.CAP_OUT_RATIO[firm_type],
                                 markup=self.INIT_MARKUP[firm_type],
                                 supplier=supplier, net_worth=net_worth,
                                 sales=0, prod=prod, wage=wage, lifetime=0)
            # Initialze competitiveness from regional average
            new_firm.competitiveness = gov.avg_comp_norm[firm_type]
        else:
            raise ValueError("Firm type not recognized in function create_new_firm().")

        # Add new firm to firms list and schedule
        self.firms[new_firm.region][type(new_firm)].append(new_firm)
        self.schedule.add(new_firm)

        gov.new_firms_resources += net_worth
        return new_firm

    def remove_household(self, household: Type[Household]) -> None:
        """Remove household from model, schedule and social network.

        Args:
            household           : Household to remove
        """
        # Remove household and its connections from social network
        self.social_networks[household.region].remove_node(household.unique_id)
        # Remove from region list
        self.households[household.region].remove(household)
        # Remove from employer
        if household.employer is not None:
            employer = household.employer
            employer.employees.remove(household)
        # Remove from schedule
        self.schedule.remove(household)

    def remove_firm(self, firm: Type[Firm]) -> None:
        """Remove firm from model. """
        self.governments[firm.region].bailout_cost += firm.net_worth
        self.firms[firm.region][type(firm)].remove(firm)
        self.schedule.remove(firm)

    def move_household(self, household: Type[Household],
                       old_region: int, new_region: int) -> None:
        """Move specified household from one region to another.

        Args:
            household       : Household agent to move
            old_region      : Region to remove household from
            new_region      : Region to add household to
        """

        # Change social network connections
        self.social_networks[old_region].remove_node(household.unique_id)
        self.social_networks[new_region].add_node(household.unique_id)
        neighbors = self.RNGs[Household].choice(self.social_networks[new_region].nodes,
                                                size=AVG_HH_CONNECTIONS, replace=False)
        for node in neighbors:
            self.social_networks[new_region].add_edge(household.unique_id, node)

        # Remove from old region agent list and add to new region agent list
        self.households[old_region].remove(household)
        self.households[new_region].append(household)

        # Remove as employee from current employer
        if household.employer is not None:
            household.employer.employees.remove(household)
            household.employer = None

        # Change region
        household.region = new_region

    def migration_regional(self, own_region: int, w: float=1.0) -> None:
        """Interregional migration (within model). Migration is directed from
           given region to another region.

        Args:
            region      : Region of origin before migration
            w           : Wage difference weight in migration probability function
        """

        # Compute probability to migrate for all other regions
        move_to_region = None
        p = 0
        for other_region in REGIONS:
            if other_region != own_region:
                unempl_diff = round(self.governments[other_region].unemployment_rate -
                                    self.governments[own_region].unemployment_rate, 2)
                wage_diff = round(self.governments[other_region].avg_wage -
                                  self.governments[own_region].avg_wage, 2)
                # If wages and unemployment are better elsewhere: consider migration
                if wage_diff > 0 and unempl_diff > 0:
                    # Compute migration probability
                    prob = 1 - np.exp(w * (- wage_diff))
                    # Update migration probability if higher for this region
                    if prob > p:
                        move_to_region = other_region
                        p = prob
        
        # If there is migration from own region: migrate fraction of households
        if p > 0:
            for household in self.households[own_region]:
                # If draw from binomial is successful: migrate to other region
                if self.RNGs["HH_migration_regional"].binomial(1, p):
                    self.move_household(household, own_region, move_to_region)

    def migration_RoW(self, region: int, noisiness: float=0.3) -> None:
        """Handles in- and outmigration processes (outside of model scope).

        Args:
            region          : Region to compute migration for
        """

        # Compute population change from average income change
        gov = self.governments[region]
        avg_income_change = (round(np.mean(gov.yearly_income_pp_change), 3)
                             if gov.yearly_income_pp_change else 0)
        hh_pop = len(self.get_households(region))

        # Scale with given ratio
        noise = self.RNGs["HH_migration_RoW"].uniform(-0.15, 0.15)
        change_factor =((1 - noisiness) * avg_income_change + noisiness * noise)
        hh_pop_change = int(change_factor * hh_pop)
        # If population change is positive: add households
        if hh_pop_change > 0:
            # Check if unemployment is low enough and avg income high enough
            if gov.unemployment_rate < self.max_unempl_immigration and gov.income_pp > 1:
                self.migration_in(region, hh_pop_change)
        # If population change is negative, remove households
        elif hh_pop_change < 0:
            # Check if unemployment is high enough
            if gov.unemployment_rate > self.min_unempl_emigration:
                self.migration_out(region, abs(hh_pop_change))

    def migration_in(self, region: int, pop_change: int) -> None:
        """Create new (randomly sampled) new households, add those to the
           model, schedule and social network.

        Args:
            region          : Region where immigration is coming to
            pop_change      : Number of households entering the region
        """
        for _ in range(pop_change):
            # Get random household attributes from attributes file
            attr_idx = self.RNGs[Household].integers(0, len(self.HH_attributes) - 1)
            attributes = self.HH_attributes.iloc[attr_idx]
            # Add new household to model
            new_household = self.add_household(region, attributes)
            # Add household to social network and connect to neighbors
            self.social_networks[region].add_node(new_household.unique_id)
            neighbors = self.RNGs[Household].choice(self.social_networks[region].nodes,
                                                    size=AVG_HH_CONNECTIONS, replace=False)
            for node in neighbors:
                self.social_networks[region].add_edge(new_household.unique_id, node)

    def migration_out(self, region: int, pop_change: int) -> None:
        """Remove (randomly sampled) household from the model, scheduler
           and social network.

        Args:
            region          : Region where immigration is coming to
            pop_change      : Number of households leaving the region
        """
        # Random sample of households to be removed
        sample_out = self.RNGs[Household].choice(self.households[region],
                                                 size=pop_change, replace=False)
        for hh_to_remove in sample_out:
            self.remove_household(hh_to_remove)

    def get_firms(self, region: int, attr=None):
        """Return list of all firms of specified region and type(s).
        
        Args:
            region          : Region where firms are located
        Returns:
            firms           : List of selected firms
        """
        return list(itertools.chain(*(self.firms[region].values())))

    def get_cons_firms(self, region: int):
        """Return list of all consumption (goods and services) firms in this region.

        Args:
            region          : Region where firms are located
        Returns:
            firms           : List of selected firms
        """
        firms = (self.firms[region][ConsumptionGoodFirm] +
                 self.firms[region][ServiceFirm])
        return firms

    def get_firms_by_type(self, firm_type: type, region: int):
        """Return all firms of specified type and region.

        Args:
            region          : Region where firms are located
            type            : Firm type to select
        Returns:
            firms           : List of selected firms
        """
        return self.firms[region][firm_type]

    def get_households(self, region: int):
        """Return list of all households in this region.

        Args:
            region          : Region where household are located
        Returns:
            firms           : List of households
        """
        return self.households[region]

    def step(self) -> None:
        """Defines a single model step in the CRAB model. """

        # -- HOUSEHOLD MIGRATION -- #
        if self.schedule.steps > 20:
            for region in REGIONS:
                # Migration between model regions
                if self.migration["Regional"]:
                    self.migration_regional(region)
                # Migration to the rest of the world (outside this model)
                if self.migration["RoW"]:
                    self.migration_RoW(region)
            
        # -- FLOOD SHOCK -- #
        if self.schedule.time == self.flood_timing:
            self.flood_now = True
            self.flood_return = self.flood_intensity
        else:
            self.flood_now = False
            self.flood_return = 0

        # -- MODEL STEP: see stages in agent classes for more details -- #
        self.schedule.step()

        # -- REMOVE BANKRUPT FIRMS -- #
        for region in REGIONS:
            # Remove bankrupt firms
            for firm in self.firms_to_remove[region]:
                self.remove_firm(firm)

            # Create new firms
            for firm_type in N_FIRMS[region].keys():
                # n_new_firms = abs(round(self.RNGs[firm_type].normal(2, 1)))
                N = self.RNGs[firm_type].poisson(lam=N_NEW_FIRMS[region][firm_type])
                for _ in range(N):
                    self.create_new_firm(firm_type, region)

            self.governments[region].bailout_cost = 0
            self.governments[region].new_firms_resources = 0

        # -- OUTPUT COLLECTION -- #
        self.datacollector.collect(self)

        for region in REGIONS:
            self.firms_to_remove[region] = []
