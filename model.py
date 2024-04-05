# !/usr/bin/env python3.11
# -*- coding: utf-8 -*-

"""

@author: lizverbeek, TabernaA

The model class for the Climate-economy Regional Agent-Based (CRAB) model).
This class is based on the MESA Model class.

"""

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
N_REGIONS = 1                                      # Number of regions
REGIONS = range(N_REGIONS)
N_HOUSEHOLDS = {REGIONS[0]: 10000}                 # Number of households per region
N_FIRMS = {REGIONS[0]: {C26: 150,
                        Agriculture: 100,        # Number of firms per type per region
                        Industry: 300,
                        Construction: 200,
                        Transport: 150,
                        Utilities: 100,
                        Private_services: 100,
                        Public_services: 100,
                        Wholesale_Retail: 100,
                        }}

# -- FIRM INITIALIZATION ATTRIBUTES -- #
INIT_NET_WORTH = {C26: 100,
                Agriculture: 50,
                Industry: 50,
                Construction: 30,
                Transport: 25,
                Utilities: 20,
                Private_services: 20,
                Public_services: 20,
                Wholesale_Retail: 20}

INIT_CAP_AMOUNT = {C26: 4,
                    Agriculture: 2,
                    Industry: 2,
                    Construction: 2,
                    Transport: 2,
                    Utilities: 2,
                    Private_services: 2,
                    Public_services: 2,
                    Wholesale_Retail: 2}

INIT_N_MACHINES = { C26: 20,  
                    Agriculture: 10,       # Initial number of machines
                    Industry: 15,
                    Construction: 10,
                    Transport: 10,
                    Utilities: 10,
                    Private_services: 10,
                    Public_services: 10,
                    Wholesale_Retail: 10}

INIT_KL_RATIO = {   C26: 2,  
                    Agriculture: 1.4,           # Initial capital-labor ratio
                    Industry: 1.5,
                    Construction: 1.1,
                    Transport: 1.4,
                    Utilities: 1.5,
                    Private_services: 1.2,
                    Public_services: 1,
                    Wholesale_Retail: 1}
INIT_MK  = {        C26: 0.25,  
                    Agriculture : 0.25,               # Initial markup
                    Industry: 0.25,
                    Construction: 0.25,
                    Transport: 0.25,
                    Utilities: 0.25,
                    Private_services: 0.25,
                    Public_services: 0.25,
                    Wholesale_Retail: 0.25}


# -- ADAPTATION ATTRIBUTES -- #
AVG_HH_CONNECTIONS = 7


class CRAB_Model(Model):
    """Model class for the CRAB model. """

    def __init__(self, random_seed: int, HH_attributes: pd.DataFrame,
                 firm_flood_depths: pd.DataFrame, PMT_weights: pd.DataFrame,
                 firms_RD: bool=True, flood_when: dict={},
                 CCA: bool=True, social_net: bool=True) -> None:
        """Initialization of the CRAB model.

        Args:
            random_seed         : Random seed for model 
            HH_attributes       : Household attributes from synthetic population file
            firm_flood_depths   : Firm buildings flood depths per return period
            PMT_weights         : Weights for household CCA decision-making
            firms_RD            : Boolean (firms research and development on/off)
            flood_when          : Dict (year: return period) of flood occurrence(s)
            CCA                 : Boolean (climate change adaptation on/off)
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
        # ------------------------------

        # -- FLOOD and ADAPTATION ATTRIBUTES -- #
        self.flood_when = flood_when
        self.CCA = CCA
        if self.CCA:
            self.social_net = social_net
            self.PMT_weights = PMT_weights

        # -- INITIALIZE AGENTS -- #
        self.governments = defaultdict(list)
        self.firms = defaultdict(list)
        self.households = defaultdict(list)
        # Add households and firms per region
        for region in REGIONS:
            # -- CREATE FIRMS -- #
            self.firms[region] = {}
            for firm_type, N in N_FIRMS[region].items():
                self.firms[region][firm_type] = []
                # Take random subsample from synthetic population
                idx = self.RNGs[firm_type].choice(firm_flood_depths.index, N)
                for _, flood_depths in firm_flood_depths.loc[idx].iterrows():
                    flood_depths = {int(RP.lstrip("Flood depth RP")): depth
                                    for RP, depth in flood_depths.items()}
                    self.add_firm(firm_type, region=region,
                                  flood_depths=flood_depths,
                                  market_share=1/N_FIRMS[region][firm_type],
                                  net_worth=INIT_NET_WORTH[firm_type],
                                  init_n_machines=INIT_N_MACHINES[firm_type],
                                  init_cap_amount=INIT_CAP_AMOUNT[firm_type],
                                  cap_out_ratio=INIT_KL_RATIO[firm_type],
                                  markup=INIT_MK[firm_type],
                                  )
            
            # -- CREATE HOUSEHOLDS -- #
            self.households[region] = []
            # Take random subsample from synthetic population
            idx = self.RNGs[Household].choice(HH_attributes.index,
                                              N_HOUSEHOLDS[region])
            for _, attributes in HH_attributes.loc[idx].iterrows():
                self.add_household(region, attributes)
            # -- SOCIAL NETWORK -- #

            self.G = nx.watts_strogatz_graph(n=N_HOUSEHOLDS[region],
                                             k=AVG_HH_CONNECTIONS, p=0)
            # Relabel nodes for consistency with agent IDs
            self.G = nx.relabel_nodes(self.G, lambda x: x +
                                      sum(N_FIRMS[region].values()) + 1)
            
            # -- CREATE GOVERNMENT -- #
            self.add_government(region)

        # -- CONNECT SUPPLIERS TO CAPITAL FIRMS -- #
        # NOTE: cannot be done during initialization of firms, since
        #       CapitalFirms have to be connected to themselves.
        for region in REGIONS:
            # NOTE: assumes suppliers within same region.
            cap_firms = self.get_cap_firms(region)
            for firm in cap_firms:
                # Get supplier
                suppliers = self.get_firms_by_type(C26, region)
                firm.supplier = self.RNGs[type(firm)].choice(suppliers)
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

    def add_household(self, region: int, attributes: pd.DataFrame) -> None:
        """Add new household to the CRAB model and scheduler. """
        hh = Household(self, region, attributes)
        self.households[region].append(hh)
        self.schedule.add(hh)

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

    def add_subsidiary(self, firm: Type[Firm]) -> None:
        """Create subsidiary of given firm.

        Args:
            firm        : Firm object to create subsidiary of
        """
        gov = self.governments[firm.region]

        # Initialize net worth as fraction of average net worth
        fraction_wealth = (0.9 - 0.1) * self.RNGs[type(firm)].uniform() + 0.1
        net_worth = max(gov.avg_net_worth, 1) * fraction_wealth
        # Get capital amount for new firms from government
        capital_amount = round(gov.capital_new_firm[type(firm)] * firm.cap_out_ratio)
        markup = INIT_MK[type(firm)]
        # Initialize new supplier randomly
        suppliers = self.get_firms_by_type(C26, firm.region)
        supplier = self.RNGs[type(firm)].choice(suppliers)
        
        if isinstance(firm, CapitalFirm):
            # Initialize productivity as fraction of regional top productivity
            x_low, x_up, a, b = (-0.075, 0.075, 2, 4)
            fraction_prod = 1 + x_low + self.RNGs[type(firm)].beta(a, b) * (x_up - x_low)
            machine_prod = np.around(gov.top_prod[type(firm)] * fraction_prod, 3)
            # Get brochure from best capital firm of same type
            brochure = gov.best_cap[type(firm)].brochure
            prod = brochure["prod"]
            # Initialize market share as fraction of total at beginning
            market_share = 1 / N_FIRMS[firm.region][type(firm)]
            # Create new firm
            sub = type(firm)(model=self, region=firm.region,
                             flood_depths=firm.flood_depths,
                             market_share=market_share, net_worth=net_worth,
                             init_n_machines=1, init_cap_amount=capital_amount,
                             markup = markup, cap_out_ratio=firm.cap_out_ratio,
                             sales=capital_amount, wage=firm.wage, price=firm.price,
                             prod=prod, machine_prod=machine_prod, lifetime=0)
        elif isinstance(firm, ConsumptionFirm):
            prod = supplier.brochure["prod"]
            # Create new firm
            sub = type(firm)(model=self, region=firm.region,
                             flood_depths=firm.flood_depths, market_share=0,
                             init_n_machines=1, init_cap_amount=capital_amount,
                             net_worth=net_worth, markup=markup,
                             cap_out_ratio=firm.cap_out_ratio,
                             sales=0, wage=firm.wage, price=firm.price, prod=prod,
                             lifetime=0)
            # Initialze competitiveness from regional average
            sub.competitiveness = gov.avg_comp_norm[type(firm)]
        else:
            raise ValueError("Firm type not recognized in function add_subsidiary().")

        # Set supplier to best regional capital firm
        sub.supplier = supplier
        sub.offers = {sub.supplier: sub.supplier.brochure}
        sub.supplier.clients.append(sub)

        # Add subsidiary to firms list and schedule
        self.firms[sub.region][type(sub)].append(sub)
        self.schedule.add(sub)

        gov.new_firms_resources += gov.avg_net_worth
        return sub

    def remove_firm(self, firm: Type[Firm]) -> None:
        """Remove firm from model. """
        self.governments[firm.region].bailout_cost += firm.net_worth
        self.firms[firm.region][type(firm)].remove(firm)
        self.schedule.remove(firm)

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
        firms = self.firms[region][Agriculture] + self.firms[region][Industry] + \
                self.firms[region][Construction] + self.firms[region][Transport] + \
                self.firms[region][Private_services] + self.firms[region][Public_services]  +  \
                self.firms[region][Utilities] + self.firms[region][Wholesale_Retail]
        return firms
    
    def get_cap_firms(self, region: int):
        """Return list of all capital firms in this region.

        Args:
            region          : Region where firms are located
        Returns:
            firms           : List of selected firms
        """
        firms = self.firms[region][C26]
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
    
    '''
    def get_firms_by_buyer(self, firm_type: type, region: int):
        """Return all firms of specified type and region.

        Args:
            region          : Region where firms are located
            type            : Firm type to select
        Returns:
            firms           : List of selected firms

        """
        if firm_type == C26:
            buyers = self.get_cons_firms(region)
        elif firm_type == C27:
            buyers = [Industry]
        elif firm_type == C28:
            buyers = [Industry, Construction]
        elif firm_type == C29:
            buyers = [Construction, Transport]
        elif firm_type == C30:
            buyers = [Transport]
        
        firms = []
        for buyer in buyers:
            firms += self.firms[region][buyer]
        
        return firms

    def get_firms_by_supplier(self, firm_type: type, region: int):
        """Return all firms of specified type and region.

        Args:
            region          : Region where firms are located
            type            : Firm type to select
        Returns:
            firms           : List of selected firms
        """
        if firm_type == Business_services:
            suppliers = [C26, C28]
        elif firm_type == Industry:
            suppliers = [C27, C28]
        elif firm_type == Construction:
            suppliers = [C28, C29]
        elif firm_type == Transport:
            suppliers = [C29, C30]
        elif firm_type == Information:
            suppliers = [C27]
        elif firm_type == Finance:
            suppliers = [C26]
        else:
            suppliers = [C28]  
       
        firms = []
        for supplier in suppliers:
            firms += self.firms[region][supplier]
        
        return firms
    '''
    
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


        # -- Get data to upddate form CGE model -- #
        '''
        if self.schedule.time % 4 == 0:
            KL_RATIO  =  pd.read_csv("KL_ratio.csv")
            KL_RATIO.set_index("Industry", inplace=True)
            KL_RATIO = KL_RATIO.to_dict()["KL_ratio"]
            # we need to convert the keys to the class type 
            for class_name, new_value in KL_RATIO.items():
                class_obj = globals()[class_name]
                INIT_KL_RATIO[class_obj] = new_value
            
            # -- Update firms' capital-labor ratio -- #
            for region in REGIONS:
                for firm_type in N_FIRMS[region].keys():
                    for firm in self.firms[region][firm_type]:
                        firm.cap_out_ratio = INIT_KL_RATIO[firm_type]
        '''

        

    

        # -- FLOOD SHOCK -- #
        if self.schedule.time in self.flood_when.keys():
            self.flood_now = True
            self.flood_return = self.flood_when[self.schedule.time]
        else:
            self.flood_now = False
            self.flood_return = 0

        # -- MODEL STEP: see stages in agent classes for more details -- #
        self.schedule.step()

        # -- REMOVE BANKRUPT FIRMS -- #
        for region in REGIONS:
            for firm in self.firms_to_remove[region]:
                self.remove_firm(firm)
                # Create new firms as subsidiaries of bankrupt firms
                firm_type = type(firm)
                new_firm = self.add_subsidiary(firm)
            self.governments[region].bailout_cost = 0
            self.governments[region].new_firms_resources = 0

            # TODO: put back here
            # self.firms_to_remove[region] = []

        # -- OUTPUT COLLECTION -- #
        self.datacollector.collect(self)

        # TODO: put back above
        for region in REGIONS:
            self.firms_to_remove[region] = []