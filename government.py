# !/usr/bin/env python3.11
# -*- coding: utf-8 -*-

"""

@author: lizverbeek, TabernaA

This class contains the Government class of the CRAB model. This class is
based on the Mesa Agent class and contains all stages of Government activation
of the CRAB model, as well as functions to set the minimum wage and helper functions
that collect information about the regional economy.

"""

# -- PACKAGES FOR TYPE CHECKING -- #
from __future__ import annotations
from typing import TYPE_CHECKING, Type
if TYPE_CHECKING:
    from model import CRAB_Model

from mesa import Agent

from CRAB_agents import *


import time




# -- MODEL CONSTANTS -- #
TRANSPORT_COST = 0.03
TRANSPORT_COST_RoW = 2 * TRANSPORT_COST
DEMAND_ROW = 0              # NOTE: TESTING, export turned off
FRAC_CONS_IND= 0.2
FRAC_CONS_TRANS = 0.2
FRAC_CONS_CONS = 0.15
FRAC_CONS_INFO = 0.1
FRAC_CONS_FIN = 0.1
FRAC_CONS_SERV = 0.25


FRAC_EXP = 0                # Multiplication factor for export each timestep
FRAC_EXP_INIT = 0           # Fraction of regional consumption for initial export d

FIRMS = [Firm, C26, C27, C28, C29, C30,
        Industry, Construction, Transport,
        Information, Finance, Business_services]

FIRM_TYPES = [C26, C27, C28, C29, C30,
            Industry, Construction, Transport,
            Information, Finance, Business_services]

CAP_FIRM_TYPES = [C26, C27, C28, C29, C30]

CONS_FIRM_TYPES = [Industry, Construction, Transport,
              Information, Finance, Business_services]

# -- HELPER FUNCTIONS -- #
def normalize(firms: list, attribute: str, convert_to_pos=False) -> dict:
    """Normalize firm attributes.

    Args:
        firms (list)            : List of firm objects (of same type)
        attribute (string)      : Attribute to normalize
    Returns:
        attr_norm_dict          : Dict {firm: normalized attribute}
    """

    # Get all attributes values for firms of the same type
    attr_list = [getattr(firm, attribute) for firm in firms]
    # Convert attributes to positive values (applied for competitiveness)
    if convert_to_pos:
        attr_list = attr_list + np.abs(np.min(attr_list, axis=0))
    # Return normalized attribute per firm
    attr_norm = attr_list/(np.linalg.norm(attr_list, 1, axis=0) + 1e-8)
    return dict(zip(firms, np.around(attr_norm, 8)))


def weighted_avg(firms: list, attr_norm: list, regional=False) -> float:
    """Get average attribute for each firm, weighted by market share.

    Args:
        firms           : List of firm objects (of same type)
        attr_norm       : List of normalized attributes (for all firms)
    Returns:
        norm_avg        : Weighted average (by market share)
    """

    # Check to normalize over all regions or only own region
    if regional:
        market_shares = [firm.market_share[firm.region] for firm in firms]
    else:
        # Compute average, weighted by past market share
        market_shares = [firm.market_share for firm in firms]
    norm_avg = np.sum(np.multiply(attr_norm, market_shares), axis=0)
    return np.around(norm_avg, 8)


def get_quantiles(firms: list, attribute: str,
                  tax_quintiles: list=[0.2, 0.4, 0.6, 0.8, 0.99]) -> None:
    """Get sectoral quantiles for given attribute.
    
    Args:
        firms           : List of firms
        attribute       : Attributes to compute quantiles for
        tax_quintiles   : List of probabilities for quantiles
    """
    attr_list = [getattr(firm, attribute) for firm in firms]
    return np.quantile(attr_list, tax_quintiles)



# -- GOVERNMENT CLASS -- #
class Government(Agent):
    """Government class for the CRAB Model. """

    def __init__(self, model: CRAB_Model, region: int, CCA_subsidy: bool=False) -> None:
        """Initialize Government agent. """

        super().__init__(model.next_id(), model)

        # -- GENERAL ATTRIBUTES -- #
        self.region = region

        # -- GOODS MARKET ATTRIBUTES -- #
        self.avg_prod = {k: 1 for k in FIRMS}
        self.prod_increase = {k: 0 for k in FIRMS}
        self.best_cap = {}
        self.top_prod = {}
        self.bailout_cost = 0
        self.new_firms_resources = 0
        self.regional_demands = {}
        self.export_demands = {}
        self.total_capital = {}
        self.capital_new_firm = {}
        self.prices_norm = {}       # Keep normalized prices per firm type
        self.unf_demand_norm = {}   # Keep normalized unfilled demand per firm type

        # -- Trade/export parameters -- #
        self.demand_RoW = DEMAND_ROW
        self.transport_cost = TRANSPORT_COST
        self.transport_cost_RoW = TRANSPORT_COST_RoW

        # -- LABOR MARKET ATTRIBUTES -- #
        self.min_wage = 1
        self.top_wage = 1
        self.avg_wage = 0
        self.income_pp = 0
        self.unempl_subsidy = 0

        # -- FLOOD-RELATED ATTRIBUTES -- #
        self.CCA_subsidy = CCA_subsidy
        self.total_repair_expenses = 0

    def set_min_wage(self, min_wage_frac: float=0.6,
                     unempl_subsidy_frac: float=1) -> None:
        """Sets minimum wages and unemployment subsidy. """

        # Set minimum wage to fraction of average wage
        self.min_wage = max(0.1, round(self.avg_wage, 2) * min_wage_frac)
        # Set unemployement subsidy to fraction of minimum wage
        self.unempl_subsidy = round(max(0.1, self.min_wage * unempl_subsidy_frac), 3)

        # # At beginning: keep minimum wage and unemployment subsidy constant
        #if self.model.schedule.time < :
         #    self.unempl_subsidy = 1
          #   self.min_wage = 1

    def get_best_cap(self, firm_type: type) -> CapitalFirm:
        """Get CapitalFirm with best productivity/price ratio. """

        # Get prod/price ratio for subsample of all CapitalFirms in this region
        cap_firms = self.model.get_firms_by_supplier(firm_type, self.region)
        cap_firms = self.model.RNGs[type(self)].choice(cap_firms, len(cap_firms)//3)
        # Collect capital firm productivities (of last timestep)
        firm_prod_dict = {firm: firm.old_prod / firm.price
                          for firm in cap_firms if firm.region == self.region}
        # Return firm with best prod/price ratio
        return max(firm_prod_dict, key=firm_prod_dict.get, default=None)

    def get_production(self, firm_type: type, bounds=(-0.25, 0.3125)) -> Tuple:
        """Update average production and production increase for given sector.
        
        Args:
            firm_type       : Type of firm to update production values for
                              Options:
                                    Firm: all sectors
                                    CapitalFirm: all capital firms
                                    ConsumptionGoodFirm: all consumption firms
                                    ServiceFirm: all service firms
            bounds          : Production increase bounds
        Returns:
            prod            : Tuple (avg productivity, avg productivity increase)
        """

        # Get employment, total production and old production for this sector
        if firm_type == Firm:
            firms = self.model.get_firms(self.region)
        else:
            firms = self.model.get_firms_by_type(firm_type, self.region)
        employment = sum(firm.size for firm in firms)
        prod = sum(firm.production_made for firm in firms)
        prod_old = max(1, self.avg_prod[firm_type])
        # Update average production and increase in production
        if employment > 0:
            # Get production relative to employment
            avg_prod = prod / employment
            # Get relative increase in production
            prod_increase = (avg_prod - prod_old) / prod_old
            prod_increase = max(bounds[0], min(bounds[1], prod_increase))
            return avg_prod, prod_increase
        else:
            return (0, 0)

    def get_capital(self, firm_type: type) -> Tuple:
        """Get total capital and capital for new firm starting in this sector. """
        
        firms = self.model.get_firms_by_type(firm_type, self.region)
        fraction = self.model.RNGs[type(self)].uniform(0.1, 0.9)
        total_capital = sum(sum(vintage.amount for vintage in firm.capital_vintage)
                            for firm in firms)
        new_firm_capital = round(min(5, fraction * total_capital / len(firms)), 2)
        return total_capital, new_firm_capital

    def stage1(self) -> None:
        """First stage of Government step function. """

        # Set minimum wage (and unemployment subsidy)
        self.set_min_wage()

    def stage2(self) -> None:
        pass

    def stage3(self) -> None:
        pass

    def stage4(self) -> None:
        """Fourth stage of government step function: normalization. """
        
        # Normalize per sector
        for firm_type in CONS_FIRM_TYPES:
            firms = self.model.get_firms_by_type(firm_type, self.region)
            # Normalize prices
            self.prices_norm[firm_type] = normalize(firms, "price")
            # Normalize unfilled demand
            self.unf_demand_norm[firm_type] = normalize(firms, "unfilled_demand")

    def stage5(self) -> None:
        """Fifth stage of Government step function. """

        # Get normalized competiveness and employment and sales quintiles
        self.comp_norm = {}
        self.avg_comp_norm = {}
        self.q_sector_sales = {}
        self.q_sector_employment = {}
        for firm_type in FIRM_TYPES:
            # Get all firms of this type
            firms = self.model.get_firms_by_type(firm_type, self.region)
            # For ConsumptionGood and Service firms: compute normalized competitiveness
            if firm_type not in CAP_FIRM_TYPES:
                self.comp_norm[firm_type] = normalize(firms, "competitiveness",
                                                      convert_to_pos=True)

                comp_norm_list = list(self.comp_norm[firm_type].values())
                self.avg_comp_norm[firm_type] = weighted_avg(firms, comp_norm_list)

            # Get employment and sales quintiles per sector
            self.q_sector_employment[firm_type] = get_quantiles(firms, "size")
            self.q_sector_sales[firm_type] = get_quantiles(firms, "sales")

        # Get regional average wages and unemployment rate
        firms = self.model.get_firms(self.region)
        self.avg_wage = (sum(firm.wage * firm.size for firm in firms) /
                         sum(firm.size for firm in firms))
        households = self.model.get_households(self.region)
        unemployment = sum(1 for hh in households if hh.employer is None)
        self.unemployment_rate = round(max(1, unemployment) /
                                       max(1, len(households)), 2)

        # Add firm entry and exit resources to consumption
        total_consumption = sum(hh.consumption for hh in households)
        entry_exit_resources = (self.bailout_cost - self.new_firms_resources)
        if total_consumption < entry_exit_resources:
            print("More resources than consumption")
        total_consumption += entry_exit_resources
        
        # Update export demand
        if self.model.schedule.time > 10:
            self.demand_RoW = self.demand_RoW * (1 + FRAC_EXP)
        else:
            self.demand_RoW = FRAC_EXP_INIT * total_consumption

        # Save regional and export demands per consumption sector
        # TODO: We should add this in a loop  dict
        ind_consumption = FRAC_CONS_IND * total_consumption
        cons_consumption = FRAC_CONS_CONS * total_consumption
        trans_consumption = FRAC_CONS_TRANS * total_consumption
        info_consumption = FRAC_CONS_INFO * total_consumption
        fin_consumption = FRAC_CONS_FIN * total_consumption
        buss_serv_consumption = FRAC_CONS_SERV * total_consumption

        # service_consumption = total_consumption - goods_consumption
        export_demand_buss_serv = self.demand_RoW * FRAC_CONS_SERV
        export_demand_ind = self.demand_RoW * FRAC_CONS_IND
        export_demand_cons = self.demand_RoW * FRAC_CONS_CONS
        export_demand_trans = self.demand_RoW * FRAC_CONS_TRANS
        export_demand_info = self.demand_RoW * FRAC_CONS_INFO
        export_demand_fin = self.demand_RoW * FRAC_CONS_FIN
    
        self.regional_demands[Business_services] = round(buss_serv_consumption, 3)
        self.export_demands[Business_services] = round(export_demand_buss_serv, 3)
        self.regional_demands[Industry] = round(ind_consumption, 3)
        self.export_demands[Industry] = round(export_demand_ind, 3)
        self.regional_demands[Construction] = round(cons_consumption, 3)
        self.export_demands[Construction] = round(export_demand_cons, 3)
        self.regional_demands[Transport] = round(trans_consumption, 3)
        self.export_demands[Transport] = round(export_demand_trans, 3)
        self.regional_demands[Information] = round(info_consumption, 3)
        self.export_demands[Information] = round(export_demand_info, 3)
        self.regional_demands[Finance] = round(fin_consumption, 3)
        self.export_demands[Finance] = round(export_demand_fin, 3)

    def stage6(self) -> None:
        """Sixth stage of Government step function. """

        # Normalize market shares per region
        self.market_share_norm = {}
        for firm_type in CONS_FIRM_TYPES:
            firms = self.model.get_firms_by_type(firm_type, self.region)
            self.market_share_norm[firm_type] = normalize(firms, "market_share")

    def stage7(self) -> None:
        """Seventh stage of Government step function. """

        # Update average production (and increase) of all firms and for all sectors
        for firm_type in FIRM_TYPES:
            production = self.get_production(firm_type)
            self.avg_prod[firm_type] = production[0]
            self.prod_increase[firm_type] = production[1]

        # Update average net worth
        firms = self.model.get_firms(self.region)
        self.avg_net_worth = sum(firm.net_worth for firm in firms) / (len(firms)+1)

    def stage8(self) -> None:
        """Eighth stage of Government step function. """

        # Get highest wage in this region (of ConsumptionGood and Service firms)
        self.top_wage = max(firm.wage for firm in
                            self.model.get_cons_firms(self.region))


        # Get total capital and capital for firm subsidiaries per sector
        for firm_type in FIRM_TYPES:
            capital = self.get_capital(firm_type)
            self.total_capital[firm_type] = capital[0]
            self.capital_new_firm[firm_type] = capital[1]
            top_prod = self.get_best_cap(firm_type)
            self.top_prod[firm_type] = top_prod.old_prod
            self.best_cap[firm_type] = top_prod
