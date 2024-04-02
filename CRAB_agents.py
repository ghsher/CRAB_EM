# !/usr/bin/env python3.11
# -*- coding: utf-8 -*-

"""

@author: lizverbeek, TabernaA

This file contains the Household and Firm agent classes for the CRAB model.
Both are based on a CRAB_Agent parent class, containing all shared functions
for any agent in the CRAB model.
The Household class is a simple agent that can engage in labor search to find a job
at one of the Firms.
There are three types of Firms: CapitalFirms, and two types of ConsumptionFirms:
ConsumptionGoodFirms and ServiceFirms.
The CapitalFirms supply machines to other CapitalFirms and to ConsumptionFirms,
which they use to provide goods and services to households for consumption.

"""

# -- PACKAGES FOR TYPE CHECKING -- #
from __future__ import annotations
from typing import TYPE_CHECKING, Type
if TYPE_CHECKING:
    from model import CRAB_Model

# -- OTHER PACKAGES -- #
import numpy as np
import math
import bisect
from collections import deque
from mesa import Agent


# -- FIRM INITIALIZATION VALUES -- #
PROD_DIST = (1, 0.02)  # Normal distribution (mean, std) of initial productivity
WAGE_DIST = (1, 0.02)  # Normal distribution (mean, std) of initial wages

# -- FIRM CONSTANTS -- #
INTEREST_RATE = 0.01

# -- FLOOD CONSTANTS -- #
DAMAGE_CURVES = {"Residential":
                    {"Depth": [0, 0.5, 1, 1.5, 2, 3, 4, 5],
                     "Damage": [0, 0.06, 0.13, 0.18, 0.23, 0.28, 0.32, 0.4]},
                 "Industry":
                    {"Depth": [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6],
                     "Damage": [0, 0.24, 0.37, 0.47, 0.55, 0.69, 0.82, 0.91, 1]}}
DAMAGE_REDUCTION = {"Elevation": 3, "Dry_proof": 0.4, "Wet_proof": 0.5}
# -- ADAPTATION CONSTANTS -- #
CCA_COSTS = {"Elevation": 2, "Dry_proof": 0.5, "Wet_proof": 1}


def systemic_tax(profits: list, sales: float, quintiles: list,
                 taxes: list=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) -> None:
    """Returns tax for given profits and sales.

    Args:
        profits         : List of firm profits per region
        sales           : List of firm sales per region
        quintiles       : List of sectoral sales quintiles
        taxes           : List of tax values per quintile
    """

    if sales >= quintiles[4]:
            tax_rate = taxes[5]
    elif sales < quintiles[4] and sales >= quintiles[3]:
            tax_rate =taxes[4]
    elif sales < quintiles[3] and sales >= quintiles[2]:
            tax_rate = taxes[3]
    elif sales < quintiles[2] and sales>= quintiles[1]:
            tax_rate = taxes[2]
    elif sales < quintiles[1] and sales >= quintiles[0]:
            tax_rate = taxes[1]
    elif sales <= quintiles[0]:
        tax_rate = taxes[0]
    return tax_rate * profits


def depth_to_damage(building_type: str, flood_depth: float) -> float:
    """Returns damage coefficient from flood depth for given building type.

    Args:
        building_type       : Type of building ("Residential" or "Industry")
        flood_depth         : Depth of flood
    Returns:
        damage_coef         : Damage coefficient
    """
    # Get (depth, damage) datapoints for depth-damage curve of building type
    depths = DAMAGE_CURVES[building_type]["Depth"]
    damages = DAMAGE_CURVES[building_type]["Damage"]
    # Interpolate between known datapoints
    damage_coef = np.interp(flood_depth, depths, damages)
    return damage_coef


class CRAB_Agent(Agent):
    """Class representing an actor (firm or household) in the CRAB model. """

    def __init__(self, model: CRAB_Model, region: int, flood_depths: dict) -> None:
        """Initializes an agent in the CRAB model.
        
        Args:
            model           : Model object containing the agent
            region          : Home region of this agent
        """

        super().__init__(model.next_id(), model)

        # -- GENERAL CRAB AGENT ATTRIBUTES -- #
        self.region = region
        self.lifetime = 1

        # -- FLOOD ATTRIBUTES -- #
        self.flood_depths = flood_depths
        self.monetary_damage = 0

    def stage8(self):
        """General CRAB Agent last stage of step function: increase lifetime."""
        self.lifetime += 1        


class Household(CRAB_Agent):
    """Class representing a household in the CRAB model. """

    def __init__(self, model: CRAB_Model, region: int, HH_attributes: pd.Series) -> None:
        """Initialize household agent.
        
        Args:
            model           : Model object containing the household
            region          : Home region of this household
        """

        # -- FLOOD DEPTHS -- # 
        flood_depths = HH_attributes.filter(regex="Flood depth")
        flood_depths = {int(RP.lstrip("Flood depth RP")): depth
                        for RP, depth in flood_depths.items()}
        super().__init__(model, region, flood_depths)

        # -- SOCIO-ECONOMIC ATTRIBUTES -- #
        self.education = HH_attributes["Education"]

        # -- FINANCIAL ATTRIBUTES -- #
        self.consumption = 1
        self.net_worth = HH_attributes["savings_norm"]
        self.house_income_ratio = HH_attributes["House:income ratio"]
        self.house_value = self.house_income_ratio

        # -- LABOR MARKET ATTRIBUTES -- #
        self.employer = None
        self.wage = 1

        # -- FLOOD ATTRIBUTES -- #
        self.adaptation = {"Elevation": 0,      # Elevation     (Height)
                           "Wet_proof": False,  # Wet-proofed   (True/False)
                           "Dry_proof": 0}      # Dry-proofed   (Age)
        self.repair_expenses = 0
        self.adaptation_costs = 0
        self.measure_to_impl = None
        # -- Adaptation attributes -- #
        self.perc_damage = HH_attributes["Flood damage"]
        self.perc_prob = HH_attributes["Flood probability"]
        self.worry = HH_attributes["Worry"]
        self.flood_experience = HH_attributes["Flood experience"]
        self.CCA_perc = {"Elevation": {"Response efficacy": HH_attributes["RE elevation"],
                                       "Self efficacy": HH_attributes["SE elevation"],
                                       "Perceived cost": HH_attributes["PC elevation"]},
                         "Wet_proof": {"Response efficacy": HH_attributes["RE wet"],
                                       "Self efficacy": HH_attributes["SE wet"],
                                       "Perceived cost": HH_attributes["PC wet"]},
                         "Dry_proof": {"Response efficacy": HH_attributes["RE dry"],
                                       "Self efficacy": HH_attributes["SE dry"],
                                       "Perceived cost": HH_attributes["PC dry"]}}
        self.social_exp = HH_attributes["Social expectations"]

    def labor_search(self) -> Firm:
        """ Labor search performed by Household agents.

        Returns:
            employer        : Firm object where household will be employed
        """

        # Check if there are firms with open vacancies
        vacancies_cap = [firm for firm in
                         self.model.get_firms_by_type(CapitalFirm, self.region)
                         if firm.open_vacancies]
        vacancies_cons = [firm for firm in self.model.get_cons_firms(self.region)
                          if firm.open_vacancies]

        # First try to find job at capital firm, then search at other firm types
        for vacancies in [vacancies_cap, vacancies_cons]:
            if vacancies:
                # Get subset of firm vacancies (bounded rationality)
                N = math.ceil(len(vacancies)/3)
                subset = self.model.RNGs[type(self)].choice(vacancies, N)
                # Choose firm with highest wage
                employer = subset[np.argmax([firm.wage for firm in subset])]
                # Add self to list of employees of this firm
                employer.employees.append(self)
                # Close vacancies if firm has enough employees
                if employer.desired_employees == len(employer.employees):
                    employer.open_vacancies = False
                return employer
            else:
                continue

        # Return None if search remains unsuccessful
        return None

    def flood_damage(self) -> None:
        """Calculate total damage to property for this household. """

        # Adjust flood depth for elevation level
        flood_depth = self.flood_depths[self.model.flood_return]
        if self.adaptation["Elevation"]:
            flood_depth -= self.adaptation["Elevation"]
        # Get damage coefficient from flood depth
        self.damage_coef = depth_to_damage("Residential", flood_depth)

        # Reduce damage when wet- or dry_proofed
        if self.adaptation["Dry_proof"] and flood_depth < 1:
            self.damage_coef -= (self.damage_coef * DAMAGE_REDUCTION["Dry_proof"])
        if self.adaptation["Wet_proof"]:
            self.damage_coef -= (self.damage_coef * DAMAGE_REDUCTION["Wet_proof"])
        # Compute monetary damage
        self.monetary_damage += self.house_value * self.damage_coef
        self.flood_experience = 1

    def repair_damage(self) -> None:
        """Repair remaining flood damage and adjust consumption. """

        # Reduce savings to repair damage
        repair_savings = min(self.monetary_damage, self.net_worth)
        self.net_worth = max(0, self.net_worth - repair_savings)

        # Reduce consumption by reparation expenses
        gov = self.model.governments[self.region]
        repair_consumption = max(0, self.consumption - gov.unempl_subsidy)
        self.consumption = max(0, self.consumption - repair_consumption)
        
        # Repair damages if affordable with consumption + savings
        self.repair_expenses += repair_savings + repair_consumption 
        if self.repair_expenses > self.monetary_damage:
            # Add repair expenses to regional total
            gov.total_repair_expenses += self.repair_expenses
            self.monetary_damage = 0

    def compute_PMT(self, measure: str, n_others_adapted: int,
                    other_measure_1: bool, other_measure_2: bool) -> None:
        """Household climate change adaptation (CCA) decision-making, based on
           Protection Motivation Theory (PMT).
           Weights are determined from survey data, see:
                Noll et al. (2022). https://doi.org/10.1038/s41558-021-01222-3
                Taberna et al. (2023). https://doi.org/10.1038/s41598-023-46318-2

        Args:
            measure             : CCA measure to consider
            n_others_adapted    : Fraction of other households in social network
                                  that have already taken this measure
            other_measure_1     : Other measure (1) already taken? (True/False)
            other_measure_2     : Other measure (2) already taken? (True/False)
                                     Please note the order of measure (1) and (2)
                                     should correspond to the PMT_weights file
        """

        # Get PMT weights for this measure
        weights = self.model.PMT_weights[measure]
        # Get PMT attributes (in right order to correspond to weights)
        PMT_attrs = [1,
                     self.perc_damage,
                     self.perc_prob,
                     self.worry,
                     self.worry * self.perc_damage,
                     self.CCA_perc[measure]["Response efficacy"],
                     self.CCA_perc[measure]["Self efficacy"],
                     self.CCA_perc[measure]["Perceived cost"],
                     self.flood_experience,
                     self.social_exp,
                     n_others_adapted,
                     n_others_adapted * self.social_exp,
                     other_measure_1,
                     other_measure_2]
        # Get weighted average of attributes
        y_hat = np.dot(weights.dropna(), PMT_attrs)
        # Take inverse logit function for adaptation (intention) probability
        y_hat = round(min(1, np.exp(y_hat)/(1 + np.exp(y_hat))), 2)

        # Binomial draw represents intention-action barrier
        if y_hat > 0 and self.model.RNGs["Adaptation"].binomial(1, y_hat/4):
            self.measure_to_impl = measure
            self.adaptation_costs = CCA_COSTS[measure]

            # If measure is subsidized: immediately take action
            gov = self.model.governments[self.region]
            if gov.CCA_subsidy == True:
                # Implement measure (for elevation: set equal to new height)
                self.adaptation[measure] = (DAMAGE_REDUCTION["Elevation"]
                                            if measure == "Elevation" else 1)
                self.adaptation_costs = 0
                self.measure_to_impl = None

    def implement_CCA_measure(self, measure) -> None:
        """Implement specified adaptation measure.

        Args:
            measure         : Adaptation measure
                              ("Elevation", "Dry_proof" or "Wet-proof")
        """

        # Spend costs and keep track of expenses by government
        gov = self.model.governments[self.region]
        self.net_worth -= self.adaptation_costs
        gov.total_repair_expenses += self.adaptation_costs
        # Implement measure (for elevation: set equal to new height)
        self.adaptation[measure] = (DAMAGE_REDUCTION["Elevation"]
                                    if measure == "Elevation" else 1)
        # Adjust perceived damage with response efficacy
        resp_eff = self.CCA_perc[self.measure_to_impl]["Response efficacy"]
        self.perc_damage = (self.perc_damage * resp_eff)
        # Reset adaptation implementation variables
        self.adaptation_costs = 0
        self.measure_to_impl = None

    def stage1(self) -> None:
        """First stage of household step function: flood shock. """
        
        # -- Flood shock -- #
        if self.model.flood_now:
            self.flood_damage()

    def stage2(self) -> None:
        pass

    def stage3(self) -> None:
        pass

    def stage4(self) -> None:
        """Fourth stage of Household step function: labor search """

        # Try to find a job at one of the firms
        if self.employer is None:
            self.employer = self.labor_search()
            # If no employer is found, get unemployment subsidy from government 
            if self.employer is None:
                self.wage = self.model.governments[self.region].unempl_subsidy
            else:
                # Otherwise, get wage from employer
                self.wage = self.employer.wage
        else:
            # Update wage if household is already employed
            self.wage = self.employer.wage

    def stage5(self) -> None:
        """Fifth stage of Household step function:
           1) Consumption
           2) Repair possible damages from flood
           3) Adaptation
        """

        # -- Consumption -- #
        self.consumption = self.wage
        # If HH is employed, planning to do adaptation, and there is no subsidy,
        # spend minimum amount (equal to unempl subsidy) on consumption, save rest
        gov = self.model.governments[self.region]
        if (self.employer and self.adaptation_costs > 0 and not gov.CCA_subsidy):
            self.savings = max(0, self.consumption - gov.unempl_subsidy)
            self.consumption -= self.savings
            self.net_worth += self.savings
        else:
            self.savings = 0

        # -- Repair flood damages -- #
        # Update house value relative to changes in average wage
        wage_diff = (gov.avg_wage - gov.avg_wage_prev)/gov.avg_wage_prev
        self.house_value = (self.house_value * (1 + wage_diff)
                            if wage_diff != 0 else self.house_value)
        # Check if still any damage remaining, otherwise reset repair costs
        if self.monetary_damage > 0:
            self.repair_damage()
        else:
            self.repair_expenses = 0

        # -- Adaptation -- #
        if (self.model.CCA and np.any(list(self.flood_depths.values()))):

            # Get household social network for opinion dynamics
            if self.model.social_net:
                neighbor_nodes = list(self.model.G.neighbors(self.unique_id))
                households = self.model.get_households(self.region)
                social_network = [self.model.schedule._agents[node]
                                  for node in neighbor_nodes]

            # Implement planned measure if net worth is high enough
            if self.measure_to_impl and (self.net_worth > self.adaptation_costs):
                self.implement_CCA_measure(self.measure_to_impl)
            
            # Consider taking adaptation action every year if nothing planned yet
            # and there are still measures to be implemented
            if (self.lifetime % 4 == 0 and self.measure_to_impl is None
                    and not np.all(list(self.adaptation.values()))):

                # Get all measures (in order of consideration)
                all_measures = ["Dry_proof", "Wet_proof", "Elevation"]
                for measure in all_measures:

                    # Consider measure if not taken yet
                    if not self.adaptation[measure]:
                        if self.model.social_net:
                            n_others_adapted = sum([bool(neighbor.adaptation[measure])
                                                    for neighbor in social_network])
                        else:
                            n_others_adapted = 0
                        other_measures = all_measures.copy()
                        other_measures.remove(measure)
                        self.compute_PMT(measure, n_others_adapted,
                                         self.adaptation[other_measures[0]],
                                         self.adaptation[other_measures[1]])

                    # Else increase measure lifetime (for dry-proofing)
                    elif self.adaptation[measure]:
                        # If dry-proofing implemented: increase its age
                        self.adaptation["Dry_proof"] += 1
                        # Dry-proofing only lasts 20 years, remove if older
                        if self.adaptation["Dry_proof"] >= 80:
                            self.adaptation["Dry_proof"] = 0
                            # Adjust perceived damage
                            resp_eff = self.CCA_perc["Dry_proof"]["Response efficacy"]
                            self.perc_damage = (self.perc_damage / resp_eff)

    def stage6(self) -> None:
        pass

    def stage7(self) -> None:
        pass

    def stage8(self) -> None:
        """Eighth stage of Household step function: increase lifetime. """
        super().stage8()


class Firm(CRAB_Agent):
    """Class representing a firm in the CRAB model. """

    def __init__(self, model: CRAB_Model, region: int, flood_depths: dict,
                 market_share: float, net_worth: int,
                 init_n_machines: int, init_cap_amount: int,
                 cap_out_ratio: float, supplier: Type[Firm]=None,
                 sales: int=10, wage: float=None, price: float=None,
                 prod: float=None, lifetime: int=1) -> None:
        """Initialize firm agent.

        Args:
            model               : Model object containing the firm
            region              : Home region of this firm
            flood_depths        : Firm building flood depths per flood return period
            market_share        : Initial market share
            net_worth           : Initial net worth
            init_n_machines     : Initial number of machines
            init_cap_amount     : Initial capital amount per machine
            cap_out_ratio       : Capital output ratio
            supplier            : Capital firm supplier
            sales               : Initial sales
            wage                : Initial wage
            price               : Initial price
            prod                : Initial productivity
            lifetime            : Firm lifetime at creation
        """

        # -- FLOOD ATTRIBUTES -- #
        super().__init__(model, region, flood_depths)

        # -- GENERAL FIRM ATTRIBUTES -- #
        self.lifetime = lifetime
        self.net_worth = net_worth
        self.cap_out_ratio = cap_out_ratio
        self.subsidiary_counter = 0
        
        # -- LABOR MARKET ATTRIBUTES -- #
        self.employees = []
        self.size = 0

        # -- CAPITAL GOODS MARKET ATTRIBUTES -- #
        self.prod = prod if prod else model.RNGs[type(self)].normal(PROD_DIST[0],
                                                                    PROD_DIST[1])
        self.wage = wage if wage else model.RNGs[type(self)].normal(WAGE_DIST[0],
                                                                    WAGE_DIST[1])

        self.supplier = supplier
        # If supplier given, also set offers and connect firm to supplier
        if self.supplier:
            self.offers = {supplier: supplier.brochure}
            supplier.clients.append(self)
        self.capital_vintage = [self.Vintage(self, self.prod, init_cap_amount)
                                for _ in range(init_n_machines)]
        self.price = price if price else self.wage/self.prod
        self.sales = sales
        self.quantity_ordered = 0
        self.order_canceled = False
        self.order_reduced = 0

        # -- CONSUMPTION GOODS MARKET ATTRIBUTES -- #
        self.inventories = 0
        self.real_demand = 1
        self.past_demand = deque([1,1])
        self.unfilled_demand = 0
        # Initialize market share per region + entry for export to RoW
        self.market_share = np.repeat(market_share, self.model.n_regions + 1)
        self.market_share_history = deque([])

    class Vintage:
        """Class representing a vintage that consists of multiple
           machines of the same age, lifetime and productivity. """

        def __init__(self, firm: Firm, prod: float, amount: int) -> None:
            """Initialize vintage object.

            Args:
                prod                : Machine productivity
                amount              : Number of machines in this vintage
            """
            self.prod = prod
            self.amount = amount
            self.age = 0
            self.lifetime = 15 + firm.model.RNGs[type(firm)].integers(1, 10)

    def damage_capital(self):
        """Apply flood damage to capital. """
        vin_to_remove = []
        for vintage in self.capital_vintage:
            # If draw from binomial is successful: vintage is destroyed
            if self.model.RNGs[(type(self))].binomial(1, self.damage_coef):
                vin_to_remove.append(vintage)
        for vintage in vin_to_remove:
            self.capital_vintage.remove(vintage)

    def update_capital(self) -> None:
        """Update capital:
           1) Handle flood damage to inventories
           2) Get ordered machines
           3) Remove old machines
        """

        # Handle orders if they are placed
        if self.supplier is not None and self.quantity_ordered > 0:
            # Add new vintage based on quantity ordered and supplier productivity
            new_machine = self.Vintage(self, prod=round(self.supplier.prod, 3),
                                       amount=round(self.quantity_ordered))
            self.capital_vintage.append(new_machine)
            # Reset ordered quantity
            self.quantity_ordered = 0

            self.machines_to_replace = int(round(self.machines_to_replace))
            # Replace machines according to the replacement investment
            while self.machines_to_replace > 0:
                vintage = self.capital_vintage[0]
                if self.machines_to_replace < vintage.amount:
                    vintage.amount -= self.machines_to_replace
                    self.machines_to_replace = 0
                else:
                    self.machines_to_replace -= vintage.amount
                    self.capital_vintage.remove(vintage)

        # Remove machines that are too old
        vin_to_remove = []
        for vintage in self.capital_vintage:
            vintage.age += 1
            if vintage.age > vintage.lifetime:
                vin_to_remove.append(vintage)
        for vintage in vin_to_remove:
            self.capital_vintage.remove(vintage)

        # Reset investment cost
        self.investment_cost = 0

    def capital_investment(self, inv_frac: float=0.1) -> int:
        """Calculate investment in capital from demand, inventories and
           current maximum production.

        Args:
            inv_frac        : Frac. of expected production to keep in inventories
        Returns:
            n_expansion     : Number of new machines to buy
        """

        # Update demand history (after 1st timestep, when history has length 3)
        if len(self.past_demand) > 2:
            # Remove oldest record from history
            self.past_demand.popleft()
        self.past_demand.append(self.real_demand)        
        expected_demand = self.real_demand

        # Get desired level of inventories
        des_inv_level = inv_frac * expected_demand
        
        # Get desired production from inventory levels and demand
        desired_prod = max(0, expected_demand + des_inv_level - self.inventories)
        # Bound desired production to maximum production
        prod_bound = (sum(vintage.amount for vintage in self.capital_vintage) /
                      self.cap_out_ratio)
        self.feasible_production = round(min(desired_prod, prod_bound))

        # If capital stock is too low: expand firm (buy more capital)
        if self.feasible_production < desired_prod:
            n_expansion = np.floor(round(desired_prod - self.feasible_production) *
                                   self.cap_out_ratio)
        else:
            n_expansion = 0

        # Save for datacollection
        self.n_expansion = n_expansion
        return n_expansion

    def replace_capital(self) -> int:
        """Determine number of machines to replace with better offers.

        Returns:
            n_replacements      : Number of machines to replace
        """

        # Potential replacement machines are current best offers
        if self.offers:
            # Get offer with best productivity:price ratio
            ratios = {supplier: brochure["prod"]/brochure["price"]
                      for supplier, brochure in self.offers.items()}
            best_supplier = max(ratios, key=ratios.get)
            new_prod, new_price = self.offers[best_supplier].values()
        else:
            # If there are no offers: pick random capital firm as supplier
            cap_firms = self.model.get_firms_by_type(CapitalFirm, self.region)
            supplier = self.model.RNGs[type(self)].choice(cap_firms)
            supplier.clients.append(self)
            if supplier.region == self.region:
                new_prod, new_price = supplier.brochure.values()

        n_replacements = 0
        for vintage in self.capital_vintage:
            # Compute unit cost advantage (UCA) of new machines
            UCA = self.wage/vintage.prod - self.wage/new_prod
            # Compute payback, only replace if advantage is high enough
            if UCA > 0 and new_price / UCA <= 3:
                n_replacements += vintage.amount

        # Save for datacollection
        self.n_replacements = n_replacements
        return n_replacements

    def place_order(self, n_expansion: int, n_replacements: int) -> int:
        """Choose supplier and place the order of machines.

        Args:
            n_expansion         : Amount of machines firm wants to add
            n_replacements      : Amount of machines firm wants to replace
        Returns:
            n_ordered           : Amount of machines ordered from supplier
        """

        # Reset investment cost and amount of machines ordered
        self.investment_cost = 0
        self.quantity_ordered = 0

        if self.offers:
            # Get productivity:price ratios
            ratios = {supplier: brochure["prod"]/brochure["price"]
                      for supplier, brochure in self.offers.items()}
            best_supplier = max(ratios, key=ratios.get)
            # Check difference with current ratio
            old_ratio = self.prod/self.price
            diff = (old_ratio - ratios[best_supplier])/old_ratio
            # Probability to change is related to difference in prod:price ratio
            p = max(0, 1 - np.exp(diff))
            # Change suppplier if successful or if firm does not have supplier
            if self.model.RNGs[type(self)].binomial(1, p) or not self.supplier:
                self.supplier = best_supplier

        else:
            # No offers? Pick random capital good firm as supplier
            cap_firms = self.model.get_firms_by_type(CapitalFirm, self.region)
            self.supplier = self.model.RNGs[type(self)].choice(cap_firms)

            # Add this firm to supplier client list (if it is not already)
            if self not in self.supplier.clients:
                self.supplier.clients.append(self)
                self.offers[self.supplier] = self.supplier.brochure

        # Limit desired amount of machines to what firm can afford
        n_desired = n_expansion + n_replacements
        n_affordable = max(0, self.net_worth // self.supplier.price)
        n_to_buy = min(n_desired, n_affordable)

        # If more machines desired than can be bought, make debt to invest
        if n_affordable < n_desired and self.net_worth > 0:
            # Compute affordable debt and adjust number of machines bought
            debt_affordable = self.sales * self.model.DEBT_SALES_RATIO
            n_from_debt = debt_affordable // self.supplier.price
            n_to_buy = min(n_desired, n_affordable + n_from_debt)

            # Set debt based on bought machines
            self.debt = (n_to_buy - n_affordable) * self.supplier.price
            if self.debt >= debt_affordable:
                self.credit_rationed = True

        n_ordered = int(np.ceil(n_to_buy))
        if n_ordered > 0:
            # Convert quantity into cost
            self.investment_cost = n_ordered * self.supplier.price
            # Add order to suppliers list
            self.supplier.regional_orders[self] = n_ordered
        elif n_ordered == 0:
            # No orders? Remove supplier
            self.supplier = None
        else:
            raise ValueError("Number of machines ordered can not be negative.")

        return n_ordered

    def cancel_orders(self) -> None:
        """Cancel replacement of machines; reset investment cost; reduce debt. """

        # Reset machines to replace
        self.machines_to_replace = 0
        # If firm is in debt, remove investment cost from debt (not spent)
        if self.debt > 0:
            self.debt = max(0, self.debt - self.investment_cost)
        # Reset investment cost and quantity ordered
        self.investment_cost = 0
        self.quantity_ordered = 0
        self.order_canceled = False

    def reduce_orders(self) -> None:
        """Reduce quantity ordered, machine to replace, investment and debt. """

        # Reduce quantity ordered and machine replacements
        self.quantity_ordered = max(0, self.quantity_ordered - self.order_reduced)
        self.machines_to_replace -= self.order_reduced
        # Get new investment cost and reduce debt
        self.investment_cost = self.quantity_ordered * self.supplier.price
        if self.debt > 0:
            self.debt = max(0, self.debt - self.order_reduced * self.supplier.price)
        self.order_reduced = 0

    def get_avg_prod(self) -> float:
        """Get average firm productivity.

        Returns:
            avg_prod        : Average productivity of firm machines
        """

        # Find most productive machines to satisfy feasible production
        Q = 0
        machines_used = []
        machines = sorted(self.capital_vintage, key=lambda x: x.prod, reverse=True)
        for vintage in machines:
            # Stop when desired amount is reached
            if Q < self.feasible_production:
                machines_used.append(vintage)
                Q += vintage.amount

        # Get average productivity of chosen machines
        avg_prod = (sum(vintage.amount * vintage.prod for vintage in machines_used)
                    / sum(vintage.amount for vintage in machines_used))

        return round(avg_prod, 3)

    def get_labor_demand(self) -> int:
        """Compute labor demand from feasible and average production.
        
        Returns:
            labor_demand        : Number of employees desired
        """

        if self.feasible_production > 0:
            self.prod = self.get_avg_prod()
            labor_demand = max(1, round(self.feasible_production / self.prod))
        else:
            labor_demand = 1

        return labor_demand

    def get_profits(self, RD=False) -> float:
        """Compute firm profits for this timestep.

        Returns:
            profits         : Firm profits (pre-tax)
            RD              : Boolean, research and development (On/Off)
        """

        self.sales = self.demand_filled * self.price
        total_costs = self.cost * self.demand_filled
        if RD:
            profits = round(self.sales - total_costs - self.RD_budget -
                            self.debt * (1 + INTEREST_RATE), 3)
        else:
            profits = round(self.sales - total_costs -
                            self.debt * (1 + INTEREST_RATE), 3)
        self.pre_tax_profits = profits
        return profits

    def update_net_worth(self) -> float:
        """Compute new firm net worth from earnings this timestep. """

        self.net_worth += self.profits - self.investment_cost
        # If firm made profits: pay taxes
        if self.profits > 0:
            gov = self.model.governments[self.region]
            tax = systemic_tax(self.profits, self.demand_filled * self.price,
                               gov.q_sector_sales[type(self)])
            self.net_worth -= tax

    def hire_and_fire(self, labor_demand: int) -> None:
        """Hire (open vacancies) or fire employees based on labor demand,
           current number of employees, profits and wage.

        Args:
            labor_demand        : Firm's demand for labor
        """

        # Bound desired number of employees
        self.open_vacancies = False
        self.desired_employees = max(1, labor_demand)

        # Open vacancies if less employees than desired
        if self.desired_employees > self.size:
            self.open_vacancies = True
        # Fire employees if more than desired and profits are low
        elif self.desired_employees < self.size:
            # Fire undesired employees
            n_to_fire = self.size - self.desired_employees
            for employee in self.employees[:n_to_fire]:
                employee.employer = None
            self.employees = self.employees[n_to_fire:]

    def create_subsidiaries(self) -> None:
        """Create subsidiaries for firms with profits higher than twice their
           current wage, for at least three timesteps in a row.
        """

        # Check if unemployment rate above zero and firm size not in first or last quintile
        gov = self.model.governments[self.region]
        empl_sector_q = gov.q_sector_employment[type(self)]
        if (gov.unemployment_rate > 0
                and self.size > empl_sector_q[0]
                and self.size < empl_sector_q[-1]
           ):
            if 0.8 * self.profits > self.wage * 2:
                self.subsidiary_counter += 1
                if self.subsidiary_counter > 7:
                    self.model.firm_subsidiaries[self.region].append(self)
                    self.subsidiary_counter = 0
            else:
                self.subsidiary_counter = 0

    def remove_employees(self) -> None:
        """Fire all firm employees. """
        for employee in self.employees:
            employee.employer = None
            self.employees = []

    def remove_offers(self) -> None:
        """Remove firm from client list of suppliers."""
        for supplier in self.offers.keys():
            supplier.clients.remove(self)

    def stage3(self) -> None:
        """Third stage of firm step function: set wages. """
        self.update_wage()

    def stage4(self) -> None:
        """Fourth stage of firm step function: destroy productivity (by flood). """
        if self.damage_coef > 0:
            self.prod -= self.prod * self.damage_coef


class CapitalFirm(Firm):
    """Class representing a capital firm in the CRAB model. """

    def __init__(self, machine_prod: float=None, **kwargs) -> None:
        """Initialize capital firm agent. """

        super().__init__(**kwargs)

        # -- Initialize CapitalFirm-specific attributes -- #
        # -- CAPITAL GOODS MARKET: SUPPLY SIDE -- #
        self.clients = []
        self.regional_orders = {}
        # Initialize productivity of sold machines
        self.machine_prod = machine_prod if machine_prod else self.prod
        self.brochure = {"prod": self.machine_prod, "price": self.price}

        # -- CAPITAL GOODS MARKET: DEMAND SIDE -- #
        self.offers = {}
        self.production_made = 0

    def RD(self, RD_frac=0.04, IN_frac=0.5):
        """Research and development: innovation and imitation.
        
        Args:
            RD_frac      : Fraction of sales or net worth spent on R&D
            IN_frac      : Fraction of R&D budget spent on innovation
        """
        
        # Calculate total R&D budget
        self.RD_budget = max( self.wage, (RD_frac * self.sales if self.sales > 0 else
                          (RD_frac * self.net_worth if self.net_worth >= 0 else 0)))
        # Innovation and imitation
        IN_budget = IN_frac * self.RD_budget
        IM_budget = self.RD_budget - IN_budget
        # Adopt new technologies: update productivity by innovation or imitation
        self.machine_prod = round(max(self.machine_prod,
                                  self.innovate(IN_budget/self.wage),
                                  self.imitate(IM_budget/self.wage), 1), 3)

    def innovate(self, IN_budget: float, Z: float=0.3, a: float=3, b: float=3,
                 bounds: Tuple=(-0.05, 0.05)):
        """ Innovation process perfomed by Firm agents.

        Args:
            IN_budget       : Innovation budget
            Z               : Budget scaling factor for innovation succes
            a               : Alpha parameter for Beta distribution
            b               : Beta parameter for Beta distribution
            bounds          : Bounds for productivity change
        """

        # Binomial draw to determine success of innovation
        p = 1 - math.exp(-Z * IN_budget)
        if self.model.RNGs["Firms_RD"].binomial(1, p):
            # Draw change in productivity from beta distribution
            prod_change = (1 + bounds[0] +
                           self.model.RNGs["Firms_RD"].beta(a, b)
                           * (bounds[1] - bounds[0]))
            return prod_change * self.machine_prod
        else:
            return 0

    def imitate(self, IM_budget: float, Z: float=0.3):
        """Imitation process performed by capital firms.

        Args:
            IM_budget   : Imitation budget
            Z           : Budget scaling factor for imitation success
        """

        # Binomial draw to determine success of imitation
        p = 1 - math.exp(-Z * IM_budget)
        if self.model.RNGs["Firms_RD"].binomial(1, p):
            firms = self.model.get_firms_by_type(type(self), self.region)
            with np.errstate(divide='ignore'):
                # Compute technological distances for all other capital firms
                other_prods = np.array([firm.machine_prod for firm in firms])
                # Get probabilities to imitate other firms
                IM_prob = 1/np.abs(self.machine_prod - other_prods)
                IM_prob = np.nan_to_num(IM_prob, posinf=0, neginf=0)

            if sum(IM_prob) > 0:
                # Pick firm to imitate from normalized cumulative imitation prob
                IM_prob = np.cumsum(IM_prob)/np.cumsum(IM_prob)[-1]
                j = bisect.bisect_right(IM_prob,
                                        self.model.RNGs["Firms_RD"].uniform(0, 1))
                return firms[j].machine_prod
        else:
            return 0

    def update_prod_cost(self, prod) -> None:
        """Computes the unit cost of production for this firm.
        
        Args:
            prod            : Firm productivity
        """

        # Use reduced productivity in case of flood shock
        if self.damage_coef > 0:
            prod -= prod * self.damage_coef
        self.cost = self.wage / prod

    def update_price(self, markup: float=0.4) -> None:
      """Update firm unit price. """
      self.price = round((1 + markup) * self.cost, 3)

    def advertise(self) -> None:
        """Advertise: create new brochures, select new clients, send brochures."""

        # Create brochures
        self.brochure = {"prod": self.machine_prod, "price": self.price}

        # Randomly sample other firms to become clients
        firms = list(set(self.model.get_firms(self.region)) - set([self]))
        if len(self.clients) > 1:
            # Number of new clients is fraction of number of current clients
            N = 1 + round(len(self.clients) * 0.2)
        else:
            # If no clients yet: sample fixed number of new clients
            N = 10
        new_clients = self.model.RNGs[type(self)].choice(firms, N, replace=False)
        # Add potential clients to own clients, avoid duplicates
        self.clients = list(set(self.clients + list(new_clients)))

        # Send brochure to chosen firms
        for client in self.clients:
            client.offers[self] = self.brochure

    def update_wage(self) -> None:
        """Set firm wage, based on minimum wage and current top wage """

        # Set new wage to current top wage in the region, bounded by minimum wage
        gov = self.model.governments[self.region]
        noise = self.model.RNGs[type(self)].normal(0, 0.02)
        self.wage = max(gov.min_wage, round(gov.top_wage + noise, 3))

    def accounting_orders(self) -> None:
        """Check if all orders can be satisfied, else: cancel or reduce orders. """

        # Check how much demand can be filled with production and inventories
        self.production_made = max(0, round(self.size * self.prod))
        stock_available = self.production_made + self.inventories
        self.demand_filled = min(stock_available, self.real_demand)

        # Get unfilled demand and update inventories
        self.unfilled_demand = max(0, self.real_demand - stock_available)
        self.inventories = max(0, stock_available - self.real_demand)

        # If demand cannot be filled: cancel or reduce orders
        if self.demand_filled < self.real_demand:
            # Get total amount to cancel
            amount_to_cancel = self.real_demand - self.demand_filled
            # Split capital and consumption firm orders
            cap_orders = list({k: v for k, v in self.regional_orders.items()
                               if isinstance(k, CapitalFirm)}.items())
            cons_orders = list({k: v for k, v in self.regional_orders.items()
                                if not isinstance(k, CapitalFirm)}.items())
            # Shuffle orders per firm type
            self.model.RNGs[type(self)].shuffle(cap_orders)
            self.model.RNGs[type(self)].shuffle(cons_orders)
            # Set capital firms to back of the orders (to cancel) list
            orders = cons_orders + cap_orders

            # Delete or reduce orders based on amount to cancel
            for order in orders:
                buyer = order[0]
                # If more should be canceled: remove full order
                if order[1] <= amount_to_cancel:
                    canceled_amount = order[1]
                    buyer.order_canceled = True
                # If order is bigger than amount to cancel: reduce order
                else:
                    canceled_amount = min(order[1], amount_to_cancel)
                    buyer.order_reduced = canceled_amount
                amount_to_cancel -= canceled_amount

        # Reset orders
        self.regional_orders = {}

    def remove_supplier(self) -> None:
        """Remove firm as supplier of all its clients. """
        for client in self.clients:
            if client.supplier == self:
                client.supplier = None

    def stage1(self) -> None:
        """First stage of capital firm step function:
           1) Compute flood damage
           2) Research and development
           3) Update capital, cost and price
           4) Advertise own machines.
        """

        # -- FLOOD SHOCK: compute damage coefficient -- #
        if self.model.flood_now:
            flood_depth = self.flood_depths[self.model.flood_return]
            self.damage_coef = depth_to_damage("Industry", flood_depth)
        else:
            self.damage_coef = 0
        # In case of flood damage: destroy (part of) capital
        if self.damage_coef > 0:
            self.damage_capital()

        # -- RESEARCH AND DEVELOPMENT -- #
        if self.model.firms_RD:
            self.RD()
        # Update capital and reset debt
        self.update_capital()
        self.debt = 0

        # CapitalFirms-specific: update production cost + price and advertise
        self.update_prod_cost(self.prod)
        self.update_price()
        self.advertise()

    def stage2(self) -> None:
        """Second stage of capital firm step function: expand and replace machines."""

        # Decide whether to expand
        n_expansion = self.capital_investment()
        # If flood: destroy part of inventories
        if self.damage_coef > 0:
            self.inventories -= self.damage_coef * self.inventories
        # Decide whether to replace capital
        n_replacements = self.replace_capital()

        # Check if net worth allows to make investments
        if n_replacements > 0 or n_expansion > 0:
            if self.net_worth > (self.size * self.wage):
                # Order new machines for expansion and replacement
                self.quantity_ordered = self.place_order(n_expansion, n_replacements)
                # Set number of machines to be replaced after expansion
                self.machines_to_replace = max(0, self.quantity_ordered - n_expansion)

    def stage3(self) -> None:
        """Third stage of capital firm step function. """

        # Inherit from Firm class: set wages
        super().stage3()

        # Update demand from regional orders
        self.real_demand = sum(self.regional_orders.values())

        labor_demand = self.get_labor_demand()
        # For capital firms: bound labor demand by real demand and productivity
        labor_demand = min(labor_demand, round(self.real_demand / self.prod))
        # Add labor demand for R&D
        if self.model.firms_RD:
            labor_demand += round(self.RD_budget / self.wage)

        # Open vacancies and fire employees
        self.hire_and_fire(labor_demand)

    def stage4(self) -> None:
        """Fourth stage of capital firm step function: destroy productivity."""
        super().stage4()

    def stage5(self) -> None:
        """Fifth stage of Capital firm step function. """
        
        # Save firm size
        self.size = len(self.employees)
        # Check if demand can be filled, cancel or reduce orders if needed
        self.accounting_orders()

    def stage6(self) -> None:
        """Sixth stage of Capital firm step function. """

        # -- ACCOUNTING -- #
        # If order is canceled: reset order and investment cost
        if self.order_canceled:
            self.cancel_orders()

        # If order is reduced: reduce quantity, investment cost and machine replacements
        if self.order_reduced > 0 and self.supplier is not None:
            self.reduce_orders()

        # Update profits and add earnings to net worth
        self.profits = self.get_profits(RD=self.model.firms_RD)
        # Update profits and add earnings to net worth
        self.update_net_worth()
        # If new worth is positive firm is not credit constrained
        if self.net_worth > 0:
            self.credit_rationed = False

    def stage7(self) -> None:
        pass

    def stage8(self) -> None:
        """Eighth stage of Capital firm step function. """

        if self.lifetime > 1:
            self.create_subsidiaries()

            # Remove firm from model if net worth or demand is too low
            if self.net_worth <= 0 and sum(self.past_demand) < 1:
                # Check that enough firms of this type still exist
                firms = self.model.get_firms_by_type(type(self), self.region)
                if len(firms) > 10:
                    # Save firm in model to remove at end of timestep
                    self.model.firms_to_remove[self.region].append(self)
                    # Fire employees and remove firm from client list of suppliers
                    self.remove_employees()
                    self.remove_offers()
                    # Remove firm as supplier of its clients
                    self.remove_supplier()

        self.offers = {}
        # Inherit CRAB Agent stage 8 dynamic: increase lifetime
        super().stage8()


class ConsumptionFirm(Firm):
    """Class representing a consumption goods firm in the CRAB model. """

    def __init__(self, competitiveness: float=1, **kwargs) -> None:
        """Initialize capital firm agent. """

        super().__init__(**kwargs)

        # -- CAPITAL GOODS MARKET: DEMAND SIDE -- #
        # Randomly pick an initial supplier from all CapitalFirms
        cap_firms = self.model.get_firms_by_type(CapitalFirm, self.region)
        self.supplier = self.model.RNGs[type(self)].choice(cap_firms)
        self.offers = {self.supplier: self.supplier.brochure}
        self.supplier.clients.append(self)

        # -- INITIAL MARKET PARAMETERS -- #
        self.competitiveness = np.repeat(competitiveness, self.model.n_regions+1)
        self.production_made = 1
        self.old_prod = self.prod

    def update_wage(self, prod_bounds=(-0.25, 0.25)) -> None:
        """Set firm wage, based on average regional productivity
           and regional minimum wage.

        Args:
            prod_bounds     : Own productivity change bounds
        """

        # Read wage sensititivities from model
        sensitivity_prod = self.model.WAGE_SENSITIVITY_PROD
        sensitivity_avg_prod = 1-sensitivity_prod
        # Get (bounded) productivity change
        prod_diff = max(prod_bounds[0], min(prod_bounds[1],
                        (self.prod - self.old_prod)/self.old_prod))
        # Bound wage by minimum wage (determined by government)
        prod_diff = max(-0.25, min(0.25, (self.prod - self.old_prod)/self.old_prod))
        # Regional productivity change, calculated by government
        gov = self.model.governments[self.region]
        avg_prod_diff = gov.prod_increase[type(self)]
        # Recalculate wage level
        self.wage = max(gov.min_wage,
                        round(self.wage *
                              (1 + 
                               sensitivity_prod * prod_diff +
                               (1 - sensitivity_avg_prod) * avg_prod_diff),
                              3))

    def compete_and_sell(self, v: float=0.05):
        """Updates firm cost, markup and price.

        Args:
            v           : Markup/market_share ratio
        """

        # Update cost, avoiding division by zero
        self.cost = self.wage / self.prod if (self.prod > 0) else self.wage

        # Compute (bounded) markup from market share history
        if len(self.market_share_history) > 1 and self.market_share_history[-2] > 0:
            self.markup = max(0.01, min(0.4,
                              round(self.markup * (1 + v *
                                    ((self.market_share_history[-1] -
                                      self.market_share_history[-2]) /
                                      self.market_share_history[-2])), 5)))
        else:
            self.markup = 0.35

        # Adjust price based on new cost and markup, bounded between
        # 0.7 and 1.3 times the old price to avoid large oscillations
        self.price = max(0.7 * self.price,
                         min(1.3 * self.price,
                             round((1 + self.markup) * self.cost, 8)))

    def update_market_share(self, chi: float=1.0) -> float:
        """Compute firm market share from competitiveness.

        Args:
            chi                 : Scaling factor for level of competitiveness
        """
        if self.lifetime == 0:
            # Initial market shares for newly created firms
            cap_amount = sum(vintage.amount for vintage in self.capital_vintage)
            cap_stock = cap_amount / self.cap_out_ratio
            cap_regional = self.model.governments[self.region].total_capital
            MS = np.repeat(max(cap_stock/cap_regional[type(self)], 1e-4), 2)
        else:
            # Compute new market share from competitiveness
            avg_comp = self.model.governments[self.region].avg_comp_norm[type(self)]
            with np.errstate(divide="ignore", invalid="ignore"):
                # Ignore zero division error, replace NaN with 0
                MS = (self.market_share *
                      (1 + chi * (self.competitiveness - avg_comp) / avg_comp))
                MS[np.isnan(MS)] = 0
        self.market_share = np.around(MS, 8)

    def stage1(self) -> None:
        """First stage of consumption firm step function:
           First stage of capital firm step function:
           1) Compute flood damage
           2) Update capital, cost and price
           3) Advertise own machines.
        """

        # -- FLOOD SHOCK: compute damage coefficient -- #
        if self.model.flood_now:
            flood_depth = self.flood_depths[self.model.flood_return]
            self.damage_coef = depth_to_damage("Industry", flood_depth)
        else:
            self.damage_coef = 0

        # In case of flood damage: destroy (part of) capital
        if self.damage_coef > 0:
            self.damage_capital()

        # Update capital and reset debt
        self.update_capital()
        self.debt = 0

        # Decide whether to expand
        n_expansion = self.capital_investment()
        # If flood: destroy part of inventories
        if self.damage_coef > 0:
            self.inventories -= self.damage_coef * self.inventories
        # Decide whether to replace capital
        n_replacements = self.replace_capital()

        # Check if net worth allows to make investments
        if n_replacements > 0 or n_expansion > 0:
            if self.net_worth > (self.size * self.wage):
                # Order new machines for expansion and replacement
                self.quantity_ordered = self.place_order(n_expansion, n_replacements)
                # Set number of machines to be replaced after expansion
                self.machines_to_replace = max(0, self.quantity_ordered - n_expansion)

    def stage2(self) -> None:
        pass

    def stage3(self) -> None:
        """Third stage of consumption firm step function. """

        super().stage3()
        # Save old production
        self.old_prod = self.prod
        labor_demand = self.get_labor_demand()
        self.hire_and_fire(labor_demand)

    def stage4(self) -> None:
        """Fourth stage of consumption firm step function:
           1) Destroy productivity
           2) Compete and sell
        """
        self.compete_and_sell()

    def stage5(self) -> None:
        """Fifth stage of Consumption firm step function. """
        
        # Save firm size
        self.size = len(self.employees)
        
        # Get normalized prices and demand
        gov = self.model.governments[self.region]
        self.price_norm = round(gov.prices_norm[type(self)][self], 8)
        self.unfilled_demand = round(gov.unf_demand_norm[type(self)][self], 8)

        # Get competitiveness (model regions, Export)
        transport_costs = np.ones(len(self.competitiveness))
        transport_costs[-1] += gov.transport_cost_RoW
        self.competitiveness = (transport_costs * self.price_norm
                                - np.array(self.unfilled_demand))

    def stage6(self) -> None:
        """Sixth stage of Consumption firm step function. """

        # Update competitiveness
        gov = self.model.governments[self.region]
        self.competitiveness = gov.comp_norm[type(self)][self]
        # Update market share from new competitiveness
        self.update_market_share()

    def stage7(self) -> None:
        """Seventh stage of consumption firm step function. """

        # Normalize and update market share
        gov = self.model.governments[self.region]
        self.market_share = gov.market_share_norm[type(self)][self]

        # Remove old market share history (after 2 timesteps)
        if len(self.market_share_history) > 2:
            self.market_share_history.popleft()
        # Save market share history
        self.market_share_history.append(round(sum(self.market_share), 5))

        # Compute real demand from regional demand and market shares
        self.real_demand = np.round(sum([gov.regional_demands[type(self)],
                                         gov.export_demands[type(self)]] 
                                        * self.market_share / self.price), 3)
        # Check available stock
        self.production_made = self.size * self.prod
        stock_available = self.production_made + self.inventories
        # Get filled and unfilled demand and update inventories
        self.demand_filled = min(stock_available, self.real_demand)
        self.unfilled_demand = max(0, self.real_demand - stock_available)
        self.inventories = max(0, stock_available - self.real_demand)

        # -- ACCOUNTING -- #
        # If order is canceled: reset order and investment cost
        if self.order_canceled:
            self.cancel_orders()

        # If order is reduced: reduce quantity, investment cost and machine replacements
        if self.order_reduced > 0 and self.supplier is not None:
            self.reduce_orders()

        # Update profits and add earnings to net worth
        self.profits = self.get_profits()
        # Update profits and add earnings to net worth
        self.update_net_worth()
        # If new worth is positive firm is not credit constrained
        if self.net_worth > 0:
            self.credit_rationed = False

    def stage8(self):
        """Eighth stage of consumption firm step function. """

        if self.lifetime > 1:
            self.create_subsidiaries()

            # Remove firm from model if market share or demand is too low
            if (self.market_share[self.region] < 1e-6
                or sum(self.past_demand) < 1):
                # Check that enough firms of this type still exist
                firms = self.model.get_firms_by_type(type(self), self.region)
                if len(firms) > 10:
                    self.model.firms_to_remove[self.region].append(self)
                    # Fire employees and remove offers from suppliers
                    self.remove_employees()
                    self.remove_offers()

        # Reset offers
        self.offers = {}
        # Inherit CRAB Agent stage 8 dynamic: increase lifetime
        super().stage8()


class ConsumptionGoodFirm(ConsumptionFirm):
    """Class representing a Consumption Goods firm in the CRAB model. """

    def __init__(self, **kwargs) -> None:
        """Initialize consumption goods firm agent. """

        super().__init__(**kwargs)


class ServiceFirm(ConsumptionFirm):
    """Class representing a Consumption Services firm in the CRAB model. """

    def __init__(self, **kwargs) -> None:
        """Initialize Service firm agent. """

        super().__init__(**kwargs)
