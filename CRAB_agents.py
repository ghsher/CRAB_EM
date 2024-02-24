# !/usr/bin/env python3.11
# -*- coding: utf-8 -*-

"""

@author: lizverbeek, TabernaA

This file contains the Household and Firm agent classes for the CRAB model.
Both are based on a CRAB_Agent parent class, containing all shared functions
for any agent in the CRAB model.
The Household class is a simple agent that can engage in labor search to find a job
at one of the Firms.
There are three types of Firms: CapitalFirms, ConsumptionGoodFirms and ServiceFirms.
The CapitalFirms supply machines to the Consum

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
PROD_DIST = (1.05, 0.02)  # Normal distribution (mean, std) of initial wages
WAGE_DIST = (1, 0.02)  # Normal distribution (mean, std) of initial productivity

# -- FIRM CONSTANTS -- #
INTEREST_RATE = 0.01
DEBT_SALES_RATIO = 2   # Ratio affordable debt : sales


def systemic_tax(profits: list, sales: float, quintiles: list,
                 taxes: list=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) -> None:
    """

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


class CRAB_Agent(Agent):
    """Class representing an actor (firm or household) in the CRAB model. """

    def __init__(self, model: CRAB_Model, region: int) -> None:
        """Initializes an agent in the CRAB model.
        
        Args:
            model           : Model object containing the agent
            region          : Home region of this agent
        """

        super().__init__(model.next_id(), model)

        # -- GENERAL CRAB AGENT ATTRIBUTES -- #
        self.region = region
        self.lifetime = 1

    def stage8(self):
        """General CRAB Agent last stage of step function: increase lifetime."""
        self.lifetime += 1        


class Household(CRAB_Agent):
    """Class representing a household in the CRAB model. """

    def __init__(self, model: CRAB_Model, region: int) -> None:
        """Initialize household agent.
        
        Args:
            model           : Model object containing the household
            region          : Home region of this household
        """
        super().__init__(model, region)

        # -- SOCIO-ECONOMIC ATTRIBUTES -- #
        self.education = self.model.RNGs[type(self)].integers(0, 5)

        # -- FINANCIAL ATTRIBUTES -- #
        self.net_worth = 0
        self.consumption = 1
        self.house_income_ratio = self.model.RNGs[type(self)].uniform(4, 80)
        self.house_value = self.house_income_ratio

        # -- LABOR MARKET ATTRIBUTES -- #
        self.employer = None
        self.wage = 1

        # -- FLOODING ATTRIBUTES -- #
        self.monetary_damage = 0

    def labor_search(self) -> Firm:
        """ Labor search performed by Household agents.

        Returns:
            employer        : Firm object where household will be employed
        """

        # Check if there are firms with open vacancies
        vacancies_cap = [firm for firm in
                         self.model.get_firms_by_type(CapitalFirm, self.region)
                         if firm.open_vacancies]
        vacancies_all = [firm for firm in self.model.get_firms(self.region)
                         if firm.open_vacancies]

        # First try to find job at capital firm, then search at all firm types
        for vacancies in [vacancies_cap, vacancies_all]:
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

    def stage1(self) -> None:
        pass

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
        """Fifth stage of Household step function: consumption. """

        # -- Consumption -- #
        gov = self.model.governments[self.region]
        self.consumption = self.wage
        self.savings = 0
        self.house_value = gov.avg_wage * self.house_income_ratio

    def stage6(self) -> None:
        pass

    def stage7(self) -> None:
        pass

    def stage8(self) -> None:
        """Eighth stage of Household step function: increase lifetime. """
        super().stage8()


class Firm(CRAB_Agent):
    """Class representing a firm in the CRAB model. """

    def __init__(self, model: CRAB_Model, region: int, market_share: float,
                 net_worth: int, init_n_machines: int, init_cap_amount: int,
                cap_out_ratio: float, markup: float,
                 sales: int=10, wage: float=None, price: float=None,
                 prod: float=None, old_prod: float=None, lifetime: int=1) -> None:
        """Initialize firm agent.

        Args:
            model               : Model object containing the firm
            region              : Home region of this firm
        """

        super().__init__(model, region)

        # -- GENERAL FIRM ATTRIBUTES -- #
        self.lifetime = lifetime
        self.net_worth = net_worth
        self.cap_out_ratio = cap_out_ratio
        self.markup = markup
        self.subsidiary_counter = 0
        
        # -- LABOR MARKET ATTRIBUTES -- #
        self.employees = []
        self.size = 0

        # -- CAPITAL GOODS MARKET ATTRIBUTES -- #
        self.wage = wage if wage else model.RNGs[type(self)].normal(WAGE_DIST[0], WAGE_DIST[1])
        self.prod =  prod if prod else model.RNGs[type(self)].normal(PROD_DIST[0], PROD_DIST[1])


        self.old_prod = old_prod if old_prod else self.prod
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

        # -- FLOOD DAMAGE ATTRIBUTES -- #
        self.damage_coeff = 0
        

    class Vintage:
        """Class representing a vintage that consists of multiple
           machines of the same age, lifetime and productivity. """

        def __init__(self, firm: Firm, prod: float, amount: int) -> None:
            """Initialize vintage object.

            Args:
                prod                : Machine productivity
                amount              : Number of machines in this vintage.
            """
            self.prod = prod
            self.amount = amount
            self.age = 0
            self.lifetime = firm.model.RNGs[type(firm)].normal(20, 10)

    def update_capital(self) -> None:
        """Update capital: get ordered machines and remove old machines."""

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
                    del vintage

        # Remove machines that are too old
        for vintage in self.capital_vintage:
            vintage.age += 1
            if vintage.age > vintage.lifetime:
                self.capital_vintage.remove(vintage)
                del vintage
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

        expected_demand = sum(self.past_demand) / len(self.past_demand)

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
            n_expansion = round( (desired_prod - self.feasible_production) *
                           self.cap_out_ratio)
        else:
            n_expansion = 0
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
            UCA = self.wage / vintage.prod - self.wage / new_prod
            # Compute payback, only replace if advantage is high enough
            if UCA > 0 and new_price / UCA <= 3:
                n_replacements += vintage.amount
        return n_replacements

    def place_order(self, n_expansion: int, n_replacements: int):
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
            ratios = [brochure["prod"]/brochure["price"]
                      for brochure in self.offers.values()]
            # Get normalized cumulative sum of all offer ratios
            sup_prob = np.cumsum(ratios)/np.cumsum(ratios)[-1]
            j = bisect.bisect_right(sup_prob, self.model.RNGs[type(self)].uniform(0, 1))
            self.supplier = list(self.offers.keys())[j]

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
            debt_affordable = self.sales * DEBT_SALES_RATIO
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

    def get_avg_prod(self):
        """Get average firm productivity.

        Returns:
            avg_prod        : Average productivity of firm machines
        """

        # Find most productive machines to satisfy feasible production
        Q = 0
        machines_used = []
        # Loop through stock backwards (latest machines are most productive)
        for vintage in self.capital_vintage[::-1]:
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
        """

        sales = self.demand_filled * self.price
        total_costs = self.cost * self.demand_filled
        profits = round(sales - total_costs -
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
            # gov.tax_revenues += tax
            self.net_worth -= tax

    def hire_and_fire(self, labor_demand: int):
        """Hire (open vacancies) or fire employees based on labor demand,
           current number of employees, profits and wage.

        Args:
            labor_demand        : Firm's demand for labor
        """

        # Reset vacancies and bound desired number of employees
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

    def stage1(self) -> None:
        """First stage of firm step function: update capital and reset debt. """
        self.update_capital()
        self.debt = 0

    def stage3(self) -> None:
        """Third stage of firm step function: set wages. """
        self.set_wage()


class CapitalFirm(Firm):
    """Class representing a capital firm in the CRAB model. """

    def __init__(self, **kwargs) -> None:
        """Initialize capital firm agent. """

        super().__init__(**kwargs)

        # -- Initialize CapitalFirm-specific attributes -- #
        # -- CAPITAL GOODS MARKET: SUPPLY SIDE -- #
        self.clients = []

        self.regional_orders = {}
        self.brochure = {"prod": self.prod, "price": self.price}
        self.cap_out_ratio = 1  # Capital output ratio

        # -- CAPITAL GOODS MARKET: DEMAND SIDE -- #
        self.offers = {}
        self.production_made = 0

    def update_prod_cost(self) -> None:
      """Computes the unit cost of production for this firm. """
      self.cost = self.wage / self.prod

    def update_price(self, markup: float=0.3) -> None:
      """Update firm unit price. """
      self.price = round((1 + markup) * self.cost, 3)

    def advertise(self) -> list[int]:
        """Advertise products: """

        # Create brochures
        self.brochure = {"prod": self.prod, "price": self.price}

        # Randomly sample other firms to become clients
        firms = self.model.get_firms(self.region)
        firms.remove(self)
        if len(self.clients) > 1:
            # Number of new clients is fraction of number of current clients
            N = 1 + round(len(self.clients) * 0.2)
        else:
            # If no clients yet: sample fixed number of new clients
            N = 10
        new_clients = self.model.RNGs[type(self)].choice(firms, N)

        # Add potential clients to own clients, avoid duplicates
        new_clients = new_clients[~np.isin(new_clients, self.clients)]
        self.clients = list(np.append(self.clients, new_clients))

        # Send brochure to chosen firms
        for client in self.clients:
            client.offers[self] = self.brochure

    def set_wage(self) -> None:
        """Set firm wage, based on minimum wage and current top wage """

        # Set new wage to current top wage in the region, bounded by minimum wage
        gov = self.model.governments[self.region]
        return max(gov.min_wage, gov.top_wage)

    def accounting_orders(self) -> None:
        """Check if all orders can be satisfied, else: cancel or reduce orders. """

        # Check how much demand can be filled with production and inventories
        self.production_made = max(0, round(self.size * self.prod, 2))
        stock_available = self.production_made + self.inventories
        self.demand_filled = min(stock_available, self.real_demand)

        # Get unfilled demand and update inventories
        self.unfilled_demand = max(0, self.real_demand - stock_available)
        self.inventories = max(0, stock_available - self.real_demand)

        #if self.lifetime  < 10:
         #   self.inventories = 0.1 * self.real_demand
          #  self.unfilled_demand = 0
    

        # If demand cannot be filled: cancel or reduce orders
        if self.demand_filled < self.real_demand:
            orders = list(self.regional_orders.items())
            self.model.RNGs[type(self)].shuffle(orders)
            amount_to_cancel = self.real_demand - self.demand_filled

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
           Update capital, cost, price and advertise own machines. """

        # First inherit functionality of Firm agents (here: update capital)
        super().stage1()

        # CapitalFirms-specific: update production cost + price and advertise
        self.update_prod_cost()
        self.update_price()
        self.advertise()

    def stage2(self) -> None:
        """Second stage of capital firm step function: expand and replace machines."""

        # Check options to expand or replace machines
        n_expansion = self.capital_investment()
        n_replacements = self.replace_capital()

        # Check if net worth allows to make investments
        if n_replacements > 0 or n_expansion > 0:
            if self.net_worth > (self.size * self.wage):
                # Order new machines for expansion and replacement
                self.quantity_ordered = self.place_order(n_expansion,
                                                         n_replacements)
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

        # Open vacancies and fire employees
        self.hire_and_fire(labor_demand)

    def stage4(self) -> None:
        pass

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
        self.profits = self.get_profits()
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

    def set_wage(self, b: float=0.2) -> None:
        """Set firm wage, based on average regional productivity
           and regional minimum wage.

        Args:
            b       : Productivity scaling factor
        """

        # Bound wage by minimum wage (determined by government)
        prod_diff = max(-0.25, min(0.25, (self.prod - self.old_prod)/self.old_prod))
        # Regional productivity change, calculated by government
        gov = self.model.governments[self.region]
        avg_prod_diff = gov.prod_increase[type(self)]
        self.wage = max(gov.min_wage,
                        round(self.wage *
                              (1 + b * prod_diff + (1 - b) * avg_prod_diff),
                              3))

    def compete_and_sell(self, v: float=0.05):
        """Updates firm cost, markup and price.

        Args:
            v           : Markup/market_share ratio
        """

        # Update cost, avoiding division by zero
        self.cost = self.wage / self.prod if (self.prod > 0) else self.wage

        # Compute (bounded) markup from market share history
        if len(self.market_share_history) > 1:
            self.markup = max(0.01, min(0.4,
                              round(self.markup * (1 + v *
                                    ((self.market_share_history[-1] -
                                      self.market_share_history[-2]) /
                                      self.market_share_history[-2])), 5)))
        else:
            self.markup = 0.2

        # Adjust price based on new cost and markup, bounded between
        # 0.7 and 1.3 times the old price to avoid large oscillations
        self.price = max(0.7 * self.price, min(1.3 * self.price,
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
        """First stage of consumption firm step function. """

        # Inherit Firm class function: update capital
        super().stage1()

        # Check options to expand or replace machines
        n_expansion = self.capital_investment()
        n_replacements = self.replace_capital()

        # Check if net worth allows to make investments
        if n_replacements > 0 or n_expansion > 0:
            if self.net_worth > (self.size * self.wage):
                # Order new machines for expansion and replacement
                self.quantity_ordered = self.place_order(n_expansion,
                                                         n_replacements)
                # Set number of machines to be replaced after expansion
                self.machines_to_replace = max(0, self.quantity_ordered -
                                                  n_expansion)

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
        """Fourth stage of consumption firm step function. """
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

        # Compute real demand from regional demand, market shares and price
        self.real_demand = np.round(sum([gov.regional_demands[type(self)],
                                         gov.export_demands[type(self)]]   * self.market_share) / self.prod, 3)
        # Check available stock
        self.production_made = self.size * self.prod
        stock_available = self.production_made + self.inventories
        # Get filled and unfilled demand and update inventories
        self.demand_filled = min(stock_available, self.real_demand)
        self.unfilled_demand = max(0, self.real_demand - stock_available)
        self.inventories = max(0, stock_available - self.real_demand)

        #if self.life < 10:
         #   self.inventories = 0.1 * self.real_demand
          #  self.unfilled_demand = 0



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


class Agriculture(ConsumptionFirm):
    """Class representing a Consumption Goods firm in the CRAB model. """

    def __init__(self, **kwargs) -> None:
        """Initialize consumption goods firm agent. """

        super().__init__(**kwargs)

        # -- CAPITAL GOODS MARKET: DEMAND SIDE -- #
        #self.cap_out_ratio = 1.5  # Capital output ratio


class Industry(ConsumptionFirm):
    """Class representing a Consumption Services firm in the CRAB model. """

    def __init__(self, **kwargs) -> None:
        """Initialize Service firm agent. """

        super().__init__(**kwargs)

        # -- CAPITAL GOODS MARKET: DEMAND SIDE -- #
        #self.cap_out_ratio = 2  # Capital output ratio

class Construction(ConsumptionFirm):
    """Class representing a Consumption Services firm in the CRAB model. """

    def __init__(self, **kwargs) -> None:
        """Initialize Service firm agent. """

        super().__init__(**kwargs)

        # -- CAPITAL GOODS MARKET: DEMAND SIDE -- #
        #self.cap_out_ratio = 2  # Capital output ratio

class Transport(ConsumptionFirm):
    """Class representing a Consumption Services firm in the CRAB model. """

    def __init__(self, **kwargs) -> None:
        """Initialize Service firm agent. """

        super().__init__(**kwargs)

        # -- CAPITAL GOODS MARKET: DEMAND SIDE -- #
        #self.cap_out_ratio = 2  # Capital output ratio
class Information(ConsumptionFirm):
    """Class representing a Consumption Services firm in the CRAB model. """

    def __init__(self, **kwargs) -> None:
        """Initialize Service firm agent. """

        super().__init__(**kwargs)

        # -- CAPITAL GOODS MARKET: DEMAND SIDE -- #
        #self.cap_out_ratio = 2  # Capital output ratio
class Finance(ConsumptionFirm):
    """Class representing a Consumption Services firm in the CRAB model. """

    def __init__(self, **kwargs) -> None:
        """Initialize Service firm agent. """

        super().__init__(**kwargs)

        # -- CAPITAL GOODS MARKET: DEMAND SIDE -- #
        #self.cap_out_ratio = 2  # Capital output ratio

class Recreation(ConsumptionFirm):
    """Class representing a Consumption Services firm in the CRAB model. """

    def __init__(self, **kwargs) -> None:
        """Initialize Service firm agent. """

        super().__init__(**kwargs)

        # -- CAPITAL GOODS MARKET: DEMAND SIDE -- #
        #self.cap_out_ratio = 2  # Capital output ratio