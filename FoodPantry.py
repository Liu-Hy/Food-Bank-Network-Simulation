from typing import Dict, Tuple

import math
import numpy as np
import pandas as pd
import numba
import time

#from FoodBank import Foodbank
from constants import *
from utils import Food, mod_beta_random

rng = np.random.default_rng()

class FoodPantry:
    def __init__(self, parent, num_households=100):
        self.parent = parent
        self.num_households = num_households
        self.clients = self.generate_clients()
        self.food = Food()
        self.operation_day = rng.integers(0, 7)
        self.previous_record = None

    def generate_clients(self) -> pd.DataFrame:
        """For each client's family, generate the weekly physical demand for food, and the baseline level of proportion
        of demand they can already secure through purchasing.
        :return: a dataframe storing the information of families, including fields that will be used later.
        >>> pantry = FoodPantry(None, num_households=3)
        >>> demo = pantry.generate_clients()
        >>> demo.iloc[:] = 0
        >>> demo  # doctest: +NORMALIZE_WHITESPACE
          num_people staples               ...    protein
                       total base_secured  ... demand_alt purchased_fresh purchased_packaged
        0          0       0            0  ...          0               0                  0
        1          0       0            0  ...          0               0                  0
        2          0       0            0  ...          0               0                  0
        <BLANKLINE>
        [3 rows x 20 columns]
        >>> demo[FV]
           total  base_secured  ...  purchased_fresh  purchased_packaged
        0      0             0  ...                0                   0
        1      0             0  ...                0                   0
        2      0             0  ...                0                   0
        <BLANKLINE>
        [3 rows x 7 columns]
        >>> pantry = FoodPantry(None)
        >>> df = pantry.generate_clients().round(2)
        >>> ((1 <= df[("num_people", "")]) & (df[("num_people", "")] <= 10)).all()
        True
        >>> bools = []
        >>> for typ, stat in PERSONAL_WEEKLY_DEMAND.items():
        ...     mean = stat["mean"] * pantry.num_households * 2.4503
        ...     bools.append(0.85 * mean < df[(typ, "total")].sum() < 1.15 * mean)
        >>> all(bools)
        True
        """
        columns = [("num_people", ""), (STP, "total"), (STP, "base_secured"), (STP, "secured"), (STP, "demand"),
                   (STP, "purchased"),
                   (FV, "total"), (FV, "base_secured"), (FV, "secured"), (FV, "demand"), (FV, "demand_alt"),
                   (FV, "purchased_fresh"), (FV, "purchased_packaged"),
                   (PT, "total"), (PT, "base_secured"), (PT, "secured"), (PT, "demand"), (PT, "demand_alt"),
                   (PT, "purchased_fresh"), (PT, "purchased_packaged")]
        clients = pd.DataFrame(columns=pd.MultiIndex.from_tuples(columns))
        clients[("num_people", "")] = rng.choice(range(1, 11), self.num_households, p=FAMILY_DISTRIBUTION)
        clients.loc[:, (slice(None), "base_secured")] = rng.uniform(0.3, 0.9, (self.num_households, 3))
        for typ, stat in PERSONAL_WEEKLY_DEMAND.items():
            mean, std = stat["mean"], stat["std"]
            low, high = 0.5 * mean, 2 * mean
            clients[(typ, "total")] = mod_beta_random(low, high, mean, std, self.num_households) * clients[
                ("num_people", "")]
        return clients

    def initialize_weekly_demand(self):
        """Generate each client's proportion of food secured this week in response to price fluctuation, and their
        demand to the food bank.
        Changes self.clients in place
        :return:
        >>> pantry = FoodPantry(None)
        >>> pantry.initialize_weekly_demand()
        >>> (pantry.clients.loc[:, (slice(None), ["demand_alt", "purchased", "purchased_fresh", "purchased_packaged"])]\
        == 0).all(axis=None)
        True
        """
        # price_ratio = Global.price.ratio
        price_ratio = {STP: 1.1, FV: 1.2, PT: 0.9}
        factor = {k: (v ** ELASTICITY[k]) for k, v in price_ratio.items()}
        for typ in ELASTICITY.keys():
            self.clients[(typ, "secured")] = self.clients[(typ, "base_secured")] * factor[typ]
            self.clients[(typ, "demand")] = self.clients[(typ, "total")] * (
                    1 - self.clients[(typ, "secured")]) + rng.normal(0, 100, self.num_households)
            if "demand_alt" in self.clients[typ]:
                self.clients[(typ, "demand_alt")] = 0.
        # remove the purchase record of the previous week
        self.clients.loc[:, (slice(None), ["purchased", "purchased_fresh", "purchased_packaged"])] = 0.

    def estimate_demand(self) -> Dict[str, float]:
        """Predict client demand this week based on prior experience
        :return: A dictionary storing the quantity needed for each type of food.
        """
        if self.previous_record is None:
            num_clients = self.clients[("num_people", "")].sum()
            est_demand = {k: (v["mean"] * num_clients) for k, v in PERSONAL_WEEKLY_DEMAND.items()}
        else:
            #NotImplemented
            num_clients = self.clients[("num_people", "")].sum()
            est_demand = {k: (v["mean"] * num_clients) for k, v in PERSONAL_WEEKLY_DEMAND.items()}
        return est_demand

    def make_order(self, demand: Dict[str, float], stock: Dict[str, float], bank_stock: Dict[str, float]) -> \
            Tuple[dict, dict]:
        """Make an order to the food bank based on estimated demand and the current inventory. Request fresh food first,
        and request packaged food to meet the remaining demand. Based on the demand and the supplimented inventory, set
        limits on fresh food per client
        :param demand: A dictionary storing the demand (or its estimation) for each type of food.
        :param stock: the current inventory of the food pantry
        :param bank_stock: the current inventory of the food bank
        :return: A dictionary storing the quantity of each type of food requested to the food bank, and a dictionary
        storing the limit on fresh food per type.
        >>> pantry = FoodPantry(None, num_households=8)
        >>> demand = {STP: 100, FV: 100, PT: 100}
        >>> stock = {STP: 30, FFV: 5, PFV: 30, FPT: 5, PPT: 20}
        >>> bank_stock1 = {STP: 500, FFV: 500, PFV: 500, FPT: 500, PPT: 500}
        >>> pantry.make_order(demand, stock, bank_stock1)[0]  # doctest: +NORMALIZE_WHITESPACE
        {'staples': 70, 'fresh_fruits_and_vegetables': 65, 'packaged_fruits_and_vegetables': 0, 'fresh_protein': 75,
        'packaged_protein': 0}
        >>> bank_stock2 = {STP: 500, FFV: 10, PFV: 500, FPT: 15, PPT: 500}
        >>> pantry.make_order(demand, stock, bank_stock2)[0]  # doctest: +NORMALIZE_WHITESPACE
        {'staples': 70, 'fresh_fruits_and_vegetables': 10, 'packaged_fruits_and_vegetables': 55, 'fresh_protein': 15,
        'packaged_protein': 60}
        >>> stock2 = {STP: 100, FFV: 0, PFV: 100, FPT: 0, PPT: 100}
        >>> ordr = pantry.make_order(demand, stock2, bank_stock1)[0]
        >>> list(ordr.values())
        [0, 0, 0, 0, 0]
        """
        types = {FV: [FFV, PFV], PT: [FPT, PPT]}
        stock[FV] = stock[FFV] + stock[PFV]
        stock[PT] = stock[FPT] + stock[PPT]
        gap = {typ: max(demand - stock[typ], 0) for typ, demand in demand.items()}
        limits = dict()
        order = dict()
        order[STP] = min(gap[STP], bank_stock[STP])
        for typ, subtypes in types.items():
            fresh, packaged = subtypes
            order[fresh] = min(gap[typ], bank_stock[fresh])
            order[packaged] = min(gap[typ] - order[fresh], bank_stock[packaged])
            # Calculate the limit on fresh food of this type
            fresh_qty = stock[fresh] + order[fresh]
            if fresh_qty < demand[typ]:
                limits[typ] = fresh_qty * 1.2 / self.num_households
            else:
                limits[typ] = float("inf")
        return order, limits

    def func(self, data: pd.Series, typ="exp", param=0.7) -> pd.Series:
        """
        Food utility as a function of the proportion of demand satisfied. It should map 0 to 1, 1 to 1, and be concave
        to reflect diminished marginal effect.
        :param data: a pd.Series or np.ndarray storing the proportion of food demand that is satisfied per household
        :param typ: the type of the function, either exponential, logarithm or quadratic.
        :param param: one additional parameter to decide the shape of the curve, e.g. the power in exponential function,
        the quadracitc coefficient in a quadratic function.
        :return: The element-wise utility value
        >>> pantry = FoodPantry(None)
        >>> portion = pd.Series(np.arange(0, 1.2, 0.2))
        >>> portion.round(2).values.tolist()
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        >>> pantry.func(portion).round(2).values.tolist()
        [0.0, 0.32, 0.53, 0.7, 0.86, 1.0]
        >>> pantry.func(portion, "quad", -0.5).round(2).values.tolist()
        [0.0, 0.28, 0.52, 0.72, 0.88, 1.0]
        >>> pantry.func(portion, "log", 3).round(2).values.tolist()
        [0.0, 0.34, 0.57, 0.74, 0.88, 1.0]
        """
        assert typ in ["exp", "log", "quad"]
        if typ == "exp":
            return np.power(data, param)
        elif typ == "log":
            return np.log(data * param + 1) / math.log(param + 1)
        elif typ == "quad":
            assert -1 <= param < 0, "The quadratic coefficient should be between -1 and 0 for quadratic functions!"
            return param * np.square(data) + (1 - param) * data

    def utility_per_type(self, typ: str) -> float:
        """After a pantry activity, estimate the increment in the utility of one type of food per household.
        :param typ: The type of food for which to calculate utility increment
        :return:
        """
        assert typ in [STP, FV, PT]
        family_size = self.clients[("num_people", "")]
        total = self.clients[(typ, "total")]
        secured = self.clients[(typ, "secured")]
        if typ == STP:
            delta = self.clients[(STP, "purchased")] / total
            inc = (self.func(secured + delta) - self.func(secured)) * family_size
        else:
            fresh_pct = self.clients[(typ, "purchased_fresh")] / self.clients[(typ, "total")]
            pckg_pct = self.clients[(typ, "purchased_packaged")] / self.clients[(typ, "total")]
            pct_with_fresh = secured + fresh_pct
            pct_total = pct_with_fresh + pckg_pct
            inc = (0.7 * self.func(pct_total) + 0.3 * self.func(pct_with_fresh) - self.func(secured)) * family_size
        return inc

    def get_utility(self) -> float:
        """Estimate the increment in total food utility after a pantry activity
        :return:
        """
        num_clients = self.clients[("num_people", "")].sum()
        tot_util = pd.Series(np.zeros(self.num_households))
        for typ in [STP, FV, PT]:
            tot_util += self.utility_per_type(typ)
        return tot_util.sum() / num_clients

    @classmethod
    def allocate_food(cls, food, demand) -> Tuple[pd.Series, pd.DataFrame]:
        """Clients line up to purchase one type of food. Record their purchase and update the pantry inventory.
        :param food: the dataframe of some type of food
        :param demand: a pd.Series object storing the demand of clients in the queue
        :return: a pd.Series storing the amount purchased by clients, and a pd.DataFrame storing the remaining food
        >>> demand = pd.Series([10.] * 5)
        >>> total = demand.sum() / TYPES[STP]["proportion"]
        >>> food = Food(total + 1).select(STP).df
        >>> purchased, remain = FoodPantry.allocate_food(food, demand)
        >>> list(purchased.round(2))
        [10.0, 10.0, 10.0, 10.0, 10.0]
        >>> remain.round(2)
                type  remaining_days  quantity
        178  staples             179      0.02
        179  staples             180      0.28
        >>> food2 = Food(total - 1).select(STP).df
        >>> purchased2, remain2 = FoodPantry.allocate_food(food2, demand)
        >>> list(purchased2.round(2))
        [10.0, 10.0, 10.0, 10.0, 9.7]
        >>> remain2.empty
        True
        >>> food0 = Food().select(STP).df
        >>> purchased0, remain0 = FoodPantry.allocate_food(food0, demand)
        >>> purchased0.sum() == 0 and remain0.empty
        True
        >>> demand0 = pd.Series([0.] * 5)
        >>> purchased_0, remain_0 = FoodPantry.allocate_food(food, demand0)
        >>> purchased_0.sum() == 0
        True
        """
        if isinstance(food, Food):
            food = food.df
        num_households = len(demand)
        cum_stock = food["quantity"].cumsum()
        tot_stock = cum_stock.iat[-1] if len(cum_stock) >= 1 else 0
        cum_demand = demand.cumsum()
        tot_demand = cum_demand.iat[-1] if len(cum_demand) >= 1 else 0
        purchased = pd.Series(np.zeros(num_households))
        if tot_stock >= tot_demand:
            # Get the index of the last batch of food before all demand is satisfied
            pivot = cum_stock.ge(tot_demand - 1e-7).idxmax()  # Due to float precision, loosen the condition a bit
            food.loc[pivot, "quantity"] = cum_stock[pivot] - tot_demand
            food = food[pivot:]
            purchased = demand
        else:
            food = pd.DataFrame()
            # Get the index of the first client who cannot get enough food
            pivot = cum_demand.gt(tot_stock).idxmax()
            purchased[:pivot] = demand[:pivot]
            purchased[pivot] = demand[pivot] - (cum_demand[pivot] - tot_stock)
        return purchased, food

    def hold_pantry(self, limits: Dict[str, float]):
        """Hold a pantry activity. Although in reality one client shops multiple types of food at once, to avoid
        unnecessary computation, we transform it to the equivalent process of allocating food multiple times, once for
        each type of food.
        Changes self.clients and self.food in place
        """
        types = {STP: [STP], FV: [FFV, PFV], PT: [FPT, PPT]}
        remains = []
        for typ, options in types.items():
            if len(options) == 1:
                purchased, remain = FoodPantry.allocate_food(self.food.select(options[0]), self.clients[(typ, "demand")])
                self.clients[(typ, "purchased")] = purchased
                remains.append(remain)
            elif len(options) == 2:
                fresh, packaged = options
                # Transfer out-of-limit demand for fresh food to packaged food
                limit = limits[typ]
                mask = (self.clients[(typ, "demand")] > limit)
                self.clients.loc[mask, (typ, "demand_alt")] = self.clients.loc[mask, (typ, "demand")] - limit
                self.clients.loc[mask, (typ, "demand")] = limit
                purchased, remain = FoodPantry.allocate_food(self.food.select(fresh), self.clients[(typ, "demand")])
                self.clients[(typ, "purchased_fresh")] = purchased
                remains.append(remain)
                # Add the unmet demand on fresh food to packaged food
                self.clients[(typ, "demand_alt")] += (
                        self.clients[(typ, "demand")] - self.clients[(typ, "purchased_fresh")])
                purchased, remain = FoodPantry.allocate_food(self.food.select(packaged), self.clients[(typ, "demand_alt")])
                self.clients[(typ, "purchased_packaged")] = purchased
                remains.append(remain)
        self.food.df = pd.concat(remains).reset_index(drop=True)

    def run_one_day(self) -> Tuple[Dict[str, float], Dict[str, float], float]:
        """ Run the simulation for one day.
        Changes self.clients, self.food and self.parent.food in place.
        :return:
        >>> pantry = FoodPantry(None)
        >>> waste, order, utility = pantry.run_one_day()
        """
        if (Global.get_day() % 7) != self.operation_day:
            return
        # Prepare food
        waste = self.food.quality_control(num_days=7)
        self.food.sort_by_freshness()
        est_demand = self.estimate_demand()
        order, limits = FoodPantry.make_order(est_demand, self.food.get_quantity(), self.parent.food.get_quantity())
        #order, limits = self.make_order(est_demand, self.food.get_quantity(), Food(7000).get_quantity())
        suppl = self.parent.food.subtract(order)  # Modifies foodbank.food in-place!
        #suppl = Food(7000).subtract(order)
        self.food.add(suppl)
        # The demand and queuing order of clients change every week
        self.initialize_weekly_demand()
        self.clients = self.clients.sample(frac=1).reset_index(drop=True)

        self.hold_pantry(limits)
        utility = self.get_utility()
        #print(utility)
        return waste, order, utility

if __name__ == '__main__':
    # we may need to drastically reduce the number of pantries to make it computationally feasible
    # about 10 million households in total
    utilities = []
    wastes = []
    num_days = 100
    pantry = FoodPantry(None, num_households=100)
    start = time.time()
    for i in range(num_days):
        waste, order, utility = pantry.run_one_day()
        utilities.append(utility)
        wastes.append(waste)
    end = time.time()
    print(end - start)
    print(sum(utilities) / len(utilities))
    tot_waste = dict()
    for typ in TYPES:
        tot_waste[typ] = sum(w[typ] for w in wastes) / num_days
    print(tot_waste)

