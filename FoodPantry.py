import math
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from FoodBank import FoodBank
from Global import *
from utils import Food, mod_beta_random



class FoodPantry:
    def __init__(self, parent, num_households=100, config=Global.config):
        self.parent = parent
        self.num_households = num_households
        self.config = config
        self.clients = self.generate_clients()
        self.num_people = self.clients[("num_people", "")].sum()
        # Inferred from interview and statistics, clients on average need 40% of their demand from foodbanks
        self.mean_demand = {k: (v["mean"] * self.num_people * 0.4) for k, v in PERSONAL_WEEKLY_DEMAND.items()}
        self.food = Food()
        self.operation_day = rng.integers(0, 7)
        self.previous_record = []

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
        clients = pd.DataFrame(columns=pd.MultiIndex.from_tuples(columns)).astype("float")
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
                    1 - self.clients[(typ, "secured")]) + rng.normal(0, 1, self.num_households)
            if "demand_alt" in self.clients[typ]:
                self.clients[(typ, "demand_alt")] = 0.
        # remove the purchase record of the previous week
        self.clients.loc[:, (slice(None), ["purchased", "purchased_fresh", "purchased_packaged"])] = 0.

    def estimate_demand(self) -> Dict[str, float]:
        """Predict client demand this week based on prior experience
        :return: A dictionary storing the quantity needed for each type of food.
        """
        if not self.config["pantry"]["use_real_demand"]:
            if not self.previous_record:
                est_demand = self.mean_demand
            else:
                est_demand = self.previous_record[-1]
        else:
            # est_demand = {k: v * 1 for k, v in est_demand.items()}  # scale the estimation?
            est_demand = dict()
            for typ in PERSONAL_WEEKLY_DEMAND:
                est_demand[typ] = self.clients[(typ, "demand")].sum().item()
        return est_demand

    def make_order(self, demand: Dict[str, float], stock: Dict[str, float], bank_stock: Dict[str, float]) -> \
            Tuple[dict, dict]:
        """Make an order to the food bank based on estimated demand and the current inventory of the pantry and the
        foodbank. Request fresh food first, and request packaged food to meet the remaining demand. Set limits on fresh
        food per client based on the demand and the replenished inventory.
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
            limits[typ] = float("inf")
            if fresh_qty < demand[typ]:
                if self.config["pantry"]["set_limit"]:
                    limits[typ] = fresh_qty * 1.3 / self.num_people
        return order, limits

    @classmethod
    def func(cls, data: pd.Series, typ="exp", param=0.7) -> pd.Series:
        """
        Food utility as a function of the proportion of demand satisfied. It should map 0 to 1, 1 to 1, and be concave
        to reflect diminished marginal effect.
        :param data: a pd.Series or np.ndarray storing the proportion of food demand that is satisfied per household
        :param typ: the type of the function, either exponential, logarithm or quadratic.
        :param param: one additional parameter to decide the shape of the curve, e.g. the power in exponential function,
        the quadracitc coefficient in a quadratic function.
        :return: The element-wise utility value
        >>> portion = pd.Series(np.arange(0, 1.2, 0.2))
        >>> portion.round(2).values.tolist()
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        >>> FoodPantry.func(portion).round(2).values.tolist()
        [0.0, 0.32, 0.53, 0.7, 0.86, 1.0]
        >>> FoodPantry.func(portion, "quad", -0.5).round(2).values.tolist()
        [0.0, 0.28, 0.52, 0.72, 0.88, 1.0]
        >>> FoodPantry.func(portion, "log", 3).round(2).values.tolist()
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

    def utility_per_type(self, typ: str) -> pd.Series:
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
            inc = (FoodPantry.func(secured + delta) - FoodPantry.func(secured)) * family_size
        else:
            fresh_pct = self.clients[(typ, "purchased_fresh")] / self.clients[(typ, "total")]
            pckg_pct = self.clients[(typ, "purchased_packaged")] / self.clients[(typ, "total")]
            pct_with_fresh = secured + fresh_pct
            pct_total = pct_with_fresh + pckg_pct
            inc = (0.7 * FoodPantry.func(pct_total) + 0.3 * FoodPantry.func(pct_with_fresh) - FoodPantry.func(secured)) \
                  * family_size
        return inc

    def get_utility(self) -> float:
        """Estimate the increment in total food utility after a pantry activity
        :return:
        """
        tot_util = pd.Series(np.zeros(self.num_households))
        for typ in [STP, FV, PT]:
            tot_util += self.utility_per_type(typ)
        return tot_util.sum() / 3 #/ (self.num_people * 3)

    def allocate_food(self, food, demand) -> Tuple[pd.Series, pd.DataFrame, int]:
        """Clients line up to purchase one type of food. Record their purchase and update the pantry inventory.
        :param food: the dataframe of some type of food
        :param demand: a pd.Series object storing the demand of clients in the queue
        :return: a pd.Series storing the amount purchased by clients, a pd.DataFrame storing the remaining food, and the
        number of clients who get enough food.
        >>> pantry = FoodPantry(None)
        >>> demand = pd.Series([10.] * 5)
        >>> total = demand.sum() / TYPES[STP]["proportion"]
        >>> food = Food(total + 2.0).select(STP).df  # a bit more food than needed
        >>> purchased, remain, served = pantry.allocate_food(food, demand)
        >>> list(purchased.round(2))
        [10.0, 10.0, 10.0, 10.0, 10.0]
        >>> remain.round(2)
                type  remaining_days  quantity
        177  staples             178      0.04
        178  staples             179      0.28
        179  staples             180      0.28
        >>> served
        5
        >>> food2 = Food(total - 40.0).select(STP).df  # less food than needed
        >>> purchased2, remain2, served2 = pantry.allocate_food(food2, demand)
        >>> list(purchased2.round(2))
        [10.0, 10.0, 10.0, 8.0, 0.0]
        >>> remain2.empty
        True
        >>> served2
        3
        >>> food0 = Food().select(STP).df
        >>> purchased0, remain0, served0 = pantry.allocate_food(food0, demand)
        >>> purchased0.sum() == 0 and remain0.empty
        True
        >>> served0
        0
        >>> demand0 = pd.Series([0.] * 5)
        >>> purchased_0, remain_0, served_0 = pantry.allocate_food(food, demand0)
        >>> purchased_0.sum() == 0
        True
        >>> served_0
        5
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
            served = num_households
        else:
            food = Food().df
            # Get the index of the first client who cannot get enough food
            pivot = cum_demand.gt(tot_stock).idxmax()
            purchased[:pivot] = demand[:pivot]
            purchased[pivot] = demand[pivot] - (cum_demand[pivot] - tot_stock)
            served = pivot  # (served+1) clients get some food
        return purchased, food, served

    def hold_pantry(self, limits: Dict[str, float]) -> Tuple[int, int]:
        """Hold a pantry activity. Although in reality one client shops multiple types of food at once, to avoid
        unnecessary computation, we transform it to the equivalent process of allocating food multiple times, once for
        each type of food.
        Changes self.clients and self.food in place
        """
        types = {STP: [STP], FV: [FFV, PFV], PT: [FPT, PPT]}
        remains = []
        served_per_type = []
        est_demand = dict()
        for typ, options in types.items():
            if len(options) == 1:
                purchased, remain, served = self.allocate_food(self.food.select(options[0]),
                                                               self.clients[(typ, "demand")])
                self.clients[(typ, "purchased")] = purchased
                remains.append(remain)
                served_per_type.append(served)
                if served < min(20, 0.2 * self.num_households):
                    # Sample size too small for a good estimation. Use statistics instead
                    est_demand[typ] = self.mean_demand[typ]
                elif served == self.num_households:
                    est_demand[typ] = purchased.sum().item()
                elif served < self.num_households:
                    est_demand[typ] = purchased.sum().item() / served * self.num_households
                else:
                    raise ValueError
            elif len(options) == 2:
                fresh, packaged = options
                # Transfer out-of-limit demand for fresh food to packaged food
                quota = self.clients[("num_people", "")] * limits[typ]
                mask = (self.clients[(typ, "demand")] > quota)
                self.clients.loc[mask, (typ, "demand_alt")] = self.clients.loc[mask, (typ, "demand")] - quota
                self.clients.loc[mask, (typ, "demand")] = quota
                purchased_fresh, remain, served = self.allocate_food(self.food.select(fresh),
                                                                     self.clients[(typ, "demand")])
                #print(served)
                self.clients[(typ, "purchased_fresh")] = purchased_fresh
                remains.append(remain)
                served_per_type.append(served)
                # Add the unmet demand on fresh food to packaged food
                self.clients[(typ, "demand_alt")] += (
                        self.clients[(typ, "demand")] - self.clients[(typ, "purchased_fresh")])
                purchased_pckg, remain, served = self.allocate_food(self.food.select(packaged),
                                                                    self.clients[(typ, "demand_alt")])
                self.clients[(typ, "purchased_packaged")] = purchased_pckg
                remains.append(remain)
                served_per_type.append(served)
                # Demand on fresh food is not observable due to the quota, but those who get enough packaged food must
                # have fulfilled their demand in this type.
                if served < min(20, 0.2 * self.num_households):
                    est_demand[typ] = self.mean_demand[typ]
                elif served == self.num_households:
                    est_demand[typ] = (purchased_fresh + purchased_pckg).sum().item()
                elif served < self.num_households:
                    est_demand[typ] = (purchased_fresh + purchased_pckg)[
                                      :served].sum().item() / served * self.num_households
                else:
                    raise ValueError
        self.food.df = pd.concat(remains).reset_index(drop=True)
        self.previous_record.append(est_demand)
        # number of clients who get all their demand satisfied (either fresh or packaged)
        all_served = min(served_per_type[0], served_per_type[2], served_per_type[4])
        # number of clients who get at least some food
        partly_served = min(max(served_per_type) + 1, self.num_households)
        return all_served, partly_served

    def run_one_day(self) -> Tuple[Dict[str, float], Dict[str, float], float, Tuple[int, int], Dict[str, float]]:
        """ Run the simulation for one day.
        Changes self.clients, self.food and self.parent.food in place.
        :return:
        >>> pantry = FoodPantry(None)
        >>> waste, order, utility, num_served = pantry.run_one_day()
        """
        if (Global.get_day() % 7) != self.operation_day:
            return
        self.initialize_weekly_demand()
        waste = self.food.quality_control(num_days=7)
        est_demand = self.estimate_demand()
        order, limits = self.make_order(est_demand, self.food.get_quantity(), self.parent.get_food_quantity())
        #order, limits = self.make_order(est_demand, self.food.get_quantity(), Food(1500).get_quantity())
        suppl = self.parent.get_food_order(order)
        #suppl = Food(1500).subtract(order)
        self.food.add(suppl)
        self.food.sort_by_freshness()
        self.clients = self.clients.sample(frac=1).reset_index(drop=True)
        num_served = self.hold_pantry(limits)
        utility = self.get_utility()
        return waste, order, utility, num_served, est_demand


if __name__ == '__main__':
    # We may need to drastically reduce the number of pantries to make it computationally feasible
    # There are about 10 million households in total
    utilities = []
    wastes = []
    served_ls = []
    households = 100
    num_days = 100
    pantry = FoodPantry(None, num_households=households)
    start = time.time()
    for i in range(num_days):
        waste, order, utility, num_served, _ = pantry.run_one_day()
        utilities.append(utility)
        wastes.append(waste)
        served_ls.append(num_served)
    all_served, partly_served = list((zip(*served_ls)))
    end = time.time()
    print(f"It took {end - start} seconds")
    print(f"Total average utility {sum(utilities) / len(utilities)}")
    print("{:.2%} of clients get all demand satisfied".format(sum(all_served) / (households * num_days)))
    print("{:.2%} of clients get at least some food".format(sum(partly_served) / (households * num_days)))
    waste_per_type = dict()
    for typ in TYPES:
        waste_per_type[typ] = sum(w[typ] for w in wastes) / num_days
    waste_qty = sum(v for v in waste_per_type.values())
    print(f"Daily waste per type: {waste_per_type}")
    print(f"Daily total waste: {waste_qty}")
