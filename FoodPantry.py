from typing import Dict

import math
import numpy as np
import pandas as pd

from FoodBank import FoodBank
from constants import *
from utils import Food, mod_beta_random

rng = np.random.default_rng()


class FoodPantry:
    def __init__(self, parent: FoodBank, num_households=100):
        self.parent = parent
        self.num_households = num_households
        self.food = Food()
        self.clients = self.generate_clients()
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

    def run_one_day(self):
        """ Run the simulation for one day.
        Changes self.clients, self.food and self.parent.food in place.
        :return:
        >>> pantry = FoodPantry(None)
        >>> waste, order, utility = pantry.run_one_day()
        >>> waste
        """
        #if (Global.get_day() % 7) != self.operation_day:
            #return
        # Prepare food
        waste = self.food.quality_control(num_days=7)
        self.food.sort_by_freshness()
        est_demand = self.estimate_demand()
        order = FoodPantry.make_order(est_demand, self.food.get_quantity(), self.parent.food.get_quantity())
        suppl = self.parent.food.subtract(order)  # Modifies foodbank.food in-place!
        self.food.add(suppl)
        # The demand and queuing order of clients change every week
        self.initialize_weekly_demand()
        self.clients = self.clients.sample(frac=1).reset_index(drop=True)
        # TO DO: test the method
        self.hold_pantry()

        utility = self.get_utility()
        return waste, order, utility

    def estimate_demand(self) -> Dict[str, float]:
        """Predict client demand this week based on prior experience
        :return: A dictionary storing the quantity needed for each type of food.
        """
        if self.previous_record is None:
            num_clients = self.clients[("num_people", "")].sum()
            est_demand = {k: (v["mean"] * num_clients) for k, v in PERSONAL_WEEKLY_DEMAND.items()}
        else:
            NotImplemented
        return est_demand

    @classmethod
    def make_order(cls, demand: Dict[str, float], stock: Dict[str, float], bank_stock: Dict[str, float]) -> \
            Dict[str, float]:
        """Make an order to the food bank based on estimated demand and the current inventory.
        Request fresh food first, and request packaged food to meet the remaining demand.
        :param demand: A dictionary storing the demand (or its estimation) for each type of food.
        :param stock: the current inventory of the food pantry
        :param bank_stock: the current inventory of the food bank
        :return: A dictionary storing the quantity of each type of food requested to the food bank.
        >>> demand = {STP: 100, FV: 100, PT: 100}
        >>> stock = {STP: 30, FFV: 5, PFV: 30, FPT: 5, PPT: 20}
        >>> bank_stock1 = {STP: 500, FFV: 500, PFV: 500, FPT: 500, PPT: 500}
        >>> FoodPantry.make_order(demand, stock, bank_stock1)  # doctest: +NORMALIZE_WHITESPACE
        {'staples': 70, 'fresh_fruits_and_vegetables': 65, 'packaged_fruits_and_vegetables': 0, 'fresh_protein': 75,
        'packaged_protein': 0}
        >>> bank_stock2 = {STP: 500, FFV: 10, PFV: 500, FPT: 15, PPT: 500}
        >>> FoodPantry.make_order(demand, stock, bank_stock2)  # doctest: +NORMALIZE_WHITESPACE
        {'staples': 70, 'fresh_fruits_and_vegetables': 10, 'packaged_fruits_and_vegetables': 55, 'fresh_protein': 15,
        'packaged_protein': 60}
        >>> stock2 = {STP: 100, FFV: 0, PFV: 100, FPT: 0, PPT: 100}
        >>> ordr = FoodPantry.make_order(demand, stock2, bank_stock1)
        >>> list(ordr.values())
        [0, 0, 0, 0, 0]
        """
        stock[FV] = stock[FFV] + stock[PFV]
        stock[PT] = stock[FPT] + stock[PPT]
        gap = {typ: max(demand - stock[typ], 0) for typ, demand in demand.items()}
        order = dict()
        order[STP] = min(gap[STP], bank_stock[STP])
        order[FFV] = min(gap[FV], bank_stock[FFV])
        order[PFV] = min(gap[FV] - order[FFV], bank_stock[PFV])
        order[FPT] = min(gap[PT], bank_stock[FPT])
        order[PPT] = min(gap[PT] - order[FPT], bank_stock[PPT])
        # To do: set limit on fresh food based on est_demand and order
        return order

    def allocate_food(self, typ, dmd_col, pcs_col) -> pd.DataFrame:
        """
        Clients line up to purchase one type of food. Record their purchase and update the pantry inventory.
        Changes self.clients and self.food in place.
        :param typ: the type of food to allocate
        :param dmd_col: the name of the column from which we read the client demand
        :param pcs_col: The name of the column where we record the purchases of clients
        :return:
        """
        type_df = self.food.df[self.food.df["type"] == typ].reset_index(drop=True)
        cum_stock = type_df["quantity"].cumsum()
        stock = cum_stock.iat[-1]
        cum_demand = self.clients[dmd_col].cumsum()
        demand = cum_demand.iat[-1]
        if stock >= demand:
            # Get the index of the last batch of food before all demand is satisfied
            pivot = cum_stock.ge(demand - 1e-7).idxmax()  # Due to float precision, loosen the condition a bit
            type_df.loc[pivot, "quantity"] = cum_stock[pivot] - demand
            type_df = type_df[pivot:]
            self.clients.loc[:, pcs_col] = self.clients.loc[:, dmd_col]
        else:
            type_df = pd.DataFrame()
            # Get the index of the first client who cannot get enough food
            pivot = cum_demand.gt(stock).idxmax()
            self.clients.loc[:pivot, pcs_col] = self.clients.loc[:pivot, dmd_col]
            self.clients.loc[pivot, pcs_col] = self.clients.loc[pivot, dmd_col] - (cum_demand[pivot] - stock)
        return type_df

    def hold_pantry(self):
        """Hold a pantry activity. Although in reality one client shops multiple types of food at once, to avoid
        unnecessary computation, we transform it to the equivalent process of allocating food multiple times, once for
        each type of food.
        Changes self.clients and self.food in place
        """
        limit = 20

        types = {STP: [STP], FV: [FFV, PFV], PT: [FPT, PPT]}
        remains = []
        for typ, options in types.items():
            if len(options) == 1:
                remains.append(self.allocate_food(typ, (typ, "demand"), (typ, "purchased")))
            elif len(options) == 2:
                priority, alt = options
                # Transfer out-of-limit demand for fresh food to packaged food
                self.clients.loc[self.clients[(typ, "demand")] > limit, (typ, "demand_alt")] = self.clients.loc[self.clients[(
                    typ, "demand")] > limit, (typ, "demand")] - limit
                self.clients.loc[self.clients[(typ, "demand")] > limit, (typ, "demand")] = limit
                remains.append(self.allocate_food(priority, (typ, "demand"), (typ, "purchased_fresh")))
                # Add the unmet demand on fresh food to packaged food
                self.clients[(typ, "demand_alt")] += (
                        self.clients[(typ, "demand")] - self.clients[(typ, "purchased_fresh")])
                remains.append(self.allocate_food(alt, (typ, "demand_alt"), (typ, "purchased_packaged")))
        self.food.df = pd.concat(remains).reset_index(drop=True)

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
        get_utility = []
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

