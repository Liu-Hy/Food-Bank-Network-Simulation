from typing import List, Tuple, Dict
import pandas as pd
from FoodPantry import FoodPantry
from Global import Global, TYPES
from utils import Food
import time
import random
import statistics


class FoodBank:
    def __init__(self, initial_storage: float):
        """Food bank constructor

        :param initial_storage: Initial storage of food in pounds. Value given to Food class
        """
        # we assume half of the food insecure people actually use the bank
        num_pantries = 33 # 166
        self.pantries: List['FoodPantry'] = [FoodPantry(self, num_households=500) for _ in range(num_pantries)]
        self.total_utility = None
        self.total_waste = None
        self.total_served = None

        food_types = Global.get_food_types()
        self.pantry_demand = dict(zip(food_types, [0] * len(food_types)))

        self.storage = Food(initial_storage)

    def food_storage(self):
        """API for retreaving food storage dataframe

        :return: storage dataframe
        """
        return self._storage.df.copy()


    def run_one_day(self, budget: float, food_donations: float) -> Tuple[Dict[str, float], Dict[str, float], float]:
        """Runs simulation for the day. Also calls `run_one_day` for each pantry it serves.

        :param budget: Budget for the day
        :param food_donations: Food donations in pounds of food
        :return: Overall waste, demand (based on orders) and utility of all pantries
        """
        new_food = Food.generate_donation(food_donations)
        self._storage.add(new_food)

        """for pantry in self.pantries:
            result = pantry.run_one_day()
            if result is not None:
                waste, order, utility, num_served = pantry.run_one_day()
                self.total_utility = FoodBank.increment_utility(self.total_utility, utility)
                self.total_waste = FoodBank.increment_waste(self.total_waste, waste)
                self.total_served = FoodBank.increment_served(self.total_served, num_served)
                self.update_demand(order)

        # self.purchase_food(budget)
        return self.total_waste, self.pantry_demand, self.total_utility, self.total_served"""

    @classmethod
    def increment_served(cls, total_served, num_served):
        if total_served is None:
            return num_served
        else:
            return (total_served[i] + num_served[i] for i in range(2))

    @classmethod
    def increment_waste(cls, total_waste, new_waste):
        if total_waste is None:
            return new_waste
        else:
            return {food: (total_waste[food] + waste) for food, waste in new_waste}

    def purchase_food(self, budget: float):
        """Purchases food using given budget

        :param budget: budget in dollars
        """
        demand = self.get_pantry_demand_proportion()
        types = demand.keys()
        remaining_days = [TYPES[t]['max_days'] for t in types]
        quantity = [demand[t] * budget * Global._base_prices[t] for t in types]

        purchase = pd.DataFrame({"type": types, "remaining_days": remaining_days, "quantity": quantity})
        self._storage.add(purchase)

    def get_pantry_demand_proportion(self):
        """Returns demand in proportions. Used to decide what food to buy next.

        :return: demand proportions
        """
        total = sum(self.pantry_demand.values())
        return {food: (amount / total) for (food, amount) in self.pantry_demand.items()}

    def update_demand(self, order):
        """Updates pantry demand values

        :param order: order made by a pantry
        """
        for food, amount in order.items():
            self.pantry_demand[food] += amount

    def get_food_quantity(self):
        """Returns quantity of food in storage

    :return:
    """
        return self.storage.get_quantity()

    def get_food_order(self, order):
        """Fulfills given order

    :param order:
    :return: order result
    """
        return self.storage.subtract(order)

    @classmethod
    def increment_utility(cls, total_utility: float, utility: float):
        """Increments total utility

        :param total_utility:
        :param utility:
        :return: new total utility
        >>> FoodBank(0, 0).increment_utility(10)
        10
        """
        if total_utility is None:
            return utility
        else:
            return total_utility + utility

if __name__ == '__main__':
    tot_util = []
    tot_waste = []
    tot_all_served = []
    tot_partly_served = []
    start = time.time()
    for run in range(10):
        Global._current_day = 0
        utilities = []
        wastes = []
        served_ls = []
        num_days = 7
        foodbank = FoodBank(initial_storage=500000)
        #foodbank.pantries[0].config = {"pantry": {"set_limit": False, "use_real_demand": False}}
        num_panties = len(foodbank.pantries)
        pantry_order = list(range(num_panties))
        for i in range(num_days):
            Global.add_day()
            waste = foodbank.storage.quality_control(1)
            wastes.append(waste)  # comment this line to trace a specific pantry
            new_food = Food.generate_donation(33333)
            foodbank.storage.add(new_food)
            random.shuffle(pantry_order)
            for p in pantry_order:
                result = foodbank.pantries[p].run_one_day()
                if result is not None:
                    #if p == 0:
                    waste, order, utility, num_served, _ = result
                    utilities.append(utility)
                    wastes.append(waste)
                    served_ls.append(num_served)

        waste_per_type = dict()
        for typ in TYPES:
            waste_per_type[typ] = sum(w[typ] for w in wastes) / num_days
        waste_qty = sum(v for v in waste_per_type.values())
        # print(f"Daily waste per type: {waste_per_type}")
        # print(f"Daily waste in pounds: {waste_qty}")
        avg_utility = sum(utilities) / len(utilities)

        all_served, partly_served = list((zip(*served_ls)))

        all_served_rate = sum(all_served) / (100 * 166) / num_days * 7 # (100 * 166) when simulation all pantries. 100 for single pantry
        partly_served_rate = sum(partly_served) / (100 * 166) / num_days * 7 # (100 * 166) when simulation all pantries. 100 for single pantry

        tot_waste.append(waste_qty)
        tot_util.append(avg_utility)
        tot_all_served.append(all_served_rate)
        tot_partly_served.append(partly_served_rate)

    end = time.time()
    print(f"It took {end - start} seconds")

    print(f"Total utility {statistics.mean(tot_util)} +- {statistics.stdev(tot_util)}")  # 除以squrt(100)
    print(f"{statistics.mean(tot_all_served)*100} +- {statistics.stdev(tot_all_served)*100} % of clients get all demand satisfied")
    print(f"{statistics.mean(tot_partly_served)*100} +- {statistics.stdev(tot_partly_served)*100} % of clients get at least some food")
    print(f"Total waste {statistics.mean(tot_waste)} +- {statistics.stdev(tot_waste)}")
