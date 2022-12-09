from FoodPantry import *
import pandas as pd
from utils import Food
from typing import List, Dict, Tuple, Any
from Global import Global, TYPES


class FoodBank:
    def __init__(self, food_insecure_pop: int, initial_storage: float,
                 households_per_pantry: int = Global.households_per_pantry):
        """Food bank constructor. Modify `food_insecure_pop` and `households_per_pantry` to simulate crisis.

    :param food_insecure_pop: Number of food insecure people. Used to estimate number of pantries
    :param initial_storage: Initial storage of food in pounds. Value given to Food class
    :param households_per_pantry: default to global number
    """
        # we assume half of the food insecure people actually use the bank
        num_pantries = int(.5 * food_insecure_pop / households_per_pantry)
        self.pantries: List[FoodPantry] = [FoodPantry(self, num_households=households_per_pantry) for _ in
                                           range(num_pantries)]

        food_types = Global.get_food_types()
        self.pantry_demand = dict(zip(food_types, [0] * len(food_types)))

        self.storage = Food(initial_storage)

        self.last_week_demand: List[Dict[str, int]] = []

    def next_week_demand_estimate(self):

        pass

    # food that will be going bad soon with current level of demand
    # projection of how much will be ordered
    # food in demand with not enough supply
    # orders that won't be met (based on true order)

    def food_storage(self):
        """API for retreaving food storage dataframe

    :return: storage dataframe
    """
        return self.storage.df.copy()

    def run_one_day(self, budget: float, food_donations: float) -> tuple[
      dict | None | Any, dict[Any, int], int | float, tuple[int, int]]:
        """Runs simulation for the day. Also calls `run_one_day` for each pantry it serves.

    :param budget: Budget for the day
    :param food_donations: Food donations in pounds of food
    :return: Overall waste, demand (based on orders) and utility of all pantries
    """
        new_food = Food.generate_donation(food_donations)
        self.storage.add(new_food)

        total_utility = 0
        total_waste = None

        day_order_increment = None
        for pantry in self.pantries:
            pantry_output = pantry.run_one_day()
            if pantry_output is None:
                continue  # if pantry wasn't held, we skip
            waste, _, utility, tuple_served, true_order = pantry_output
            total_utility = FoodBank.increment_utility(total_utility, utility)
            total_waste = FoodBank.increment_food_dict(total_waste, waste)
            day_order_increment = FoodBank.increment_food_dict(day_order_increment, true_order)
            self.update_demand(true_order)

        self.update_weekly_demand(day_order_increment)

        self.purchase_food(budget)

        return total_waste, self.pantry_demand, total_utility, tuple_served

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
    def increment_food_dict(total_food_dict, new_food_dict):
        if total_food_dict is None:
            return new_food_dict
        return {food: (total_food_dict[food] + food_dict) for food, food_dict in new_food_dict}

    def purchase_food(self, budget: float):
        """Purchases food using given budget

    :param budget: budget in dollars
    """
        demand = self.last_week_pantry_demand_proportion()
        types = demand.keys()
        remaining_days = [TYPES[t]['max_days'] for t in types]
        quantity = [demand[t] * budget * Global._base_prices[t] for t in types]

        purchase = pd.DataFrame({"type": types, "remaining_days": remaining_days, "quantity": quantity})
        self.storage.add(purchase)

    def last_week_pantry_demand_proportion(self):
        """Returns demand in proportions. Used to decide what food to buy next.
    Calculation based on last week's demand

    :return: demand proportions
    """
        total = sum([sum(day_order.values()) for day_order in self.last_week_demand])
        return {food: (amount / total) for (food, amount) in self.pantry_demand.items()}

    def update_demand(self, order):
        """Updates pantry demand values

    :param order: order made by a pantry
    """
        for food, amount in order.items():
            self.pantry_demand[food] += amount

    def update_weekly_demand(self, order):
        if len(self.last_week_demand) >= 7:
            self.last_week_demand.pop(0)
        self.last_week_demand.append(order)

    @classmethod
    def increment_utility(self, total_utility: float, utility: float):
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
    food_bank = FoodBank(10_000, 1000)
    # Global.add_day()
    Global._base_prices = {
        STP: 2,
        FFV: 3,
        PFV: 2,
        FPT: 2,
        PPT: 2,
    }
    for day in range(50):
        food_bank.run_one_day(1000, 10)
        food_bank
        Global.add_day()
