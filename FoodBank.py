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

    def run_one_day(self, budget: float, food_donations: float):
        """Runs simulation for the day. Also calls `run_one_day` for each pantry it serves.

    :param budget: Budget for the day
    :param food_donations: Food donations in pounds of food
    :return: Overall waste, demand (based on orders) and utility of all pantries
    """
        new_food = Food.generate_donation(food_donations)
        self.storage.add(new_food)

        purchase = self.purchase_food(budget)

        total_utility = []
        total_waste = None

        day_order_increment = None

        pantries_run = False

        for pantry in self.pantries:
            pantry_output = pantry.run_one_day()
            if pantry_output is None:
                continue  # if pantry wasn't held, we skip
            pantries_run = True
            waste, _, utility, _, true_order = pantry_output
            total_utility.append(utility)
            # total_utility = FoodBank.increment_utility(total_utility, utility)
            total_waste = FoodBank.increment_food_dict(total_waste, waste)
            refined_true_order = self.refine_true_order(true_order)
            day_order_increment = FoodBank.increment_food_dict(day_order_increment, refined_true_order)
            self.update_demand(refined_true_order)

        if not pantries_run:
            return total_waste, self.pantry_demand, 0, None

        self.update_weekly_demand(day_order_increment)

        return total_waste, self.pantry_demand, sum(total_utility) / len(total_utility), None

    @classmethod
    def refine_true_order(cls, order: Dict[str, float]) -> Dict[str, float]:
        return {STP: order[STP],
                FFV: (FoodBank.price_demand_choice(order, FFV, PFV, FV))[0],
                PFV: (FoodBank.price_demand_choice(order, FFV, PFV, FV))[1],
                FPT: (FoodBank.price_demand_choice(order, FPT, PPT, PT))[0],
                PPT: (FoodBank.price_demand_choice(order, FPT, PPT, PT))[1]}

    @classmethod
    def price_demand_choice(cls, order: Dict[str, float], food_a: str, food_b: str, food_type: str):
        if Global.price_for(food_a) < Global.price_for(food_b):
            return order[food_type], 0

        return 0, order[food_type]

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
    def increment_food_dict(cls, total_food_dict, new_food_dict):
        if total_food_dict is None:
            return new_food_dict
        return {food: (total_food_dict[food] + food_amount) for food, food_amount in new_food_dict.items()}

    def purchase_food(self, budget: float):
        """Purchases food using given budget

    :param budget: budget in dollars
    """
        demand = self.last_week_pantry_demand_proportion()
        types = demand.keys()
        remaining_days = [TYPES[t]['max_days'] for t in types]
        quantity = [demand[t] * budget / Global.price_for(t) for t in types]

        purchase = pd.DataFrame({"type": types, "remaining_days": remaining_days, "quantity": quantity})
        self.storage.add(purchase)
        return sum(quantity)

    def last_week_pantry_demand_proportion(self):
        """Returns demand in proportions. Used to decide what food to buy next.
    Calculation based on last week's demand

    :return: demand proportions
    """
        total = sum([sum(day_order.values()) for day_order in self.last_week_demand])
        pantry_demand = {}
        for demand in self.last_week_demand:
            for food, amount in demand.items():
                if food not in pantry_demand:
                    pantry_demand[food] = amount
                    continue
                pantry_demand[food] += amount
        return {food: 1 / len(Global.get_food_types()) if total == 0 else (amount / total) for (food, amount) in pantry_demand.items()}

    def update_demand(self, order):
        """Updates pantry demand values

    :param order: order made by a pantry
    """
        for food, amount in order.items():
            self.pantry_demand[food] += amount

    def update_weekly_demand(self, order):
        if order is None:
            return
        if len(self.last_week_demand) >= 7:
            self.last_week_demand.pop(0)
        self.last_week_demand.append(order)

    @classmethod
    def increment_utility(cls, total_utility: float, utility: float):
        """Increments total utility

    :param total_utility: 
    :param utility: 
    :return: new total utility
    """
        if total_utility is None:
            return utility
        else:
            return total_utility + utility


if __name__ == '__main__':
    food_bank = FoodBank(80_000, 500_000)
    # Global.add_day()
    Global._base_prices = {
        STP: 2,
        FFV: 3,
        PFV: 2,
        FPT: 2,
        PPT: 2,
    }
    import matplotlib.pyplot as plt
    utility_history = []
    bank_storage = []
    for day in range(100):
        # print('current day: ', day)
        _, _, utility, _ = food_bank.run_one_day(10000, 100)
        bank_storage.append(sum(food_bank.storage.get_quantity().values()))
        utility_history.append(utility)
        Global.add_day()
    plt.plot(utility_history)
    plt.title('Utility history')
    plt.figure()

    plt.plot(bank_storage, 'r')
    plt.title('Bank storage history')
    plt.show()
