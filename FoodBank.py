import FoodPantry
import pandas as pd
from utils import Food
from typing import List, Dict, Tuple
from Global import Global, TYPES


class FoodBank:
  def __init__(self, food_insecure_pop: int, initial_storage: float):
    """Food bank constructor

    :param food_insecure_pop: Number of food insecure people. Used to estimate number of pantries
    :param initial_storage: Initial storage of food in pounds. Value given to Food class
    """
    # we estimated this number from real data of the Eastern Food Bank
    ppl_per_pantry = 245
    # we assume half of the food insecure people actually use the bank
    num_pantries = int(.5 * food_insecure_pop / ppl_per_pantry)
    self.pantries:List['FoodPantry'] = [FoodPantry(self) for _ in range(num_pantries)]
    self.total_utility = None
    self.total_waste = None

    food_types = Global.get_food_types()
    self.pantry_demand = dict(zip(food_types, [0] * len(food_types)))

    self._storage = Food(initial_storage)

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

    for pantry in self.pantries:
      waste, order, utility = pantry.run_one_day()
      self.total_utility = FoodBank.increment_utility(self.total_utility, utility)
      self.total_waste = FoodBank.increment_waste(self.total_waste, waste)
      self.update_demand(order)

    self.purchase_food(budget)

    return self.total_waste, self.pantry_demand, self.total_utility

  @classmethod
  def increment_waste(total_waste, new_waste):
    return { food: (total_waste[food] + waste) for food, waste in new_waste}

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
    return { food: (amount/total) for (food, amount) in self.pantry_demand.items() }

  def update_demand(self, order):
    """Updates pantry demand values

    :param order: order made by a pantry
    """
    for food, amount in order.items():
      self.pantry_demand[food] += amount

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
