import FoodPantry
import pandas as pd
from utils import Food

class FoodBank:
  def __init__(self, food_insecure_pop: int):
    ppl_per_pantry =  260
    # we assume half of the food insecure people actually use the bank
    num_pantries = int(.5 * food_insecure_pop / ppl_per_pantry)
    self.pantries = [FoodPantry(self) for i in range(num_pantries)]
    # set proportion of pantries from food_insecure

  def food_storage(self) -> pd.DataFrame:
    """Returns food storage stats as a dataframe

    :return: storage dataframe copy
    """
    return self._storage.df.copy()

  def receive_dotations(self, amount: float):
    """Receives donations and allocates it to buying specific products

    :param amount: amount of donations in dollars
    """
  
  def buy_food(self, donations: float):
    """Buys food based on donations and demand from pantries.

    :param donations: donations in dollars
    """
    # predicted supply and predicted demand
    # open to receiving food
    # food available to donate
    # feeding america tax
