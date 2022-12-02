import FoodPantry
import pandas as pd
from utils import Food

class FoodBank:
  def __init__(self, num_pantries: int, per_capita_wealth: float, initial_storage: Food):
    self.pantries = [FoodPantry(self) for _ in range(num_pantries)]
    self._storage = initial_storage
  
  def request_food(self, request:dict):
    """Function used by pantries to request food

    :param request: dataframe with food keys and quantities
    """
    pass

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