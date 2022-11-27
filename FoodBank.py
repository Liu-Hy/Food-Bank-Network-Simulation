import FoodPantry

class FoodBank:
  def __init__(self, num_pantries: int, per_capita_wealth: float):
    self.pantries = [FoodPantry(self) for i in range(num_pantries)]
    self.wealth = per_capita_wealth


