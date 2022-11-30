import FoodPantry

class FoodBank:
  def __init__(self, food_insecure_pop: int):
    ppl_per_pantry =  260
    # we assume half of the food insecure people actually use the bank
    num_pantries = int(.5 * food_insecure_pop / ppl_per_pantry)
    self.pantries = [FoodPantry(self) for i in range(num_pantries)]
    # set proportion of pantries from food_insecure
