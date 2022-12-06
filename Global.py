STP = "staples"
FFV = "fresh_fruits_and_vegetables"
PFV = "packaged_fruits_and_vegetables"
FPT = "fresh_protein"
PPT = "packaged_protein"

FV = "fruits_and_vegetables"
PT = "protein"

"""Only PERSON_WEEKLY_DEMAND and FAMILY-DISTRIBUTION are accurate. Others are fabricated. """
TYPES = {STP: {"proportion": 0.3, "max_days": 180},
         FFV: {"proportion": 0.1, "max_days": 14},
         PFV: {"proportion": 0.25, "max_days": 360},
         FPT: {"proportion": 0.1, "max_days": 10},
         PPT: {"proportion": 0.25, "max_days": 180}}

PERSONAL_WEEKLY_DEMAND = {STP: {"mean": 5.125, "std": 0.4},
                          FV: {"mean": 13.25, "std": 1.1},
                          PT: {"mean": 12.875, "std": 0.9}}

# 1.24% households have 7 or more persons. Assume the max number is 10 person, and 7-10 persons are equally likely
FAMILY_DISTRIBUTION = [0.2845, 0.3503, 0.1503, 0.1239, 0.0583, 0.0203] + ([0.0031] * 4)

ELASTICITY = {STP: -0.3, FV: -0.5, PT: -0.6}

config = {"pantry": {"set_limit": True, "use_real_demand": False}}

class Global:
  """Setters should only be used by simulation. Getters can be used by other classes.
  """
  _current_day:int = 0
  _price_inflation_pct: float  # dictionary with TYPE (str) float pairs. Set by Simulation.
  _base_prices = { # base prices for each food type
    STP: 0,
    FFV: 0,
    PFV: 0,
    FPT: 0,
    PPT: 0,
  }

  @classmethod
  def add_day(self):
    """Updates current day, only to be used by simulation.
    """
    self._current_day += 1

  @classmethod
  def get_day(self):
    return self._current_day

  @classmethod
  def price_for(self, food_type: str):
    """Returns price for given food type

    :param food_type: 
    :return: 
    """
    return self._base_prices[food_type]

  @classmethod
  def get_food_types(self):
    return self._base_prices.keys()


if __name__ == '__main__':
  print(Global.get_day())
  print(Global.add_day())
  print(Global.get_day())

  print(Global.get_food_types())