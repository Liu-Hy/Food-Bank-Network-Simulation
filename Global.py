import numpy as np

RANDOM_SEED = 1

STP = "staples"
FFV = "fresh_fruits_and_vegetables"
PFV = "packaged_fruits_and_vegetables"
FPT = "fresh_protein"
PPT = "packaged_protein"
GAS = "gas_price"

FOOD_GOODS = [STP, FFV, PFV, FPT, PPT]

FV = "fruits_and_vegetables"
PT = "protein"

rng = np.random.default_rng()

"""Only PERSONAL_WEEKLY_DEMAND and FAMILY_DISTRIBUTION are accurate. Others are educated guess. """
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

DEMAND_RATIO = 0.4  # from interview and statistics, clients on average need 40% of their demand from food banks

ELASTICITY = {STP: -0.3, FV: -0.5, PT: -0.6}

STOCKPILE_RATIO = 1 / 52  # foodbanks have 1 week of stored food (info from interview)

FOOD_PURCHASE_BUDGET_RATIO = 0.20  # foodbanks use roughly 20% of annual budget for food purchase (info from disclosure)

PACKAGED_COST_RATIO = 0.95  # packaged food slightly cheaper than fresh food

BASELINE_PRICE = {STP: 0.881335917, FV: 1.231762712,
                  PT: 2.843222868}  # set baseline prices based on mean of observed values

# from (https://www.pensketruckrental.com/commercial-truck-rental/commercial-trucks/refrigerated-trucks/18-26-foot-refrigerated-truck-cdl-required/)
POUNDS_PER_TRUCK = 5_000
TRUCK_MPG = 10

DONATION_BOOST = 2  # ratio to boost food donations to enable testing of network

POPULATION_FACTOR = 0.1  # set lower to raise global speed and increase food supply


class Global:
    """Setters should only be used by simulation. Getters can be used by other classes.
  """
    # we originally estimated this number from real data of the Eastern Illinois Food Bank
    # we then decided on increasing that number to have less pantries and increase efficiency
    households_per_pantry = 500
    _current_day: int = 0
    _price_inflation_pct: float = 1  # dictionary with TYPE (str) float pairs. Set by Simulation.
    _base_prices = {  # base prices for each food type
        STP: 1,
        FFV: 1,
        PFV: 1,
        FPT: 1,
        PPT: 1,
    }
    _gas_price = 0
    config = {
        "pantry": {"set_limit": True, "use_real_demand": False}
    }

    @classmethod
    def set_price(cls, good_type: str, price: float):
        """
        sets price for food, used by simulation
        :param good_type: string name of food type
        :param price: float price
        :return: None
        """
        cls._base_prices[good_type] = price

    @classmethod
    def add_day(cls):
        """Updates current day, only to be used by simulation.
    """
        cls._current_day += 1

    @classmethod
    def get_food_demand_types(cls):
        return [STP, FV, PT]

    @classmethod
    def get_day(cls):
        return cls._current_day

    @classmethod
    def price_for(cls, food_type: str):
        """Returns price for given food type

    :param food_type: 
    :return: 
    """
        return cls._base_prices[food_type]

    @classmethod
    def get_food_types(cls):
        return cls._base_prices.keys()

    @classmethod
    def base_prices(cls):
        """Applies inflation to base prices

    :return: base prices dictionary with inflation applied
    """
        return {k: v * cls._price_inflation_pct for k, v in cls._base_prices.items()}


if __name__ == '__main__':
    print(Global.get_day())
    print(Global.add_day())
    print(Global.get_day())

    print(Global.get_food_types())
