import pandas as pd
from Global import *
from utils import *
from FoodBank import *
from FoodPantry import *
from geographiclib.geodesic import Geodesic
import matplotlib.pyplot as plt

geod = Geodesic.WGS84


def initialize_food_banks(food_bank_df: pd.DataFrame) -> list[FoodBank]:
    """
    Use a combination of previous group data (food insecure population and geocoding) and manually annotated data
    (pounds of food distributed and programming budget) to initialize food banks
    :param food_bank_df: dataframe with food bank data per row
    :return: list of food_banks initialized with pre-generated input data
    >>> input_csv=pd.read_csv("input.csv").head(1)
    >>> len(initialize_food_banks(input_csv)[0].pantries) #(323340/2)/245 = 659 rounded down
    659


    """

    banks = []
    for row in food_bank_df.itertuples():
        print("Creating food bank in: "+ row.city)
        stored_food = row.estimated_pounds_per_year * STOCKPILE_RATIO
        banks.append(FoodBank(row.food_insecure_pop, stored_food))
    return banks


def generate_distance_matrix(food_bank_df: pd.DataFrame) -> np.ndarray:
    """

    :param food_bank_df: food_bank_df: dataframe with food bank locations
    :return: array of distances (km) between food bank pairs (foodbank x foodbank)

    >>> input_csv=pd.read_csv("input.csv").head(2)
    >>> generate_distance_matrix(input_csv)[0,1]# doctest: +ELLIPSIS
    269.74...

    """
    lat = food_bank_df["latitude"]
    long = food_bank_df["longitude"]
    num_bank = len(lat)
    distance = np.empty([num_bank, num_bank])

    for i in range(0, num_bank):
        for j in range(0, num_bank):
            pair_dist = geod.Inverse(lat[i], long[i], lat[j], long[j])['s12']
            pair_dist = pair_dist / 1609.34  # convert meters to miles
            distance[i, j] = pair_dist

    return distance


def generate_funds_distribution(food_bank_df: pd.DataFrame, num_days: int) -> np.ndarray:
    """
    generates randomized distribution for daily food purchase budget
    daily budget distributed randomly but approach annual budget
    :param num_days: number of days to run simulation
    :param food_bank_df: dataframe with food bank data per row
    :return: numpy array, beta distribution of food purchase funds per day (foodbank x days)

    >>> input_csv=pd.read_csv("input.csv").head(5)
    >>> test=generate_funds_distribution(input_csv, 365)[0]
    >>> np.mean(test)
    100
    >>> np.sum(test)<14094249
    """
    budgets = food_bank_df["estimated_budget"]
    ret = np.zeros(shape=(len(budgets), num_days))
    for i in range(0, len(budgets)):
        annual = budgets[i]
        total = annual * FOOD_PURCHASE_BUDGET_RATIO  # annual budget for food purchases
        daily_avg = total / 365
        min_val = daily_avg * 0.5  # min is half daily avg
        max_val = daily_avg * 2  # max is twice daily avg
        std_dev = daily_avg * 0.2  # estimate standard deviation as 20%
        ret[i] = mod_beta_random(low=min_val, high=max_val, mean=daily_avg, std=std_dev, samples=num_days)
    return ret


def generate_food_distribution(food_bank_df: pd.DataFrame, num_days: int) -> np.ndarray:
    """
    :param num_days: number of days to run simulation
    :param food_bank_df: dataframe with food bank data per row
    :return: beta distribution of food donations per day (foodbank x day)
    """
    amounts = food_bank_df["estimated_pounds_per_year"]
    ret = np.zeros(shape=(len(amounts), num_days))
    for i in range(0, len(amounts)):
        annual = amounts[i]
        total = annual * (1 - FOOD_PURCHASE_BUDGET_RATIO)  # 90% of food is donated
        daily_avg = total / 365
        min_val = daily_avg * 0.5  # min is half daily avg
        max_val = daily_avg * 2  # max is twice daily avg
        std_dev = daily_avg * 0.2  # estimate standard deviation as 20%
        ret[i] = mod_beta_random(low=min_val, high=max_val, mean=daily_avg, std=std_dev, samples=num_days)
    return ret


def generate_good_prices(price_summary: pd.DataFrame, num_days: int) -> dict[list[float]]:
    """
    Use statistics from Bureau of Labor Statistics to generate price distributions for each good
    :param num_days: number of days to run simulation
    :param price_path: Dataframe of summary statistics for the good of interest
    :return: dictionary of price distributions for food items and gasoline
    >>> input_df=pd.read_csv("price_summary.csv")
    >>> output=generate_good_prices(input_df, 365)
    >>> np.mean(output["gas"])
    """
    price_dict = dict()

    price_dict["gas"] = good_price_distr(price_summary, "gas", num_days)
    price_dict[FFV] = good_price_distr(price_summary, "ffv", num_days)
    price_dict[FPT] = good_price_distr(price_summary, "meat", num_days)
    price_dict[STP] = good_price_distr(price_summary, "staples", num_days)
    price_dict[PFV] = good_price_distr(price_summary, "ffv", num_days) * PACKAGED_COST_RATIO # modify packaged price
    price_dict[PPT] = good_price_distr(price_summary, "meat", num_days) * PACKAGED_COST_RATIO # modify packaged price

    return price_dict
def good_price_distr(price_summary: pd.DataFrame,good:str, num_days:int)->np.ndarray:
    """
    Generate price distribution
    starts at current price and performs a random walk
    resistance grows as price approaches good max or min
    mean change at

    :param price_summary: df of price data
    :param good: string name of row
    :param num_days: number of days
    :return: distribution of good price
    >>> prices=pd.read_csv("price_summary.csv")
    >>> good_price_distr(prices, "gas", 100)
    """
    price_summary = price_summary.set_index("good")
    mean =price_summary.loc[good]["mean"]
    real_price=price_summary.loc[good]["latest_price"]

    mean_delta = price_summary.loc[good]["mean_delta"]
    mean_delta = mean_delta/30 #monthly mean change to daily mean change
    std_delta = price_summary.loc[good]["std_delta"]
    std_delta = std_delta/30 #monthly std to daily std

    price_list=[mean] #start random walk from historical mean

    for i in range(1, num_days):
        prev=price_list[i-1] # start from previous day price
        scale_factor=(mean-prev)/mean # trend toward mean food price
        scaled_change=mean_delta*scale_factor
        print(scaled_change)
        change=np.random.normal(scaled_change,std_delta) # normal distribution centered around price change
        price_list.append(prev+change) #calculate new price
    return np.array(price_list)




def tick_day(food_banks: list):
    """

    :param food_banks:
    :return:
    """


def redistribute_food(food_banks: list, distance_mat: np.ndarray, prices:dict=Global._base_prices, payment_source:str="recipient") -> None:
    """
    Implements cr

    :param prices:
    :param payment_source:
    :param distance_mat: distance matrix between pairs of food banks
    :param food_banks: list of food banks to redistribute food between
    :return: None
    """


    for i in range(len(food_banks)):
        for j in range(len(food_banks)):
           for food in food_goods:
                Global.price_for(food)





if __name__ == "__main__":
    food_banks_df = pd.read_csv("input.csv").head(1)
    prices_df = pd.read_csv("price_summary.csv")
    num_days=365
    inflation_rate=1.08 # settable (to add)
    #initialize Global state
    global_state=Global()
    global_state._price_inflation_pct=inflation_rate

    #initialize food banks
    food_banks=initialize_food_banks(food_banks_df)

    #generate randomized distributions
    daily_budget=generate_funds_distribution(food_banks_df, num_days)
    daily_donations=generate_food_distribution(food_banks_df, num_days)
    good_prices=generate_good_prices(prices_df, num_days)

    for g in good_prices:
        x=np.arange(0,num_days)
        y=good_prices[g]
        plt.plot(x, y)
        plt.show()

    for i in range(0, num_days):
        for g in good_prices:
            Global.set_price(g,good_prices[g][i])

        for j in range(0, len(food_banks)):
            curr=food_banks[j]
            print(daily_budget[j,i])
            print(daily_donations[j,i])
            curr.run_one_day(budget=daily_budget[j,i], food_donations=daily_donations[j,i])
            #print(curr.total_waste)








