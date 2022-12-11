import pandas as pd

import Global
from Global import *
from utils import *
from FoodBank import *
from FoodPantry import *
from geographiclib.geodesic import Geodesic
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool as Pool

geod = Geodesic.WGS84


def initialize_food_banks(food_bank_df: pd.DataFrame) -> list[FoodBank]:
    """
    Use a combination of previous group data (food insecure population and geocoding) and manually annotated data
    (pounds of food distributed and programming budget) to initialize food banks
    :param food_bank_df: dataframe with food bank data per row
    :return: list of food_banks initialized with pre-generated input data
    >>> input_csv=pd.read_csv("input.csv").head(1)
    >>> len(initialize_food_banks(input_csv)[0].pantries) #(323340/5)/500 = 64 rounded down
    64


    """

    banks = []
    for row in food_bank_df.itertuples():
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
    >>> 76000<np.mean(test)<78000
    True
    >>> 14000000<np.sum(test)<142000000
    True
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
    >>> input_csv=pd.read_csv("input.csv").head(5)
    >>> test=generate_funds_distribution(input_csv, 365)[0]
    >>> 76000<np.mean(test)<78000
    True
    >>> 27000000<np.sum(test)<29000000
    True
    >>> np.max(test)<130000
    True

    """
    amounts = food_bank_df["estimated_pounds_per_year"]
    ret = np.zeros(shape=(len(amounts), num_days))
    for i in range(0, len(amounts)):
        total = amounts[i]
        daily_avg = total / 365
        min_val = daily_avg * 0.5  # min is half daily avg
        max_val = daily_avg * 2  # max is twice daily avg
        std_dev = daily_avg * 0.3  # estimate standard deviation
        ret[i] = mod_beta_random(low=min_val, high=max_val, mean=daily_avg, std=std_dev, samples=num_days)
    return ret


def generate_good_prices(price_summary: pd.DataFrame, num_days: int, ) -> dict[list[np.ndarray]]:
    """
    Use statistics from Bureau of Labor Statistics to generate price distributions for each good
    :param num_days: number of days to run simulation
    :param price_path: Dataframe of summary statistics for the good of interest
    :return: dictionary of price distributions for food items and gasoline
    >>> input_df=pd.read_csv("price_summary.csv")
    >>> output=generate_good_prices(input_df, 365)
    >>> 2.5<np.mean(output[GAS])<3.4
    True

    """
    price_dict = dict()

    price_dict[GAS] = good_price_distr(price_summary, "gas", num_days)
    price_dict[FFV] = good_price_distr(price_summary, "ffv", num_days)
    price_dict[FPT] = good_price_distr(price_summary, "meat", num_days)
    price_dict[STP] = good_price_distr(price_summary, "staples", num_days)
    price_dict[PFV] = good_price_distr(price_summary, "ffv", num_days) * PACKAGED_COST_RATIO # modify packaged price
    price_dict[PPT] = good_price_distr(price_summary, "meat", num_days) * PACKAGED_COST_RATIO # modify packaged price

    return price_dict
def good_price_distr(price_summary: pd.DataFrame,good:str, num_days:int, random_seed:int=RANDOM_SEED)->np.ndarray:
    """
    Generate price distribution
    starts at current price and performs a random walk
    resistance grows as price approaches good max or min
    mean change at

    :param random_seed: random seed
    :param price_summary: df of price data
    :param good: string name of row
    :param num_days: number of days
    :return: distribution of good price
    >>> prices=pd.read_csv("price_summary.csv")
    >>> gas=good_price_distr(prices, "gas", 365)
    >>> 2.5<np.mean(gas)<3.4
    True
    >>> np.std(gas)<0.3
    True
    """
    price_summary = price_summary.set_index("good")
    mean =price_summary.loc[good]["mean"]
    real_price=price_summary.loc[good]["latest_price"]

    mean_delta = price_summary.loc[good]["mean_delta"]
    mean_delta = mean_delta/30 #monthly mean change to daily mean change
    std_delta = price_summary.loc[good]["std_delta"]
    std_delta = std_delta/math.sqrt(30) #monthly std to daily std
    price_list=[mean] #start random walk from historical mean

    for i in range(1, num_days):
        prev=price_list[i-1] # start from previous day price
        scale_factor=(mean-prev)/mean # trend toward mean food price
        scaled_change=mean_delta*scale_factor
        change=np.random.normal(scaled_change,std_delta) # normal distribution centered around price change
        price_list.append(prev+change) #calculate new price
    return np.array(price_list)





def precalculate_supply_demand(food_banks:list[FoodBank]) ->(list[Food], list[Food]):
    """
    limit number of calls to food_going_bad, quality control, and future_unmet_demand
    :param food_banks:
    :return: tuple of list of Food objects
    >>> fb=[FoodBank(10000,3000, ), FoodBank(10000,3000, )]
    >>> sup_demand=precalculate_supply_demand(fb)
    >>> sup_demand[0][0].get_quantity()[FFV]!=0
    True
    >>> sup_demand[1][0].get_quantity()[FFV]!=0
    False


    """
    #precalculate food bank demands and supplies
    excess_supply=[]
    excess_demand=[]
    for f in food_banks:
        #precalculate supply and remove going bad food
        supply=f.food_going_bad()
        supply.quality_control()
        excess_supply.append(supply)

        #precalculate demands for each food bank
        excess_demand.append(f.future_unmet_demand())
    return (excess_supply,excess_demand)


def generate_net_demand(num_foodbank:int, excess_supply_demand:tuple[list[Food]]) -> np.ndarray:
    """

    :param num_foodbank: number of food banks
    :param excess_supply_demand: excess supply and demand Food objects per food bank
    :return: 3d numpy array of foodtype, sending foodbank, receiving food bank

    >>> fb=[FoodBank(100000,300000, ), FoodBank(100000,300000, )]
    >>> sup_demand=precalculate_supply_demand(fb)
    >>> generate_net_demand(2,sup_demand)

    """

    net_food_demand=np.zeros([len(food_goods),num_foodbank,num_foodbank]) #food type by food bank by food bank matrix
    for i in range(num_foodbank):
        available_food=excess_supply_demand[0][i].get_quantity()
        for j in range(num_foodbank):
            demanded_food=excess_supply_demand[1][j].get_quantity()
            for k in range(len(food_goods)):
                food=food_goods[k]
                if available_food[food]!=0 and -demanded_food[food]!=0:
                    print(available_food[food]!=0)
                    print(-demanded_food[food]!=0)
                net_food_demand[k,i,j]=min(available_food[food], -demanded_food[food])
    return net_food_demand


def food_network(food_banks: list, distance_mat: np.ndarray, payment_source:str="recipient",) -> None:
    """
    Implements national food sharing network
    food banks indexed by (sending,receiving)

    :param payment_source: setting to control distribution strategy,
    recipient: receiving food bank pays
    sender: sending food bank pays
    disaster: government (no transportation cost)
    :param distance_mat: distance matrix between pairs of food banks
    :param food_banks: list of food banks to redistribute food between
    :return: None

    Difficult to write doctests, as there are many dependencies in the Food Bank class and the

    """
    num_foodbank=len(food_banks)
    food_market_value=np.empty([num_foodbank, num_foodbank])

    excess_supply_demand=precalculate_supply_demand(food_banks)
    net_demand=generate_net_demand(num_foodbank, excess_supply_demand)

    pounds_to_move=np.sum(net_demand, axis=0)

    pounds_to_move[pounds_to_move<1000] = 0 # set threshold for reasonable transport
    print(np.sum(pounds_to_move))
    print(pounds_to_move)
    for i in range(len(food_goods)):
        food_market_value+=net_demand[i]*Global.price_for(food_goods[i])

    cost_per_truck=distance_mat/TRUCK_MPG*Global._gas_price * 2 #return trip for truck
    cost_per_lb_truck=cost_per_truck/POUNDS_PER_TRUCK

    total_cost=cost_per_lb_truck*pounds_to_move

    cost_effective= total_cost-food_market_value # if transport>market value of food, then feasible
    cost_effective[cost_effective < 0] = 0

def food_subset(food_supply, food_to_remove:dict):
    """

    :param food_supply:
    :param food_to_remove:
    :return: new food object
    """


def run_one_bank(arg_tuple:Tuple) -> tuple:
    """
    wrapper function for multiprocessing
    :param tuple(food_bank, daily budget, daily donations, index i, index j)

    :return: tuple of tick one day output
    """

    total_waste, pantry_demand, total_utility, _ = arg_tuple[0].run_one_day(arg_tuple[1][arg_tuple[4],arg_tuple[3]], arg_tuple[2][arg_tuple[4],arg_tuple[3]])
    return total_waste,pantry_demand,total_utility

if __name__ == "__main__":
    pool = Pool() #initialize MP pool

    food_banks_df = pd.read_csv("input.csv").head(10) #number of food banks to initialized
    prices_df = pd.read_csv("price_summary.csv")

    num_days=5 #number of days to run simulation
    inflation_rate=1.08 # settable

    #initialize Global state
    global_state=Global()
    global_state._price_inflation_pct=inflation_rate

    #initialize food banks
    food_banks=initialize_food_banks(food_banks_df)
    distances=generate_distance_matrix(food_banks_df)

    #generate randomized distributions
    daily_budget=generate_funds_distribution(food_banks_df, num_days)

    daily_donations=generate_food_distribution(food_banks_df, num_days)*DONATION_BOOST
    good_prices=generate_good_prices(prices_df, num_days)
    days = np.arange(0, num_days)
    plt.plot(days, good_prices[FFV])
    plt.show()
    """
    plt.plot(days, daily_donations[0])
    plt.show()
    plt.plot(days, daily_budget[0])
    plt.show()
    """

    daily_waste=dict()
    for d in food_goods:
        daily_waste[d]=[0]*num_days
    weekly_utility=[]
    weekly_total = 0
    for i in range(0, num_days):
        if i%7==0:
            weekly_utility.append(weekly_total)
            weekly_total=0

        print("Running day "+ str(i))
        for g in good_prices:
            if g==GAS:
                Global._gas_price = good_prices[g][i]*Global._price_inflation_pct
            else:
                Global.set_price(g,good_prices[g][i])

        Global.base_prices()#apply inflation

        if i>0:
            food_network(food_banks, distances)
        args=[] #args for multiprocess
        for j in range(0, len(food_banks)):
            curr=food_banks[j]
            args.append((curr, daily_budget, daily_donations, i, j))

        output=pool.map(run_one_bank, args) #run all food bank day method calls in parallel
        for r in output:
            if r[0]:
                for w in r[0]:
                    daily_waste[w][i]+=r[0][w]
            #print(pantry_demand)
            if r[2]:
                weekly_total+=r[2]


        Global.add_day()

    weeks=np.arange(0, len(weekly_utility))

    plt.plot(weeks, weekly_utility)
    plt.show()

    for w in daily_waste:
        plt.plot(days, daily_waste[w])
    plt.show()

    pool.close()






