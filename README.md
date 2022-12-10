# Food Bank Network Simulation
## Background

## Model of clients
### Client demand
Let $T \in \mathbb{R}^3$ denote the weekly physical demand of a household for each type of food, and $q \in \mathbb{R}^3$ denote the proportion of food demand that a household meets by purchasing without the help of food banks. $Tq$ responds to the fluctuation of food price $p \in \mathbb{R}^3$. Assume that the elasticity of demand is constant in the range of price, denoted as $k \in \mathbb{R}^3$, and that the purchase of different types of food are independent. Solving an ODE 
$k = dln(Tq_i) / dln(p_i)$ we find that $q_i$ is proportional to $(p_i)^{k_i}, i = 1,2,3$. Then using baseline price $p_0^i$ and quantity $q_0^i$, $q_i = \frac{q_i^0}{{(p_i^0)}^{k_i}} * {p_i}^{k_i}$.

Assume that the individual demand $V$ conforms to a 4-parameter Beta distribution with mean and std. given by statistics, and the family size is $S$ with a given discrete distribution. Then, the demand of a household to a food pantry is given by:

$D(p) = SV * (1-q(p)) + n$, 
where $n$ is a Gaussian noise. As food price increases, people's affordability decreases and hence need more food from food pantries.

### Purchase and Utility

## Hypotheses

## Unit Food Bank Simulation
To test that our simulation is working at the Food Bank level, measured daily average utility.
In the following plot, it we observe that average utility stabilizes at around .25, which is the value we expect:

![Stable Utility Plot](plots/foodbank_utility_history_stable.png)

The dips in utility are also reasonable, as they represend days when pantries are not held.

Furthermore, when a food bank is provided with excessive amounts of donations, we have that utility is much higher, reaching around .32 (out of .4):

![Excess Donations Bank Utility](./plots/foodbank_utility_history_high_donations.png)

## Planning
### Presentation

- intro to the problem being addressed: Haoyang
  - hypothesis
- research background: Lucian & Haoyang
  - interviews
  - data
- design: Rodrigo
- code sections: All
- prelim results: All

### Who pays

- receiving foodbank
- ordering food bank

#### Suggestions from Mr. Weible:
(Office hour Dec 1)
- The receiving foodbank pays for transportation
- The economic utility function may or may not be valid. Also make intuitive statistics, e.g. the proportion of clients\
who get no food at all
- Compare the setting of sharing food between foodbanks v.s no sharing
- The simulation doesn't need to be too complicated.

