# Food Bank Network Simulation
## Background

## Model of clients
Let $D \in \mathbb{R}^3$ denote the total physical demand for each type of food in a household, and $q \in \mathbb{R}^3$ denote the proportion of food demand that a household meets by purchasing without the help of food banks. $Dq$ responds to the fluctuation of food price $p \in \mathbb{R}^3$. Assume that the elasticity of demand is constant in the range of price, denoted as $k \in \mathbb{R}^3$, and that the purchase of different types of food are independent. Solving an ODE 
$k = dln(Dq_i) / dln(p_i)$
we find that $q_i$ is proportional to $(p_i)^k, k = 1,2,3$.

The demand of a household to a food pantry is given by:
Actual demand =  random individual demand * family size * (1 - q(p)) + Gaussian r.v. 

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

