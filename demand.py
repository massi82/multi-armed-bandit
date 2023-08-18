import random
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import bernoulli

# parameter 'a' for the demand curve
a = 2
# candidate prices
prices = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
prices = [20, 30, 40, 50, 60]

# context map (context -> b, where b is the 'b' parameter of the demand curve)
PARAMETERS = {
    0: {
        0: 0.0255,
        1: 0.0364,
        2: 0.051,
        3: 0.0315,
    },
    1: {
        0: 0.0284,
        1: 0.0639,
        2: 0.0425,
        3: 0.0232,
    }
}

def demand_curve(price, a=0.5, b=0.05):
    return a / (1 + np.exp(b * price))

def revenue_derivative(x, a=0.5, b=0.05):
    # derivative of the function x*demand_curve
    return a / (np.exp(b * x) + 1) - (a * b * x * np.exp(b * x)) / (np.exp(b * x) + 1) ** 2

def get_optimal_price(a=0.5, b=0.05):
    return fsolve(revenue_derivative, 0, args=(a, b))[0]

def get_reward(price, a=0.5, b=0.05):
    prob = demand_curve(price, a, b)
    return bernoulli.rvs(prob)

def generate_context():
    # Randomly select a geographical position
    geographical_position = random.choice([0, 1])

    # Draw age from a Gaussian distribution
    # age = int(random.gauss(35, 10))
    age = int(random.uniform(0, 90))

    # Clip the age to make sure it's non-negative
    age = max(15, age)
    # Determine the corresponding age bucket
    if age <= 25:
        age = 0
    elif age <= 45:
        age = 1
    elif age <= 65:
        age = 2
    else:
        age = 3

    # Retrieve the corresponding 'a' and 'b' values
    b = PARAMETERS[geographical_position][age]

    return (geographical_position, age), a, b

optimal_prices = {}
for geo in PARAMETERS:
    for age in PARAMETERS[geo]:
        b = PARAMETERS[geo][age]
        # print(demand_curve(prices[0], a, b))
        root = fsolve(revenue_derivative, 0, args=(a, b))
        optimal_prices[(geo, age)] = root[0]

if __name__ == "__main__":
    for geo in PARAMETERS:
        for age in PARAMETERS[geo]:
            b = PARAMETERS[geo][age]
            # print(demand_curve(prices[0], a, b))
            root = get_optimal_price(a, b)
            print("Value for context %s of x for which the expression goes to zero: %.4f" %((geo, age), root))
            # print("Price probabilities for %s: %s" %((geo, age),[demand_curve(p, a, b) for p in prices]))
