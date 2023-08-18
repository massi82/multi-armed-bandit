import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from demand import get_reward, prices, demand_curve, revenue_derivative

a = 2  
b = 0.042

optimal_price = fsolve(revenue_derivative, 0, args=(a, b))[0]
optimal_probability = demand_curve(optimal_price, a, b)
print(optimal_price)
print(optimal_probability)

best_price_index = index = (np.abs(np.array(prices) - optimal_price)).argmin()

print("----->", [demand_curve(p, a, b) for p in prices])

def greedy(arm_avg_reward):
    if np.all(arm_avg_reward == 0):
        # if we have not gotten any reward, go random
        arm = np.random.choice(len(arm_avg_reward))
    else:
        # else choose the arm with the highest average reward
        arm = np.argmax(arm_avg_reward)
    return arm

def epsilon_greedy(arm_avg_reward, epsilon=0.1):
    if np.random.rand() < epsilon:
        # with probability epsilon choose a random arm
        arm = np.random.choice(len(arm_avg_reward))
    elif np.all(arm_avg_reward == 0):
        # if we have not gotten any reward, go random
        arm = np.random.choice(len(arm_avg_reward))
    else:
        # else choose the arm with the highest average reward
        arm = np.argmax(arm_avg_reward)
    return arm

def UCB1(arm_avg_reward, arm_counter, iteration, C=1, normalize=False):     
    if np.all(arm_avg_reward == 0):        
        # if we have not gotten any reward, go random
        arm = np.random.choice(len(arm_avg_reward))  
        return arm  
    if 0 in arm_counter:  
        # if there's an arm that hasn't been pulled yet, pull it.
        arm = np.argmin(arm_counter)
        return arm
    # Total number of times any arm has been played
    total_plays = iteration + 1  # since iteration starts from 0
    
    if normalize:
        max_reward = arm_avg_reward.max()
        arm_norm_reward = arm_avg_reward/max_reward
        # Calculate upper bounds for all arms
        ucb_values = arm_norm_reward + C * np.sqrt(2 * np.log(total_plays) / arm_counter)
        ucb_values *= max_reward
    else:        
        # calculate upper bounds for all arms
        ucb_values = arm_avg_reward + C * np.sqrt(2 * np.log(total_plays) / arm_counter)
    
    # Return the arm which has the maximum upper bound
    return np.argmax(ucb_values)

def thompson_sampling(arm_prices, successes, failures):
    # print(successes/(successes+failures))    
    samples = [np.random.beta(successes[i]+1, failures[i]+1) for i in range(len(prices))]
    samples = [s*arm_prices[i] for i, s in enumerate(samples)]
    return np.argmax(samples)

def run_simulation(prices, nstep, strategy="epsgreedy"):
    reactivity = nstep # worst case scenario initialization
    react_counter = 10 # number of steps needed to confirm that the reactivity threshold has been hit
    cum_regret = np.zeros((nstep,))
    avg_reward = 0
    arm_counter = np.zeros_like(prices, dtype=float)
    arm_avg_reward = np.zeros_like(prices, dtype=float)
    
    if strategy == "thompson":
        successes = np.zeros_like(prices, dtype=int)
        failures = np.zeros_like(prices, dtype=int) 
    
    for iteration in range(nstep):
        if strategy == "greedy":
            arm = greedy(arm_avg_reward)
        elif strategy == "epsgreedy":
            arm = epsilon_greedy(arm_avg_reward, epsilon=0.1)
        elif strategy.startswith("ucb"):
            try: 
                if strategy.endswith("-norm"):
                    normalize = True
                    _, C, _ = strategy.split("-")
                else:
                    normalize = False
                    _, C = strategy.split("-")
                C = float(C)
            except:
                C = 1
                normalize = False
            arm = UCB1(arm_avg_reward, arm_counter, iteration, C=C, normalize=normalize)
        elif strategy == "thompson":
            arm = thompson_sampling(prices, successes, failures)
        
        reward = get_reward(prices[arm], a, b)
        # compute cumulative regret using the known optimal_price
        cum_regret[iteration] = cum_regret[iteration-1]+(optimal_price*optimal_probability - prices[arm]*reward)

        if strategy == "thompson":
            if reward > 0:
                successes[arm] += 1 
            else:
                failures[arm] += 1 

        # update the value for the chosen arm using a running average
        arm_counter[arm] += 1
        reward *= prices[arm]    
        arm_avg_reward[arm] = ((arm_counter[arm] - 1) * arm_avg_reward[arm] + reward) / arm_counter[arm]  
        avg_reward = ((iteration) * avg_reward + reward) / (iteration+1) 

        # verify if the reactivity threshold has been hit
        if iteration > 100 and react_counter != 0 and avg_reward >= 0.95*optimal_price*optimal_probability:
            react_counter -= 1
            if react_counter == 0:
                reactivity = iteration+1 

    return cum_regret, reactivity, arm_counter

nstep = 10000
nepoch = 1000
regret_curves = {}
for strategy in ["greedy", "epsgreedy", "thompson", "ucb1-0.7-norm"]:#
    regret_curves[strategy] = np.zeros((nstep,)) 
    regrets = []
    reactivities = []
    arm_counters = np.zeros((len(prices),))
    for ep in range(nepoch):
        regret, reactivity, arm_counter = run_simulation(prices, nstep, strategy=strategy)
        regret_curves[strategy] += regret
        regrets.append(regret[-1])
        reactivities.append(reactivity)
        arm_counters += arm_counter/nstep
    regret_curves[strategy] /= nepoch
    arm_allocation = 100*arm_counters/nepoch
    print("-------------\nStrategy: %s" %strategy)    
    print("Regret -> mean: %.2f, median: %.2f, std: %.2f" %(np.mean(regrets), np.median(regrets), np.std(regrets)))
    print("Reactivity -> mean: %.2f, median: %.2f, std: %.2f" %(np.mean(reactivities), np.median(reactivities), np.std(reactivities)))
    print("Arm allocation -> %s" %(arm_allocation))
    
plt.figure(figsize=(12, 6))
for label in regret_curves:
    plt.plot(regret_curves[label], label=label)
plt.xlabel("Time Step")
plt.ylabel("Cumulative Regret")
plt.title("Cumulative Regret Comparison")
plt.legend()
plt.grid(True)
plt.show()