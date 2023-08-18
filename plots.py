import matplotlib.pyplot as plt

def plot_arm_allocations(arm_allocations):
    # Colors and labels
    colors = ['#FF9999', 'lightgreen', '#66B2FF', '#FFCC99', '#E6E6FA']
    labels_with_price = ["price: 20", "price: 30", "price: 40", "price: 50", "price: 60"]
    strategy_titles = ["Greedy", "ε-greedy", "Thompson Sampling", "UCB1"]

    # Pie Chart for each strategy
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    fig.suptitle("Arm Allocation", fontsize=18, y=0.99)
    axes = axes.ravel()

    for i, strategy in enumerate(strategy_titles):
        ax = axes[i]
        wedges, texts, autotexts = ax.pie(
            arm_allocations[i][::-1],
            autopct='%1.1f%%',
            startangle=90,
            colors=colors[::-1],
            pctdistance=0.85,
            # wedgeprops=dict(width=0.2, edgecolor='black', linewidth=0.5)
        )

        # Draw a center circle for 'donut' style
        # centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        # ax.add_artist(centre_circle)

        # Increase the size and weight of the percentage labels
        for text in autotexts:
            text.set(size=10)

        ax.set_title(strategy, fontsize=13, y=0.92)

    fig.legend(wedges[::-1], labels_with_price, title="Arms", loc="upper right", fontsize='large')

    plt.tight_layout(pad=0.01)
    plt.show()

# Data
# First Round
arm_allocations = [
    [33.59666, 27.59777, 18.20022, 13.00166, 7.60369],
    [11.42941, 64.58118, 18.75543, 3.00168, 2.2323],
    [4.88634, 70.08769, 18.50603, 4.38611, 2.13383],
    [41.54771, 30.43495, 17.8829, 7.61774, 2.5167]
]

# Second Round
arm_allocations = [
    [34.39607, 24.89851, 20.40018, 12.10207, 8.20317],  # greedy
    [12.1732, 64.81001, 17.89838, 2.89369, 2.22472],    # epsgreedy
    [4.85504, 69.37921, 19.35197, 4.37928, 2.0345],     # thompson
    [8.26354, 76.07635, 13.34003, 1.75242, 0.56766]     # ucb1-0.7-norm
]


# Plotting
plot_arm_allocations(arm_allocations)




# import matplotlib.pyplot as plt
# import numpy as np

# # Data
# strategies = ['greedy', 'epsgreedy', 'thompson', 'ucb1']

# # First Round
# # Regret
# means_regret = [11565.34, 4258.84, 3844.70, 9081.66]
# medians_regret = [10822.16, 3257.16, 3712.16, 10397.16]
# std_regret = [11751.26, 3414.00, 2177.11, 8694.37]

# # Reactivity
# medians_reactivity = [183.50, 181.50, 468.00, 183.50]

# # Second Round
# # Regret
# means_regret = [11943.96, 4370.22, 3839.43, 2620.22]
# medians_regret = [10982.16, 3477.16, 3627.16, 2197.16]
# std_regret = [11737.17, 3379.81, 2185.16, 2462.17]


# # Reactivity
# # Reactivity
# medians_reactivity = [179.50, 202.50, 489.00, 210.00]

# # Bar plot setup
# barWidth = 0.3
# r1 = np.arange(len(means_regret))
# r2 = [x + barWidth for x in r1]

# # Plot
# fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# # Regret plot
# ax[0].bar(r1, means_regret, width=barWidth, color='#AED6F1', edgecolor='grey', yerr=std_regret, capsize=7, label='Mean Regret')
# ax[0].bar(r2, medians_regret, width=barWidth, color='#FAD7A0', edgecolor='grey', label='Median Regret')
# ax[0].set_xlabel('Strategy', fontweight='bold')
# ax[0].set_ylabel('Regret Value', fontweight='bold')
# ax[0].set_title('Regret (Mean and Median) by Strategy', fontweight='bold')
# ax[0].set_xticks([r + barWidth for r in range(len(means_regret))])
# ax[0].set_xticklabels(strategies)
# ax[0].legend()

# # Reactivity plot
# ax[1].bar(strategies, medians_reactivity, color='#D2B4DE', edgecolor='grey')
# ax[1].set_xlabel('Strategy', fontweight='bold')
# ax[1].set_ylabel('Reactivity Value', fontweight='bold')
# ax[1].set_title('Reactivity (Median) by Strategy', fontweight='bold')

# plt.tight_layout()
# plt.show()

