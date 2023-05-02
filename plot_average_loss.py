"""
Reads the output files of an model and plots the average loss with rolling average.
"""
import numpy as np
import csv
from matplotlib import pyplot as plt


def read_csv_data(file_paths):
    data = []
    for file_path in file_paths:
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            lines = [float(line[1]) for line in reader]
            data.append(lines)
    return data


def calculate_mean(data, window_size=20):
    means = [np.mean(e) for e in zip(*data)]
    return [np.mean(means[i : i + window_size]) for i in range(len(means) - window_size + 1)]


def plot_means(lists, labels):
    means = [mean_of_csv(l) for l in lists]

    for m, l in zip(means, labels):
        plt.plot(m, label=l)

    plt.xlabel("Episode")
    plt.ylabel("Mean of loss")
    plt.legend()
    plt.show()


def plot_data(lists, labels):
    data = [read_csv_data(l) for l in lists]

    means = [calculate_mean(d) for d in data]

    for m, l in zip(means, labels):
        plt.plot(m, label=l)

    plt.xlabel("Episode", fontsize="16")
    plt.ylabel("Mean value of loss", fontsize="16")
    plt.legend(fontsize="16")
    plt.show()


num_runs = 30
# Stratified small +0.5stdev
bp = "runs/Pendulum-v1/adapt_strat_sb_pos_"
adapt_strat_sb_pos = [f"{bp}{i}/loss.csv" for i in range(num_runs)]

# Stratified large +0.5stdev
bp = "runs/Pendulum-v1/adapt_strat_lb_pos_"
adapt_strat_lb_pos = [f"{bp}{i}/loss.csv" for i in range(num_runs)]

# Stratified small -0.5stdev
bp = "runs/Pendulum-v1/adapt_strat_sb_neg_"
adapt_strat_sb_neg = [f"{bp}{i}/loss.csv" for i in range(num_runs)]

# Stratified large -0.5stdev
bp = "runs/Pendulum-v1/adapt_strat_lb_neg_"
adapt_strat_lb_neg = [f"{bp}{i}/loss.csv" for i in range(num_runs)]

#### 0.7 sampling ratio
# Stratified small +0.5stdev
bp = "runs/Pendulum-v1/adapt_strat_sb_pos_sr07_"
adapt_strat_sb_pos_sr07 = [f"{bp}{i}/loss.csv" for i in range(num_runs)]

# Stratified large +0.5stdev
bp = "runs/Pendulum-v1/adapt_strat_lb_pos_sr07_"
adapt_strat_lb_pos_sr07 = [f"{bp}{i}/loss.csv" for i in range(num_runs)]

# Stratified small -0.5stdev
bp = "runs/Pendulum-v1/adapt_strat_sb_neg_sr07_"
adapt_strat_sb_neg_sr07 = [f"{bp}{i}/loss.csv" for i in range(num_runs)]

# Stratified large -0.5stdev
bp = "runs/Pendulum-v1/adapt_strat_lb_neg_sr07_"
adapt_strat_lb_neg_sr07 = [f"{bp}{i}/loss.csv" for i in range(num_runs)]

### Softmax
# Large buffer
bp = "runs/Pendulum-v1/softmax_lb_"
softmax_lb = [f"{bp}{i}/loss.csv" for i in range(num_runs)]

# Small buffer
bp = "runs/Pendulum-v1/softmax_sb_"
softmax_sb = [f"{bp}{i}/loss.csv" for i in range(num_runs)]


####

# Basic small buffer
bp = "runs/Pendulum-v1/basic_sb_"
basic_sb = [f"{bp}{i}/loss.csv" for i in range(num_runs)]

# Basic large buffer
bp = "runs/Pendulum-v1/basic_lb_"
basic_lb = [f"{bp}{i}/loss.csv" for i in range(num_runs)]


lists = [
    adapt_strat_sb_pos,
    adapt_strat_sb_pos_sr07,
    adapt_strat_lb_pos,
    adapt_strat_lb_pos_sr07,
    adapt_strat_sb_neg,
    adapt_strat_sb_neg_sr07,
    adapt_strat_lb_neg,
    adapt_strat_lb_neg_sr07,
    softmax_lb,
    softmax_sb,
    basic_sb,
    basic_lb,
]
labels = [
    "strat: sb, +0.5std,",
    "strat: sb, +0.5std, sr 0.7",
    "strat: lb, +0.5std,",
    "strat: lb, +0.5std, sr 0.7",
    "strat: sb, -0.5std,",
    "strat: sb, -0.5std, sr 0.7",
    "strat: lb, -0.5std,",
    "strat: lb, -0.5std, sr 0.7",
    "softmax: lb",
    "softmax: sb",
    "basic: sb",
    "basic: lb",
]
"""
num_runs = 6

# Stratified large +0.5stdev
bp = "runs/BipedalWalker-v3/adapt_strat_lb_pos_"
adapt_strat_lb_pos = [f"{bp}{i}/loss.csv" for i in range(num_runs)]

# Stratified large -0.5stdev
bp = "runs/BipedalWalker-v3/adapt_strat_lb_neg_"
adapt_strat_lb_neg = [f"{bp}{i}/loss.csv" for i in range(num_runs)]

### Softmax
# Large buffer
bp = "runs/BipedalWalker-v3/softmax_lb_"
softmax_lb = [f"{bp}{i}/loss.csv" for i in range(num_runs)]

####

# Basic large buffer
bp = "runs/BipedalWalker-v3/basic_lb_"
basic_lb = [f"{bp}{i}/loss.csv" for i in range(num_runs)]


lists = [
    adapt_strat_lb_pos,
    adapt_strat_lb_neg,
    softmax_lb,
    basic_lb,
]
labels = [
    "strat: lb, +0.5std,",
    "strat: lb, -0.5std,",
    "softmax: lb",
    "basic: lb",
]
"""
plot_data(lists, labels)
