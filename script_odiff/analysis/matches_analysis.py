import os.path

import numpy as np
import matplotlib.pyplot as plt

import argparse

def read_file(file_name: str):
    file = np.load(file_name)
    matches_x0_y = file["arr_0"]
    matches_x0_xt = file["arr_1"]
    matches_xt_y = file["arr_2"]
    return matches_x0_y, matches_x0_xt, matches_xt_y

def plot_lines(matches_x0_y, matches_x0_xt, matches_xt_y, file_name):
    x = np.arange(0, 250)
    plt.plot(x, matches_x0_y, label="matches_x0_y")
    plt.plot(x, matches_xt_y, label="matches_xt_y")
    plt.plot(x, matches_x0_xt, label="matches_x0_xt")
    plt.legend()
    plt.savefig(file_name)
    plt.close()

def plot_file(file_name):
    matches_x0_y, matches_x0_xt, matches_xt_y = read_file(file_name)
    parent_folder = os.path.dirname(file_name)
    fig_name = os.path.join(parent_folder, "matches.png")
    plot_lines(matches_x0_y, matches_x0_xt, matches_xt_y, fig_name)

parser = argparse.ArgumentParser(description="Process some integers")
parser.add_argument("--file", type=str,
                    default="runs/sampling_ots/IMN64/unconditional/scale14.0_jointtemp0.3_margtemp0.3_mocov2_meanclose_obs/reference/matches.npz")




if __name__ == '__main__':
    args = parser.parse_args()
    file_name = args.file
    plot_file(file_name)