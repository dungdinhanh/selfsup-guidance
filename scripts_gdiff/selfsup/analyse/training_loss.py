import os

import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default="training_scratch")
parser.add_argument("--logpath", type=str, default="runs/selfsup_training/scratch/logs/progress.csv")


def read_csv_log(log_file: str):
    df = pd.read_csv(log_file)
    return df


def read_log_train_loss(df_reader: pd.DataFrame):
    training_loss = df_reader['train_loss']
    training_loss_q0 = df_reader['train_loss_q0']
    training_loss_q1 = df_reader['train_loss_q1']
    training_loss_q2 = df_reader['train_loss_q2']
    training_loss_q3 = df_reader['train_loss_q3']
    return training_loss, training_loss_q0, training_loss_q1, training_loss_q2, training_loss_q3


def visualize_loss(df_reader: pd.DataFrame, folder_file:str, file:str):
    file_path = os.path.join(folder_file, file)
    df_reader.plot(y=["train_loss", "train_loss_q0", "train_loss_q1", "train_loss_q2", "train_loss_q3"])
    plt.savefig(file_path)


if __name__ == '__main__':
    args = parser.parse_args()
    folder = "runs/visualize_training_ss/"
    os.makedirs(folder, exist_ok=True)
    df_reader = read_csv_log(args.logpath)
    visualize_loss(df_reader, folder, args.filename)