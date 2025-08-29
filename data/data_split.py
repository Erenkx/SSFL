"""
This module provides utility functions for simulating data splits 
with varying degrees of non-IID heterogeneity using the Dirichlet 
distribution.

It includes functions for:
- Splitting datasets into subsets across clients with controlled label
  heterogeneity.
- Visualizing class distributions across clients in each split.

Originally adapted from the SSL-FL codebase:
https://github.com/rui-yan/SSL-FL

Expected directory structure:
-----------------------------
To use the functions in this module, your dataset folder should be 
organized as follows:

data/
└── dataset_name/
    ├── train/          # Folder containing training image files.
    ├── test/           # Folder containing testing image files.
    ├── train.csv       # List of training sample filenames.
    ├── test.csv        # List of testing sample filenames.
    ├── labels.csv      # CSV mapping sample filenames to labels.
    └── central/
        └── train.csv   # Duplicate of train.csv for centralized 
                          training reference.   
"""

import os
import csv
import glob
from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def dirichlet_split(
    labels: np.ndarray, 
    n_clients: int, 
    n_classes: int, 
    beta: float = 100
) -> dict:
    """
    Generates non-IID client data indices using Dirichlet distribution.

    Args:
        labels (np.ndarray): Array of labels for training samples.
        n_clients (int): Number of clients to split data into.
        n_classes (int): Total number of classes in the dataset.
        beta (float): Dirichlet concentration parameter. Smaller values
            yield higher heterogeneity. Default is 100.
    
    Returns:
        dict: A dictionary mapping client indices to lists of data 
            indices. Keys are client IDs starting from 1.
    """
    np.random.seed(2022)
    N = labels.shape[0]
    clients = {}

    min_client_size = 0
    min_required_size = 10
    while min_client_size < min_required_size:
        splits = [[] for _ in range(n_clients)]

        for label in range(n_classes):
            label_indices = np.where(labels == label)[0]
            np.random.shuffle(label_indices)

            proportions = np.random.dirichlet(np.repeat(beta, n_clients))
            proportions = np.array([
                p * (len(split) < N / n_clients)
                for p, split in zip(proportions, splits)
            ])

            norm_proportions = proportions / proportions.sum()
            split_points = (
                np.cumsum(norm_proportions) * len(label_indices)
            ).astype(int)[:-1]
            
            for i, split in enumerate(np.split(label_indices, split_points)):
                splits[i].extend(split.tolist())
            
        min_client_size = min([len(split) for split in splits])

    for i in range(n_clients):
        np.random.shuffle(splits[i])
        clients[i + 1] = splits[i]

    return clients


def split_data(
    data_path: str,
    n_clients: int,
    n_classes: int,
    beta_list: list[float] = [100, 1, 0.5]
) -> None:
    """
    Simulates Dirichlet-based data splits and saves client-wise sample
    lists to CSV files under the given dataset directory.

    Args:
        data_path (str): Path to the dataset directory. Must contain
            'central/train.csv' and 'labels.csv'.
        n_clients (int): Number of simulated clients.
        n_classes (int): Number of distinct classes in the dataset.
        beta_list (list[float]): List of Dirichlet concentration 
            parameters to generate multiple splits. Smaller values yield
            higher heterogeneity. Default is [100, 1, 0.5].

    Returns:
        None: Client split files are saved to:
            {data_path}/{n_clients}_clients/split_{i}/client_{k}.csv
    """
    # Load training sample filenames
    train_path = os.path.join(data_path, 'central', 'train.csv')
    train_samples = {
        line.strip().split(',')[0]: None
        for line in open(train_path)
    }
    
    # Load full label mapping
    label_path = os.path.join(data_path, 'labels.csv')
    labels = {
        line.strip().split(',')[0]: int(float(line.strip().split(',')[1]))
        for line in open(label_path)
    }

    # Extract labels for training samples only
    train_labels = {
        sample: label 
        for sample, label in labels.items() 
        if sample in train_samples
    }
    
    # Generate one split per beta value
    for split_id, beta in enumerate(beta_list, start=1):
        print(f'\n------- split_{split_id} (beta={beta}) -------')

        split_path = os.path.join(
            data_path, f'{n_clients}_clients', f'split_{split_id}'
        )
        os.makedirs(split_path, exist_ok=True)

        # Perform Dirichlet split
        clients = dirichlet_split(
            labels=np.array(list(train_labels.values())),
            n_clients=n_clients,
            n_classes=n_classes,
            beta=beta
        )

        train_fnames = np.array(list(train_labels.keys()))
        train_values = np.array(list(train_labels.values()))
        for client_id, sample_indices in clients.items():
            client_samples = train_fnames[sample_indices]
            client_labels = train_values[sample_indices]

            print(f'Client {client_id}: {dict(Counter(client_labels))}')

            client_path = os.path.join(split_path, f'client_{client_id}.csv')
            with open(client_path, 'w') as fout:
                writer = csv.writer(fout, delimiter='\n')
                writer.writerow(client_samples.tolist())

    print()


def view_split(
    data_path: str,
    n_clients: int = 5,
    save_plot: bool = False
) -> dict:
    """
    Loads and visualizes client-wise label distributions for all 
    Dirichelet-based data splits under a given dataset path. 

    Args:
        data_path (str): Path to the dataset directory. Should contain
            '{n_clients}_clients/split_{i}/client_{k}.csv' and 
            'labels.csv'.
        n_clients (int): Number of clients. Default is 5.
        save_plot (bool): If True, saves bar plots of class distribution
            for each split. Default is False.

    Returns:
        dict: Nested dictionary of label counts per client for each 
            split. 
            Format: dict[split_id][client_id] = Counter({label: count})
    """
    # Load global labels
    label_path = os.path.join(data_path, 'labels.csv')
    labels = {
        line.strip().split(',')[0]: int(float(line.strip().split(',')[1]))
        for line in open(label_path)
    }
    
    # Locate split directories
    client_root = os.path.join(data_path, f'{n_clients}_clients')
    split_folders = sorted([
        d for d in os.listdir(client_root)
        if os.path.isdir(os.path.join(client_root, d)) 
            and d.startswith('split_')
    ])
    split_summary = {}

    # Collect label counts per client per split
    for split in split_folders:
        client_label_counter = {}
        client_folder = os.path.join(client_root, split)
        client_files = sorted(
            glob.glob(os.path.join(client_folder, 'client_*.csv'))
        )
        
        for client_path in client_files:
            client_name = os.path.basename(client_path).split('.')[0]

            with open(client_path, 'r') as fin:
                samples = [line.strip().split(',')[0] for line in fin]
            
            sample_labels = [
                labels[sample] for sample in samples 
                if sample in labels
            ]
            client_label_counter[client_name] = Counter(sample_labels)

        split_summary[split] = client_label_counter

    # Plot distributions
    sns.set_theme(font_scale=1.8, style='white')

    df = pd.DataFrame(split_summary)
    fig, axes = plt.subplots(
        1, len(split_folders), figsize=(6 * len(split_folders), 5)
    )

    if len(split_folders) == 1:
        axes = [axes]   # make sure axes is iterable

    for i, split_id in enumerate(split_folders):
        df_split = df.loc[:, split_id].apply(pd.Series)
        df_split = df_split.reindex(sorted(df_split.columns), axis=1)
        df_split = df_split.sort_index(axis=0)
        df_split["Client ID"] = df_split.index
        df_split = df_split.rename(columns={0: "normal", 1: "diseased"})

        df_split.plot(
            x="Client ID",
            kind="barh",
            stacked=True,
            cmap="tab20c",
            title=split_id,
            ax=axes[i],
            legend=(i == len(split_folders) - 1)
        )

        if i > 0:
            axes[i].set_yticks([])
            axes[i].set(ylabel=None)

        axes[i].set_title(split_id)

    fig.tight_layout()

    if save_plot:
        save_path = os.path.join(client_root, 'split_distribution.png')
        fig.savefig(save_path)
        print(f'Saved distribution plot to: {save_path}')

    plt.show()

    return split_summary


if __name__ == '__main__':
    dataset = 'Retina'
    data_path = os.path.join(os.getcwd(), dataset)
    split_data(data_path=data_path, n_clients=5, n_classes=2)
    view_split(data_path=data_path, save_plot=True)
