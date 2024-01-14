import torch
import numpy as np
import random
import pandas as pd
import scipy.io
from abc import ABC, abstractmethod
from scipy.io import arff
from sklearn.model_selection import train_test_split
import os
import sys
from folktables import *
import shutil
import zipfile


# ==================================================================================================
# General utilities
def create_contaminated_train_test_splits(
    X, y, contamination_factor, test_size=0.2, random_state=163, verbose=False
):
    """
    Create train and test splits from the dataset with a specified contamination factor in the train set.

    Parameters:
    X (np.array): Feature data.
    y (np.array): Labels (0 = normal, 1 = anomaly).
    contamination_factor (float): Desired ratio of anomalies in the train set.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random state for reproducibility.
    verbose (bool): If True, prints detailed information about the dataset and splits.

    Returns:
    X_train (np.array): Feature data for the train set.
    y_train (np.array): Labels for the train set.
    X_test (np.array): Feature data for the test set.
    y_test (np.array): Labels for the test set.
    """

    # Divide X and y into normal and anomalous data
    normal_data = X[y == 0]
    normal_labels = y[y == 0]
    anomalous_data = X[y == 1]
    anomalous_labels = y[y == 1]

    # Create a train split using only normal data
    X_train_normal, X_test_normal, y_train_normal, y_test_normal = train_test_split(
        normal_data,
        normal_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=normal_labels,
    )

    # Calculate the number of anomalies to add to the train set to achieve the desired contamination ratio
    desired_anomalies_in_train = int(len(X_train_normal) * contamination_factor)
    actual_anomalies_to_add = min(desired_anomalies_in_train, len(anomalous_data))

    # Add anomalies to the train set
    X_train_anomalies = anomalous_data[:actual_anomalies_to_add]
    y_train_anomalies = anomalous_labels[:actual_anomalies_to_add]

    # Remaining anomalies go to the test set
    X_test_anomalies = anomalous_data[actual_anomalies_to_add:]
    y_test_anomalies = anomalous_labels[actual_anomalies_to_add:]

    # Combine normal and anomalous data for train and test sets
    X_train = np.concatenate([X_train_normal, X_train_anomalies])
    y_train = np.concatenate([y_train_normal, y_train_anomalies])
    X_test = np.concatenate([X_test_normal, X_test_anomalies])
    y_test = np.concatenate([y_test_normal, y_test_anomalies])

    # Calculate actual contamination ratios
    train_contamination = np.mean(y_train)
    test_contamination = np.mean(y_test)

    # Print the summary of the splits
    total_anomalies = len(anomalous_labels)
    anomalies_in_train_percentage = (len(y_train_anomalies) / total_anomalies) * 100
    anomalies_in_test_percentage = (len(y_test_anomalies) / total_anomalies) * 100

    # Verbose output
    if verbose:
        print("\n=====================")
        print("Split creation verbose:")
        print(f"Training dataset size: {len(X_train)}")
        print(f"Training dataset contamination: {train_contamination:.2}")
        print(f"Number of anomalies in training dataset: {len(y_train_anomalies)}")
        print(
            f"Percentage of total anomalies in training dataset: {anomalies_in_train_percentage:.2f}%"
        )
        print(f"Test dataset size: {len(X_test)}")
        print(f"Test dataset contamination: {test_contamination:.2}")
        print(f"Number of anomalies in test dataset: {len(y_test_anomalies)}")
        print(
            f"Percentage of total anomalies in test dataset: {anomalies_in_test_percentage:.2f}%"
        )
        print("=====================\n")

    return X_train, y_train, X_test, y_test


# ===================================================================================================
# ODDS datasets
class ODDS_dataLoader:
    def __init__(self):
        pass

    def get_dataset(self, dataset_name, contamination_factor):
        script_dir = os.path.dirname(__file__)
        rel_path = os.path.join("data/ODDS", dataset_name)
        abs_file_path = os.path.join(script_dir, rel_path)
        mat_files = [
            "annthyroid",
            "arrhythmia",
            "breastw",
            "cardio",
            "forest_cover",
            "glass",
            "ionosphere",
            "letter",
            "lympho",
            "mammography",
            "mnist",
            "musk",
            "optdigits",
            "pendigits",
            "pima",
            "satellite",
            "satimage",
            "shuttle",
            "speech",
            "thyroid",
            "vertebral",
            "vowels",
            "wbc",
            "wine",
        ]
        ##########################
        script_dir = os.path.dirname(__file__)
        ##########################
        if dataset_name in mat_files:
            print("generic mat file")
            return self.build_train_test_generic_matfile(
                abs_file_path, contamination_factor
            )

        if dataset_name == "seismic":
            print("seismic")
            return self.build_train_test_seismic(
                abs_file_path + ".arff", contamination_factor
            )

        if dataset_name == "mulcross":
            print("mullcross")
            return self.build_train_test_mulcross(
                abs_file_path + ".arff", contamination_factor
            )

        if dataset_name == "abalone":
            print("abalone")
            return self.build_train_test_abalone(
                abs_file_path + ".data", contamination_factor
            )

        if dataset_name == "ecoli":
            print("ecoli")
            return self.build_train_test_ecoli(
                abs_file_path + ".data", contamination_factor
            )

        if dataset_name == "kdd":
            print("kdd")
            return self.build_train_test_kdd(
                script_dir + "/Data/kddcup.data_10_percent_corrected.zip",
                contamination_factor,
            )

        if dataset_name == "kddrev":
            print("kddrev")
            return self.build_train_test_kdd_rev(
                script_dir + "/Data/kddcup.data_10_percent_corrected.zip",
                contamination_factor,
            )

        if dataset_name == "mulcross_mini":
            print("mullcross_mini")
            return self.build_train_test_mulcross_mini(
                abs_file_path + ".csv", contamination_factor
            )

        if dataset_name == "mulcross_micro":
            print("mullcross_micro")
            return self.build_train_test_mulcross_micro(
                abs_file_path + ".csv", contamination_factor
            )

        if dataset_name == "bank":
            print("bank from ADRepository")
            return self.build_train_test_bank(
                abs_file_path + ".csv", contamination_factor
            )
        sys.exit("No such dataset!")

    def build_train_test_generic_matfile(
        self, name_of_file, contamination_factor
    ):  # takes a matrice that contains the entire dataset, and creates a trainset with 50% of the data of all normals, and the rest are test
        print("Name of file: {}".format(name_of_file))
        dataset = scipy.io.loadmat(name_of_file)  # Load .mat file
        X = dataset["X"]  # Loading the records (X) in a numpy ndarray
        classes = dataset["y"]  # Loading the Labels (Classes) in a numpy ndarray
        jointXY = torch.cat(
            (
                torch.tensor(X, dtype=torch.double),
                torch.tensor(classes, dtype=torch.double),
            ),
            dim=1,
        )  # Concatinating over the columns of X and Y into a pytorch tensor, same number of rows, columns contatinated together
        #####################################
        # print("Dataset shape = {}, chosen Contamination Factor {}%.".format(X.shape, contamination_factor))
        print(
            "Dataset consists of {} entries/rows, and {} features/columns, chosen Contamination Factor {}%.".format(
                X.shape[0], (X.shape[1] + 1), contamination_factor
            )
        )
        #####################################
        normals = jointXY[
            jointXY[:, -1] == 0
        ]  # Separating the normal records, by the last columb
        anomalies = jointXY[jointXY[:, -1] == 1]  # Separating the anomalies records
        #####################################
        no_all_anomalies = anomalies.shape[0]
        all_anomalies_per = int((no_all_anomalies / jointXY.shape[0]) * 100)
        print(
            "Dataset contains {} anomalies, {}% of the dataset.".format(
                no_all_anomalies, all_anomalies_per
            )
        )
        no_train_anomalies = int((no_all_anomalies / 100) * contamination_factor)
        no_test_anomalies = no_all_anomalies - no_train_anomalies
        #####################################
        normals = normals[
            torch.randperm(normals.shape[0])
        ]  # Shuffeling the normal records together
        train, test_norm = torch.split(
            normals, int(normals.shape[0] / 2) + 1
        )  # Splitting the normal records in 1/2, to train and test
        #####################################
        anomalies_df = pd.DataFrame(anomalies)

        train_anomalies = anomalies_df.sample(frac=(contamination_factor / 100))
        test_anomalies = anomalies_df.drop(train_anomalies.index)

        train_anomalies_tensor = torch.tensor((train_anomalies.to_numpy()))
        test_anomalies_tensor = torch.tensor((test_anomalies.to_numpy()))

        if len(train_anomalies_tensor) > 0:
            train = torch.cat((train, train_anomalies_tensor))
            test = torch.cat((test_norm, test_anomalies_tensor))
        else:
            test = torch.cat((test_norm, anomalies))
        #####################################
        # test = torch.cat((test_norm, anomalies)) #Creating the test set by concatinating over the rows, the anomalies and 1/2 of the normal data
        test = test[torch.randperm(test.shape[0])]  # Shuffling the test data
        train = train[torch.randperm(train.shape[0])]  # Re-shuffling the train data
        test_classes = test[:, -1].view(
            -1, 1
        )  # Loading the test labels into a 1 columb tensor
        train = train[
            :, 0 : train.shape[1] - 1
        ]  # reshape the train tensor to remove the labels (last column)
        test = test[
            :, 0 : test.shape[1] - 1
        ]  # reshape the test tensor to remove the labels (last column)
        #####################################
        print(
            "Train dataset shape = {}, contains {} anomalies, contamination factor {}%.".format(
                train.size(),
                no_train_anomalies,
                int((no_train_anomalies / train.size()[0]) * 100),
            )
        )
        print(
            "Test dataset shape = {}, contains {} anomalies, contamination factor {}%.".format(
                test.size(),
                no_test_anomalies,
                int((no_test_anomalies / test.size()[0]) * 100),
            )
        )
        #####################################
        return (
            train,
            test,
            test_classes,
        )  # return train and test tensors and the test labels tensor

    def build_train_test_seismic(
        self, name_of_file, contamination_factor
    ):  # takes a matrice that contains the entire dataset, and creates a trainset with 50% of the data of all normals, and the rest are test
        print("Name of file: {}".format(name_of_file))  ###
        dataset, meta = arff.loadarff(name_of_file)
        dataset = pd.DataFrame(dataset)
        #####################################
        print(
            "Dataset consists of {} entries/rows, and {} features/columns, chosen Contamination Factor {}%.".format(
                dataset.shape[0], dataset.shape[1], contamination_factor
            )
        )
        #####################################
        classes = dataset.iloc[:, -1]
        dataset = dataset.iloc[:, :-1]
        dataset = pd.get_dummies(dataset.iloc[:, :-1])
        dataset = pd.concat((dataset, classes), axis=1)
        normals = dataset[dataset.iloc[:, -1] == b"0"].values
        anomalies = dataset[dataset.iloc[:, -1] == b"1"].values
        ######################################
        no_all_anomalies = anomalies.shape[0]
        all_anomalies_per = round(((no_all_anomalies / dataset.shape[0]) * 100), 2)
        print(
            "Dataset contains {} anomalies, {}% of the dataset.".format(
                no_all_anomalies, all_anomalies_per
            )
        )
        ######################################
        normals = normals[torch.randperm(normals.shape[0])]
        normals = torch.tensor(normals[:, :-1].astype("float32"))
        anomalies = torch.tensor(anomalies[:, :-1].astype("float32"))
        normals = torch.cat((normals, torch.zeros(normals.shape[0]).view(-1, 1)), dim=1)
        anomalies = torch.cat(
            (anomalies, torch.ones(anomalies.shape[0]).view(-1, 1)), dim=1
        )

        no_train_anomalies = int((no_all_anomalies / 100) * contamination_factor)
        no_test_anomalies = no_all_anomalies - no_train_anomalies

        train, test_norm = torch.split(normals, int(normals.shape[0] / 2) + 1)

        anomalies_df = pd.DataFrame(anomalies)

        train_anomalies = anomalies_df.sample(frac=(contamination_factor / 100))
        test_anomalies = anomalies_df.drop(train_anomalies.index)

        train_anomalies_tensor = torch.tensor((train_anomalies.to_numpy()))
        test_anomalies_tensor = torch.tensor((test_anomalies.to_numpy()))

        if len(train_anomalies_tensor) > 0:
            train = torch.cat((train, train_anomalies_tensor))
            test = torch.cat((test_norm, test_anomalies_tensor))
        else:
            test = torch.cat((test_norm, anomalies))
        ###

        train, test_norm = torch.split(normals, int(normals.shape[0] / 2) + 1)
        test = torch.cat((test_norm, anomalies))
        test = test[torch.randperm(test.shape[0])]
        train = train[torch.randperm(train.shape[0])]
        test_classes = test[:, -1].view(-1, 1)
        train = train[:, 0 : train.shape[1] - 1]
        test = test[:, 0 : test.shape[1] - 1]
        ##########################
        print(
            "Train dataset shape = {}, contains {} anomalies, contamination factor {}%.".format(
                train.size(),
                no_train_anomalies,
                int((no_train_anomalies / train.size()[0]) * 100),
            )
        )
        print(
            "Test dataset shape = {}, contains {} anomalies, contamination factor {}%.".format(
                test.size(),
                no_test_anomalies,
                int((no_test_anomalies / test.size()[0]) * 100),
            )
        )
        ##########################
        return (train, test, test_classes)

    def build_train_test_mulcross(self, name_of_file, contamination_factor):
        # takes a matrix that contains the entire dataset, and creates a train set
        #  with 50% of the data of all normals, and the rest are test
        print("Name of file: {}".format(name_of_file))
        dataset, _ = arff.loadarff(name_of_file)
        dataset = pd.DataFrame(dataset)
        #####################################
        print(
            "Dataset consists of {} entries/rows, and {} features/columns, chosen Contamination Factor {}%.".format(
                dataset.shape[0], dataset.shape[1], contamination_factor
            )
        )
        #####################################
        normals = dataset[dataset.iloc[:, -1] == b"Normal"].values
        anomalies = dataset[dataset.iloc[:, -1] == b"Anomaly"].values
        #####################################
        no_all_anomalies = anomalies.shape[0]
        all_anomalies_per = round(((no_all_anomalies / dataset.shape[0]) * 100), 2)
        print(
            "Dataset contains {} anomalies, {}% of the dataset.".format(
                no_all_anomalies, all_anomalies_per
            )
        )
        ######################################
        normals = normals[torch.randperm(normals.shape[0])]
        normals = torch.tensor(normals[:, :-1].astype("float32"))
        anomalies = torch.tensor(anomalies[:, :-1].astype("float32"))
        normals = torch.cat(
            (normals, torch.zeros(normals.shape[0]).view(-1, 1)), dim=1
        )  ### One hot coding of the labels, 0 = Normal
        anomalies = torch.cat(
            (anomalies, torch.ones(anomalies.shape[0]).view(-1, 1)), dim=1
        )  ###  One hot coding of the labels, 1 = Anomaly

        no_train_anomalies = int((no_all_anomalies / 100) * contamination_factor)
        no_test_anomalies = no_all_anomalies - no_train_anomalies

        train, test_norm = torch.split(normals, int(normals.shape[0] / 2) + 1)

        anomalies_df = pd.DataFrame(anomalies)

        train_anomalies = anomalies_df.sample(frac=(contamination_factor / 100))
        test_anomalies = anomalies_df.drop(train_anomalies.index)

        train_anomalies_tensor = torch.tensor((train_anomalies.to_numpy()))
        test_anomalies_tensor = torch.tensor((test_anomalies.to_numpy()))

        if len(train_anomalies_tensor) > 0:
            train = torch.cat((train, train_anomalies_tensor))
            test = torch.cat((test_norm, test_anomalies_tensor))
        else:
            test = torch.cat((test_norm, anomalies))
        ###
        test = test[torch.randperm(test.shape[0])]
        train = train[torch.randperm(train.shape[0])]
        test_classes = test[:, -1].view(-1, 1)
        train = train[:, 0 : train.shape[1] - 1]
        test = test[:, 0 : test.shape[1] - 1]
        ##########################
        print(
            "Train dataset shape = {}, contains {} anomalies, contamination factor {}%.".format(
                train.size(),
                no_train_anomalies,
                int((no_train_anomalies / train.size()[0]) * 100),
            )
        )
        print(
            "Test dataset shape = {}, contains {} anomalies, contamination factor {}%.".format(
                test.size(),
                no_test_anomalies,
                int((no_test_anomalies / test.size()[0]) * 100),
            )
        )
        #####################################
        return (train, test, test_classes)

    def build_train_test_mulcross_mini(
        self, name_of_file, contamination_factor
    ):  # takes a matrice that contains the entire dataset, and creates a trainset with 50% of the data of all normals, and the rest are test
        #####################################
        print("Name of file: {}".format(name_of_file))
        #####################################
        # dataset, _ = arff.loadarff(name_of_file)
        # dataset = pd.DataFrame(dataset)
        dataset = pd.read_csv(name_of_file)
        #####################################
        print(
            "Dataset consists of {} entries/rows, and {} features/columns, chosen Contamination Factor {}%.".format(
                dataset.shape[0], dataset.shape[1], contamination_factor
            )
        )
        #####################################
        normals = dataset[dataset.iloc[:, -1] == "b'Normal'"].values
        anomalies = dataset[dataset.iloc[:, -1] == "b'Anomaly'"].values
        #####################################
        no_all_anomalies = anomalies.shape[0]
        all_anomalies_per = int((no_all_anomalies / dataset.shape[0]) * 100)
        print(
            "Dataset contains {} anomalies, {}% of the dataset.".format(
                no_all_anomalies, all_anomalies_per
            )
        )
        #####################################
        normals = normals[torch.randperm(normals.shape[0])]
        normals = torch.tensor(normals[:, :-1].astype("float32"))
        anomalies = torch.tensor(anomalies[:, :-1].astype("float32"))
        normals = torch.cat(
            (normals, torch.zeros(normals.shape[0]).view(-1, 1)), dim=1
        )  ### One hot coding of the labels, 0 = Normal
        anomalies = torch.cat(
            (anomalies, torch.ones(anomalies.shape[0]).view(-1, 1)), dim=1
        )  ###  One hot coding of the labels, 1 = Anomaly
        #####################################
        no_train_anomalies = int((no_all_anomalies / 100) * contamination_factor)
        no_test_anomalies = no_all_anomalies - no_train_anomalies

        train, test_norm = torch.split(normals, int(normals.shape[0] / 2) + 1)

        anomalies_df = pd.DataFrame(anomalies)

        train_anomalies = anomalies_df.sample(frac=(contamination_factor / 100))
        test_anomalies = anomalies_df.drop(train_anomalies.index)

        train_anomalies_tensor = torch.tensor((train_anomalies.to_numpy()))
        test_anomalies_tensor = torch.tensor((test_anomalies.to_numpy()))

        if len(train_anomalies_tensor) > 0:
            train = torch.cat((train, train_anomalies_tensor))
            test = torch.cat((test_norm, test_anomalies_tensor))
        else:
            test = torch.cat((test_norm, anomalies))
        #####################################
        test = test[torch.randperm(test.shape[0])]
        train = train[torch.randperm(train.shape[0])]
        test_classes = test[:, -1].view(-1, 1)
        train = train[:, 0 : train.shape[1] - 1]
        test = test[:, 0 : test.shape[1] - 1]
        #####################################
        print(
            "Train dataset shape = {}, contains {} anomalies, contamination factor {}%.".format(
                train.size(),
                no_train_anomalies,
                int((no_train_anomalies / train.size()[0]) * 100),
            )
        )
        print(
            "Test dataset shape = {}, contains {} anomalies, contamination factor {}%.".format(
                test.size(),
                no_test_anomalies,
                int((no_test_anomalies / test.size()[0]) * 100),
            )
        )
        #####################################
        return (train, test, test_classes)

    def build_train_test_mulcross_micro(
        self, name_of_file, contamination_factor
    ):  # takes a matrice that contains the entire dataset, and creates a trainset with 50% of the data of all normals, and the rest are test
        #####################################
        print("Name of file: {}".format(name_of_file))
        #####################################
        # dataset, _ = arff.loadarff(name_of_file)
        # dataset = pd.DataFrame(dataset)
        dataset = pd.read_csv(name_of_file)
        #####################################
        print(
            "Dataset shape = {}, chosen Contamination Factor {}%.".format(
                dataset.shape, contamination_factor
            )
        )
        #####################################
        normals = dataset[dataset.iloc[:, -1] == "b'Normal'"].values
        anomalies = dataset[dataset.iloc[:, -1] == "b'Anomaly'"].values
        #####################################
        no_all_anomalies = anomalies.shape[0]
        all_anomalies_per = int((no_all_anomalies / dataset.shape[0]) * 100)
        print(
            "Dataset contains {} anomalies, {}% of the dataset.".format(
                no_all_anomalies, all_anomalies_per
            )
        )
        #####################################
        normals = normals[torch.randperm(normals.shape[0])]
        normals = torch.tensor(normals[:, :-1].astype("float32"))
        anomalies = torch.tensor(anomalies[:, :-1].astype("float32"))
        normals = torch.cat(
            (normals, torch.zeros(normals.shape[0]).view(-1, 1)), dim=1
        )  ### One hot coding of the labels, 0 = Normal
        anomalies = torch.cat(
            (anomalies, torch.ones(anomalies.shape[0]).view(-1, 1)), dim=1
        )  ###  One hot coding of the labels, 1 = Anomaly
        #####################################
        no_train_anomalies = int((no_all_anomalies / 100) * contamination_factor)
        no_test_anomalies = no_all_anomalies - no_train_anomalies

        train, test_norm = torch.split(normals, int(normals.shape[0] / 2) + 1)

        anomalies_df = pd.DataFrame(anomalies)

        train_anomalies = anomalies_df.sample(frac=(contamination_factor / 100))
        test_anomalies = anomalies_df.drop(train_anomalies.index)

        train_anomalies_tensor = torch.tensor((train_anomalies.to_numpy()))
        test_anomalies_tensor = torch.tensor((test_anomalies.to_numpy()))

        if len(train_anomalies_tensor) > 0:
            train = torch.cat((train, train_anomalies_tensor))
            test = torch.cat((test_norm, test_anomalies_tensor))
        else:
            test = torch.cat((test_norm, anomalies))
        #####################################
        test = test[torch.randperm(test.shape[0])]
        train = train[torch.randperm(train.shape[0])]
        test_classes = test[:, -1].view(-1, 1)
        train = train[:, 0 : train.shape[1] - 1]
        test = test[:, 0 : test.shape[1] - 1]
        #####################################
        print(
            "Train dataset shape = {}, contains {} anomalies, contamination factor {}%.".format(
                train.size(),
                no_train_anomalies,
                int((no_train_anomalies / train.size()[0]) * 100),
            )
        )
        print(
            "Test dataset shape = {}, contains {} anomalies, contamination factor {}%.".format(
                test.size(),
                no_test_anomalies,
                int((no_test_anomalies / test.size()[0]) * 100),
            )
        )
        #####################################
        return (train, test, test_classes)
        # return (train, test, test_classes, train_classes)

    def build_train_test_ecoli(
        self, name_of_file, contamination_factor
    ):  # takes a matrice that contains the entire dataset, and creates a trainset with 50% of the data of all normals, and the rest are test
        #####################################
        print("Name of file: {}".format(name_of_file))
        #####################################
        dataset = pd.read_csv(name_of_file, header=None, sep="\s+")
        #####################################
        print(
            "Dataset shape = {}, chosen Contamination Factor {}%.".format(
                dataset.shape, contamination_factor
            )
        )
        #####################################
        dataset = dataset.iloc[:, 1:]
        anomalies = np.array(
            dataset[
                (dataset.iloc[:, 7] == "omL")
                | (dataset.iloc[:, 7] == "imL")
                | (dataset.iloc[:, 7] == "imS")
            ]
        )[:, :-1]
        normals = np.array(
            dataset[
                (dataset.iloc[:, 7] == "cp")
                | (dataset.iloc[:, 7] == "im")
                | (dataset.iloc[:, 7] == "pp")
                | (dataset.iloc[:, 7] == "imU")
                | (dataset.iloc[:, 7] == "om")
            ]
        )[:, :-1]
        normals = torch.tensor(normals.astype("double"))
        anomalies = torch.tensor(anomalies.astype("double"))
        normals = torch.cat(
            (normals, torch.zeros(normals.shape[0], 1, dtype=torch.double)), dim=1
        )
        anomalies = torch.cat(
            (anomalies, torch.ones(anomalies.shape[0], 1, dtype=torch.double)), dim=1
        )
        normals = normals[torch.randperm(normals.shape[0])]
        anomalies = anomalies[torch.randperm(anomalies.shape[0])]
        #####################################
        no_all_anomalies = anomalies.shape[0]
        all_anomalies_per = int((no_all_anomalies / dataset.shape[0]) * 100)

        print(
            "Dataset contains {} anomalies, {}% of the dataset.".format(
                no_all_anomalies, all_anomalies_per
            )
        )
        no_train_anomalies = int((no_all_anomalies / 100) * contamination_factor)
        no_test_anomalies = no_all_anomalies - no_train_anomalies

        train, test_norm = torch.split(normals, int(normals.shape[0] / 2) + 1)

        anomalies_df = pd.DataFrame(anomalies)

        train_anomalies = anomalies_df.sample(frac=(contamination_factor / 100))
        test_anomalies = anomalies_df.drop(train_anomalies.index)

        train_anomalies_tensor = torch.tensor((train_anomalies.to_numpy()))
        test_anomalies_tensor = torch.tensor((test_anomalies.to_numpy()))

        if len(train_anomalies_tensor) > 0:
            train = torch.cat((train, train_anomalies_tensor))
            test = torch.cat((test_norm, test_anomalies_tensor))
        else:
            test = torch.cat((test_norm, anomalies))
        #####################################
        test = torch.cat((test_norm, anomalies))
        test = test[torch.randperm(test.shape[0])]
        test_classes = test[:, -1].view(-1, 1)
        train = train[:, :-1]
        test = test[:, :-1]
        #####################################
        print(
            "Train dataset shape = {}, contains {} anomalies, contamination factor {}%.".format(
                train.size(),
                no_train_anomalies,
                int((no_train_anomalies / train.size()[0]) * 100),
            )
        )
        print(
            "Test dataset shape = {}, contains {} anomalies, contamination factor {}%.".format(
                test.size(),
                no_test_anomalies,
                int((no_test_anomalies / test.size()[0]) * 100),
            )
        )
        #####################################
        return (train, test, test_classes)

    def build_train_test_abalone(
        self, path, contamination_factor
    ):  # takes a matrice that contains the entire dataset, and creates a trainset with 50% of the data of all normals, and the rest are test
        #####################################
        print("Name of file: {}".format("abalone"))
        #####################################
        data = pd.read_csv(path, header=None, sep=",")
        #####################################
        print(
            "Dataset shape = {}, chosen Contamination Factor {}%.".format(
                data.shape, contamination_factor
            )
        )
        #####################################
        data = data.rename(columns={8: "y"})
        data["y"].replace([8, 9, 10], -1, inplace=True)
        data["y"].replace([3, 21], 0, inplace=True)
        data.iloc[:, 0].replace("M", 0, inplace=True)
        data.iloc[:, 0].replace("F", 1, inplace=True)
        data.iloc[:, 0].replace("I", 2, inplace=True)
        test = data[data["y"] == 0]  ### Anomalies
        normal = data[data["y"] == -1].sample(frac=1)  ### Normals
        #####################################
        no_all_anomalies = test.shape[0]
        all_anomalies_per = int((no_all_anomalies / data.shape[0]) * 100)

        print(
            "Dataset contains {} anomalies, {}% of the dataset.".format(
                no_all_anomalies, all_anomalies_per
            )
        )
        no_train_anomalies = int((no_all_anomalies / 100) * contamination_factor)
        no_test_anomalies = no_all_anomalies - no_train_anomalies
        #
        num_normal_samples_test = normal.shape[0] // 2
        # train, test_norm = torch.split(normal, int(normal.shape[0] / 2) + 1)
        #
        anomalies_df = pd.DataFrame(test)
        #
        train_anomalies = anomalies_df.sample(frac=(contamination_factor / 100))
        test_anomalies = anomalies_df.drop(train_anomalies.index)
        #
        # train_anomalies_tensor = torch.tensor((train_anomalies.to_numpy()))
        # test_anomalies_tensor = torch.tensor((test_anomalies.to_numpy()))
        #
        if len(train_anomalies) > 0:
            # train = torch.cat((train, train_anomalies))
            # test = torch.cat((test_norm, test_anomalies))
            test_data = np.concatenate(
                (
                    test_anomalies.drop("y", axis=1),
                    normal[:num_normal_samples_test].drop("y", axis=1),
                ),
                axis=0,
            )
            # train = np.concatenate((train_anomalies, normal[num_normal_samples_test:]), axis=0)
            train = pd.concat(
                [train_anomalies, normal[num_normal_samples_test:]], axis=0
            )
        else:
            test_data = np.concatenate(
                (
                    test.drop("y", axis=1),
                    normal[:num_normal_samples_test].drop("y", axis=1),
                ),
                axis=0,
            )
            train = normal[num_normal_samples_test:]
        #####################################
        # num_normal_samples_test = normal.shape[0] // 2
        # test_data = np.concatenate((test.drop('y', axis=1), normal[:num_normal_samples_test].drop('y', axis=1)), axis=0)
        # train = normal[num_normal_samples_test:]
        train_data = train.drop("y", axis=1).values
        test_labels = np.concatenate(
            (test["y"], normal[:num_normal_samples_test]["y"].replace(-1, 1)), axis=0
        )
        for i in range(test_labels.shape[0]):
            if test_labels[i] == 0:
                test_labels[i] = 1
            else:
                test_labels[i] = 0
        train_data = torch.tensor(train_data.astype("double"))
        test_data = torch.tensor(test_data.astype("double"))
        test_labels = torch.tensor(test_labels.astype("double"))
        #####################################
        print(
            "Train dataset shape = {}, contains {} anomalies, contamination factor {}%.".format(
                train_data.size(),
                no_train_anomalies,
                int((no_train_anomalies / train_data.size()[0]) * 100),
            )
        )
        print(
            "Test dataset shape = {}, contains {} anomalies, contamination factor {}%.".format(
                test_data.size(),
                no_test_anomalies,
                int((no_test_anomalies / test_data.size()[0]) * 100),
            )
        )
        #####################################
        return (train_data, test_data, test_labels)

    def build_train_test_kdd(
        self, name_of_file, contamination_factor
    ):  # takes a matrice that contains the entire dataset, and creates a trainset with 50% of the data of all normals, and the rest are test
        zf = zipfile.ZipFile(name_of_file)
        #####################################
        print("Name of file: {}".format(name_of_file))
        #####################################
        kdd_loader = pd.read_csv(
            zf.open("kddcup.data_10_percent_corrected"), delimiter=","
        )
        entire_set = np.array(kdd_loader)
        revised_pd = pd.DataFrame(entire_set)
        revised_pd = pd.concat(
            (revised_pd, pd.get_dummies(revised_pd.iloc[:, 1], prefix="new1")), axis=1
        )
        revised_pd = pd.concat(
            (revised_pd, pd.get_dummies(revised_pd.iloc[:, 2], prefix="new2")), axis=1
        )
        revised_pd = pd.concat(
            (revised_pd, pd.get_dummies(revised_pd.iloc[:, 3], prefix="new3")), axis=1
        )
        revised_pd = pd.concat(
            (revised_pd, pd.get_dummies(revised_pd.iloc[:, 6], prefix="new6")), axis=1
        )
        revised_pd = pd.concat(
            (revised_pd, pd.get_dummies(revised_pd.iloc[:, 11], prefix="new11")), axis=1
        )
        revised_pd = pd.concat(
            (revised_pd, pd.get_dummies(revised_pd.iloc[:, 21], prefix="new21")), axis=1
        )
        revised_pd.drop(
            revised_pd.columns[[1, 2, 3, 6, 11, 20, 21]], inplace=True, axis=1
        )
        new_columns = [
            0,
            "new1_icmp",
            "new1_tcp",
            "new1_udp",
            "new2_IRC",
            "new2_X11",
            "new2_Z39_50",
            "new2_auth",
            "new2_bgp",
            "new2_courier",
            "new2_csnet_ns",
            "new2_ctf",
            "new2_daytime",
            "new2_discard",
            "new2_domain",
            "new2_domain_u",
            "new2_echo",
            "new2_eco_i",
            "new2_ecr_i",
            "new2_efs",
            "new2_exec",
            "new2_finger",
            "new2_ftp",
            "new2_ftp_data",
            "new2_gopher",
            "new2_hostnames",
            "new2_http",
            "new2_http_443",
            "new2_imap4",
            "new2_iso_tsap",
            "new2_klogin",
            "new2_kshell",
            "new2_ldap",
            "new2_link",
            "new2_login",
            "new2_mtp",
            "new2_name",
            "new2_netbios_dgm",
            "new2_netbios_ns",
            "new2_netbios_ssn",
            "new2_netstat",
            "new2_nnsp",
            "new2_nntp",
            "new2_ntp_u",
            "new2_other",
            "new2_pm_dump",
            "new2_pop_2",
            "new2_pop_3",
            "new2_printer",
            "new2_private",
            "new2_red_i",
            "new2_remote_job",
            "new2_rje",
            "new2_shell",
            "new2_smtp",
            "new2_sql_net",
            "new2_ssh",
            "new2_sunrpc",
            "new2_supdup",
            "new2_systat",
            "new2_telnet",
            "new2_tftp_u",
            "new2_tim_i",
            "new2_time",
            "new2_urh_i",
            "new2_urp_i",
            "new2_uucp",
            "new2_uucp_path",
            "new2_vmnet",
            "new2_whois",
            "new3_OTH",
            "new3_REJ",
            "new3_RSTO",
            "new3_RSTOS0",
            "new3_RSTR",
            "new3_S0",
            "new3_S1",
            "new3_S2",
            "new3_S3",
            "new3_SF",
            "new3_SH",
            4,
            5,
            "new6_0",
            "new6_1",
            7,
            8,
            9,
            10,
            "new11_0",
            "new11_1",
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            "new21_0",
            "new21_1",
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
        ]
        revised_pd = revised_pd.reindex(columns=new_columns)
        ######################################
        print(
            "Dataset shape = {}, chosen Contamination Factor {}%.".format(
                revised_pd.shape, contamination_factor
            )
        )
        ######################################
        revised_pd.loc[revised_pd[41] != "normal.", 41] = 0.0
        revised_pd.loc[revised_pd[41] == "normal.", 41] = 1.0
        kdd_normal = np.array(revised_pd.loc[revised_pd[41] == 0.0], dtype=np.double)
        kdd_anomaly = np.array(revised_pd.loc[revised_pd[41] == 1.0], dtype=np.double)
        ######################################
        no_all_anomalies = kdd_anomaly.shape[0]
        all_anomalies_per = round(((no_all_anomalies / revised_pd.shape[0]) * 100), 2)
        print(
            "Dataset contains {} anomalies, {}% of the dataset.".format(
                no_all_anomalies, all_anomalies_per
            )
        )
        ######################################
        kdd_normal = torch.tensor(kdd_normal)
        kdd_anomaly = torch.tensor(kdd_anomaly)
        kdd_normal = kdd_normal[torch.randperm(kdd_normal.shape[0])]
        kdd_anomaly = kdd_anomaly[torch.randperm(kdd_anomaly.shape[0])]
        ######################################
        no_train_anomalies = int((no_all_anomalies / 100) * contamination_factor)
        no_test_anomalies = no_all_anomalies - no_train_anomalies

        train, test_norm = torch.split(kdd_normal, int(kdd_normal.shape[0] / 2) + 1)

        anomalies_df = pd.DataFrame(kdd_anomaly)

        train_anomalies = anomalies_df.sample(frac=(contamination_factor / 100))
        test_anomalies = anomalies_df.drop(train_anomalies.index)

        train_anomalies_tensor = torch.tensor((train_anomalies.to_numpy()))
        test_anomalies_tensor = torch.tensor((test_anomalies.to_numpy()))

        if len(train_anomalies_tensor) > 0:
            train = torch.cat((train, train_anomalies_tensor))
            test = torch.cat((test_norm, test_anomalies_tensor))
        else:
            test = torch.cat((test_norm, kdd_anomaly))
        ######################################
        train, test_norm = torch.split(kdd_normal, int(kdd_normal.shape[0] / 2) + 1)
        test = torch.cat((test_norm, kdd_anomaly))
        test = test[torch.randperm(test.shape[0])]
        test_classes = test[:, -1].view(-1, 1)
        train = train[:, 0 : train.shape[1] - 1]
        test = test[:, 0 : test.shape[1] - 1]
        ######################################
        print(
            "Train dataset shape = {}, contains {} anomalies, contamination factor {}%.".format(
                train.size(),
                no_train_anomalies,
                int((no_train_anomalies / train.size()[0]) * 100),
            )
        )
        print(
            "Test dataset shape = {}, contains {} anomalies, contamination factor {}%.".format(
                test.size(),
                no_test_anomalies,
                int((no_test_anomalies / test.size()[0]) * 100),
            )
        )
        ######################################
        return (train, test, test_classes)

    def build_train_test_kdd_rev(
        self, name_of_file, contamination_factor
    ):  # takes a matrice that contains the entire dataset, and creates a trainset with 50% of the data of all normals, and the rest are test
        zf = zipfile.ZipFile(name_of_file)
        #####################################
        print("Name of file: {}".format(name_of_file))
        #####################################
        kdd_loader = pd.read_csv(
            zf.open("kddcup.data_10_percent_corrected"), delimiter=","
        )
        entire_set = np.array(kdd_loader)
        revised_pd = pd.DataFrame(entire_set)
        revised_pd = pd.concat(
            (revised_pd, pd.get_dummies(revised_pd.iloc[:, 1], prefix="new1")), axis=1
        )
        revised_pd = pd.concat(
            (revised_pd, pd.get_dummies(revised_pd.iloc[:, 2], prefix="new2")), axis=1
        )
        revised_pd = pd.concat(
            (revised_pd, pd.get_dummies(revised_pd.iloc[:, 3], prefix="new3")), axis=1
        )
        revised_pd = pd.concat(
            (revised_pd, pd.get_dummies(revised_pd.iloc[:, 6], prefix="new6")), axis=1
        )
        revised_pd = pd.concat(
            (revised_pd, pd.get_dummies(revised_pd.iloc[:, 11], prefix="new11")), axis=1
        )
        revised_pd = pd.concat(
            (revised_pd, pd.get_dummies(revised_pd.iloc[:, 21], prefix="new21")), axis=1
        )
        revised_pd.drop(
            revised_pd.columns[[1, 2, 3, 6, 11, 20, 21]], inplace=True, axis=1
        )
        new_columns = [
            0,
            "new1_icmp",
            "new1_tcp",
            "new1_udp",
            "new2_IRC",
            "new2_X11",
            "new2_Z39_50",
            "new2_auth",
            "new2_bgp",
            "new2_courier",
            "new2_csnet_ns",
            "new2_ctf",
            "new2_daytime",
            "new2_discard",
            "new2_domain",
            "new2_domain_u",
            "new2_echo",
            "new2_eco_i",
            "new2_ecr_i",
            "new2_efs",
            "new2_exec",
            "new2_finger",
            "new2_ftp",
            "new2_ftp_data",
            "new2_gopher",
            "new2_hostnames",
            "new2_http",
            "new2_http_443",
            "new2_imap4",
            "new2_iso_tsap",
            "new2_klogin",
            "new2_kshell",
            "new2_ldap",
            "new2_link",
            "new2_login",
            "new2_mtp",
            "new2_name",
            "new2_netbios_dgm",
            "new2_netbios_ns",
            "new2_netbios_ssn",
            "new2_netstat",
            "new2_nnsp",
            "new2_nntp",
            "new2_ntp_u",
            "new2_other",
            "new2_pm_dump",
            "new2_pop_2",
            "new2_pop_3",
            "new2_printer",
            "new2_private",
            "new2_red_i",
            "new2_remote_job",
            "new2_rje",
            "new2_shell",
            "new2_smtp",
            "new2_sql_net",
            "new2_ssh",
            "new2_sunrpc",
            "new2_supdup",
            "new2_systat",
            "new2_telnet",
            "new2_tftp_u",
            "new2_tim_i",
            "new2_time",
            "new2_urh_i",
            "new2_urp_i",
            "new2_uucp",
            "new2_uucp_path",
            "new2_vmnet",
            "new2_whois",
            "new3_OTH",
            "new3_REJ",
            "new3_RSTO",
            "new3_RSTOS0",
            "new3_RSTR",
            "new3_S0",
            "new3_S1",
            "new3_S2",
            "new3_S3",
            "new3_SF",
            "new3_SH",
            4,
            5,
            "new6_0",
            "new6_1",
            7,
            8,
            9,
            10,
            "new11_0",
            "new11_1",
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            "new21_0",
            "new21_1",
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
        ]
        revised_pd = revised_pd.reindex(columns=new_columns)
        ######################################
        print(
            "Dataset shape = {}, chosen Contamination Factor {}%.".format(
                revised_pd.shape, contamination_factor
            )
        )
        ######################################
        revised_pd.loc[revised_pd[41] != "normal.", 41] = 1.0
        revised_pd.loc[revised_pd[41] == "normal.", 41] = 0.0
        kdd_anomaly = np.array(revised_pd.loc[revised_pd[41] == 1.0], dtype=np.double)
        kdd_normal = np.array(revised_pd.loc[revised_pd[41] == 0.0], dtype=np.double)
        ######################################
        no_all_anomalies = kdd_anomaly.shape[0]
        all_anomalies_per = round(((no_all_anomalies / revised_pd.shape[0]) * 100), 2)
        print(
            "Dataset contains {} anomalies, {}% of the dataset.".format(
                no_all_anomalies, all_anomalies_per
            )
        )
        ######################################
        kdd_normal = torch.tensor(kdd_normal)
        kdd_anomaly = torch.tensor(kdd_anomaly)
        kdd_anomaly = kdd_anomaly[
            random.sample(range(kdd_anomaly.shape[0]), int(kdd_normal.shape[0] / 4)), :
        ]
        kdd_normal = kdd_normal[torch.randperm(kdd_normal.shape[0])]
        ######################################
        no_train_anomalies = int((no_all_anomalies / 100) * contamination_factor)
        no_test_anomalies = no_all_anomalies - no_train_anomalies

        train, test_norm = torch.split(kdd_normal, int(kdd_normal.shape[0] / 2) + 1)

        anomalies_df = pd.DataFrame(kdd_anomaly)

        train_anomalies = anomalies_df.sample(frac=(contamination_factor / 100))
        test_anomalies = anomalies_df.drop(train_anomalies.index)

        train_anomalies_tensor = torch.tensor((train_anomalies.to_numpy()))
        test_anomalies_tensor = torch.tensor((test_anomalies.to_numpy()))

        if len(train_anomalies_tensor) > 0:
            train = torch.cat((train, train_anomalies_tensor))
            test = torch.cat((test_norm, test_anomalies_tensor))
        else:
            test = torch.cat((test_norm, kdd_anomaly))
        ######################################
        train, test_norm = torch.split(kdd_normal, int(kdd_normal.shape[0] / 2) + 1)
        test = torch.cat((test_norm, kdd_anomaly))
        test = test[torch.randperm(test.shape[0])]
        test_classes = test[:, -1].view(-1, 1)
        train = train[:, 0 : train.shape[1] - 1]
        test = test[:, 0 : test.shape[1] - 1]
        ######################################
        print(
            "Train dataset shape = {}, contains {} anomalies, contamination factor {}%.".format(
                train.size(),
                no_train_anomalies,
                int((no_train_anomalies / train.size()[0]) * 100),
            )
        )
        print(
            "Test dataset shape = {}, contains {} anomalies, contamination factor {}%.".format(
                test.size(),
                no_test_anomalies,
                int((no_test_anomalies / test.size()[0]) * 100),
            )
        )
        ######################################
        return (train, test, test_classes)

    def build_train_test_bank(self, name_of_file, contamination_factor):
        print("Name of file: {}".format(name_of_file))
        dataset = pd.read_csv(name_of_file)
        normals = dataset[dataset.iloc[:, -1] == 0].values
        anomalies = dataset[dataset.iloc[:, -1] == 1].values
        normals = normals[torch.randperm(normals.shape[0])]
        normals = torch.tensor(normals[:, :-1].astype("float32"))
        anomalies = torch.tensor(anomalies[:, :-1].astype("float32"))
        normals = torch.cat(
            (normals, torch.zeros(normals.shape[0]).view(-1, 1)), dim=1
        )  ### One hot coding of the labels, 0 = Normal
        anomalies = torch.cat(
            (anomalies, torch.ones(anomalies.shape[0]).view(-1, 1)), dim=1
        )  ###  One hot coding of the labels, 1 = Anomaly
        ##########################
        no_all_anomalies = anomalies.shape[0]
        all_anomalies_per = round(((no_all_anomalies / dataset.shape[0]) * 100), 2)
        print(
            "Dataset contains {} anomalies, {}% of the dataset.".format(
                no_all_anomalies, all_anomalies_per
            )
        )
        ##########################

        no_train_anomalies = int((no_all_anomalies / 100) * contamination_factor)
        no_test_anomalies = no_all_anomalies - no_train_anomalies

        print(
            "Dataset contains {} anomalies, {}% of the dataset.".format(
                no_all_anomalies, all_anomalies_per
            )
        )

        train, test_norm = torch.split(normals, int(normals.shape[0] / 2) + 1)

        anomalies_df = pd.DataFrame(anomalies)

        train_anomalies = anomalies_df.sample(frac=(contamination_factor / 100))
        test_anomalies = anomalies_df.drop(train_anomalies.index)

        train_anomalies_tensor = torch.tensor((train_anomalies.to_numpy()))
        test_anomalies_tensor = torch.tensor((test_anomalies.to_numpy()))

        if len(train_anomalies_tensor) > 0:
            train = torch.cat((train, train_anomalies_tensor))
            test = torch.cat((test_norm, test_anomalies_tensor))
        else:
            test = torch.cat((test_norm, anomalies))
        ###

        train, test_norm = torch.split(normals, int(normals.shape[0] / 2) + 1)
        test = torch.cat((test_norm, anomalies))
        test = test[torch.randperm(test.shape[0])]
        train = train[torch.randperm(train.shape[0])]
        test_classes = test[:, -1].view(-1, 1)
        train = train[:, 0 : train.shape[1] - 1]
        test = test[:, 0 : test.shape[1] - 1]
        ##########################
        print(
            "Train dataset shape = {}, contains {} anomalies, contamination factor {}%.".format(
                train.size(),
                no_train_anomalies,
                int((no_train_anomalies / train.size()[0]) * 100),
            )
        )
        print(
            "Test dataset shape = {}, contains {} anomalies, contamination factor {}%.".format(
                test.size(),
                no_test_anomalies,
                int((no_test_anomalies / test.size()[0]) * 100),
            )
        )
        ##########################
        return (train, test, test_classes)


# ===================================================================================================
# ADBench datasets
class ADBench_dataLoader:
    def __init__(self):
        self.dataset_list_classical = self.generate_dataset_list(
            "../data/adbench", "Classical"
        )
        self.dataset_list_cv = self.generate_dataset_list(
            "../data/adbench", "CV_by_ResNet18"
        )
        self.dataset_list_nlp = self.generate_dataset_list(
            "../data/adbench", "NLP_by_BERT"
        )

    def generate_dataset_list(self, base_path, sub_path):
        return [
            os.path.splitext(_)[0]
            for _ in os.listdir(os.path.join(base_path, sub_path))
            if os.path.splitext(_)[1] == ".npz"
        ]

    def print_dataset_list(self):
        print("Classical datasets:")
        for dataset in self.dataset_list_classical:
            print(dataset)

        print("\nCV datasets:")
        for dataset in self.dataset_list_cv:
            print(dataset)

        print("\nNLP datasets:")
        for dataset in self.dataset_list_nlp:
            print(dataset)

    def load_data(
        self, dataset_name, cv_embedding="ResNet18", nlp_embedding="BERT", verbose=False
    ):
        print(
            "For more information on the dataset, please visit: https://github.com/Minqi824/ADBench/tree/main/adbench/datasets"
        )

        # Loading data

        if dataset_name in self.dataset_list_classical:
            file_path = "../data/adbench/Classical/" + dataset_name + ".npz"
            data = np.load(file_path, allow_pickle=True)
            X, y = data["X"], data["y"]
        elif dataset_name in self.dataset_list_cv:
            if cv_embedding == "ResNet18":
                file_path = "../data/adbench/CV_by_ResNet18/" + dataset_name + ".npz"
                data = np.load(file_path, allow_pickle=True)
                X, y = data["X"], data["y"]
            elif cv_embedding == "ViT":
                file_path = "../data/adbench/CV_by_ViT/" + dataset_name + ".npz"
                data = np.load(file_path, allow_pickle=True)
                X, y = data["X"], data["y"]
        elif dataset_name in self.dataset_list_nlp:
            if nlp_embedding == "BERT":
                file_path = "../data/adbench/NLP_by_BERT/" + dataset_name + ".npz"
                data = np.load(file_path, allow_pickle=True)
                X, y = data["X"], data["y"]
            elif nlp_embedding == "RoBERTa":
                file_path = "../data/adbench/NLP_by_RoBERTa/" + dataset_name + ".npz"
                data = np.load(file_path, allow_pickle=True)
                X, y = data["X"], data["y"]
        else:
            print("Dataset not found")
            X, y = None, None

        if verbose:
            # Calculate maximum contamination factor assuming an 80/20 train/test split
            num_anomalies = np.sum(y)
            total_instances = len(y)
            max_contamination_factor = min(
                num_anomalies, int(0.8 * total_instances)
            ) / (0.8 * total_instances)

            # Print dataset summary
            print(f"Data loading verbose:")
            print(f"Dataset: {dataset_name}")
            print(f"Total instances: {total_instances}")
            print(f"Number of anomalies: {num_anomalies}")
            print(f"Feature dimension: {X.shape[1]}")
            print(
                f"Maximum contamination factor in train set (80/20 split): {max_contamination_factor:.2f}\n"
            )
        return X, y

    # Data loading and splitting
    def load_and_split_data(self, dataset_name, contamination_factor, verbose):
        # Loading and splitting data
        X, y = self.load_data(dataset_name, verbose=verbose)
        X_train, y_train, X_test, y_test = create_contaminated_train_test_splits(
            X, y, contamination_factor, verbose=verbose
        )

        return X_train, y_train, X_test, y_test


# ===================================================================================================
# ACS Folktables datasets

# Writing our own ACS tasks keeping all common features.
# This allows us to trace the intersections between subsets of instances per task.

# features common to all tasks
common_features = [
    "AGEP",
    "COW",
    "HINS2",
    "SCHL",
    "MAR",
    "OCCP",
    "POBP",
    "RELP",
    "WKHP",
    "SEX",
    "RAC1P",
    "DIS",
    "ESP",
    "CIT",
    "MIG",
    "MIL",
    "ANC",
    "NATIVITY",
    "DEAR",
    "DEYE",
    "DREM",
    "RACAIAN",
    "RACASN",
    "RACBLK",
    "RACNH",
    "RACPI",
    "RACSOR",
    "RACWHT",
    "PINCP",
    "ESR",
    "ST",
    "FER",
    "PUMA",
    "JWTR",
    "POWPUMA",
    "POVPIP",
    "GCL",
]


def adult_filter(data):
    """Mimic the filters in place for Adult data.

    Adult documentation notes: Extraction was done by Barry Becker from
    the 1994 Census database. A set of reasonably clean records was extracted
    using the following conditions:
    ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
    """
    df = data
    df = df[df["AGEP"] > 16]
    df = df[df["PINCP"] > 100]
    df = df[df["WKHP"] > 0]
    df = df[df["PWGTP"] >= 1]
    return df


ACSIncome_common = folktables.BasicProblem(
    features=common_features,
    target="PINCP",
    target_transform=lambda x: x > 50000,
    group="RAC1P",
    preprocess=adult_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

ACSEmployment_common = folktables.BasicProblem(
    features=common_features,
    target="ESR",
    target_transform=lambda x: x == 1,
    group="RAC1P",
    preprocess=lambda x: x,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

ACSHealthInsurance_common = folktables.BasicProblem(
    features=common_features,
    target="HINS2",
    target_transform=lambda x: x == 1,
    group="RAC1P",
    preprocess=lambda x: x,
    postprocess=lambda x: np.nan_to_num(x, -1),
)


def public_coverage_filter(data):
    """
    Filters for the public health insurance prediction task; focus on low income Americans, and those not eligible for Medicare
    """
    df = data
    df = df[df["AGEP"] < 65]
    df = df[df["PINCP"] <= 30000]
    return df


ACSPublicCoverage_common = folktables.BasicProblem(
    features=common_features,
    target="PUBCOV",
    target_transform=lambda x: x == 1,
    group="RAC1P",
    preprocess=public_coverage_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)


def travel_time_filter(data):
    """
    Filters for the employment prediction task
    """
    df = data
    df = df[df["AGEP"] > 16]
    df = df[df["PWGTP"] >= 1]
    df = df[df["ESR"] == 1]
    return df


ACSTravelTime_common = folktables.BasicProblem(
    features=common_features,
    target="JWMNP",
    target_transform=lambda x: x > 20,
    group="RAC1P",
    preprocess=travel_time_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

ACSMobility_common = folktables.BasicProblem(
    features=common_features,
    target="MIG",
    target_transform=lambda x: x == 1,
    group="RAC1P",
    preprocess=lambda x: x.drop(x.loc[(x["AGEP"] <= 18) | (x["AGEP"] >= 35)].index),
    postprocess=lambda x: np.nan_to_num(x, -1),
)


def employment_filter(data):
    """
    Filters for the employment prediction task
    """
    df = data
    df = df[df["AGEP"] > 16]
    df = df[df["AGEP"] < 90]
    df = df[df["PWGTP"] >= 1]
    return df


ACSEmploymentFiltered_common = folktables.BasicProblem(
    features=common_features,
    target="ESR",
    target_transform=lambda x: x == 1,
    group="RAC1P",
    preprocess=employment_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

ACSIncomePovertyRatio_common = folktables.BasicProblem(
    features=common_features,
    target="POVPIP",
    target_transform=lambda x: x < 250,
    group="RAC1P",
    preprocess=lambda x: x,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

ACSAll_common = folktables.BasicProblem(
    features=common_features,
    target="POVPIP",  # some target that we will ignore.
    # target_transform=lambda x: x < 250,
    group="RAC1P",
    preprocess=lambda x: x,
    postprocess=lambda x: np.nan_to_num(x, -1),
)


class ACS_DataLoader:
    def __init__(
        self,
        year=2018,
        survey="person",
        horizon="1-Year",
        states=["AL"],
        data_directory="../data/ACS/",
        download_to_current=False,
        verbose=False,
    ):
        self.year = year
        self.survey = survey
        self.horizon = horizon
        self.states = states
        self.verbose = verbose
        # Set the data directory
        self.data_directory = data_directory
        self.download_to_current = download_to_current
        self.data_source = ACSDataSource(
            survey_year=self.year,
            horizon=self.horizon,
            survey=self.survey,
        )
        self.dataset = self.load_acs_data()

    def load_acs_data(self):
        download_directory = "./data/"
        self.data_directory = "../data/ACS/"

        # Ensure the target directory exists
        os.makedirs(self.data_directory, exist_ok=True)

        # Load the ACS data using folktables with specified parameters
        data = self.data_source.get_data(
            states=self.states,
            download=True,
        )

        if not self.download_to_current:
            try:
                # Move the downloaded files from the download directory to the target directory
                for file in os.listdir(download_directory):
                    shutil.move(
                        os.path.join(download_directory, file), self.data_directory
                    )

                # Delete the contents of the download directory
                shutil.rmtree(download_directory)

            except Exception as e:
                print(
                    "Error moving downloaded files. If download doesn't work, add download_to_current = True and move the downloaded files manually."
                )
                print(f"Error message: {str(e)}")

        return data

    def generate_features_labels_df(self, task, common=False):
        # Generate features and labels based on the specified task
        if task == "employment":
            if common:
                features, labels, _ = ACSEmployment_common.df_to_numpy(self.dataset)
                task_features = ACSEmployment_common.features
            else:
                features, labels, _ = ACSEmployment.df_to_numpy(self.dataset)
                task_features = ACSEmployment.features
        elif task == "income":
            if common:
                features, labels, _ = ACSIncome_common.df_to_numpy(self.dataset)
                task_features = ACSIncome_common.features
            else:
                features, labels, _ = ACSIncome.df_to_numpy(self.dataset)
                task_features = ACSIncome.features
        elif task == "mobility":
            if common:
                features, labels, _ = ACSMobility_common.df_to_numpy(self.dataset)
                task_features = ACSMobility_common.features
            else:
                features, labels, _ = ACSMobility.df_to_numpy(self.dataset)
                task_features = ACSMobility.features
        elif task == "health_insurance":
            if common:
                features, labels, _ = ACSHealthInsurance_common.df_to_numpy(
                    self.dataset
                )
                task_features = ACSHealthInsurance_common.features
            else:
                features, labels, _ = ACSHealthInsurance.df_to_numpy(self.dataset)
                task_features = ACSHealthInsurance.features
        elif task == "travel_time":
            if common:
                features, labels, _ = ACSTravelTime_common.df_to_numpy(self.dataset)
                task_features = ACSTravelTime_common.features
            else:
                features, labels, _ = ACSTravelTime.df_to_numpy(self.dataset)
                task_features = ACSTravelTime.features
        elif task == "all":
            features, labels, _ = ACSAll_common.df_to_numpy(self.dataset)
            task_features = ACSAll_common.features
        else:
            print("Specified task not supported")
            return None

        df = pd.DataFrame(features, columns=task_features)

        return df, features, labels

    def get_all_task_dfs(self, verbose=False, common=True):
        tasks = [
            "all",
            "employment",
            "income",
            "mobility",
            "health_insurance",
            "travel_time",
        ]
        dfs = {}

        for task in tasks:
            df, features, task_labels = self.generate_features_labels_df(task, common)
            dfs[task] = {"df": df, "features": features, "labels": task_labels}

            if verbose:
                print(f"\nTask: {task}")
                print(f"Dimensions: {df.shape}\n")

        return dfs

    def print_dataset_summary(self):
        # Print a summary of the dataset
        print("\n--------------------------------------")
        print(f"ACS Data Year: {self.year}")
        print(f"ACS Survey: {self.survey}")
        print(f"ACS Horizon: {self.horizon}")
        if self.states:
            print(f"States: {', '.join(self.states)}")
        print(f"Total instances: {len(self.dataset)}")
        print(f"Total features: {len(self.dataset.columns)}")
        print("--------------------------------------\n")
