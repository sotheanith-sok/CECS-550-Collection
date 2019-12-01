import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns


def run():
    data = load_data()
    data = process_data(data)

    plot_features_distribution(data)
    plot_features_heatmap(data)

    data = features_selection(data)

    pass


def load_data():
    print("Loading data...")
    data = pd.read_csv(
        "data\\breast-cancer-wisconsin.data",
        header=None,
        names=[
            "Sample ID",
            "Clump Thickness",
            "Uniformity of Cell Size",
            "Uniformity of Cell Shape",
            "Marginal Adhesion",
            "Single Epithelial Cell Size",
            "Bare Nuclei",
            "Bland Chromatin",
            "Normal Nucleoli",
            "Mitoses",
            "Class",
        ],
    )
    data = data.drop("Sample ID", axis=1)
    return data


def process_data(data):
    print("Processing data...")
    # Replace "?" with NaN
    data = data.replace("?", np.nan).astype(np.float64)

    # Fill NaN with mean of columns
    data = data.fillna(data.mean())

    # Scale values between 0 and 1
    scaler = preprocessing.MinMaxScaler()
    data[:] = scaler.fit_transform(data)

    return data


def plot_features_distribution(data):
    print("Plot features distributions...")
    fig, ax = plt.subplots(3, 3)
    fig.suptitle("Features Distributions")

    kind = "kde"
    bw_method = "scott"

    data.groupby("Class")["Clump Thickness"].plot(
        kind=kind,
        bw_method=bw_method,
        title="Clump Thickness",
        legend=True,
        ax=ax[0, 0],
    )
    data.groupby("Class")["Uniformity of Cell Size"].plot(
        kind=kind,
        bw_method=bw_method,
        title="Uniformity of Cell Size",
        legend=True,
        ax=ax[0, 1],
    )
    data.groupby("Class")["Uniformity of Cell Shape"].plot(
        kind=kind,
        bw_method=bw_method,
        title="Uniformity of Cell Shape",
        legend=True,
        ax=ax[0, 2],
    )
    data.groupby("Class")["Marginal Adhesion"].plot(
        kind=kind,
        bw_method=bw_method,
        title="Marginal Adhesion",
        legend=True,
        ax=ax[1, 0],
    )
    data.groupby("Class")["Single Epithelial Cell Size"].plot(
        kind=kind,
        bw_method=bw_method,
        title="Single Epithelial Cell Size",
        legend=True,
        ax=ax[1, 1],
    )
    data.groupby("Class")["Bare Nuclei"].plot(
        kind=kind, bw_method=bw_method, title="Bare Nuclei", legend=True, ax=ax[1, 2]
    )
    data.groupby("Class")["Bland Chromatin"].plot(
        kind=kind,
        bw_method=bw_method,
        title="Bland Chromatin",
        legend=True,
        ax=ax[2, 0],
    )
    data.groupby("Class")["Normal Nucleoli"].plot(
        kind=kind,
        bw_method=bw_method,
        title="Normal Nucleoli",
        legend=True,
        ax=ax[2, 1],
    )
    data.groupby("Class")["Mitoses"].plot(
        kind=kind, bw_method=bw_method, title="Mitoses", legend=True, ax=ax[2, 2]
    )

    pass


def plot_features_heatmap(data):
    print("Plot features heatmap...")
    plt.figure()
    corr = data.drop("Class", axis=1).corr()
    corr = corr.transpose()
    ax = sns.heatmap(
        corr, vmin=-1.0, vmax=1.0, center=0, annot=True, cmap="Blues", square=True
    )
    ax.set_ylim(0, len(corr))
    ax.set_title("Features Correlation Heatmap")


def features_selection(data):
    print("Perform k best features selection...")
    X = data[data.columns[0:9]]
    y = data[data.columns[9:]]

    selector = SelectKBest(f_classif, k=5)
    selector.fit(X, y.values.ravel())
    features_indicies = selector.get_support(indices=True)

    return data[data.columns[np.append(features_indicies, 9)]]

def pca(data):
    print("Perform PCA...")
    pass

run()
