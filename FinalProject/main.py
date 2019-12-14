import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import math
import time


def run(display_graph=False):
    """Run data processing pipeline
    """
    # starts timer
    start_time = time.time()

    # Data loading
    data = load_data()

    # Data preprocessing part 1
    data = process_data(data)

    # Features analysis
    plot_features_distribution(data)
    plot_features_heatmap(data)

    # Feature engineering
    data = features_selection(data)
    data = pca(data)

    # Features analysis
    plot_pca(data)

    # Data splitting
    (X_train, y_train), (X_test, y_test) = split_data(data)

    # KNN
    knn(X_train, y_train, X_test, y_test)

    # ANN
    ann(X_train, y_train, X_test, y_test)

    if display_graph:
        plt.show()
    
    # prints time
    print("Time elapsed:  %s seconds" % round(((time.time() - start_time)), 0))


def load_data():
    """Load data from files
    """

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
    print("Labels: benign (2), malignant (4)")
    data = data.drop("Sample ID", axis=1)
    return data


def process_data(data):
    """Processing data by replacing NaN with mean of features and scale values between min and max
    """

    print("Processing data...")
    dataInstanceCount = len(data.index)
    print('{}{}'.format("Instances of data: ", dataInstanceCount))
    print("# of times '?' occured in the data: ")
    missing = data.isin(['?']).sum()
    print(missing)
    # Replace "?" with NaN
    data = data.replace("?", np.nan).astype(np.float64)

    # Fill NaN with mean of columns
    data = data.fillna(data.mean())

    # Scale values between 0 and 1
    scaler = preprocessing.MinMaxScaler()
    data[:] = scaler.fit_transform(data)

    return data


def plot_features_distribution(data):
    """Plot features distributions. Ignore the last column as it's the classfication.
    """
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


def plot_features_heatmap(data):
    """Plot features heatmap. Ignore the last column as it's the classfication.
    """
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
    """Select 5 best features using "f_classif" algorithm
    """
    print("Perform k best features selection...")

    # Split data into features and classification
    X = data[data.columns[0:9]]
    y = data[data.columns[9:]]

    # Create featuresSelector
    selector = SelectKBest(f_classif, k=5)

    # Fit into data into featuresSelector
    selector.fit(X, y.values.ravel())

    # Find  indicies of best features
    features_indicies = selector.get_support(indices=True)

    return data[data.columns[np.append(features_indicies, 9)]]


def pca(data):
    """Performe PCA to reduce dimensionaity of dataset. Output components are decided by Minka’s MLE 
    """
    print("Perform PCA...")

    # Do PCA
    pca = PCA(n_components="mle", svd_solver="auto")
    result = pca.fit_transform(data.drop("Class", axis=1))

    # Add columns name to dataframe
    columns = []
    for i in range(np.shape(result)[1]):
        columns.append(("PCA " + str(i)))

    df = pd.DataFrame(data=result, columns=columns)

    # Add classifications to new features (PCA) dataframe
    df["Class"] = data["Class"]

    return df


def plot_pca(data):
    """Plot PCA compoenents distribution 
    """
    print("Plot PCA distributions...")
    column_count = len(data.columns)

    fig, ax = plt.subplots(column_count - 1, 1)

    fig.suptitle("PCA Distributions")

    kind = "kde"
    bw_method = "scott"

    for i in range(column_count - 1):
        data.groupby("Class")[data.columns[i]].plot(
            kind=kind, bw_method=bw_method, title=data.columns[i], legend=True, ax=ax[i]
        )


def split_data(data):
    """Splitting data into training set and testing set
    """
    print("Splitting data into training set and testing set...")
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("Class", axis=1), data["Class"], test_size=0.25
    )
    return (X_train, y_train), (X_test, y_test)


def knn(X_train, y_train, X_test, y_test):
    """Use GridSearchCV with training set to find the best estimator. Then using the best estimator to score the testing set.
    """
    print("Using GridSearchCV to find the best knn estimator...")

    # Create KNN estimator
    estimator = KNeighborsClassifier(n_neighbors=1)

    # Parameters to be search on
    param_grid = {
        "n_neighbors": range(1, 50, 1),
        "weights": ["uniform", "distance"],
        "algorithm": ["ball_tree", "kd_tree", "brute"],
    }

    # Start parameters searching
    clf = GridSearchCV(estimator, param_grid, n_jobs=-1, cv=5, iid=False)
    clf.fit(X_train, y_train)

    # Retreive the best estimator
    new_estimator = clf.best_estimator_

    # Print results
    print("Best KNN: ", new_estimator)
    kNNScore = (new_estimator.score(X_test, y_test))
    print("KNN's score: %.10f" % (kNNScore))
    kNNmisclassified = round(((1 - kNNScore) * len(X_test.index)), 2)
    print('{}{}'.format("Number of misclassified records from kNN: ", kNNmisclassified))


def ann(X_train, y_train, X_test, y_test):
    """Use GridSearchCV with training set to find the best estimator. Then using the best estimator to score the testing set.
    """
    print("Using GridSearchCV to find the best ann estimator...")

    # Create MLP estimator
    estimator = MLPClassifier(solver="adam", activation="relu", max_iter=10000)

    # Parameters to be search on
    param_grid = {
        "hidden_layer_sizes": [
            (100),
            (50, 50),
            (33, 33, 33),
            (25, 25, 25, 25),
            (20, 20, 20, 20, 20),
        ],
        "alpha": np.linspace(0.0001, 1.0, 20),
    }

    # Start parameters searching
    clf = GridSearchCV(estimator, param_grid, n_jobs=-1, cv=5, iid=False)
    clf.fit(X_train, y_train)

    # Retreive the best estimator
    new_estimator = clf.best_estimator_

    # Print results
    print("Best ANN: ", new_estimator)
    ANNScore = (new_estimator.score(X_test, y_test))
    print("ANN's score: %.10f" % ANNScore)
    ANNmisclassified = round(((1 - ANNScore) * len(X_test.index)), 2)
    print('{}{}'.format("Number of misclassified records from ANN: ", ANNmisclassified))


run()
