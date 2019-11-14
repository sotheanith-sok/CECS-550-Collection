import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
import sys
import  matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


class KNN(object):
    def __init__(self):
        pass

    def load_data(self):
        # Load data and label columns
        data = pd.read_csv(
            "./prompts/wine_quality.csv",
            skiprows=14,
            header=None,
            names=[
                "fa",
                "va",
                "ca",
                "rs",
                "chlor",
                "fsd",
                "tsd",
                "density",
                "pH",
                "sulphates",
                "alcohol",
                "quality",
            ],
        )

        # Drop row that contrains NA
        data = data.dropna()

        # Remove invalid data. 0<= quality <=10
        data = data[((data["quality"] >= 0) & (data["quality"] <= 10))]

        # Remove class that have data less than 2
        data = data[data.groupby("quality")["quality"].transform("count").ge(2)]

        # Get feature sets
        x = data.drop(columns=["quality"])

        # Get expected output set
        y = data["quality"]
        return x, y

    def run(self, k, T, R1, R2):
        print("--KNN Testing Procedures--")
        print("Parameters: \n k: %d \n T: %d \n R1: %d \n R2: %d" % (k, T, R1, R2))

        x, y = self.load_data()

        # Run Part 2 and 3: Create and Testing KNN
        self.test_KNN_model(x, y, k)

        # Run Part 4: Cross Validation
        self.cross_validation(x, y, k, T)

        # Run Part 5: Cross Validation
        self.optmization(x, y, k, T, R1, R2)
        pass

    def test_KNN_model(self, x, y, k):
        print("--Simple KNN Test--")
        print(
            "Other Parameters:\n test_size: %.2f \n stratify: %s"
            % (0.2, "y")
        )
        # Split data
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, stratify=y
        )

        # Create and train model
        model = KNeighborsClassifier(n_neighbors=k, weights="distance")
        model.fit(x_train, y_train)

        # Test model
        accuracy = model.score(x_test, y_test)

        # Print accuracy
        print("Accuracy:", accuracy)
        print()

    def cross_validation(self, x, y, k, T):
        #Print other parameters
        print("--Cross Validation of KNN--")
        print(
            "Other Parameters: \n weights: %s"
            % ("distance")
        )

        #Create and score models
        model = KNeighborsClassifier(n_neighbors=k, weights="distance")
        scores = cross_val_score(model, x, y, cv=T)

        #Print results
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print()

    def optmization(self, x, y, k, T, R1, R2):
        print("--Optimizing KNN--")

        #Create model
        model = KNeighborsClassifier(n_neighbors=k, weights="distance")

        #Initialize possible parameters
        param_grid = {
            "n_neighbors": range(R1, R2, 1)
        }

        #Perform GridSearchCV on model
        clf = GridSearchCV(model, param_grid, cv=T)
        clf.fit(x, y)

        #Print result
        print("Best Score: ", clf.best_score_)
        print("Best Parameters: ", clf.best_params_)

        self.plot(clf)
        
        
    def plot(self, clf):
        data = pd.DataFrame.from_dict(clf.cv_results_["params"])
        data["mean_test_score"]=clf.cv_results_["mean_test_score"]
        data.plot(x="n_neighbors", y= "mean_test_score")
        plt.show()




# Set default argument

m = KNN()
m.run(
    int(sys.argv[1]) if len(sys.argv) > 1 else 5,
    int(sys.argv[2]) if len(sys.argv) > 2 else 5,
    int(sys.argv[3]) if len(sys.argv) > 3 else 1,
    int(sys.argv[4]) if len(sys.argv) > 4 else 5,
)
