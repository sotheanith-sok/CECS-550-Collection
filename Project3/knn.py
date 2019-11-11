import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
pd.set_option('display.max_rows', None)


class KNN(object):
    def __init__(self):
        pass

    def load_data(self):
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

        data = data.fillna(value = 0)

        x = data.drop(columns=["quality"])
        y = data["quality"]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.2
        )
        pass

    def run(self):
        self.model = KNeighborsClassifier(n_neighbors=5, weights="distance")
        self.model.fit(self.x_train, self.y_train)
        accuracy = self.model.score(self.x_test, self.y_test)
        predicted_result = self.model.predict(self.x_test)
        print(pd.DataFrame({"predicted_result":predicted_result, "expected_result":self.y_test}))
        print(accuracy)
        # start of part 4
        clf = svm.SVC(kernel='linear', C=1)
        scores = cross_val_score(clf, self.data, cv=10)
        # start of part 5
        # not sure how to alter the n_neighbor value through the parameter
        parameters = [{}]
        # also not sure about cross_val_score 'cv'
        grid_search = GridSearchCV(estimator = self.model,
                            param_grid = parameters,
                            scoring = 'accuracy',
                            cv = 10,
                            n_jobs = -1)
        grid_search - grid_search.fit(self.x_train, self.y_train)
        accuracy = grid_search.best_score_
        print(accuracy)
        print(grid_search.best_params_)
        # end of part 5
        pass


m=KNN()
m.load_data()
m.run()
