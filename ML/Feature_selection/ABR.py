from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostRegressor as ABR
import pandas as pd

class skl:
    def __init__(self):
        self.model = make_pipeline(
            MinMaxScaler(),
            ABR()
                )
        self.cv = LeaveOneOut()  
        self.parameters = {
                            # 'randomforestregressor__n_estimators':[1, 10, 100], # 0.1
                            # 'histgradientboostingregressor__l2_regularization': [0.1, 1, 10], # 100
                            # 'randomforestregressor__max_depth':[1, 3, 5, 7, 9], # 3
                            # 'gradientboostingregressor__min_samples_split': [2, 4, 6, 8, 10], # 2
                            # 'gradientboostingregressor__min_samples_leaf': [1, 2, 3, 4, 5], # 1
                            # 'gradientboostingregressor__max_features': [1, 10] # 'None'
                           }
        self.njobs = 32 
        self.scoring = 'neg_root_mean_squared_error'

    def training(self, X, y):
      
        grid_search = GridSearchCV(estimator=self.model, param_grid=self.parameters,
                                   scoring=self.scoring, 
                                   n_jobs=self.njobs, cv=self.cv)
        grid_search.fit(X, y)

      
        best_parameters = grid_search.best_params_

        
        scores = -grid_search.best_score_

        return scores, best_parameters

