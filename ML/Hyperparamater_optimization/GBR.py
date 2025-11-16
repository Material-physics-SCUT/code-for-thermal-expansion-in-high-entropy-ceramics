from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, LeaveOneOut, KFold
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class skl:
    def __init__(self, model):
       
        self.model = make_pipeline(
            MinMaxScaler(),
            model
        )
        
        self.cv = LeaveOneOut()   
 
        self.parameters = {
            'gradientboostingregressor__loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
            'gradientboostingregressor__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'gradientboostingregressor__n_estimators': [1, 50, 100, 150, 200],
            'gradientboostingregressor__min_samples_split': [1, 2, 5, 10, 15, 20],
            'gradientboostingregressor__max_depth': np.round(np.arange(1, 7, 1), 0).tolist(),
            'gradientboostingregressor__random_state': [876543],
        }
        self.njobs = 32 
        self.scoring = 'neg_root_mean_squared_error'

    def training(self, X, y, model_name):
       
        grid_search = GridSearchCV(estimator=self.model, param_grid=self.parameters,
                                   scoring=self.scoring, 
                                   n_jobs=self.njobs, cv=self.cv)
        grid_search.fit(X, y)

        
        best_parameters = grid_search.best_params_

       
        best_cv_score = -grid_search.best_score_

        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        return best_cv_score, best_parameters, rmse, r2
        

data = pd.read_excel('criteria_train.xlsx')  
X = data.iloc[:, 2:].apply(lambda x: round(x, 4))  
y = data.iloc[:, 1].values  


models = {
    'gradientboostingregressor': GradientBoostingRegressor(),
}


output_file = "model_results.txt"
with open(output_file, "w") as f:
   
    for model_name, (model, parameters) in models.items():
        model_cv = skl(model, parameters)
        cv_score, best_parameters, train_rmse, train_r2 = model_cv.training(X, y)
        
       
        f.write(f"{model_name}\n")
        f.write(f"{cv_score}\n")
        f.write(f"{best_parameters}\n")
        f.write(f"{train_rmse}\n")
        f.write(f"{train_r2}\n\n")
        
        
        print(f"{model_name}")
        print(f"{cv_score}")
        print(f"{best_parameters}")
        print(f" {train_rmse}")
        print(f"{train_r2}\n")
        
print(f"{output_file}")

