from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from time import time
from joblib import parallel_backend

params={
    'svm' : {
        'model' : SVR(tol=1e-4),
        'params' : {
            'kernel' : ['poly','rbf','sigmoid'],
            'degree' : [1,2,3],
            'gamma' : ['scale','auto'],
            'max_iter' : [1000,10000,100000]
        }
    },
    'ridge' : {
        'model' : Ridge(tol=1e-4,normalize=True),
        'params' : {
            'alpha' : [0,0.25,0.5,0.75,1],
            'solver' : ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
            'max_iter' : [1000,10000,100000]
        }
    }, 
    'elastic_net' : {
        'model' : ElasticNet(tol=1e-4,normalize=True),
        'params' : {
            'alpha' : [0,0.25,0.5,0.75,1],
            'max_iter' : [1000,10000,100000],
            'selection' : ['cyclic','random']
            
        }
    }, 
    'gradient_boosting' : {
        'model' : GradientBoostingRegressor(tol=1e-4),
        'params' : {
            'loss' : ['ls','lad','huber','quantile'],
            'learning_rate' : [1e-3,1e-4,1e-5],
            'n_estimators' : [100,200,300,400,500],
            'criterion' : ['friedman_mse', 'mse', 'mae'],
            'max_features' : [None,'auto', 'sqrt', 'log2']
        }
    },
    'mlp' : {
        'model' : MLPRegressor(early_stopping=True,tol=1e-4,activation='relu',solver='adam'),
        'params' : {
            'hidden_layer_sizes' : [[8],[32],[8,4]],
            #'activation' : ['identity', 'logistic', 'tanh', 'relu'],
            #'solver' : ['lbfgs', 'sgd', 'adam'],
            'alpha' : [1,0.1,1e-2,1e-3,1e-4,1e-5,1e-6],
            'learning_rate' : ['constant','invscaling','adaptive'],
            'learning_rate_init' : [1e-3,1e-4,1e-5,1e-6],
            'momentum' : [0,0.25,0.5,0.75,1]
            
        }
    } 
}

def hyperParameterOptimizer(X,Y):
    scores={
            'model' : [],
            'Best Score' : [],
            'Best Params' :  []
        }
    for model_name,model_params in params.items():
        start=time()
        with parallel_backend('threading',n_jobs=-1):
            hyperParamOpt=GridSearchCV(model_params['model'],model_params['params'],cv=3,return_train_score=False,n_jobs=-1)
            hyperParamOpt.fit(X,Y)
        scores['model'].append(model_name)
        scores['Best Score'].append(hyperParamOpt.best_score_)
        scores['Best Params'].append(hyperParamOpt.best_params_)
        print(f"{model_name} => {hyperParamOpt.best_score_} | {hyperParamOpt.best_params_} took {time() - start} seconds to complete.")
    return scores





