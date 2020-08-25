import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from pycaret.regression import *

class RegressorPyCaret(BaseEstimator):
    def __init__(self, metric="RMSE", numeric_features=None, categorical_features=None, numeric_imputation='mean',
                 normalize=True, normalize_method='zscore', handle_unknown_categorical=True, 
                 unknown_categorical_method='least_frequent', feature_selection=False, feature_selection_threshold=0.8,
                 feature_interaction=False, folds=5, **kwargs):
        
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.numeric_imputation = numeric_imputation
        self.normalize = normalize
        self.normalize_method = normalize_method
        self.handle_unknown_categorical = handle_unknown_categorical
        self.unknown_categorical_method = unknown_categorical_method
        self.feature_selection = feature_selection
        self.feature_selection_threshold = feature_selection_threshold
        self.feature_interaction = feature_interaction
        self.metric = metric
        self.folds = folds
        self.params = kwargs
        self.trained = False
        self.__model_loaded = None
        self.MODELS_TO_STR = {
             'Random Forest': 'rf',
             'Gradient Boosting Regressor': 'gbr',
             'Extra Trees Regressor': 'et',
             'CatBoost Regressor': 'catboost',
             'Extreme Gradient Boosting': 'xgboost',
             'Light Gradient Boosting Machine': 'lightgbm',
             'AdaBoost Regressor': 'ada',
             'K Neighbors Regressor': 'knn',
             'Decision Tree': 'dt',
             'Ridge Regression': 'ridge',
             'Least Angle Regression': 'lar',
             'Linear Regression': 'lr',
             'Bayesian Ridge': 'br',
             'Huber Regressor': 'huber',
             'TheilSen Regressor': 'tr',
             'Random Sample Consensus': 'ransac',
             'Lasso Regression': 'lasso',
             'Orthogonal Matching Pursuit': 'omp',
             'Elastic Net': 'en',
             'Support Vector Machine': 'svm',
             'Passive Aggressive Regressor': 'par',
             'Lasso Least Angle Regression': 'llar'
        }
        
    def fit(self, X, y, *args):
        if not isinstance(X, pd.DataFrame):
            cols = ['X_'+str(i+1) for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=cols)
        else:
            df = X.copy()

        if isinstance(y, pd.Series):
            name = "target"
            y.name=name
        elif isinstance(y, pd.DataFrame):
            y = y.iloc[:,0]
            name = y.name
        else:
            name = "target"
            y = pd.Series(y, name=name)

        df[name] = y

        if self.__model_loaded is not None:
            return self.__model_loaded

        if self.categorical_features is None:
            if self.numeric_features is None:
                self.categorical_features = [i for i in df.dtypes[df.dtypes=='object'].index.tolist() if i!=name]
                self.numeric_features =  [i for i in df.dtypes[df.dtypes!='object'].index if i!=name]
            else:
                self.categorical_features = [i for i in df.columns if i!=name and i not in self.numeric_features]
        else:
            if self.numeric_features is None:
                self.numeric_features = [i for i in df.columns if i!=name and i not in self.categorical_features]
            
        self.env_setup = setup(data=df,
                               target=name,
                               train_size=0.9,
                               numeric_features = self.numeric_features,
                               categorical_features = self.categorical_features,
                               numeric_imputation = self.numeric_imputation,
                               normalize = self.normalize,
                               normalize_method = self.normalize_method,
                               handle_unknown_categorical = self.handle_unknown_categorical,
                               unknown_categorical_method = self.unknown_categorical_method,
                               feature_selection = self.feature_selection,
                               feature_selection_threshold = self.feature_selection_threshold,
                               feature_interaction = self.feature_interaction,
                               **self.params)
        from pycaret.regression import prep_pipe
        self.all_models = compare_models(fold=self.folds, sort=self.metric)
        self.__name_model = self.all_models.data.iloc[0, 0]
        str_model = self.MODELS_TO_STR[self.__name_model]
        metric_to_optimize = self.metric.lower() if self.metric.lower() in ["r2", "mse", "mae"] else "mse"
        best_model = tune_model(str_model, fold=self.folds, n_iter=50, optimize=metric_to_optimize)
        self.best_model = finalize_model(best_model)
        self.preprocessor = prep_pipe
        self.trained = True
        
        return self.all_models
        
    def predict(self, X, postprocess=None):
        if not self.trained:
            raise ValueError("Model not fitted yet!")
        if not isinstance(X, pd.DataFrame):
            cols = ['X_'+str(i+1) for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=cols)
        X_transformed = self.preprocessor.transform(X)
        output = self.best_model.predict(X_transformed)
        if postprocess is not None:
            output = postprocess(output)
        return output
    
    def preprocess(self, X):
        if not isinstance(X, pd.DataFrame):
            cols = ['X_'+str(i+1) for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=cols)
        return self.preprocessor.transform(X)
    
    def evaluate(self, X, y_true, postprocess=None):
        if not self.trained:
            raise ValueError("Model not fitted yet!")
        METRICS = ["MAE", "MSE", "RMSE", "R2", "RMLSE", "MAPE"]
        FUNCTIONS = [mean_absolute_error, mean_squared_error, rmse, r2_score, rmlse, mape]
        metrics = {}
        y_pred = self.predict(X, postprocess=postprocess)
        for metric, function in zip(METRICS, FUNCTIONS):
            metrics[metric] = function(y_true, y_pred)

        return pd.DataFrame(metrics, index=[self.__name_model])
    
    def plot_analysis(self, X, y_true, postprocess=None, name=None):
        _, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,figsize=(14,3))
        y_pred = self.predict(X, postprocess=postprocess)
        residuals = y_true - y_pred

        if name is not None:
            txt_to_show = f"{name} | RMSE: {rmse(y_true, y_pred):.3f} | R2: {r2_score(y_true, y_pred):.3f}"
            print(len(txt_to_show)*'-')
            print(txt_to_show)
            print(len(txt_to_show)*'-')
        ax1.set_title("Histogram", fontsize=13)
        ax1.hist(y_true, color="lightblue", edgecolor="navy", alpha=1, label="True")
        ax1.hist(y_pred, color="red", edgecolor="red", alpha=0.6, label="Predicted")
        ax1.legend()

        ax2.set_title("Residuals Plot", fontsize=13)
        sns.regplot(x=y_pred, y=residuals, ci=0, 
                    scatter_kws={"color":"lightblue", "linewidth":1, "edgecolors":"navy"}, 
                    line_kws={"color": "red"}, ax=ax2)
        ax2.plot([min(y_pred), max(y_pred)], [0, 0], color="black", linestyle="--")
        ax2.set_xlabel("Fitted Values")
        ax2.set_ylabel("Residuals")

        standardized_residuals = (np.sort(residuals)-np.mean(residuals))/np.std(residuals, ddof=1)
        n = len(standardized_residuals)
        theoretical_quantiles = [scipy.stats.norm.ppf(p) for p in np.linspace(0.01, 0.99, len(standardized_residuals))]
        ax3.set_title("QQ-Plot", fontsize=13)
        sns.regplot(x=theoretical_quantiles, y=standardized_residuals, fit_reg=False,
                    scatter_kws={"color":"lightblue", "linewidth":1, "edgecolors":"navy"}, 
                    line_kws={"color": "red"}, ax=ax3)
        ax3.plot([min(theoretical_quantiles), max(theoretical_quantiles)],
                [min(theoretical_quantiles), max(theoretical_quantiles)], color="red", linestyle="--")
        ax3.set_xlim((min(theoretical_quantiles), max(theoretical_quantiles)))
        ax3.set_ylim((min(theoretical_quantiles), max(theoretical_quantiles)))
        ax3.set_ylabel("Standardized Residuals")
        ax3.set_xlabel("Theoretical Quantiles")

        ax4.set_title("Predicted x True", fontsize=13)
        min_val = min([y_pred.min(), y_true.min()])
        max_val = max([y_pred.max(), y_true.max()])
        sns.scatterplot(y_pred, y_true, color="lightblue", edgecolor="navy", ax=ax4)
        ax4.plot([min_val, max_val], [min_val, max_val], color="red", ls="dashed")
        ax4.set_xlim([min_val, max_val])
        ax4.set_ylim([min_val, max_val])
        ax4.set_xlabel("Predicted Values")
        ax4.set_ylabel("True Values")

        plt.tight_layout()
        plt.show()
        
    def save_pipeline(self, filename=None):
        final_pipeline = Pipeline([
            ("preprocessing", self.preprocessor),
            ("model", self.best_model)])
        if filename is None:
            filename = "regressor_"+datetime.now().strftime("%Y%b%d_%H%M")+".pkl"

        with open(filename, "wb") as file:
            pickle.dump(final_pipeline, file)

    def load_pipeline(self, filename):
        with open(filename, "rb") as file:
             self.__model_loaded = pickle.load(file)
             self.preprocessor = self.__model_loaded[0]
             self.best_model = self.__model_loaded[1]
             self.__name_model = "Loaded Model"
             self.trained = True


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def rmlse(y_true, y_pred):
    log_term = np.log((1+y_pred)/(1+y_true))
    return np.sqrt(np.mean(log_term**2))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))
        