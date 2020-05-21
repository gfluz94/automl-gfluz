import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, roc_auc_score, 
                             f1_score, roc_curve, precision_recall_curve)
import pickle
from pycaret.classification import *

class ClassifierPyCaret(BaseEstimator):
    def __init__(self, metric="F1", numeric_features=None, categorical_features=None, numeric_imputation='mean',
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
            'K Neighbors Classifier': 'knn',
            'Ada Boost Classifier': 'ada',
            'Extreme Gradient Boosting': 'xgboost',
            'CatBoost Classifier': 'catboost',
            'Quadratic Discriminant Analysis': 'qda',
            'Extra Trees Classifier': 'et',
            'Naive Bayes': 'nb',
            'Random Forest Classifier': 'rf',
            'Light Gradient Boosting Machine': 'lightgbm',
            'Gradient Boosting Classifier': 'gbc',
            'Decision Tree Classifier': 'dt',
            'Logistic Regression': 'lr',
            'SVM - Linear Kernel': 'svm',
            'Linear Discriminant Analysis': 'lda',
            'Ridge Classifier': 'ridge',
            'Multi Level Perceptron': 'mlp',
            'SVM (Linear)': 'svm',
            'SVM (RBF)': 'rbfsvm',
            'Gaussian Process': 'gpc'
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
        from pycaret.classification import prep_pipe
        self.all_models = compare_models(fold=self.folds, sort=self.metric)
        self.__name_model = self.all_models.data.iloc[0, 0]
        str_model = self.MODELS_TO_STR[self.__name_model]
        best_model = tune_model(str_model, fold=self.folds, n_iter=50, optimize=self.metric)
        self.best_model = finalize_model(best_model)
        self.preprocessor = prep_pipe
        self.trained = True
        
        return self.all_models
        
    def predict(self, X):
        if not self.trained:
            raise ValueError("Model not fitted yet!")
        if not isinstance(X, pd.DataFrame):
            cols = ['X_'+str(i+1) for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=cols)
        X_transformed = self.preprocessor.transform(X)
        return self.best_model.predict(X_transformed)
    
    def predict_proba(self, X):
        if not self.trained:
            raise ValueError("Model not fitted yet!")
        if not isinstance(X, pd.DataFrame):
            cols = ['X_'+str(i+1) for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=cols)
        X_transformed = self.preprocessor.transform(X)
        try:
            output = self.best_model.predict_proba(X_transformed)
        except:
            raise ValueError("Model does not return probabilities!")
        return output
    
    def preprocess(self, X):
        if not isinstance(X, pd.DataFrame):
            cols = ['X_'+str(i+1) for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=cols)
        return self.preprocessor.transform(X)
    
    def evaluate(self, X, y_true):
        if not self.trained:
            raise ValueError("Model not fitted yet!")
        METRICS = ["Accuracy", "Recall", "Precision", "F1-Score", "AUC-ROC"]
        metrics = {}
        y_pred = self.predict(X)
        try:
            y_proba = self.predict_proba(X)[:,1]
        except:
            y_proba = None
        if len(np.unique(y_true))==2:
            for metric in METRICS:
                if metric=="Accuracy":
                    metrics[metric] = accuracy_score(y_true, y_pred)
                elif metric=="Precision":
                    metrics[metric] = precision_score(y_true, y_pred)
                elif metric=="Recall":
                    metrics[metric] = recall_score(y_true, y_pred)
                elif metric=="F1-Score":
                    metrics[metric] = f1_score(y_true, y_pred)
                elif metric=="AUC-ROC" and y_proba is not None:
                    metrics[metric] = roc_auc_score(y_true, y_proba)
        else:
            for metric in METRICS:
                if metric=="Accuracy":
                    metrics[metric] = accuracy_score(y_true, y_pred)
                elif metric=="Precision":
                    metrics[metric] = precision_score(y_true, y_pred, average="weighted")
                elif metric=="Recall":
                    metrics[metric] = recall_score(y_true, y_pred, average="weighted")
                elif metric=="F1-Score":
                    metrics[metric] = f1_score(y_true, y_pred, average="weighted")

        return pd.DataFrame(metrics, index=[self.__name_model])
    
    def binary_evaluation_plot(self, X, y_true):
        if not self.trained:
            raise ValueError("Model not fitted yet!")
        try:
            y_proba = self.predict_proba(X)[:,1]
        except:
            raise ValueError("Model does not return probabilities!")

        if len(np.unique(y_true))!=2:
            raise ValueError("Multiclass Problem!")

        fig, ax = plt.subplots(2,2,figsize=(12,8))
        self._plot_roc(y_true, y_proba, ax[0][0])
        self._plot_pr(y_true, y_proba, ax[0][1])
        self._plot_cap(y_true, y_proba, ax[1][0])
        self._plot_ks(y_true, y_proba, ax[1][1])
        plt.tight_layout()
        plt.show()
        
    def _plot_cap(self, y_test, y_proba, ax):
        cap_df = pd.DataFrame(data=y_test, index=y_test.index)
        cap_df["Probability"] = y_proba

        total = cap_df.iloc[:, 0].sum()
        perfect_model = (cap_df.iloc[:, 0].sort_values(ascending=False).cumsum()/total).values
        current_model = (cap_df.sort_values(by="Probability", ascending=False).iloc[:, 0].cumsum()/total).values

        max_area = 0
        covered_area = 0
        h = 1/len(perfect_model)
        random = np.linspace(0, 1, len(perfect_model))
        for i, (am, ap) in enumerate(zip(current_model, perfect_model)):
            try:
                max_area += (ap-random[i]+perfect_model[i+1]-random[i+1])*h/2
                covered_area += (am-random[i]+current_model[i+1]-random[i+1])*h/2
            except:
                continue
        accuracy_ratio = covered_area/max_area

        ax.plot(np.linspace(0, 1, len(current_model)), current_model, 
                            color="green", label=f"{self.__name_model}: AR = {accuracy_ratio:.3f}")
        ax.plot(np.linspace(0, 1, len(perfect_model)), perfect_model, color="red", label="Perfect Model")
        ax.plot([0,1], [0,1], color="navy")
        ax.set_xlabel("Individuals", fontsize=12)
        ax.set_ylabel("Target Individuals", fontsize=12)
        ax.set_xlim((0,1))
        ax.set_ylim((0,1.01))
        ax.legend(loc=4, fontsize=10)
        ax.set_title("CAP Analysis", fontsize=13)
        
    def _plot_roc(self, y_test, y_proba, ax):
        fpr, tpr, _ = roc_curve(y_test, y_proba)

        ax.plot(fpr, tpr, color="red", label=f"{self.__name_model} (AUC = {roc_auc_score(y_test, y_proba):.3f})")
        ax.plot([0,1], [0,1], color="navy")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_xlim((0,1))
        ax.set_ylim((0,1.001))
        ax.legend(loc=4)
        ax.set_title("ROC Analysis", fontsize=13)
        
    def _plot_pr(self, y_test, y_proba, ax):
        precision, recall, _ = precision_recall_curve(y_test, y_proba)

        ax.plot(recall, precision, color="red", label=f"{self.__name_model}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim((0,1))
        ax.set_ylim((0,1.001))
        ax.legend(loc=4)
        ax.set_title("Precision-Recall Analysis", fontsize=13)
        
    def _plot_ks(self, y_test, y_proba, ax):
        prediction_labels = pd.DataFrame(y_test.values, columns=["True Label"])
        prediction_labels["Probabilities"] = y_proba
        prediction_labels["Thresholds"] = prediction_labels["Probabilities"].apply(lambda x: np.round(x, 2))
        df = prediction_labels.groupby("Thresholds").agg(["count", "sum"])[["True Label"]]
        ks_df = pd.DataFrame(df["True Label"]["sum"]).rename(columns={"sum":"Negative"})
        ks_df["Positive"] = df["True Label"]["count"]-df["True Label"]["sum"]
        ks_df["Negative"] = ks_df["Negative"].cumsum()/ks_df["Negative"].sum()
        ks_df["Positive"] = ks_df["Positive"].cumsum()/ks_df["Positive"].sum()
        ks_df["KS"] = ks_df["Positive"]-ks_df["Negative"]
        ks_df.loc[0.0, :] = [0.0, 0.0, 0.0]
        ks_df = ks_df.sort_index()
        max_ks_thresh = ks_df.KS.idxmax()

        ks_df.drop("KS", axis=1).plot(ax=ax)
        ax.set_title("KS Analysis", fontsize=13)
        ax.plot([max_ks_thresh, max_ks_thresh], 
                [ks_df.loc[max_ks_thresh,"Negative"], ks_df.loc[max_ks_thresh,"Positive"]], label="Max KS")
        ax.text(max_ks_thresh-0.15, 0.5, f"KS={ks_df.loc[max_ks_thresh,'KS']:.3f}", fontsize=12, color="green")
        ax.legend()

    def save_pipeline(self, filename=None):
        final_pipeline = Pipeline([
            ("preprocessing", self.preprocessor),
            ("model", self.best_model)])
        if filename is None:
            filename = "classifier_"+datetime.now().strftime("%Y%b%d_%H%M")+".pkl"

        with open(filename, "wb") as file:
            pickle.dump(final_pipeline, file)

    def load_pipeline(self, filename):
        with open(filename, "rb") as file:
             self.__model_loaded = pickle.load(file)
             self.preprocessor = self.__model_loaded[0]
             self.best_model = self.__model_loaded[1]
             self.__name_model = "Loaded Model"
             self.trained = True
        