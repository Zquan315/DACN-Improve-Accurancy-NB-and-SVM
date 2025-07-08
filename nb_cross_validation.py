import warnings
from abc import ABCMeta, abstractmethod
import os
from numbers import Integral, Real
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from scipy.special import logsumexp
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils._param_validation import Interval
from numbers import Real
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    _fit_context,
)
from sklearn.preprocessing import LabelBinarizer, binarize, label_binarize, StandardScaler, LabelEncoder
from sklearn.utils._param_validation import Interval
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.multiclass import _check_partial_fit_first_call
from sklearn.utils.validation import (
    _check_n_features,
    _check_sample_weight,
    check_is_fitted,
    check_non_negative,
    validate_data,
)

__all__ = [
    "BernoulliNB",
    "CategoricalNB",
    "ComplementNB",
    "GaussianNB",
    "MultinomialNB",
]

class _BaseNB(ClassifierMixin, BaseEstimator, metaclass=ABCMeta):
    """Abstract base class for naive Bayes estimators"""

    @abstractmethod
    def _joint_log_likelihood(self, X):
        """Compute the unnormalized posterior log probability of X"""

    @abstractmethod
    def _check_X(self, X):
        """Validate X, used only in predict* methods."""

    def predict_joint_log_proba(self, X):
        check_is_fitted(self)
        X = self._check_X(X)
        return self._joint_log_likelihood(X)

    def predict(self, X):
        check_is_fitted(self)
        X = self._check_X(X)
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_log_proba(self, X):
        check_is_fitted(self)
        X = self._check_X(X)
        jll = self._joint_log_likelihood(X)
        # normalize by P(x) = P(f_1, ..., f_n)
        log_prob_x = logsumexp(jll, axis=1)
        return jll - np.atleast_2d(log_prob_x).T

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

class GaussianNB(_BaseNB):
    _parameter_constraints: dict = {
        "priors": ["array-like", None],
        "var_smoothing": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(self, *, priors=None, var_smoothing=1e-9):
        self.priors = priors
        self.var_smoothing = var_smoothing

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        X, y = self._validate_data(X, y)

        # Sử dụng LabelEncoder để mã hóa nhãn
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        self.classes_ = label_encoder.classes_

        # Tính toán số lượng mẫu cho mỗi lớp
        class_counts = np.bincount(y, weights=sample_weight)
        self.class_count_ = class_counts

        # Tính toán xác suất tiên nghiệm (class_prior_)
        if self.priors is None:
            self.class_prior_ = class_counts / class_counts.sum()
        else:
            self.class_prior_ = np.asarray(self.priors)

        # Khởi tạo các tham số theta_ và var_
        n_features = X.shape[1]
        n_classes = len(self.classes_)
        self.theta_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))

        # Tính toán mean và variance cho từng lớp
        for i in range(n_classes):
            X_i = X[y == i]
            self.theta_[i, :] = X_i.mean(axis=0)
            self.var_[i, :] = X_i.var(axis=0)

        return self
    def _check_X(self, X):
        """Validate X, used only in predict* methods."""
        return validate_data(self, X, reset=False)
    
    @staticmethod
    def _update_mean_variance(n_past, mu, var, X, sample_weight=None):
        if X.shape[0] == 0:
            return mu, var

        # Compute (potentially weighted) mean and variance of new datapoints
        if sample_weight is not None:
            n_new = float(sample_weight.sum())
            if np.isclose(n_new, 0.0):
                return mu, var
            new_mu = np.average(X, axis=0, weights=sample_weight)
            new_var = np.average((X - new_mu) ** 2, axis=0, weights=sample_weight)
        else:
            n_new = X.shape[0]
            new_var = np.var(X, axis=0)
            new_mu = np.mean(X, axis=0)

        if n_past == 0:
            return new_mu, new_var

        n_total = float(n_past + n_new)

        # Combine mean of old and new data, taking into consideration
        # (weighted) number of observations
        total_mu = (n_new * new_mu + n_past * mu) / n_total
        # Combine variance of old and new data, taking into consideration
        # (weighted) number of observations. This is achieved by combining
        # the sum-of-squared-differences (ssd)
        old_ssd = n_past * var
        new_ssd = n_new * new_var
        total_ssd = old_ssd + new_ssd + (n_new * n_past / n_total) * (mu - new_mu) ** 2
        total_var = total_ssd / n_total

        return total_mu, total_var
    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        return self._partial_fit(
            X, y, classes, _refit=False, sample_weight=sample_weight
        )
    def _partial_fit(self, X, y, classes=None, _refit=False, sample_weight=None):
        if _refit:
            self.classes_ = None
        first_call = _check_partial_fit_first_call(self, classes)
        X, y = validate_data(self, X, y, reset=first_call)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        # If the ratio of data variance between dimensions is too small, it
        # will cause numerical errors. To address this, we artificially
        # boost the variance by epsilon, a small fraction of the standard
        # deviation of the largest dimension.
        self.epsilon_ = self.var_smoothing * np.var(X, axis=0).max()
        if first_call:
            # This is the first call to partial_fit:
            # initialize various cumulative counters
            n_features = X.shape[1]
            n_classes = len(self.classes_)
            self.theta_ = np.zeros((n_classes, n_features))
            self.var_ = np.zeros((n_classes, n_features))

            self.class_count_ = np.zeros(n_classes, dtype=np.float64)

            # Initialise the class prior
            # Take into account the priors
            if self.priors is not None:
                priors = np.asarray(self.priors)
                # Check that the provided prior matches the number of classes
                if len(priors) != n_classes:
                    raise ValueError("Number of priors must match number of classes.")
                # Check that the sum is 1
                if not np.isclose(priors.sum(), 1.0):
                    raise ValueError("The sum of the priors should be 1.")
                # Check that the priors are non-negative
                if (priors < 0).any():
                    raise ValueError("Priors must be non-negative.")
                self.class_prior_ = priors
            else:
                # Initialize the priors to zeros for each class
                self.class_prior_ = np.zeros(len(self.classes_), dtype=np.float64)
        else:
            if X.shape[1] != self.theta_.shape[1]:
                msg = "Number of features %d does not match previous data %d."
                raise ValueError(msg % (X.shape[1], self.theta_.shape[1]))
            # Put epsilon back in each time
            self.var_[:, :] -= self.epsilon_

        classes = self.classes_

        unique_y = np.unique(y)
        unique_y_in_classes = np.isin(unique_y, classes)

        if not np.all(unique_y_in_classes):
            raise ValueError(
                "The target label(s) %s in y do not exist in the initial classes %s"
                % (unique_y[~unique_y_in_classes], classes)
            )

        for y_i in unique_y:
            i = classes.searchsorted(y_i)
            X_i = X[y == y_i, :]

            if sample_weight is not None:
                sw_i = sample_weight[y == y_i]
                N_i = sw_i.sum()
            else:
                sw_i = None
                N_i = X_i.shape[0]

            new_theta, new_sigma = self._update_mean_variance(
                self.class_count_[i], self.theta_[i, :], self.var_[i, :], X_i, sw_i
            )

            self.theta_[i, :] = new_theta
            self.var_[i, :] = new_sigma
            self.class_count_[i] += N_i

        self.var_[:, :] += self.epsilon_

        # Update if only no priors is provided
        if self.priors is None:
            # Empirical prior, with sample_weight taken into account
            self.class_prior_ = self.class_count_ / self.class_count_.sum()

        return self
    def _joint_log_likelihood(self, X):
        joint_log_likelihood = []
        for i in range(np.size(self.classes_)):
            jointi = np.log(self.class_prior_[i])
            n_ij = -0.5 * np.sum(np.log(2.0 * np.pi * self.var_[i, :]))
            n_ij -= 0.5 * np.sum(((X - self.theta_[i, :]) ** 2) / (self.var_[i, :]), 1)
            joint_log_likelihood.append(jointi + n_ij)

        joint_log_likelihood = np.array(joint_log_likelihood).T
        return joint_log_likelihood
    

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
import pickle

def scale_dataset(dataframe, oversample=False, k_features=40):
    df = dataframe.copy()
    for col in df.columns[:-1]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isnull().all():
            df.drop(columns=[col], inplace=True)
    df = df.dropna()
    x = df[df.columns[:-1]].values
    y = df[df.columns[-1]].values
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    
    selector = SelectKBest(score_func=mutual_info_classif, k=k_features)
    x = selector.fit_transform(x, y)
    selected_indices = selector.get_support(indices=True)
    selected_features = df.columns[:-1][selected_indices].tolist()
    print("Selected features:", selected_features, "\n")
    
    if oversample:
        ros = SMOTE(sampling_strategy=0.95, random_state=42)
        x, y = ros.fit_resample(x, y)
    return x, y, selected_features

def main(folder_path):
    selected_features = [
        'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Std',
        'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
        'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
        'Fwd IAT Total', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
        'Bwd IAT Total', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
        'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
        'Fwd Header Length', 'Bwd Header Length', 'Min Packet Length', 'Max Packet Length',
        'Packet Length Std', 'Packet Length Mean', 'Packet Length Variance',
        'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
        'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
        'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
        'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
        'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'min_seg_size_forward',
        'Active Mean', 'Idle Mean'
    ]
    with open("log_result_cross_validation.txt", "a") as log_file:
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".csv"):
                file_path = os.path.join(folder_path, file_name)
                print(f"Processing file: {file_name}")
                cols = ["Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
                        "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
                        "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
                        "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
                        "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
                        "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
                        "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
                        "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
                        "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
                        "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
                        "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
                        "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
                        "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
                        "Avg Fwd Segment Size", "Avg Bwd Segment Size","Fwd Avg Bytes/Bulk",
                        "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk",
                        "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets",
                        "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
                        "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std",
                        "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Label"]
                df = pd.read_csv(file_path, names=cols, skipinitialspace=True)
                df["Label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)
                
                # Only keep 56 features + Label
                available_features = [f for f in selected_features if f in df.columns]
                if len(available_features) != len(selected_features):
                    missing_features = set(selected_features) - set(available_features)
                    print(f"Warning: The following features are not in the dataset: {missing_features}")
                keep_cols = available_features + ['Label']
                df = df[keep_cols]
                
                x, y, selected_features_40 = scale_dataset(df, oversample=True, k_features=40)
                print("Selected 40 features:", selected_features_40)
                print("Scaling and oversampling datasets...")
                print("Training model with 5-fold cross-validation...")
                
                nb_model = GaussianNB()
                param_grid = {'var_smoothing': [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                grid_search = GridSearchCV(nb_model, param_grid, cv=cv, scoring='f1', return_train_score=True)
                grid_search.fit(x, y)
                best_nb_model = grid_search.best_estimator_
                
                # model_name = f"nb_model_cross_validation.pkl"
                # with open(model_name, 'wb') as f:
                #     pickle.dump(best_nb_model, f)
                # print(f"Saved model to {model_name}")
                # print("Best parameters:", grid_search.best_params_)
                
                mean_test_score = grid_search.cv_results_['mean_test_score'][grid_search.best_index_]
                std_test_score = grid_search.cv_results_['std_test_score'][grid_search.best_index_]
                
                log_file.write(f"Result of file: {file_name}\n")
                log_file.write(f"Classification Report:\n{classification_report(y, best_nb_model.predict(x))}")
                log_file.write(f"Accuracy: {accuracy_score(y, best_nb_model.predict(x))*100:.2f}%\n")
                log_file.write(f"ROC-AUC: {roc_auc_score(y, best_nb_model.predict_proba(x)[:, 1]):.2f}\n")
                log_file.write(f"Best parameters: {grid_search.best_params_}\n")
                log_file.write(f"Mean F1 Score (5-fold CV): {mean_test_score:.4f} (+/- {std_test_score * 2:.4f})\n")
                log_file.write(f"Selected features: {selected_features_40}\n")
                log_file.write("Model training complete.\n")
                log_file.write("---------------------------------------------------\n\n")
                
                print(f"\nMean F1 Score (5-fold CV): {mean_test_score:.4f} (+/- {std_test_score * 2:.4f})")
                print(f"Processing file: {file_name}")
                print(f"Classification Report:\n{classification_report(y, best_nb_model.predict(x))}")
                print(f"Accuracy: {accuracy_score(y, best_nb_model.predict(x))*100:.2f}%")               
                print(f"ROC-AUC: {roc_auc_score(y, best_nb_model.predict_proba(x)[:, 1]):.2f}")
                print(f"Best parameters: {grid_search.best_params_}")
                print("Number of features: ", x.shape[1])
                print("Model trained successfully!")
                print("--------------------------------------------------------------\n\n")
                del x, y, best_nb_model

if __name__ == "__main__":
    folder_path = r"D:\TO_CONG_QUAN\University_Documents\Nam_3\HK2\DACN"
    main(folder_path)