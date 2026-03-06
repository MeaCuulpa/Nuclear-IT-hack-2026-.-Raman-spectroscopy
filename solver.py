import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score, root_mean_squared_log_error
from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from DNN import DNN

from utils import set_seed
from dataset import Dataset_Titanic, Dataset_HousePrices


class Solver:
    def __init__(self, config):
        self.config = config
        set_seed(self.config.seed)
        if self.config.task.titanic:
            self.dataset = Dataset_Titanic(self.config)
        elif self.config.task.house_pricing:
            self.dataset = Dataset_HousePrices(self.config)
            self.model_name = []


    def train_one_model(self, model, model_name):

        if self.config.task.titanic:
            data = pd.read_csv(self.config.path.train_titanic)
            X = data.drop('Survived', axis=1)
            y = data['Survived']
        elif self.config.task.house_pricing:
            data = pd.read_csv(self.config.path.train_pricing)
            X = data.drop('SalePrice', axis=1)
            y = np.log(data['SalePrice'])

        X = self.dataset.preprocessing(X)

        Path(self.config.path.data_checkpoint).mkdir(parents=True, exist_ok=True)

        if self.config.task.titanic:
            sgkf = StratifiedKFold(**self.config.validation.StratifiedKFold)
            path_to_save_data_ckpt = os.path.join(self.config.path.data_checkpoint,
                                    f"data_checkpoint_titanic.pickle")

        elif self.config.task.house_pricing:
            oof_preds = np.zeros(len(X))
            sgkf = KFold(**self.config.validation.KFold)
            path_to_save_data_ckpt = os.path.join(self.config.path.data_checkpoint,
                                                  f"data_checkpoint_pricing.pickle")

        with open(path_to_save_data_ckpt, "wb") as f:
            pickle.dump(self.dataset, f)

        accuracy = []
        roc_auc = []
        rmse = []

        Path(self.config.path.models).mkdir(parents=True, exist_ok=True)

        for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y)):

            fold_model = clone(model)

            X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

            fold_model.fit(X_train, y_train)

            Path(os.path.join(self.config.path.models, f"{model_name}")).mkdir(parents=True, exist_ok=True)

            path_to_save = os.path.join(self.config.path.models,
                                        f"{model_name}/{model_name}_{fold}.pickle")

            with open(path_to_save, "wb") as f:
                pickle.dump(fold_model, f)

            preds = fold_model.predict(X_val)

            if self.config.task.titanic:
                accuracy.append(accuracy_score(y_val, preds))
                roc_auc.append(roc_auc_score(y_val, preds))

            if self.config.task.house_pricing:
                path_oof = os.path.join(self.config.path.models, f"{model_name}/{model_name}_oof.npy")
                oof_preds[test_idx] = preds
                rmse.append(root_mean_squared_log_error(y_val, preds))

        if self.config.task.titanic:
            return np.asarray(accuracy).mean(), np.asarray(roc_auc).mean()
        elif self.config.task.house_pricing:
            np.save(path_oof, oof_preds)
            return np.asarray(rmse).mean()

    def inference_one_model(self, preds, model_name, x_test):

        cnt = 0
        preds[model_name] = np.zeros(len(x_test))
        path_model_ckpt = os.path.join(self.config.path.models, f"{model_name}")
        for file_path in Path(path_model_ckpt).iterdir():
            if 'oof' not in str(file_path):
                with open(file_path, 'rb') as f:
                    model = pickle.load(f)

                if self.config.task.house_pricing:
                    preds[model_name] += model.predict(x_test)
                elif self.config.task.titanic:
                    preds[model_name] += model.predict_proba(x_test)[:, 1]
                cnt += 1
        preds[model_name] /= cnt
        return preds

    def fit(self):
        if self.config.mode.LinearRegression:
            model = LinearRegression(**self.config.params.LinearRegression)
            self.model_name.append('LinearRegression')
            rmse = self.train_one_model(model, 'LinearRegression')
            print(rmse, 'LinearRegression', '\n')

        if self.config.mode.Lasso:
            model = Lasso(**self.config.params.Lasso)
            self.model_name.append('Lasso')
            rmse = self.train_one_model(model, 'Lasso')
            print(rmse, 'Lasso', '\n')

        if self.config.mode.Ridge:
            model = Ridge(**self.config.params.Ridge)
            self.model_name.append('Ridge')
            rmse = self.train_one_model(model, 'Ridge')
            print(rmse, 'Ridge', '\n')

        if self.config.mode.ElasticNet:
            model = ElasticNet(**self.config.params.ElasticNet)
            self.model_name.append('ElasticNet')
            rmse = self.train_one_model(model, 'ElasticNet')
            print(rmse, 'ElasticNet', '\n')

        if self.config.mode.KNN:
            if self.config.task.titanic:
                model = KNeighborsClassifier(**self.config.params.KNN)
                accuracy, roc_auc = self.train_one_model(model, 'KNN')
                print(accuracy, roc_auc, 'KNN', '\n')
            if self.config.task.house_pricing:
                model = KNeighborsRegressor(**self.config.params.KNN)
                self.model_name.append('KNN')
                rmse = self.train_one_model(model, 'KNN')
                print(rmse, 'KNN', '\n')

        if self.config.mode.DecisionTree:
            if self.config.task.titanic:
                model = DecisionTreeClassifier(**self.config.params.DecisionTreeClassifier)
                accuracy, roc_auc = self.train_one_model(model, 'DecisionTreeClassifier')
                print(accuracy, roc_auc, 'DeisionTreeClassifier', '\n')
            if self.config.task.house_pricing:
                model = DecisionTreeRegressor(**self.config.params.DecisionTreeClassifier)
                self.model_name.append('DecisionTreeRegressor')
                rmse = self.train_one_model(model, 'DecisionTreeRegressor')
                print(rmse, 'DeisionTreeRegressor', '\n')

        if self.config.mode.RandomForest:
            if self.config.task.titanic:
                model = RandomForestClassifier(**self.config.params.RandomForestClassifier)
                accuracy, roc_auc = self.train_one_model(model, 'RandomForestClassifier')
                print(accuracy, roc_auc, 'RandomForestClassifier', '\n')
            if self.config.task.house_pricing:
                model = RandomForestRegressor(**self.config.params.RandomForestClassifier)
                self.model_name.append('RandomForestRegressor')
                rmse = self.train_one_model(model, 'RandomForestRegressor')
                print(rmse, 'RandomForestRegressor', '\n')

        if self.config.mode.XGB:
            if self.config.task.titanic:
                model = XGBClassifier(**self.config.params.XGBClassifier)
                accuracy, roc_auc = self.train_one_model(model, 'XGBClassifier')
                print(accuracy, roc_auc, 'XGBClassifier', '\n')
            if self.config.task.house_pricing:
                model = XGBRegressor(**self.config.params.XGBClassifier)
                self.model_name.append('XGBRegressor')
                rmse = self.train_one_model(model, 'XGBRegressor')
                print(rmse, 'XGBRegressor', '\n')

        if self.config.mode.CatBoost:
            if self.config.task.titanic:
                model = CatBoostClassifier(**self.config.params.CatBoostClassifier)
                accuracy, roc_auc = self.train_one_model(model, 'CatBoostClassifier')
                print(accuracy, roc_auc, 'CatBoostClassifier', '\n')
            if self.config.task.house_pricing:
                model = CatBoostRegressor(**self.config.params.CatBoostClassifier)
                self.model_name.append('CatBoostRegressor')
                rmse = self.train_one_model(model, 'CatBoostRegressor')
                print(rmse, 'CatBoostRegressor', '\n')


        if self.config.mode.LGBM:
            if self.config.task.titanic:
                model = LGBMClassifier(**self.config.params.LGBMClassifier)
                accuracy, roc_auc = self.train_one_model(model, 'LGBMClassifier')
                print(accuracy, roc_auc, 'LGBMClassifier', '\n')
            if self.config.task.house_pricing:
                model = LGBMRegressor(**self.config.params.LGBMClassifier)
                self.model_name.append('LGBMRegressor')
                rmse = self.train_one_model(model, 'LGBMRegressor')
                print(rmse, 'LGBMRegressor', '\n')

        if self.config.mode.DNN:
            model = DNN(**self.config.params.DNN)
            accuracy, roc_auc = self.train_one_model(model, 'DNN')
            print(accuracy, roc_auc, 'DNN', '\n')

        if self.config.task.house_pricing:

            meta_features = []

            for name in self.model_name:
                oof_path = os.path.join(self.config.path.models, f"{name}/{name}_oof.npy")
                if os.path.exists(oof_path):
                    meta_features.append(np.load(oof_path))
                else:
                    print(f"Skip {name}. Missing")

            X_meta = np.column_stack(meta_features)

            data = pd.read_csv(self.config.path.train_pricing)
            y_meta = np.log(data['SalePrice'])

            meta_model = LinearRegression()

            meta_model.fit(X_meta, y_meta)

            Path(os.path.join(self.config.path.models, "StackingMetaModel")).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(self.config.path.models, "StackingMetaModel/meta_model.pickle"), "wb") as f:
                pickle.dump(meta_model, f)

    def predict(self):
        preds = {}
        try:
            if self.config.task.titanic:
                path_to_load_data_ckpt = os.path.join(self.config.path.data_checkpoint,
                                                  f"data_checkpoint_titanic.pickle")
            elif self.config.task.house_pricing:
                path_to_load_data_ckpt = os.path.join(self.config.path.data_checkpoint,
                                                      f"data_checkpoint_pricing.pickle")
            with open(path_to_load_data_ckpt, 'rb') as f:
                dataset = pickle.load(f)
        except:
            print("Trouble to unpickle dataset.pickle")
            return

        if self.config.task.titanic:
            x_test = pd.read_csv(self.config.path.test_titanic)
            passenger_ids = x_test['PassengerId']
        elif self.config.task.house_pricing:
            x_test = pd.read_csv(self.config.path.test_pricing)
            ids = x_test['Id']

        x_test = dataset.preprocessing(x_test)

        if self.config.mode.LinearRegression:
            preds = self.inference_one_model(preds, 'LinearRegression', x_test)

        if self.config.mode.Lasso:
            preds = self.inference_one_model(preds, 'Lasso', x_test)

        if self.config.mode.Ridge:
            preds = self.inference_one_model(preds, 'Ridge', x_test)

        if self.config.mode.ElasticNet:
            preds = self.inference_one_model(preds, 'ElasticNet', x_test)

        if self.config.mode.KNN:
            preds = self.inference_one_model(preds, 'KNN', x_test)

        if self.config.mode.DecisionTree:
            if self.config.task.titanic:
                preds = self.inference_one_model(preds, 'DecisionTreeClassifier', x_test)
            if self.config.task.house_pricing:
                preds = self.inference_one_model(preds, 'DecisionTreeRegressor', x_test)

        if self.config.mode.RandomForest:
            if self.config.task.titanic:
                preds = self.inference_one_model(preds, 'RandomForestClassifier', x_test)
            if self.config.task.house_pricing:
                preds = self.inference_one_model(preds, 'RandomForestRegressor', x_test)

        if self.config.mode.CatBoost:
            if self.config.task.titanic:
                preds = self.inference_one_model(preds, 'CatBoostClassifier', x_test)
            if self.config.task.house_pricing:
                preds = self.inference_one_model(preds, 'CatBoostRegressor', x_test)


        if self.config.mode.XGB:
            if self.config.task.titanic:
                preds = self.inference_one_model(preds, 'XGBClassifier', x_test)
            if self.config.task.house_pricing:
                preds = self.inference_one_model(preds, 'XGBRegressor', x_test)

        if self.config.mode.LGBM:
            if self.config.task.titanic:
                preds = self.inference_one_model(preds, 'LGBMClassifier', x_test)
            if self.config.task.house_pricing:
                preds = self.inference_one_model(preds, 'LGBMRegressor', x_test)

        if self.config.task.titanic:
            preds_df = pd.DataFrame(preds)

            corr_matrix = preds_df.corr()
            print(corr_matrix.round(3))

            high_corr = []
            cols = corr_matrix.columns
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    if corr_matrix.iloc[i, j] > 0.96:
                        high_corr.append(f"{cols[i]} и {cols[j]} ({corr_matrix.iloc[i, j]:.3f})")

            if high_corr:
                print("\nEnsemble check didn't pass! High model correlation")
                for pair in high_corr:
                    print(f" - {pair}")
            else:
                print("\nEnsemble check passed!")

        if self.config.task.titanic:
            mean_probabilities = np.mean(list(preds.values()), axis=0)
            final_predictions = (mean_probabilities >= self.config.inference.ensemble.threshold).astype(int)
            submission = pd.DataFrame({
                'PassengerId': passenger_ids,
                'Survived': final_predictions
            })
            submission.to_csv(self.config.path.output_titanic, index=False)
        elif self.config.task.house_pricing:
            if self.config.inference.ensemble.is_stacking:
                print("\nInference Stacking")

                meta_model_path = os.path.join(self.config.path.models, "StackingMetaModel/meta_model.pickle")
                with open(meta_model_path, "rb") as f:
                    meta_model = pickle.load(f)
                models_for_stacking = list(preds.keys())

                try:
                    meta_features_test = [preds[name] for name in models_for_stacking]
                    X_meta_test = np.column_stack(meta_features_test)

                    mean_predictions = meta_model.predict(X_meta_test)
                except KeyError as e:
                    print(
                        f"Stacking error: model {e} not found in preds.")
                    return

            else:
                print("\nInference Soft Voting")
                mean_predictions = np.mean(list(preds.values()), axis=0)

            final_predictions = np.expm1(mean_predictions)

            submission = pd.DataFrame({
                'Id': ids,
                'SalePrice': final_predictions
            })
            submission.to_csv(self.config.path.output_pricing, index=False)
            print(f"Файл успешно сохранен: {self.config.path.output_pricing}")