import category_encoders as ce
import pandas as pd
from sklearn.model_selection import StratifiedKFold


class TargetEncodingCallback:

    def __init__(self, cat_feat_to_encode, folds=5):
        self.cat_feat_to_encode = cat_feat_to_encode
        self.folds = folds

    def before_general_data_processing(self, learner_context, X, X_test, holdout_frac, num_bagging_folds, label):
        oof = pd.DataFrame([])
        y = X[label]
        for tr_idx, oof_idx in StratifiedKFold(n_splits=self.folds, random_state=0, shuffle=True).split(X, y):
            encoder = ce.TargetEncoder(cols=self.cat_feat_to_encode)
            encoder.fit(X[self.cat_feat_to_encode].iloc[tr_idx, :], y.iloc[tr_idx])
            oof = oof.append(encoder.transform(X[self.cat_feat_to_encode].iloc[oof_idx, :]), ignore_index=False)
        predict_encoder = ce.TargetEncoder(cols=self.cat_feat_to_encode)
        predict_encoder.fit(X[self.cat_feat_to_encode], y)
        learner_context['dataset_target_encoder'] = predict_encoder
        oof = oof.loc[X.index.values].add_suffix('_mean')
        X = pd.concat([X, oof], axis=1)
        return learner_context, X, X_test, holdout_frac, num_bagging_folds, label

    def before_predict_proba(self, learner_context, X_test, model):
        encoder = learner_context['dataset_target_encoder']
        X_test_encoded = encoder.transform(X_test[self.cat_feat_to_encode]).add_suffix('_mean')
        X_test = pd.concat([X_test, X_test_encoded], axis=1)
        return learner_context, X_test, model

    def before_get_feature_importance(self, learner_context, model, X, y, features, raw, subsample_size, silent, label):
        _, X, _ = self.before_predict_proba(learner_context, X, model)
        return learner_context, model, X, y, features, raw, subsample_size, silent, label
