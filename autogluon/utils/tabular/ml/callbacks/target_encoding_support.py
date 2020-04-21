import category_encoders as ce

from autogluon.utils.tabular.ml.models.lgb.hyperparameters.parameters import MULTICLASS


class TargetEncodingCallback:

    def __init__(
            self,
            low_cat_counts_threshold: int = 20,
            max_cat_uniq_values_frac: float = 0.20,
            min_samples_leaf: int = 20,
            smoothing: float = 1.0,
    ):
        """

        Parameters
        ----------
        low_cat_counts_threshold
            min number of categories for feature to be considered for target encoding
        max_cat_uniq_values_frac
            max ratio of number of categories to total len of the dataset (filter-out extremely high cardinality features)
        min_samples_leaf: int
            minimum samples to take category average into account.
        smoothing: float
            smoothing effect to balance categorical average vs prior. Higher value means stronger regularization.
            The value must be strictly bigger than 0.
        """
        self.low_cat_counts_threshold = low_cat_counts_threshold
        self.max_cat_uniq_values_frac = max_cat_uniq_values_frac
        self.min_samples_leaf = min_samples_leaf
        self.smoothing = smoothing

    # Callback method
    def stacker_preprocessing_config(self, context):
        # Disable stacker data preprocessing - we have fold-specific pre-processing - it is not cacheable.
        context['preprocess_in_stacker'] = False
        context['preprocess_in_model'] = False
        return context

    # Callback method
    def before_folds(self, training_context, X, y):
        feature_types_metadata = training_context['feature_types_metadata']
        cols_to_encode = feature_types_metadata.get('object', [])
        uniq_counts = X[cols_to_encode].nunique()
        uniq_frac = uniq_counts / len(X)
        cats_count_above_min = uniq_counts[uniq_counts > self.low_cat_counts_threshold]  # Ignore low-cardinality categories: <20 values
        cats_frac_below_max = uniq_frac[uniq_frac < self.max_cat_uniq_values_frac]  # Ignore categories with >20% unique values
        cols_to_encode = [c for c in cols_to_encode if (c in cats_count_above_min) and (c in cats_frac_below_max)]
        training_context['cats_for_target_encoding'] = cols_to_encode
        print(f'Categories targeted for target encoding: {cols_to_encode}')

        return training_context, X, y

    # Callback method
    def before_fold_fit(self, training_context, fold_context, fold_model, X_train, y_train, X_test, y_test):
        X_train = X_train.copy()
        X_test = X_test.copy()

        cols_to_encode = training_context['cats_for_target_encoding']
        fold_context['cols_to_encode'] = cols_to_encode

        if training_context['problem_type'] == MULTICLASS:
            classes = y_train.value_counts()[:-1].index.values  # Drop last one: it is redundant
            fold_context['classes'] = classes
            encoders = []
            for i, cls in enumerate(classes):
                # TODO: refactor duplicated code
                y_cls = (y_train == cls).astype(int)  # one vs all
                encoder = ce.TargetEncoder(cols=cols_to_encode, min_samples_leaf=self.min_samples_leaf)  # , smoothing=5)
                encoder.fit(X_train[cols_to_encode], y_cls)
                encoders.append(encoder)
                X_train_encoded = encoder.transform(X_train[cols_to_encode]).add_suffix(f'_{i}')
                X_test_encoded = encoder.transform(X_test[cols_to_encode]).add_suffix(f'_{i}')
                for c in X_train_encoded.columns:
                    X_train[c] = X_train_encoded[c]
                    X_test[c] = X_test_encoded[c]

            # Drop original columns
            X_train = X_train.drop(columns=cols_to_encode)
            X_test = X_test.drop(columns=cols_to_encode)
            fold_context['encoders'] = encoders
        else:
            encoder = ce.TargetEncoder(cols=cols_to_encode, min_samples_leaf=self.min_samples_leaf)  # , smoothing=5)
            fold_context['encoder'] = encoder
            encoder.fit(X_train, y_train)
            X_train_encoded = encoder.transform(X_train)[cols_to_encode]
            X_test_encoded = encoder.transform(X_test)[cols_to_encode]
            for c in X_train_encoded.columns:
                X_train[c] = X_train_encoded[c]
                X_test[c] = X_test_encoded[c]

        X_train = fold_model.preprocess(X_train)
        X_test = fold_model.preprocess(X_test)

        return training_context, fold_context, fold_model, X_train, y_train, X_test, y_test

    # Callback method
    def before_fold_predict_proba(self, training_context, fold_context, model, X):
        X = X.copy()
        if training_context['problem_type'] == MULTICLASS:
            if ('encoders' in fold_context) & ('classes' in fold_context) & ('cols_to_encode' in fold_context):
                classes = fold_context['classes']
                encoders = fold_context['encoders']
                cols_to_encode = fold_context['cols_to_encode']
                for i, c in enumerate(classes):
                    encoder = encoders[i]
                    X_encoded = encoder.transform(X[cols_to_encode])
                    X = X.join(X_encoded, rsuffix=f'_{i}')
                X = X.drop(columns=cols_to_encode)
        else:
            if ('encoder' in fold_context) & ('cols_to_encode' in fold_context):
                X_train_encoded = fold_context['encoder'].transform(X)
                for c in fold_context['cols_to_encode']:
                    X[c] = X_train_encoded[c]
        X = model.preprocess(X)

        return training_context, fold_context, model, X
