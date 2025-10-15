from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
import openml
import numpy as np
from scipy.sparse import issparse
import logging


logger = logging.getLogger('lcdb')



def config_prepipeline(X_feature):
    ### check feature type
    # dtypes = pd.concat([X_feature, y_label], axis=1).dtypes
    dtypes = X_feature.dtypes
    categorical = dtypes[(dtypes == 'object') | (dtypes == 'category')].index.tolist()
    numerical = dtypes[~((dtypes == 'object') | (dtypes == 'category'))].index.tolist()

    # imputer for numerical column
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('to_array', FunctionTransformer(lambda x: x.toarray() if hasattr(x, 'toarray') else x)),
    ])

    # imputer and onehot for categorical column
    transformers = []
    for col in categorical:
            transformers.append((f'onehot_{col}', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
        ]), [col]))

    # Combine numerical and categorical pipelines using ColumnTransformer
    preprocessor = ColumnTransformer(
        [('num', numerical_pipeline, numerical),] + transformers)
   
    return Pipeline([('preprocessor', preprocessor)])



def get_dataset(openmlid): 
    dataset = openml.datasets.get_dataset(openmlid, download_data=True, download_qualities=True)
    # Fetch the data and target (features and labels)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    print(f"Loading raw data from OpenML ID {openmlid}")
    return X, y



def get_class( kls ):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__( module )
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def get_outer_split(X, y, seed):
    test_samples_at_90_percent_training = int(X.shape[0] * 0.1)
    if test_samples_at_90_percent_training <= 5000:
        return train_test_split(X, y, train_size = 0.9, random_state=seed, stratify=y)
    else:
        return train_test_split(X, y, train_size = X.shape[0] - 5000, test_size = 5000, random_state=seed, stratify=y)


def get_inner_split(X, y, outer_seed, inner_seed):
    X_learn, X_test, y_learn, y_test = get_outer_split(X, y, outer_seed)
    
    validation_samples_at_90_percent_training = int(X_learn.shape[0] * 0.1)
    if validation_samples_at_90_percent_training <= 5000:
        X_train, X_valid, y_train, y_valid = train_test_split(X_learn, y_learn, train_size = 0.9, random_state=inner_seed, stratify=y_learn)
    else:
        logger.info(f"Creating sample with instances: {X_learn.shape[0] - 5000}")
        X_train, X_valid, y_train, y_valid = train_test_split(X_learn, y_learn, train_size = X_learn.shape[0] - 5000, test_size = 5000, random_state=inner_seed, stratify=y_learn)
                                                                                      
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def get_splits_for_anchor(X, y, anchor, outer_seed, inner_seed):
    if issparse(y):
        y = y.toarray()
    y = np.ravel(y)
    X_train, X_valid, X_test, y_train, y_valid, y_test = get_inner_split(X, y, outer_seed, inner_seed)
    if anchor > X_train.shape[0]:
        raise ValueError(f"Invalid anchor {anchor} when available training instances are only {X_train.shape[0]}.")
    return X_train[:anchor], X_valid, X_test, y_train[:anchor], y_valid, y_test



def get_entry_learner(learner_name, learner_params, X, y, anchor, outer_seed, inner_seed):
    
    # get learner
    learner_class = get_class(learner_name)
    learner_inst = learner_class(**learner_params)

    # create a random split based on the seed
    X_train, X_valid, X_test, y_train, y_valid, y_test = get_splits_for_anchor(X, y, anchor, outer_seed, inner_seed)

    # fit the model
    pipeline_inst = Pipeline([
            ('preprocessor', config_prepipeline(X_train)), 
            ('learner', learner_inst),
            ])

    pipeline_inst.fit(X_train, y_train)

    logger.info(f"Training ready. Obtaining predictions for {X_test.shape[0]} instances.")

    # compute predictions on train data
    y_hat_train = pipeline_inst.predict(X_train)
    train_error_rate = 1 - accuracy_score(y_train , y_hat_train)

    # compute predictions on validation data
    y_hat_valid = pipeline_inst.predict(X_valid)
    valid_error_rate = 1 - accuracy_score(y_valid , y_hat_valid)

    # compute predictions on test data
    y_hat_test = pipeline_inst.predict(X_test)
    test_error_rate = 1 - accuracy_score(y_test , y_hat_test)
   
    # compute info entry
    info = {
        "size_train": anchor,
        "outer_seed": outer_seed,
        "inner_seed": inner_seed,
        "training_error_rate": np.round(train_error_rate, 4),
        "validation_error_rate": np.round(valid_error_rate, 4),
        "test_error_rate": np.round(test_error_rate, 4),
    }
    
    return info
