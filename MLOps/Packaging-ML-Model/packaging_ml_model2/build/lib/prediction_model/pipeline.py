from sklearn.pipeline import Pipeline
from prediction_model.config import config
import prediction_model.processing.preprocessing as pp
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

classification_pipeline = Pipeline(
    [
        ('Mean Imputation',pp.MeanImputter(variables = config.NUM_FEATURES)),
        ('Mode Imputation',pp.ModeImputter(variables = config.CAT_FEATURES)),
        ('DomainProcessing',pp.DomainProcessing(variables_to_modify=config.FEATURE_TO_MODIFY, variables_to_add=config.FEATURE_TO_ADD)),
        ('Drop Features',pp.DropColumns(variables_to_drop=config.DROP_FEATURES)),
        ('Label Encoder', pp.CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE)),
        ('LogTransform',pp.LogTransforms(variables=config.LOG_FEATURES)),
        ('MinMaxScaler',MinMaxScaler()),
        ('LogisticClassifier',LogisticRegression(random_state=0))

    ]
)