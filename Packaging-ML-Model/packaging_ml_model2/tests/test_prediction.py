# output from predict script not null
# output from predict script is str data type
# the output is Y for an example data


import pytest
from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset
from prediction_model.predict import generate_predictions


@pytest.fixture
def single_prediction():
    test_dataset = load_dataset(config.TEST_FILE)
    single_row = test_dataset[0:1]
    result = generate_predictions(single_row)
    return result


# we'll use fixtures from pytest --> fixtures are basically functions which are ran before execution of each test case , in this case single_prediction is our fixture
# so we basically want that single prediction to run before test case. we use @pytest.fixture decorator to make it fixture.


def test_single_pred_not_none(single_prediction):
    assert single_prediction is not None

def test_single_pred_str_type(single_prediction):
    assert isinstance(single_prediction.get('Predictions')[0],str)

def test_single_pred_validate(single_prediction):
    assert single_prediction.get('Predictions')[0] == 'Y'