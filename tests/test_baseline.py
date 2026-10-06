import numpy as np

from src.config import CATEGORIES
from src.models.evaluate import compute_metrics
from src.nlp.baseline import train_and_evaluate


def test_baseline_training_is_reproducible(fixture_split):
    train, test = fixture_split
    a = train_and_evaluate(train, test, plot=False)
    b = train_and_evaluate(train, test, plot=False)
    for name in a:
        np.testing.assert_array_equal(a[name]["y_pred"], b[name]["y_pred"])
        assert a[name]["metrics"]["macro_f1"] == b[name]["metrics"]["macro_f1"]
    assert set(a) == {"logreg_ovr", "decision_tree", "gradient_boosting"}


def test_split_is_disjoint_and_covers_all(fixture_df, fixture_split):
    train, test = fixture_split
    assert not set(train.passage_id) & set(test.passage_id)
    assert len(train) + len(test) == len(fixture_df)


def test_logreg_learns_fixture_categories(fixture_split):
    train, test = fixture_split
    res = train_and_evaluate(train, test, plot=False)
    assert res["logreg_ovr"]["metrics"]["macro_f1"] > 0.7


def test_metrics_are_multilabel():
    y_true = np.array([[1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0]])
    y_pred = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
    m = compute_metrics(y_true, y_pred)
    assert set(m["per_class"]) == set(CATEGORIES)
    assert m["per_class"]["cyber"]["recall"] == 0.0
    assert m["subset_accuracy"] == 0.5


def test_multilabel_probabilities_are_independent(baseline_classifier):
    """Sigmoid/OvR outputs: a passage can score high on two categories at once."""
    text = (
        "Cyberattacks could compromise the security of our information systems. "
        "Climate change regulation could increase the cost of our operations. "
        "A data breach could expose confidential customer information. "
        "Carbon pricing mechanisms could increase our compliance costs."
    )
    probs = baseline_classifier.predict_proba([text])[0]
    assert probs.shape == (len(CATEGORIES),)
    assert probs.sum() > 1.0  # softmax would force this to 1
