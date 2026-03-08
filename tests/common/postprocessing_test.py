import numpy as np
from fuzzy_cnn.common.postprocessing import postprocess, softmax

def test_softmax_sums_to_one():
    logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    probs = softmax(logits)

    assert probs.shape == (10,)
    assert np.allclose(probs.sum(), 1.0)
    assert (probs >= 0).all()  # no negative probabilities

def test_softmax_numerical_stability():
    # Check that really big logits don't overflow
    logits = np.array([1000.0, 1001.0, 1002.0, 999.0, 998.0, 997.0, 996.0, 995.0, 994.0, 993.0])
    probs = softmax(logits)

    assert not np.any(np.isnan(probs))  # no NaN from overflow
    assert np.allclose(probs.sum(), 1.0)

def test_postprocess_returns_top_k():
    logits = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    result = postprocess(logits, top_k=3)

    assert len(result) == 3
    assert result[0]["prob"] >= result[1]["prob"] >= result[2]["prob"]

def test_postprocess_correct_labels():
    # Make one clear winner for the logit
    logits = np.zeros(10)
    logits[3] = 100.0  # index 3 = "cat"
    result = postprocess(logits, top_k=1)

    assert result[0]["label"] == "cat"