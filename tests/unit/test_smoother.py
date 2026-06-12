from src.lesson.smoother import PredictionSmoother


def test_returns_none_until_warmup():
    s = PredictionSmoother(window=15)
    for _ in range(6):  # < window // 2
        s.update(0, 0.9)
    assert s.smoothed() is None
    s.update(0, 0.9)
    s.update(0, 0.9)
    assert s.smoothed() is not None


def test_modal_class_wins():
    s = PredictionSmoother(window=15)
    for _ in range(10):
        s.update(0, 0.9)  # 'A'
    for _ in range(5):
        s.update(1, 0.9)  # 'B'
    idx, _ = s.smoothed()
    assert idx == 0


def test_confidence_average_class_only():
    s = PredictionSmoother(window=10)
    for _ in range(6):
        s.update(0, 0.8)
    for _ in range(4):
        s.update(1, 0.2)
    idx, conf = s.smoothed()
    assert idx == 0
    assert abs(conf - 0.8) < 1e-6  # only the modal class's confidences averaged


def test_window_eviction():
    s = PredictionSmoother(window=4)
    s.update(1, 0.9)  # will be evicted
    for _ in range(4):
        s.update(0, 0.9)
    idx, _ = s.smoothed()
    assert idx == 0
