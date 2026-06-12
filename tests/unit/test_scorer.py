from src.lesson.scorer import Light, TrafficLightScorer


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


TARGET = 5


def _scorer(clock=None):
    return TrafficLightScorer(
        target_idx=TARGET, hold_seconds=1.0, clock=clock or (lambda: 0.0)
    )


def test_wrong_class_is_red():
    light, done = _scorer().evaluate(TARGET + 1, conf=0.99)
    assert light is Light.RED and not done


def test_correct_low_conf_is_red():
    light, _ = _scorer().evaluate(TARGET, conf=0.4)
    assert light is Light.RED


def test_correct_mid_conf_is_amber():
    light, _ = _scorer().evaluate(TARGET, conf=0.65)
    assert light is Light.AMBER


def test_correct_high_conf_is_green_not_yet_complete():
    light, done = _scorer().evaluate(TARGET, conf=0.9)
    assert light is Light.GREEN and not done


def test_completion_after_hold():
    clk = FakeClock()
    s = _scorer(clock=clk)
    light, done = s.evaluate(TARGET, 0.9)
    assert light is Light.GREEN and not done
    clk.advance(1.0)
    light, done = s.evaluate(TARGET, 0.9)
    assert light is Light.GREEN and done


def test_amber_resets_green_timer():
    clk = FakeClock()
    s = _scorer(clock=clk)
    s.evaluate(TARGET, 0.9)  # start green
    clk.advance(0.6)
    s.evaluate(TARGET, 0.65)  # drop to amber -> reset
    clk.advance(0.6)
    light, done = s.evaluate(TARGET, 0.9)
    assert light is Light.GREEN and not done  # timer restarted, < 1.0s elapsed


def test_no_completion_just_before_threshold():
    clk = FakeClock()
    s = _scorer(clock=clk)
    s.evaluate(TARGET, 0.9)
    clk.advance(1.0 - 1e-3)
    _, done = s.evaluate(TARGET, 0.9)
    assert not done
    clk.advance(1e-3)
    _, done = s.evaluate(TARGET, 0.9)
    assert done
