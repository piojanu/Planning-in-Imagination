import numpy as np
import utils


class PreprocessingTest(np.testing.TestCase):
    def test_given_pong_state_should_return_preprocessed_float_vector(self):
        pong_state = np.zeros((210, 160, 3))
        pong_state[:35] = np.random.rand(35, 160, 3)   # Check top cropping
        pong_state[195:] = np.random.rand(15, 160, 3)  # Check bottom cropping
        pong_state[99] = np.ones((1, 160, 3)) * 144    # Check type 1 background erasing
        pong_state[119] = np.ones((1, 160, 3)) * 109   # Check type 2 background erasing
        vec = utils.preprocess_pong_state(pong_state)
        np.testing.assert_array_equal(vec, np.zeros(6400))
