import numpy as np
import torch
from torch.autograd import Variable

from model import PolicyGradientModel

torch.manual_seed(1)


class PolicyGradientModelTest(np.testing.TestCase):
    def setUp(self):
        self.pgm = PolicyGradientModel(input_size=5, hidden_size=3, output_size=2)
        self.pgm_summary = ['Linear(in_features=5, out_features=3)',
                            'ReLU()',
                            'Linear(in_features=3, out_features=2)',
                            'Softmax()']

    def test_model_has_correct_architecture(self):
        for layer, correct_layer in zip(self.pgm.model.children(), self.pgm_summary):
            self.assertEqual(repr(layer), correct_layer)

    def test_forward_propagation_on_batch_of_zeroes_works_correctly(self):
        x = Variable(torch.zeros(2, 5))
        out = self.pgm.forward(x).data.numpy()
        np.testing.assert_array_equal(out.shape, [2, 2])
        np.testing.assert_array_equal(out[0], out[1])
        np.testing.assert_almost_equal(out[0].sum(), 1)
