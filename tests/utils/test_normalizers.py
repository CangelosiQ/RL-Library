import unittest
import numpy as np

from rl_library.utils.normalizer import RunningMeanStd


class TestRunningMeanStd(unittest.TestCase):

    def setUp(self) -> None:
        self.rms = RunningMeanStd(shape=(33,))

    def test_setUp(self):
        self.assertEqual(self.rms.mean.shape, np.zeros((33,)).shape)

    def update(self):
        batch = np.ones((64, 33))


if __name__ == '__main__':
    unittest.main()
