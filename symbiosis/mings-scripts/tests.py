import unittest
from main import get_hs_bins
import numpy as np

class TestHSHistogram(unittest.TestCase):

    def test_red(self):
        # test an all red image
        image = np.array([[[255, 0, 0] for i in range(100)] for j in range(100)])

        # expect full saturation
        saturation_bins = [0] * 9 + [1]
        # expect 0 degree hue
        hue_bins = [1] + [0] * 9

        expected_output = hue_bins + saturation_bins

        self.assertCountEqual(expected_output, get_hs_bins(image))

    def test_green(self):
        # test an all red image
        image = np.array([[[0, 255, 0] for i in range(100)] for j in range(100)])

        # expect full saturation
        saturation_bins = [0] * 9 + [1]
        # expect 0 degree hue
        hue_bins = [0] * 10
        # 0.3333 is hue for green
        hue_bins[3] = 1

        expected_output = hue_bins + saturation_bins

        self.assertCountEqual(expected_output, get_hs_bins(image))

    def test_random(self):
        for i in range(20):
            # run multiple times to test

            # get a random image
            image = np.random.randint(0, 255, (100, 100, 3))

            output = get_hs_bins(image)

            # check that the values sum to 1 since we normalize the hues and saturations
            self.assertAlmostEqual(output[:10].sum(), 1)
            self.assertAlmostEqual(output[10:].sum(), 1)
