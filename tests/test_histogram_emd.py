import unittest

import torch

from util.loss_functions import HistogramEMDLoss


class HistogramEMDLossTest(unittest.TestCase):
    def test_identical_images_have_zero_loss(self):
        image = torch.rand(2, 1, 8, 8)
        loss = HistogramEMDLoss()(image, image)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_unit_intensity_shift_has_unit_emd(self):
        zeros = torch.zeros(1, 1, 4, 4)
        ones = torch.ones(1, 1, 4, 4)
        loss = HistogramEMDLoss()(zeros, ones)
        self.assertAlmostEqual(loss.item(), 1.0, places=5)

    def test_batch_matches_individual_samples(self):
        torch.manual_seed(7)
        prediction = torch.rand(2, 1, 8, 8)
        target = torch.rand(2, 1, 8, 8)
        criterion = HistogramEMDLoss(reduction="none")
        batch_loss = criterion(prediction, target)
        individual_loss = torch.stack(
            [criterion(prediction[i : i + 1], target[i : i + 1])[0] for i in range(2)]
        )
        self.assertTrue(torch.allclose(batch_loss, individual_loss, atol=1e-6))

    def test_gradient_is_finite(self):
        prediction = torch.rand(2, 1, 8, 8, requires_grad=True)
        target = torch.rand(2, 1, 8, 8)
        loss = HistogramEMDLoss()(prediction, target)
        loss.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())


if __name__ == "__main__":
    unittest.main()
