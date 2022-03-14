"""
Test Input Data function (pre-CNN)
"""

import unittest
import pandas as pd
from scipy import stats
import numpy as np
from load_train_data import split_data, make_data_loader
import torch

class TestSplitData(unittest.TestCase):
    """
    Test class for function: split_data(X, y)
    """
    def test_split_data(self):
        """
        Test whether X and y have identical number of examples
        """
        X = torch.rand(2000, 500, 20)
        y = torch.rand(1999, 1)
        with self.assertRaises(AssertionError):
          split_data(X,y)
        
        return None


suite = unittest.TestLoader().loadTestsFromTestCase(TestSplitData)
_ = unittest.TextTestRunner().run(suite)


class TestMakeDataLoader(unittest.TestCase):
  """
  Test class for function: make_dataloader(trainset, testset, batchsize)
  """
  def test_batchsize_float(self):
    """
    Test whether error is raised when batchsize is a float 
    """
    X = torch.rand(2000, 500, 20)
    y = torch.rand(2000, 1)
    trainset, testset = split_data(X, y)
    batchsize = 100.5

    with self.assertRaises(AssertionError):
          make_data_loader(trainset, testset, batchsize)

    return None


suite = unittest.TestLoader().loadTestsFromTestCase(TestMakeDataLoader)
_ = unittest.TextTestRunner().run(suite)
