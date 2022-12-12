import numpy as np
import pandas as pd
import random

class Node:
    def __init__(self, predicted_class):
        self.predictedClass = predicted_class
        self.featureIndex = 0
        self.threshold = 0
        self.left = None
        self.right = None