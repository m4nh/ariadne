import math
import cv2
import numpy as np
from scipy.stats import norm


class DistancePredictor(object):
    """ 
    Base class for distance Predictors

    Parameters
    ----------
    ariadne : Ariadne 
              object associated with a source image

    """

    def __init__(self, ariadne):
        self.ariadne = ariadne

    def computeScore(self, n1, n2):
        return 1.0


class DistancePredictorMaxDistance(DistancePredictor):
    """
    Distance predictor with max distance computation

    Parameters
    ----------
    ariadne : Ariadne 
        object associated with a source image

    max_distance : float
        max allowed distance

    bradford_coefficient : float
        c parameter of bradford distribudtion

    """

    def __init__(self, ariadne, max_distance=100, bradford_coefficient=0.2):
        super(DistancePredictorMaxDistance, self).__init__(ariadne)

        self.max_distance = max_distance
        self.bradford_coefficient = bradford_coefficient

    def computeScore(self, n1, n2):
        """
        Computes distance score of a target distance

        Parameters
        ----------
        n1 : int
            first node/region
        n2 : array
            second node/region
        """
        p1 = self.ariadne.graph.centroidPoint(n1)
        p2 = self.ariadne.graph.centroidPoint(n2)
        distance = np.linalg.norm(p1 - p2)
        norm_distance = distance / self.max_distance
        if norm_distance > 1:
            return 0.0
        else:
            c = self.bradford_coefficient
            return c / (math.log(1 + c) * (1 + c * norm_distance))
            # return bradford.pdf(norm_distance, c=self.bradford_coefficient)
