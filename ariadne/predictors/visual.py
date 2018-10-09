import math
import cv2
import numpy as np
from scipy.stats import norm
from scipy.stats import uniform


class VisualPredictor(object):
    """ 
    Base class for visual Predictors

    Parameters
    ----------
    ariadne : Ariadne 
              object associated with a source image

    """

    def __init__(self, ariadne):
        self.ariadne = ariadne

    def computeScore(self, n1, n2, reference_value=None):
        return 1.0


class VisualPredictorColor2DHistogram(VisualPredictor):
    """
    2D Histogram visual predictor

    Parameters
    ----------
    ariadne : Ariadne 
        object associated with a source image

    bins : Array
        Bins for each dimension, default: [32,32]

    normalization_sigma : float
        Sigma for normal distribution of histogram distances

    """

    def __init__(self, ariadne, bins=[8, 8, 8], normalization_sigma=0.4, enable_cache=False):
        super(VisualPredictorColor2DHistogram, self).__init__(ariadne)

        self.img = (ariadne.image * 255.0).astype(np.uint8)
        #self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
        self.hsv = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)
        self.h, self.s, self.v = cv2.split(self.hsv)

        self.bins = bins
        self.normalization_sigma = normalization_sigma
        self.histogram_map = {}
        self.enable_cache = enable_cache

        # if precompute_histograms:
        #     nodes = self.ariadne.graph.nodes()
        #     for i, n in enumerate(nodes):
        #         region = self.ariadne.graph.region(n)
        #         histo = self.regionHistogram(region)
        #         self.histogram_map[region] = histo
        #         # print(histo)
        #         print("Histogram percentage", 100.0 *
        #               float(i) / float(len(nodes)))

    def regionHistogram(self, n):
        """
        Computes histogram for the region with node id 'n'

        Parameters
        ----------
        n : int
            node/region index

        """
        if self.enable_cache:
            if n in self.histogram_map:
                return self.histogram_map[n]

        hist = cv2.calcHist(
            [self.hsv],
            [0, 1, 2],
            self.ariadne.graph.maskImage(n),
            self.bins,
            [0, 180, 0, 256, 0, 256]
        )
        hist = hist / np.sum(hist.ravel())

        if self.enable_cache:
            self.histogram_map[n] = hist
        return hist

    def histogramComparison(self, h1, h2):
        """
        Comparison function between two histograms

        Parameters
        ----------
        h1 : array
            first histogram
        h2 : array
            second histogram

        """

        return cv2.compareHist(h1, h2, cv2.HISTCMP_INTERSECT)

    def computeNormalProbability(self, value):
        """
        Computes an approximation of the probability in a normal distribution
        """
        delta = 0.05
        p1 = norm.pdf(value + delta, scale=self.normalization_sigma)
        p2 = norm.pdf(value - delta, scale=self.normalization_sigma)
        return (delta / 2.0) * (p1 + p2)

    def computeScore(self, n1, n2, reference_value=None):
        """
        Computes visual (match) score between two nodes/regions

        Parameters
        ----------
        n1 : int
            first node/region
        n2 : array
            second node/region
        reference_value : float
            reference value to compute the normal probability for the target score
        """

        h1 = self.regionHistogram(n1)
        h2 = self.regionHistogram(n2)
        score = self.histogramComparison(h1, h2)
        if reference_value is None:
            return score
        else:
            return self.computeNormalProbability(math.fabs(score - reference_value))


class VisualPredictorColor3DHistogram(VisualPredictor):
    """
    3D Histogram visual predictor

    Parameters
    ----------
    ariadne : Ariadne 
        object associated with a source image

    bins : Array
        Bins for each dimension, default: [32,32]

    normalization_sigma : float
        Sigma for normal distribution of histogram distances

    """

    def __init__(self, ariadne, bins=[8, 8, 8], distribution_parameters=[0.5], enable_cache=False):
        super(VisualPredictorColor3DHistogram, self).__init__(ariadne)

        self.img = (ariadne.image * 255.0).astype(np.uint8)
        #self.hsv = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
        self.hsv = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)
        self.h, self.s, self.v = cv2.split(self.hsv)

        self.bins = bins
        self.distribution_parameters = distribution_parameters
        self.histogram_map = {}
        self.enable_cache = enable_cache

    def regionHistogram(self, n):
        """
        Computes histogram for the region with node id 'n'

        Parameters
        ----------
        n : int
            node/region index

        """
        if self.enable_cache:
            if n in self.histogram_map:
                return self.histogram_map[n]

        hist = cv2.calcHist(
            [self.hsv],
            [0, 1, 2],
            self.ariadne.graph.maskImage(n),
            self.bins,
            [0, 180, 0, 256, 0, 256]
        )
        hist = hist / np.sum(hist.ravel())

        if self.enable_cache:
            self.histogram_map[n] = hist
        return hist

    def histogramComparison(self, h1, h2):
        """
        Comparison function between two histograms

        Parameters
        ----------
        h1 : array
            first histogram
        h2 : array
            second histogram

        """

        return cv2.compareHist(h1, h2, cv2.HISTCMP_INTERSECT)

    def computeNormalProbability(self, value):
        """
        Computes the probability of histogram differences in a custom distribution
        """
        c = self.distribution_parameters[0]
        return c / (math.log(1 + c) * (1 + c * (1.0 - value)))
        # return bradford.pdf(1.0 - value, c=self.distribution_parameters[0])

    def computeScore(self, n1, n2, reference_value=None):
        """
        Computes visual (match) score between two nodes/regions

        Parameters
        ----------
        n1 : int
            first node/region
        n2 : array
            second node/region
        reference_value : float
            reference value to compute the normal probability for the target score
        """

        h1 = self.regionHistogram(n1)
        h2 = self.regionHistogram(n2)
        score = self.histogramComparison(h1, h2)
        return self.computeNormalProbability(score)
