import math
import numpy as np
import skimage.color
from scipy.stats import multivariate_normal
from scipy.stats import norm
import cv2

class VisualMatcher(object):

    def __init__(self, ariadne):
        self.ariadne = ariadne

    def computeCost(self, n1, n2):
        return 0.0


class SimpleColorMatcher(VisualMatcher):

    def __init__(self, ariadne):
        super(SimpleColorMatcher, self).__init__(ariadne)

    def computeCost(self, n1, n2):
        node1 = self.ariadne.graph.node(n1)
        node2 = self.ariadne.graph.node(n2)
        return np.linalg.norm(node1['mean color'] - node2['mean color'])


class LabHistogramMatcher(VisualMatcher):

    def __init__(self, ariadne, bins=[8, 8, 8], precompute_histograms=False):
        super(LabHistogramMatcher, self).__init__(ariadne)
        self.bins = bins

        self.max_distance = math.sqrt(np.sum(np.array(bins).ravel()))
        self.histogram_map = {}
        if precompute_histograms:
            nodes = self.ariadne.graph.nodes()
            for i, n in enumerate(nodes):
                region = self.ariadne.graph.region(n)
                histo = self.regionHistogram(region)
                self.histogram_map[region] = histo
                print("Histogram percentage", 100.0 *
                      float(i) / float(len(nodes)))

    def regionHistogram(self, region):
        if region in self.histogram_map:
            return self.histogram_map[region]

        colors_list = []
        coords = region['coords']
        for p1 in coords:
            color = self.ariadne.image[p1[0], p1[1], :]
            # colors.append(color)
            color_lab = skimage.color.rgb2lab(color.reshape((1, 1, 3)))
            colors_lab.append(color_lab)

        # colors = np.array(colors)
        colors_lab = np.array(colors_lab).reshape((len(coords), 3))
        l = colors_lab[:, 0]
        a = colors_lab[:, 1]
        b = colors_lab[:, 2]
        hl, _ = np.histogram(l, self.bins[0], range=(0, 100.0))
        ha, _ = np.histogram(a, self.bins[1], range=(-127., 128.))
        hb, _ = np.histogram(b, self.bins[2], range=(-127., 128.))

        hl = np.array(hl).astype(float)
        ha = np.array(ha).astype(float)
        hb = np.array(hb).astype(float)
        hl = hl / np.max(hl)
        ha = ha / np.max(ha)
        hb = hb / np.max(hb)
        histo = np.concatenate((hl, ha, hb))
        self.histogram_map[region] = histo
        return histo

    def computeCost(self, n1, n2):
        node1 = self.ariadne.graph.node(n1)
        node2 = self.ariadne.graph.node(n2)
        r1 = self.ariadne.graph.region(n1)
        r2 = self.ariadne.graph.region(n2)

        h1 = self.regionHistogram(r1)
        h2 = self.regionHistogram(r2)
        return np.linalg.norm(h1 - h2) / self.max_distance

        # return np.linalg.norm(node1['mean color'] - node2['mean color'])


class PathVisualMatcher(object):

    def __init__(self, ariadne, path):
        self.ariadne = ariadne
        self.path = path

    def computePDF(self, n1):
        return 0.0


class PathLabHistogramMatcher(PathVisualMatcher):

    def __init__(self, ariadne, path, bins=[8, 8, 8], precompute_histograms=False):
        super(PathLabHistogramMatcher, self).__init__(ariadne, path)
        self.bins = bins
        self.max_distance = math.sqrt(np.sum(np.array(bins).ravel()))
        self.histogram_map = {}
        if precompute_histograms:
            nodes = self.ariadne.graph.nodes()
            for i, n in enumerate(nodes):
                region = self.ariadne.graph.region(n)
                histo = self.regionHistogram(region)
                self.histogram_map[region] = histo
                # print(histo)
                print("Histogram percentage", 100.0 *
                      float(i) / float(len(nodes)))

    def regionHistogram(self, region):
        if region in self.histogram_map:
            return self.histogram_map[region]

        colors_lab = []
        coords = region['coords']
        for p1 in coords:
            color = self.ariadne.image[p1[0], p1[1], :]
            # colors.append(color)
            color_lab = skimage.color.rgb2lab(color.reshape((1, 1, 3)))
            colors_lab.append(color_lab)

        # colors = np.array(colors)
        colors_lab = np.array(colors_lab).reshape((len(coords), 3))
        l = colors_lab[:, 0]
        a = colors_lab[:, 1]
        b = colors_lab[:, 2]
        hl, _ = np.histogram(l, self.bins[0], range=(0, 100.0))
        ha, _ = np.histogram(a, self.bins[1], range=(-127., 128.))
        hb, _ = np.histogram(b, self.bins[2], range=(-127., 128.))

        # print("HISTOGRAMSSSSS", hl, ha, hb)

        hl = hl / float(len(coords))
        hb = hb / float(len(coords))
        ha = ha / float(len(coords))

        hl = np.array(hl).astype(float)
        ha = np.array(ha).astype(float)
        hb = np.array(hb).astype(float)

        # print("HISTOGRAMSSSSS ** ", hl, ha, hb)
        histo = np.concatenate((hl, ha, hb))
        self.histogram_map[region] = histo
        return histo

    def computePDF(self, n2):
        if self.path.size() == 0:
            return 1.0

        r1 = self.ariadne.graph.region(self.path.last_node)
        r2 = self.ariadne.graph.region(n2)

        h1 = self.regionHistogram(r1)
        h2 = self.regionHistogram(r2)
        mn = multivariate_normal(h1)
        return mn.pdf(h2)

        # return np.linalg.norm(node1['mean color'] - node2['mean color'])


class ColorPathHistogramMatcher(PathVisualMatcher):
    DESCRIPTOR_CEILAB = "DESCRIPTOR_CEILAB"
    DESCRIPTOR_BALLARD = "DESCRIPTOR_BALLARD"

    def __init__(self, ariadne, path, bins=[8, 8, 8], precompute_histograms=False, descriptor="DESCRIPTOR_CEILAB",instogram_intersection=True,single_part_normalization=False):
        super(ColorPathHistogramMatcher, self).__init__(ariadne, path)
        self.bins = bins
        self.max_distance = math.sqrt(np.sum(np.array(bins).ravel()))
        self.histogram_map = {}
        self.descriptor = descriptor
        self.histogram_intersection = instogram_intersection
        self.single_part_normalization = single_part_normalization

        if precompute_histograms:
            nodes = self.ariadne.graph.nodes()
            for i, n in enumerate(nodes):
                region = self.ariadne.graph.region(n)
                histo = self.regionHistogram(region)
                self.histogram_map[region] = histo
                # print(histo)
                print("Histogram percentage", 100.0 *
                      float(i) / float(len(nodes)))

    def colorFeature(self, color_input):
        if self.descriptor == ColorPathHistogramMatcher.DESCRIPTOR_CEILAB:
            return skimage.color.rgb2lab(color_input.reshape((1, 1, 3)))
        elif self.descriptor == ColorPathHistogramMatcher.DESCRIPTOR_BALLARD:
            ballard = np.array([
                color_input[0]+color_input[1]+color_input[2],
                color_input[0]-color_input[1],
                2.0*color_input[2]-color_input[0]-color_input[1]
            ])
            return ballard
    
    def colorFeatureRange(self):
        if self.descriptor == ColorPathHistogramMatcher.DESCRIPTOR_CEILAB:
            return [(0, 100.0),(-127., 128.),(-127., 128.)]
        elif self.descriptor == ColorPathHistogramMatcher.DESCRIPTOR_BALLARD:
            return [(0.0,3.0),(-1.0,1.0),(-2.0,2.0)]

    def regionHistogram(self, region):
        if region in self.histogram_map:
            return self.histogram_map[region]

        colors_list = []
        coords = region['coords']
        for p1 in coords:
            color = self.ariadne.image[p1[0], p1[1], :]
            # colors.append(color)
            #color_lab = skimage.color.rgb2lab(color.reshape((1, 1, 3)))
            colors_list.append(self.colorFeature(color))

        # colors = np.array(colors)
        colors_list = np.array(colors_list).reshape((len(coords), 3))
        l = colors_list[:, 0]
        a = colors_list[:, 1]
        b = colors_list[:, 2]

        ranges = self.colorFeatureRange()
        hl, _ = np.histogram(l, self.bins[0], range=ranges[0])
        ha, _ = np.histogram(a, self.bins[1], range=ranges[1])
        hb, _ = np.histogram(b, self.bins[2], range=ranges[2])

        # print("HISTOGRAMSSSSS", hl, ha, hb)

        if self.single_part_normalization:
            hl = hl / float(len(coords))
            hb = hb / float(len(coords))
            ha = ha / float(len(coords))

        hl = np.array(hl).astype(float)
        ha = np.array(ha).astype(float)
        hb = np.array(hb).astype(float)

        # print("HISTOGRAMSSSSS ** ", hl, ha, hb)
        histo = np.concatenate((hl, ha, hb))
        self.histogram_map[region] = histo
        return histo


    def return_intersection(self,hist_1, hist_2):
        minima = np.minimum(hist_1, hist_2)
        intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
        return intersection

    def computePDF(self, n2):
        if self.path.size() == 0:
            return 1.0

        r1 = self.ariadne.graph.region(self.path.last_node)
        r2 = self.ariadne.graph.region(n2)

        h1 = self.regionHistogram(r1)
        h2 = self.regionHistogram(r2)
        mn = multivariate_normal(h1)
        if self.histogram_intersection:
            return self.return_intersection(h1,h2)
        else:
            return  mn.pdf(h2)

        # return np.linalg.norm(node1['mean color'] - node2['mean color'])


class Color2DHistogramMatcher(PathVisualMatcher):
    DESCRIPTOR_CEILAB = "DESCRIPTOR_CEILAB"
    DESCRIPTOR_BALLARD = "DESCRIPTOR_BALLARD"

    def __init__(self, ariadne,  bins=[32,32], precompute_histograms=False, descriptor="DESCRIPTOR_CEILAB",instogram_intersection=True,single_part_normalization=False):
        super(Color2DHistogramMatcher, self).__init__(ariadne, None)
        
        self.img = (ariadne.image * 255.0).astype(np.uint8)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
        self.hsv = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)
        self.h, self.s, self.v = cv2.split(self.hsv)

        self.bins = bins
        self.max_distance = math.sqrt(np.sum(np.array(bins).ravel()))
        self.histogram_map = {}
        self.descriptor = descriptor
        self.histogram_intersection = instogram_intersection
        self.single_part_normalization = single_part_normalization

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
        hist = cv2.calcHist(
            [self.hsv], 
            [0, 1], 
            self.ariadne.graph.maskImage(n), 
            [32, 32], 
            [0, 180, 0, 256]
            )
        hist = hist / np.sum(hist.ravel())
        return hist


    def histogramComparison(self,h1, h2):
        return cv2.compareHist(h1, h2, cv2.HISTCMP_INTERSECT)
        #return cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
    

    def compare(self, n1,n2):
        h1 = self.regionHistogram(n1)
        h2 = self.regionHistogram(n2)
        return self.histogramComparison(h1,h2)

    def normalizeComparison(self,current_value,max_value):
        dx = math.fabs(current_value-max_value)

        delta = 0.05
        p1 = norm.pdf(dx+delta,scale=0.02)
        p2 = norm.pdf(dx-delta,scale=0.02)
        return (delta / 2.0) * (p1 + p2)
        #return math.exp(-((current_value - max_value)**2.0)/(2*(0.2**2)))

