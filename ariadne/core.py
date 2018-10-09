#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from skimage.segmentation import slic, felzenszwalb, quickshift, watershed
from skimage.measure import regionprops
from skimage import draw

import skimage.future.graph as graph


from skimage.segmentation import mark_boundaries


from skimage.util import img_as_float
from skimage.color import rgb2gray
import skimage.color
from skimage.filters import sobel

from skimage import io

import scipy.io

import matplotlib.pyplot as plt
import argparse
import math
import numpy as np
import cv2
import os

import networkx as nx
import itertools


class ImageSegmentator(object):
    SEGMENTATOR_SLIC = "SLIC"
    SEGMENTATOR_SLIC_MONO = "SLIC_MONO"
    SEGMENTATOR_QUICKSHIFT = "QUICKSHIFT"
    SEGMENTATOR_FELZENSZWALB = "FELZENSZWALB"
    SEGMENTATOR_WATERSHED = "WATERSHED"

    def __init__(self, segmentator_type=None):
        if segmentator_type is None:
            segmentator_type = ImageSegmentator.SEGMENTATOR_SLIC
        self.segmentator_type = segmentator_type
        self.options_map = {}

    def segments(self, image, options=None):
        if options is None:
            options = self.options_map
        print("ACTIVE OPTIONS", options)
        if self.segmentator_type == ImageSegmentator.SEGMENTATOR_SLIC:
            return slic(
                image,
                sigma=options['sigma'],
                n_segments=options['n_segments'],
                convert2lab=True,
                multichannel=True,
                compactness=options['compactness']
            )
        elif self.segmentator_type == ImageSegmentator.SEGMENTATOR_SLIC_MONO:
            gray = skimage.color.rgb2grey(image)
            return slic(
                gray,
                sigma=options['sigma'],
                n_segments=options['n_segments'],
                convert2lab=False,
                multichannel=True,
                compactness=options['compactness']
            )
        elif self.segmentator_type == ImageSegmentator.SEGMENTATOR_QUICKSHIFT:
            return quickshift(
                image,
                kernel_size=3,
                max_dist=6,
                ratio=0.5
            )
        elif self.segmentator_type == ImageSegmentator.SEGMENTATOR_FELZENSZWALB:
            return felzenszwalb(image, scale=100, sigma=0.5, min_size=50)

        if self.segmentator_type == ImageSegmentator.SEGMENTATOR_WATERSHED:
            gradient = sobel(rgb2gray(image))
            return watershed(gradient, markers=options['n_segments'], compactness=0.001)


class AriadneGraph(object):

    def __init__(self, image, segments):
        self.image = image
        self.segments = segments
        self.labels = self.segments + 1
        self.regions = regionprops(self.labels)
        self.graph = graph.rag_mean_color(self.image, self.labels)

        #######################################
        # Centroid Mapping
        #######################################
        for region in self.regions:
            self.graph.node[region['label']]['centroid'] = region['centroid']
            self.graph.node[region['label']]['centroid_point'] = np.array([
                region['centroid'][1],
                region['centroid'][0]
            ], dtype=float)

    def generateBoundaryImage(self, image=None, color=(1, 1, 0)):
        if image is None:
            image = self.image
        return mark_boundaries(image, self.labels, color=color)

    def generateEdgesImage(self, image, threshold=np.inf):
        if image is None:
            image = self.image
        for edge in self.graph.edges():
            n1, n2 = edge

            r1, c1 = map(int, self.graph.node[n1]['centroid'])
            r2, c2 = map(int, self.graph.node[n2]['centroid'])

            p1 = np.array([c1, r1], dtype=float)
            p2 = np.array([c2, r2], dtype=float)

            line = draw.line(r1, c1, r2, c2)
            circle = draw.circle(r1, c1, 3)

            if self.graph[n1][n2]['weight'] < threshold:
                image[line] = 0, 0, 0
            image[circle] = 0, 0, 0
        return image

    def generateEdgesImageCV(self, image, threshold=np.inf):
        if image is None:
            image = self.image
        for edge in self.graph.edges():
            n1, n2 = edge

            r1, c1 = map(int, self.graph.node[n1]['centroid'])
            r2, c2 = map(int, self.graph.node[n2]['centroid'])

            p1 = np.array([c1, r1], dtype=int)
            p2 = np.array([c2, r2], dtype=int)

            cv2.line(image, tuple(p1), tuple(p2), np.array([110, 122, 84.]), 3)
            cv2.circle(image, tuple(p1), 10, np.array([56, 38, 50.]), -1)

        return image

    def generateHNodesImage(self, image, nodes=[], radius=5, color=(255, 0, 0), thickness=-1, showNumbers=False):
        for i, n in enumerate(nodes):
            node = self.graph.node[n]
            p = node['centroid_point'].astype(int)
            cv2.circle(
                image,
                tuple(p),
                radius=radius,
                color=color,
                thickness=thickness
            )
            if showNumbers:
                cv2.putText(
                    image,
                    "{}".format(i),
                    tuple(p + np.array([5, 0])),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.5,
                    color,
                    1
                )
        return image

    def nearestNode(self, point, match_region_points=True):
        if match_region_points:
            ipoint = point.astype(int)
            label = self.labels[ipoint[1], ipoint[0]]
            return label
        else:
            min_dist = 10000000.0
            min_region = None
            for n in self.graph.nodes():
                node = self.graph.node[n]
                region_center = np.array([
                    node['centroid'][1],
                    node['centroid'][0]
                ], dtype=float)
                dist = np.linalg.norm(region_center - point)
                if dist < min_dist:
                    min_dist = dist
                    min_region = n
            return min_region

    def adjacentNodes(self, input_nodes, excluded_nodes=[]):
        node_map = set()
        for input_node in input_nodes:
            for n in self.graph.neighbors(input_node):
                if n not in excluded_nodes:
                    node_map.add(n)
        return list(node_map)

    def getNeighbourhood(self, target_node):
        neigh = self.adjacentNodes([target_node])
        neigh2 = self.adjacentNodes(neigh, excluded_nodes=neigh + [target_node])
        return neigh + neigh2

    def deepNeighbourhood(self, target_node, level=2, forced_excluded_nodes=[], include_start_seed=False, as_simple_list=False):

        seed = [target_node]
        excluded_nodes = []
        founds = []
        if include_start_seed:
            founds = [[target_node]]

        excluded_nodes = [target_node] + forced_excluded_nodes
        while level > 0:
            neigh = self.adjacentNodes(seed, excluded_nodes=excluded_nodes)
            founds.append(neigh)
            seed = neigh
            excluded_nodes = [target_node] + forced_excluded_nodes
            for f in founds:
                excluded_nodes.extend(f)
            level -= 1
        if not as_simple_list:
            return founds
        else:
            return list(itertools.chain.from_iterable(founds))

    def getBoundaryType(self, n):
        node = self.node(n)
        mask = self.maskImage(n)
        indices = np.array(np.nonzero(mask)).T
        rows = indices[:, 0]
        cols = indices[:, 1]

        bt = ''

        max_row = mask.shape[0] - 1
        max_col = mask.shape[1] - 1

        if 0 in rows:
            bt += 'N'
        if max_col in cols:
            bt += 'W'
        if max_row in rows:
            bt += 'S'
        if 0 in cols:
            bt += 'E'

        return bt

    def isBoundary(self, n):
        return len(self.getBoundaryType(n)) > 0

    def nodes(self):
        return self.graph.nodes()

    def node(self, n):
        return self.graph.node[n]

    def centroidPoint(self, n):
        node = self.node(n)
        return node['centroid_point']

    def region(self, n):
        return self.regions[n - 1]

    def maskImage(self, n):
        mask = np.zeros(self.labels.shape, dtype=np.uint8)
        mask[self.labels == n] = 255
        return mask


class AriadnePathTip(object):

    def __init__(self, position=None, orientation=None, points=None):
        self.position = np.array([0.0, 0.0])
        self.orientation = 0.0
        if points is not None:
            p1 = np.array(points[0])
            p2 = np.array(points[1])
            direction = p1 - p2
            self.position = p1
            self.orientation = math.atan2(-direction[1], direction[0])
        else:
            self.position = np.array(position)
            self.orientation = np.array(orientation)

    def getDirection(self):
        return np.array([
            math.cos(self.orientation),
            -math.sin(self.orientation)
        ])

    def draw(self, image, color=(255, 0, 255), thickness=4, size=30):
        direction = self.getDirection()
        p2 = self.position + direction * size
        cv2.arrowedLine(image, tuple(self.position), tuple(p2.astype(int)), color, thickness, tipLength=0.5)


class AriadnePath(object):

    def __init__(self, ariadne=None, raw_points=np.array([])):
        self.ariadne = ariadne
        self.subgraph = nx.DiGraph()
        self.last_node = None
        self.first_node = None

        self.raw_points = []
        if self.ariadne is None:
            if raw_points.ndim == 1:
                raw_points = raw_points.reshape((int(len(raw_points) / 2)), 2)
            for i in range(0, raw_points.shape[0]):
                self.raw_points.append(tuple(raw_points[i, :]))

    def clone(self):
        path = AriadnePath(self.ariadne)
        path.subgraph = nx.DiGraph()
        nodes = self.asList()
        for n in nodes:
            path.addNode(n)
        return path

    def size(self):
        if self.ariadne is None:
            return len(self.as2DPoints())
        else:
            return len(self.subgraph.nodes())

    def sizeInPixel(self):
        points = self.as2DPoints()
        sizepx = 0.0
        for i in range(1, len(points)):
            p1 = np.array(points[i - 1])
            p2 = np.array(points[i])
            sizepx += np.linalg.norm(p1 - p2)
        return sizepx

    def contains(self, n):
        return self.subgraph.has_node(n)

    def addNode(self, n):
        if self.subgraph.has_node(n) == True:
            return
        self.subgraph.add_node(n)
        if self.last_node is not None:
            self.subgraph.add_edge(self.last_node, n)
        if self.first_node is None:
            self.first_node = n
        self.last_node = n

    def searchSimilar(self, direction=None, predictor=None, max_level=3):
        if self.size() == 0:
            print("Impossible to search: Path si void!")
            return None

        if self.size() == 1 and direction is None:
            print("Impossible to search: Path has only one node and no initial DIRECTION provided!")
            return None

        nodes = self.asList()
        points = self.as2DPoints()

        target_node = nodes[-1]
        neighbourhood_list = self.ariadne.graph.deepNeighbourhood(target_node, level=max_level)
        neighbourhood = list(itertools.chain.from_iterable(neighbourhood_list))

        pdfs = []

        #######################################
        # Max Computeation
        #######################################
        max_compare = 0
        for n in neighbourhood:
            compare = predictor.computeScoreVisual(self.last_node, n)
            if compare > max_compare:
                max_compare = compare

        #######################################
        # PDF Computation
        #######################################
        for n in neighbourhood:
            if self.contains(n):
                pdfs.append(0.0)
                continue

            norm_visual = predictor.computeScoreVisual(self.last_node, n, None)

            new_path = self.clone()
            new_path.addNode(n)
            direction_pdf = predictor.computeScoreCurvature(new_path, initial_direction=direction)

            # self.removeNode(n)
            if direction is None:
                cumulative_pdf = norm_visual * direction_pdf
            else:
                cumulative_pdf = direction_pdf
            #print("Match", target_node, n, norm_visual,  direction_pdf, "=", cumulative_pdf)
            pdfs.append(cumulative_pdf)

        #######################################
        # Arg max
        #######################################
        pdfs = np.array(pdfs)
        max_i = np.argmax(pdfs)
        #print("Max", pdfs[max_i], neighbourhood[max_i])
        return neighbourhood[max_i]

    def searchSimilarV2(self, direction=None, direction_matcher=None, visual_matcher=None, debug_visual_matcher=None):
        if self.size() == 0:
            print("Impossible to search: Path si void!")
            return None

        if self.size() == 1 and direction is None:
            print("Impossible to search: Path has only one node and no initial DIRECTION provided!")
            return None

        nodes = self.asList()
        points = self.as2DPoints()

        target_node = nodes[-1]
        neighbourhood = self.ariadne.graph.getNeighbourhood(target_node)

        pdfs = []

        #######################################
        # Max Computeation
        #######################################
        max_compare = 0
        for n in neighbourhood:
            compare = visual_matcher.compare(self.last_node, n)
            if compare > max_compare:
                max_compare = compare

        #######################################
        # PDF Computation
        #######################################
        for n in neighbourhood:
            if self.contains(n):
                pdfs.append(0.0)
                continue

            visual_pdf = visual_matcher.compare(self.last_node, n)
            norm_visual = visual_matcher.normalizeComparison(visual_pdf, max_compare)

            new_path = self.clone()
            new_path.addNode(n)
            direction_pdf = direction_matcher.computeScore(new_path, force_direction=direction)

            # self.removeNode(n)
            if direction is None:
                cumulative_pdf = norm_visual * direction_pdf
            else:
                cumulative_pdf = direction_pdf
            print("Match", target_node, n, visual_pdf,  direction_pdf, "=", cumulative_pdf, "|||", norm_visual, norm_visual * direction_pdf)
            pdfs.append(cumulative_pdf)

        #######################################
        # Arg max
        #######################################
        pdfs = np.array(pdfs)
        max_i = np.argmax(pdfs)
        print("Max", pdfs[max_i], neighbourhood[max_i])
        return neighbourhood[max_i]

    def removeNode(self, n):
        if self.subgraph.has_node(n) == False:
            return
        try:
            pred = self.subgraph.predecessors(n)[0]
        except:
            pred = None

        try:
            succ = self.subgraph.successors(n)[0]
        except:
            succ = None

        if pred is not None and succ is not None:
            self.subgraph.add_edge(pred, succ)

        if self.first_node == n:
            self.first_node = succ
        if self.last_node == n:
            self.last_node = pred

        self.subgraph.remove_node(n)

    def asList(self):
        if self.first_node is None:
            return []

        path_nodes = [self.first_node]
        current_node = self.first_node
        try:
            next_node = next(self.subgraph.successors(self.first_node))
            while next_node is not None:
                path_nodes.append(next_node)
                next_node = next(self.subgraph.successors(next_node))
        except:
            pass

        return path_nodes

    def getTip(self):
        points = self.as2DPoints()
        tip = AriadnePathTip(points=points[0:2])
        return tip

    def getTips(self, reverse_order=False):
        points = self.as2DPoints()
        if len(points) < 2:
            return None
        tip1 = AriadnePathTip(points=[points[0], points[1]])
        tip2 = AriadnePathTip(points=[points[-1], points[-2]])
        if reverse_order:
            return tip2, tip1
        else:
            return tip1, tip2

    def as2DPoints(self):
        if self.ariadne is None:
            return self.raw_points
        node_list = self.asList()
        points = []
        for n in node_list:
            node = self.ariadne.graph.graph.node[n]
            point = tuple(node['centroid_point'].astype(int))
            points.append(point)
        return points

    def endsInRegion(self, region_center, region_radius, squared_region=False):
        if self.size() == 0:
            return False

        points = self.as2DPoints()
        last_point = np.array(points[-1])
        center = np.array(region_center)
        dist = np.linalg.norm(center - last_point)
        return dist <= region_radius

    def draw(self, img, color=(1, 0, 0), thickness=5, offset=np.array([0, 0]), dtype=int, draw_numbers=False, draw_tips=False):

        points = self.as2DPoints()
        for i in range(1, len(points)):
            # n1 = self.ariadne.graph.graph.node[e[0]]
            # n2 = self.ariadne.graph.graph.node[e[1]]
            p1 = points[i - 1]  # n1['centroid_point'].astype(int) + offset
            p2 = points[i]  # n2['centroid_point'].astype(int) + offset
            cv2.circle(img, tuple(p1), color=color,
                       radius=thickness, thickness=-1)
            cv2.circle(img, tuple(p2), color=color,
                       radius=thickness, thickness=-1)
            if draw_numbers:
                cv2.putText(img, "{}".format(i), tuple(
                    p1 + np.array([5, 0])), cv2.FONT_HERSHEY_PLAIN, 1.5, color, 1)
            cv2.line(img, tuple(p1), tuple(p2), color=color,
                     thickness=int(thickness * 0.6))

        if draw_tips and self.size() >= 2:
            tips = self.getTips()
            tips[0].draw(img, color=(0, 0, 1))
            tips[1].draw(img, color=(0, 1, 0))

    def drawWithOffset(self, img, color=(1, 0, 0), thickness=5, offset=np.array([0, 0], dtype=int)):
        for i, e in enumerate(self.subgraph.edges()):
            n1 = self.ariadne.graph.graph.node[e[0]]
            n2 = self.ariadne.graph.graph.node[e[1]]
            p1 = n1['centroid_point'].astype(int) + offset
            p2 = n2['centroid_point'].astype(int) + offset
            cv2.circle(img, tuple(p1), color=color,
                       radius=thickness, thickness=-1)
            cv2.putText(img, "{}".format(i), tuple(
                p1 + np.array([5, 0])), cv2.FONT_HERSHEY_PLAIN, 1.5, color, 1)
            cv2.line(img, tuple(p1), tuple(p2), color=color,
                     thickness=int(thickness * 0.6))

    @staticmethod
    def loadPathsFromTXT(filename):
        f = open(filename, "r")
        lines = f.readlines()
        paths = []
        for l in lines:
            path_data = np.array(list(map(int, l.split())))
            points_data = path_data[1:]
            path = AriadnePath(ariadne=None, raw_points=points_data)
            paths.append(path)
        return paths


class AriadnePathFinder(object):

    def __init__(self, ariadne, predictor, name="PathFinder"):
        self.ariadne = ariadne
        self.predictor = predictor
        self.path = AriadnePath(ariadne)
        self.initial_direction = None
        self.initial_neighboring = None
        self.open = True
        self.name = name

    def isOpen(self):
        return self.open

    def close(self):
        self.open = False

    def open(self):
        self.open = True

    def initPathWithDirection(self, initial_node, initial_direction=None):
        self.path.addNode(initial_node)
        self.initial_direction = initial_direction
        found = self.searchNextNode(direction=initial_direction)
        if found is None:
            return False
        self.path.addNode(found)
        return True

    def initPathWithNeighboring(self, initial_node, neigh_node):
        self.path.addNode(initial_node)
        self.path.addNode(neigh_node)
        return initial_node != neigh_node

    def searchNextNode(self, direction=None, auto_add_node=True):
        if self.path.size() == 0:
            print("Unable to find next: Path si void!")
            return None

        if self.path.size() == 1 and direction is None:
            print("Unable to find next: Path has only one node and no initial DIRECTION provided!")
            return None

        nodes = self.path.asList()
        points = self.path.as2DPoints()

        target_node = nodes[-1]
        neighbourhood_list = self.ariadne.graph.deepNeighbourhood(
            target_node,
            # forced_excluded_nodes=nodes,
            level=self.predictor.getNeighbourhoodLevel(self.path.size())
        )
        neighbourhood = list(itertools.chain.from_iterable(neighbourhood_list))

        #######################################
        # Invalid Neighbourhood
        #######################################
        if len(neighbourhood) == 0:
            return None

        pdfs = []

        #######################################
        # Visual Max Score
        #######################################
        visual_max_score = 0
        for n in neighbourhood:
            visual_score = self.predictor.computeScoreVisual(self.path.last_node, n)
            if visual_score > visual_max_score:
                visual_max_score = visual_score

        #######################################
        # PDF Computation
        #######################################
        for n in neighbourhood:
            if self.path.contains(n):
                pdfs.append(0.0)
                continue

            norm_visual_pdf = self.predictor.computeScoreVisual(self.path.last_node, n, reference_value=None)

            new_path = self.path.clone()
            new_path.addNode(n)
            direction_pdf = self.predictor.computeScoreCurvature(new_path, initial_direction=direction)

            distance_pdf = self.predictor.computeScoreDistance(self.path.last_node, n)

            # if '3' in self.name:
            #     self.predictor.curvature_predictor.computeScore(new_path, debug=True)

            if direction is None:
                cumulative_pdf = norm_visual_pdf * direction_pdf
            else:
                cumulative_pdf = direction_pdf

            cumulative_pdf = cumulative_pdf * distance_pdf
            # if '0' in self.name:
            #     print("Match", target_node, n, norm_visual_pdf,  direction_pdf,distance_pdf, "=", cumulative_pdf)
            pdfs.append(cumulative_pdf)

        #print(self.name, pdfs)
        #######################################
        # Arg max
        #######################################
        pdfs = np.array(pdfs)
        max_i = np.argmax(pdfs)

        #print("Max", pdfs[max_i], neighbourhood[max_i])
        if auto_add_node:
            self.path.addNode(neighbourhood[max_i])
        return neighbourhood[max_i]


class AriadneMultiPathFinder(object):

    def __init__(self, ariadne, predictor):
        self.ariadne = ariadne
        self.predictor = predictor
        self.path_finders = []
        self.iterations = 0

    def isFinished(self):
        close_counter = 0
        for f in self.path_finders:
            if not f.isOpen():
                close_counter += 1
        return close_counter == self.size()

    def close(self):
        for f in self.path_finders:
            f.close()

    def size(self):
        return len(self.path_finders)

    def getIterations(self):
        return self.iterations

    def startSearchInNeighbourhood(self, start_node, depth=1):

        neighbourhood_raw = self.ariadne.graph.deepNeighbourhood(start_node, level=depth, as_simple_list=False)

        neighbourhood = neighbourhood_raw[-1]
        self.path_finders = []
        for i, n2 in enumerate(neighbourhood):
            path_finder = AriadnePathFinder(self.ariadne, self.predictor, name="PathFinder_{}".format(i))
            path_finder.initPathWithNeighboring(start_node, n2)
            self.path_finders.append(path_finder)
        self.iterations = 1

    def nextStep(self):
        for i, f in enumerate(self.path_finders):
            if f.isOpen():
                f.searchNextNode()
        self.iterations += 1

    def getScores(self, single_components=False):
        scores = []
        for i, f in enumerate(self.path_finders):
            if single_components:
                scores.append(self.predictor.score_function.computeScores(f.path))
            else:
                scores.append(self.predictor.score_function.computeScore(f.path))
        return scores

    def getBestPathFinder(self):
        scores = self.getScores()
        index = np.argmax(scores)
        return self.path_finders[index]


class Ariadne(object):
    GAIN_DIRECTION = 2.5
    GAIN_COLOR = 1
    PROPERTIES_FILE_EXTENSION = "mat"
    IMAGE_FILE_EXTENSION = "jpg"

    def __init__(self, image=None, image_file=None, segments_data=None, threshold=0.4, segmentator=ImageSegmentator()):

        #######################################
        # Image loading
        #######################################
        if image == None and image_file == None:
            import sys
            print("Ariadne error! Neither Image or ImageFile passed")
            sys.exit(0)

        self.image_path = None
        if image == None and image_file is not None:
            image_file = os.path.abspath(image_file)
            self.image = img_as_float(io.imread(image_file))
            self.image_path = image_file
        else:
            self.image = image
        self.threshold = threshold

        #######################################
        # Objects
        #######################################
        self.paths = []

        #######################################
        # Slic Superpixel Segmentation
        #######################################
        if segments_data is None:
            self.segments = segmentator.segments(self.image)
        else:
            self.segments = segments_data

        #######################################
        # Superpixels Graph
        #######################################
        self.graph = AriadneGraph(self.image, self.segments)

    def saveSegmentation(self, filename):
        """ Save sagmentations to file """
        np.savetxt(filename, self.segments)

    def clearPaths(self):
        self.paths = []

    def createPath(self):
        path = AriadnePath(self)
        self.paths.append(path)
        return path

    def removePath(self, path):
        self.paths.remove(path)

    def saveToFile(self, filename):
        data = {
            "image_path": self.image_path,
            "segments": self.segments,
            "paths": [],
            "paths_points": []
        }

        for p in self.paths:
            if len(p.asList()) > 0:
                data["paths"].append(p.asList())
                data["paths_points"].append(p.as2DPoints())

        scipy.io.savemat(filename, data, do_compression=True)

    @staticmethod
    def savePathsToTXT(paths, filename):
        paths_point = []
        for path in paths:
            points = np.array(path.as2DPoints()).ravel()
            points = np.insert(points, 0, path.size())
            paths_point.append(points)

        f = open(filename, "w")
        for path in paths_point:
            for p in path:
                f.write(str(p))
                f.write(" ")
            f.write("\n")
        f.close()

    @staticmethod
    def loadFromFile(filename, is_image=False):
        image_filename = None
        if is_image:
            image_filename = filename
            filename = Ariadne.getImageProperties(filename)

        data = scipy.io.loadmat(filename)
        image_path = data["image_path"][0]

        if not os.path.exists(image_path):
            image_path = os.path.splitext(filename)[0] + "." + Ariadne.IMAGE_FILE_EXTENSION

        ar = Ariadne(
            image=None,
            image_file=image_path,
            segments_data=data["segments"]
        )
        for p in data["paths"].ravel():
            path = ar.createPath()
            for n in p.ravel():
                path.addNode(n)
        return ar

    @staticmethod
    def hasPropertiesFile(filename):
        properties_file = Ariadne.getImageProperties(filename)
        return os.path.exists(properties_file)

    @staticmethod
    def getImageProperties(filename):
        properties_file = os.path.splitext(filename)[0] + "." + Ariadne.PROPERTIES_FILE_EXTENSION
        return properties_file
