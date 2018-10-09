import math
import numpy as np
from scipy.stats import vonmises


class DirectionMatcher(object):

    def __init__(self, ariadne):
        self.ariadne = ariadne

    def computeCost(self, n1, n2, direction):
        return 0.0


class SimpleDirectionMatcher(DirectionMatcher):

    def __init__(self, ariadne):
        super(SimpleDirectionMatcher, self).__init__(ariadne)

    def computeCost(self, n1, n2, prev_direction):
        node1 = self.ariadne.graph.node(n1)
        node2 = self.ariadne.graph.node(n2)
        p1 = node1['centroid_point']
        p2 = node2['centroid_point']
        direction = p2 - p1
        direction = direction / np.linalg.norm(direction)
        comp_direction = prev_direction / np.linalg.norm(prev_direction)
        return 1.0 - (np.dot(direction, comp_direction) + 1.0) / 2.0


class PathDirectionMatcher(object):

    def __init__(self, ariadne, path):
        self.ariadne = ariadne
        self.path = path

    def computeCost(self, n2, force_direction=None):
        return 0.0


class VonMisesPathDirectionMatcher(PathDirectionMatcher):

    def __init__(self, ariadne, path, kappa=4, max_angle=math.pi * 0.5):
        super(VonMisesPathDirectionMatcher, self).__init__(ariadne, path)
        self.vonmises = vonmises(kappa)
        self.max_angle = max_angle

    def computeScore(self, path,  force_direction=None):
        #######################################
        # Single Node Path
        #######################################
        if path.size() <= 1:
            return 1.0

        #######################################
        # Single Edge Path
        #######################################
        if path.size() == 2:
            if force_direction is None:
                return 1.0
            else:
                points = path.as2DPoints()
                p1 = np.array(points[0])
                p2 = np.array(points[1])
                direction = p2 - p1
                direction = direction / np.linalg.norm(direction)
                comp_direction = force_direction / \
                    np.linalg.norm(force_direction)
                angle = math.acos(np.dot(direction, comp_direction))
                return self.vonmises.pdf(angle)

        #######################################
        # Normal Path
        #######################################
        directions = []
        points = path.as2DPoints()
        for i in range(1, len(points)):
            p1 = np.array(points[i - 1])
            p2 = np.array(points[i])
            direction = p2 - p1
            direction = direction / np.linalg.norm(direction)
            directions.append(direction)

        thetas = []
        for i in range(1, len(directions)):
            d1 = directions[i - 1]
            d2 = directions[i]
            a1 = math.atan2(d1[1], d1[0])
            a2 = math.atan2(d2[1], d2[0])
            angle = a1 - a2
            angle = math.acos(math.cos(angle))
            thetas.append(angle)

        if math.fabs(thetas[-1]) > self.max_angle:
            return 0.0

        if len(thetas) == 1:
            return self.vonmises.pdf(thetas[0])
        elif len(thetas) > 1:
            posterios = []
            for i in range(1, len(thetas)):
                t1 = thetas[i - 1]
                t2 = thetas[i]
                posterios.append(self.vonmises.pdf(t1 - t2))
            return np.prod(np.array(posterios).ravel())**(1.0 / (len(points) - 3.0))

    def computePDF(self, n2, force_direction=None):
        # print("DIRECTION MATCHER COMPUTE PDF:", self.path.size())
        if self.path.size() == 0:
            return 1.0

        node1 = self.ariadne.graph.node(self.path.last_node)
        node2 = self.ariadne.graph.node(n2)
        p1 = node1['centroid_point']
        p2 = node2['centroid_point']
        direction = p2 - p1
        direction = direction / np.linalg.norm(direction)

        if force_direction is not None:
            comp_direction = force_direction / np.linalg.norm(force_direction)
            angle = math.acos(np.dot(direction, comp_direction))
            return self.vonmises.pdf(angle)
        else:
            if self.path.size() == 1:
                return 1.0

            if self.path.size() == 2:
                nodes = self.path.asList()
                node1 = self.ariadne.graph.node(nodes[0])
                node2 = self.ariadne.graph.node(nodes[1])
                node3 = self.ariadne.graph.node(n2)
                dir1 = node2['centroid_point'] - node1['centroid_point']
                dir2 = node3['centroid_point'] - node2['centroid_point']
                dir1 = dir1 / np.linalg.norm(dir1)
                dir2 = dir2 / np.linalg.norm(dir2)
                try:
                    angle = math.acos(np.dot(dir1, dir2))
                except:
                    angle = math.pi
                # print("PDF", self.vonmises.pdf(angle))
                return self.vonmises.pdf(angle)

            directions = []
            points = self.path.as2DPoints()
            target_node = self.ariadne.graph.node(n2)
            points.append(target_node['centroid_point'])
            for i in range(1, len(points)):
                p1 = np.array(points[i - 1])
                p2 = np.array(points[i])
                direction = p2 - p1
                direction = direction / np.linalg.norm(direction)
                directions.append(direction)

            thetas = []
            for i in range(1, len(directions)):
                d1 = directions[i - 1]
                d2 = directions[i]
                a1 = math.atan2(d1[1], d1[0])
                a2 = math.atan2(d2[1], d2[0])
                angle = a1 - a2
                thetas.append(angle)
            # print("DIRECTIONS", directions)
            # print("THETAS", thetas)
            posterios = []
            if len(thetas) > 1:
                for i in range(1, len(thetas)):
                    t1 = thetas[i - 1]
                    t2 = thetas[i]
                    posterios.append(self.vonmises.pdf(t1 - t2))
                # print("PDF:", posterios)

            return np.prod(np.array(posterios).ravel())**(1.0 / (len(points) - 3.0))
