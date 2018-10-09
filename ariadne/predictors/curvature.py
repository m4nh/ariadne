import math
import numpy as np
from scipy.stats import vonmises
from scipy.interpolate import splprep, splev, spalde
import cv2


class VonMisesBuffer(object):

    def __init__(self, k=3):
        self.vonmises_range = np.arange(0, math.pi * 2, 0.0001)
        self.vonmises_base = vonmises(k)
        self.vonmises_values = self.vonmises_base.pdf(self.vonmises_range)

    def pdf(self, x):

        i = int(math.fabs(x) * 10000)
        if i >= 0 and i < len(self.vonmises_values):
            return self.vonmises_values[i]
        return 0.0


class CurvaturePredictor(object):
    """
    Base class for curvature Predictors

    Parameters
    ----------
    ariadne : Ariadne
              object associated with a source image

    """

    def __init__(self, ariadne):
        self.ariadne = ariadne

    def computeScore(self, path, initial_direction=None):
        return 1.0


class CurvatureVonMisesPredictor(CurvaturePredictor):
    """
    Curvature predictor using VonMises distribution

    Parameters
    ----------
    ariadne : Ariadne
        object associated with a source image

    kappa : float
        Kappa parameter for the VonMises distribution

    max_angle : float
        max allowe angle for two consecutive edges

    """

    def __init__(self, ariadne, kappa=4, max_angle=math.pi * 0.5):
        super(CurvatureVonMisesPredictor, self).__init__(ariadne)
        self.vonmises = VonMisesBuffer(k=kappa)
        self.max_angle = max_angle

    def computeScore(self, path,  initial_direction=None, debug=False):
        """
        Computes the score for a given path. Considering also degenerate paths

        Parameters
        ----------
        path : AriadnePath
            target path

        initial_direction : np.array
            initial direction used to compute the score of single-edge paths

        """

        #######################################
        # Single Node Path
        #######################################
        if path.size() <= 1:
            return 1.0

        #######################################
        # Single Edge Path
        #######################################
        if path.size() == 2:
            if initial_direction is None:
                return 1.0
            else:
                points = path.as2DPoints()
                p1 = np.array(points[0])
                p2 = np.array(points[1])
                direction = p2 - p1
                direction = direction / np.linalg.norm(direction)
                comp_direction = initial_direction / \
                    np.linalg.norm(initial_direction)
                # print("COMPUTERING", np.dot(direction, comp_direction))
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
            # try:
            #     angle = math.acos(np.dot(d1, d2))
            # except:
            #     angle = 0
            thetas.append(angle)

        # if len(thetas) == 0:
        #     return 0.0

        if math.fabs(thetas[-1]) > self.max_angle:
            return 0.0
        if debug:
            print("Thetas", thetas)
        # print("=" * 50)
        # print("Thetas", thetas)
        if len(thetas) == 1:
            return self.vonmises.pdf(thetas[0])
        elif len(thetas) > 1:
            posterios = []
            for i in range(1, len(thetas)):
                t1 = thetas[i - 1]
                t2 = thetas[i]
                posterios.append(self.vonmises.pdf(t1 - t2))
            return np.prod(np.array(posterios).ravel())**(1.0 / (len(points) - 3.0))


class CurvatureVonMisesLastPredictor(CurvaturePredictor):
    """
    Curvature predictor using VonMises distribution only for last edge

    Parameters
    ----------
    ariadne : Ariadne
        object associated with a source image

    kappa : float
        Kappa parameter for the VonMises distribution

    max_angle : float
        max allowe angle for two consecutive edges

    """

    def __init__(self, ariadne, kappa=4, max_angle=math.pi * 0.5):
        super(CurvatureVonMisesLastPredictor, self).__init__(ariadne)
        self.vonmises = VonMisesBuffer(k=kappa)
        self.max_angle = max_angle

    def computeScore(self, path,  initial_direction=None, debug=False):
        """
        Computes the score for a given path. Considering also degenerate paths

        Parameters
        ----------
        path : AriadnePath
            target path

        initial_direction : np.array
            initial direction used to compute the score of single-edge paths

        """

        #######################################
        # Single Node Path
        #######################################
        if path.size() <= 1:
            return 1.0

        #######################################
        # Single Edge Path
        #######################################
        if path.size() == 2:
            if initial_direction is None:
                return 1.0
            else:
                points = path.as2DPoints()
                p1 = np.array(points[0])
                p2 = np.array(points[1])
                direction = p2 - p1
                direction = direction / np.linalg.norm(direction)
                comp_direction = initial_direction / \
                    np.linalg.norm(initial_direction)
                # print("COMPUTERING", np.dot(direction, comp_direction))
                angle = math.acos(np.dot(direction, comp_direction))
                return self.vonmises.pdf(angle)

        #######################################
        # Normal Path
        #######################################
        directions = []
        points = path.as2DPoints()
        for i in range(len(points) - 2, len(points)):
            p1 = np.array(points[i - 1])
            p2 = np.array(points[i])
            direction = p2 - p1
            direction = direction / np.linalg.norm(direction)
            directions.append(direction)

        try:
            angle = math.acos(np.dot(directions[0], directions[1]))
        except:
            angle = 0.0
        return self.vonmises.pdf(angle)


class CurvatureSplinePredictor(CurvaturePredictor):
    """
    Curvature predictor using B-Splines

    Parameters
    ----------
    ariadne : Ariadne
        object associated with a source image

    k : int
        BSpline Degree

    smoothing : float
        BSpline smoothing factor

    periodic: int
        If non-zero, data points are considered periodic with period x[m-1] - x[0]

    """

    def __init__(self, ariadne, k=3, smoothing=0.0, periodic=0):
        super(CurvatureSplinePredictor, self).__init__(ariadne)
        self.k = k
        self.smoothing = smoothing
        self.periodic = periodic
        # tck, u = splprep(pts.T, u=None, k=k, s=smoothin, per=periodic)

    def computeSpline(self, path):
        """
        Curvature predictor using B-Splines

        Parameters
        ----------
        path : AriadnePath
            target path

        """

        points = path.as2DPoints()
        if len(points) > self.k:
            pts = np.array(points)
            tck, u = splprep(pts.T,
                             u=None,
                             k=self.k,
                             s=self.smoothing,
                             per=self.periodic
                             )
            return {'tck': tck, 'u': u}
        return None

    def generateSplineMaskImage(self, path, radius=1, steps=1000, ref_image=None):
        if ref_image is None:
            ref_image = self.ariadne.image

        mask = np.zeros(
            (ref_image.shape[0], ref_image.shape[1]), dtype=np.uint8)
        spline = self.computeSpline(path)
        if spline is None:
            return mask
        u = spline['u']
        tck = spline['tck']
        u_new = np.linspace(u.min(), u.max(), steps)
        x_new, y_new = splev(u_new, tck, der=0)
        for i, x in enumerate(x_new):
            cv2.circle(mask, (int(x_new[i]), int(
                y_new[i])), radius, (255, 255, 255), -1)
        return mask

    def computeCurvatures(self, spline):
        u = spline['u']
        tck = spline['tck']

        der = spalde(u, tck)

        curvatures = np.zeros((len(u), 1))
        for i in range(0, len(u)):

            d1 = (der[0][i][1], der[1][i][1])
            d2 = (der[0][i][2], der[1][i][2])
            x1 = d1[0]
            y1 = d1[1]
            x2 = d2[0]
            y2 = d2[1]
            curvature = (x1 * y2 - y1 * x2) / \
                ((x1 * x1 + y1 * y1)**(3.0 / 2.0))
            curvatures[i] = curvature
        return curvatures

    def computeCurvature(self, spline, positive_curvatures=True, normalize=True):
        curvatures = self.computeCurvatures(spline)
        if positive_curvatures:
            curvatures = np.absolute(curvatures)
        summation = np.sum(curvatures)
        if normalize:
            summation = summation / float(len(curvatures))
        return summation

    def computeScore(self, path,  initial_direction=None):
        """
        Computes the score for a given path. Considering also degenerate paths

        Parameters
        ----------
        path : AriadnePath
            target path

        initial_direction : np.array
            initial direction used to compute the score of single-edge paths

        """

        #######################################
        # Path with k nodes
        #######################################
        if path.size() <= self.k:
            return 1.0

        #######################################
        # Normal Path
        #######################################
        spline = self.computeSpline(path)
        if spline is None:
            return 1.0

        u = spline['u']
        tck = spline['tck']

        der = spalde(u, tck)

        total_curvature = 0.0
        total_curvature_pos = 0.0
        for i in range(0, len(u)):

            d1 = (der[0][i][1], der[1][i][1])
            d2 = (der[0][i][2], der[1][i][2])
            x1 = d1[0]
            y1 = d1[1]
            x2 = d2[0]
            y2 = d2[1]
            curvature = (x1 * y2 - y1 * x2) / \
                ((x1 * x1 + y1 * y1)**(3.0 / 2.0))
            total_curvature += curvature
            total_curvature_pos += math.fabs(curvature)

        normalized_curvature = total_curvature_pos / float(len(u))
        final_curvature = min(1.0, 1.0 - normalized_curvature)
        return final_curvature


class LengthPredictor(CurvaturePredictor):
    """
    Length predictor 

    Parameters
    ----------
    ariadne : Ariadne
        object associated with a source image



    """

    def __init__(self, ariadne, max_length=1000):
        super(LengthPredictor, self).__init__(ariadne)
        self.max_length = max_length

    def computeScore(self, path, initial_direction=None, debug=False):
        """
        Computes the score for a given path. Considering also degenerate paths

        Parameters
        ----------
        path : AriadnePath
            target path

        initial_direction : np.array
            initial direction used to compute the score of single-edge paths

        """
        points = path.as2DPoints()
        length = 0.0
        for i in range(1, len(points)):
            pf1 = np.array(points[i]).astype(float)
            pf2 = np.array(points[i - 1]).astype(float)
            length += np.linalg.norm(pf1 - pf2)

        return 1.0+min(length / self.max_length, 1.0)
