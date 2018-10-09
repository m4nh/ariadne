
class OverallScoreFunction(object):

    def __init__(self, scores_components=[], score_function="1.0"):
        self.scores_components = scores_components
        self.score_function = score_function

    def computeScores(self, path):
        scores = []
        for sf in self.scores_components:
            sname = sf[0]
            sfunc = sf[1]
            scores.append((sname, sfunc.computeScore(path)))
        return scores

    def computeScore(self, path):
        scores = self.computeScores(path)
        for s in scores:
            sname = s[0]
            sval = s[1]
            exec("{}={}".format(sname, sval))
        return eval(self.score_function)


class OverallPredictor(object):
    """
    Aggregate of simple predictors
    """

    def __init__(self, predictors={}, options={}, neighbourhood_levels=[1, 2], start_depth=1, score_function=OverallScoreFunction()):

        self.visual_predictor = None
        self.curvature_predictor = None
        self.distance_predictor = None
        self.neighbourhood_levels = neighbourhood_levels
        self.score_function = score_function
        self.start_depth = start_depth

        #######################################
        # Predictors
        #######################################
        if 'visual' in predictors:
            self.visual_predictor = predictors['visual']

        if 'curvature' in predictors:
            self.curvature_predictor = predictors['curvature']

        if 'distance' in predictors:
            self.distance_predictor = predictors['distance']

        #######################################
        # Options
        #######################################
        if 'neighbourhood_levels' in options:
            self.neighbourhood_levels = options['neighbourhood_levels']

    def getNeighbourhoodLevel(self, index):
        if index <= 0:
            return self.neighbourhood_levels[0]
        if index < len(self.neighbourhood_levels):
            return self.neighbourhood_levels[index]
        else:
            return self.neighbourhood_levels[-1]

    def computeScoreVisual(self, n1, n2, reference_value=None):
        """
        Visual score computation between two nodes/regions

        Parameters
        ----------
        h1 : array
            first histogram
        h2 : array
            second histogram
        reference_value : float
            reference value to compute the normal probability for the target score
        """

        if self.visual_predictor is not None:
            return self.visual_predictor.computeScore(n1, n2, reference_value=reference_value)
        else:
            return 1.0

    def computeScoreCurvature(self, path, initial_direction=None):
        """
        Computes the score for a given path. Considering also degenerate paths

        Parameters
        ----------
        path : AriadnePath
            target path

        initial_direction : np.array
            initial direction used to compute the score of single-edge paths

        """
        if self.curvature_predictor is not None:
            return self.curvature_predictor.computeScore(path,  initial_direction=initial_direction)
        else:
            return 1.0

    def computeScoreDistance(self, n1, n2):
        """
        Computes the score for a given distance

        Parameters
        ----------
        n1 : int
            first node/region
        n2 : array
            second node/region

        """
        if self.distance_predictor is not None:
            return self.distance_predictor.computeScore(n1, n2)
        else:
            return 1.0

    def computeScores(self, path):
        scores = []
        for sf in self.scores_functions:
            scores.append(sf.computeScore(path))
        return scores
