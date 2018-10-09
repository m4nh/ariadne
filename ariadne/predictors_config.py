

from ariadne.predictors.visual import VisualPredictor, VisualPredictorColor2DHistogram
from ariadne.predictors.overall import OverallPredictor, OverallScoreFunction
from ariadne.predictors.visual import VisualPredictorColor2DHistogram, VisualPredictorColor3DHistogram
from ariadne.predictors.distance import DistancePredictorMaxDistance
from ariadne.predictors.curvature import *
import sys


def buildPredictorFromConfig(ariadne, config_parser, config_name):
    try:
        score_function = config_parser.get(config_name, 'predictors_function')
        histogram_bins = int(config_parser.get(config_name, 'predictors_visual_bins'))
        start_depth = int(config_parser.get(config_name, 'predictors_start_depth'))
        predictors_flags = config_parser.get(config_name, 'predictors_flags')
        neigh_level = int(config_parser.get(config_name, 'predictors_neighbours'))
    except:
        print("Error parsing  configuration: ")
        sys.exit(0)

    overall_score_function = OverallScoreFunction(
        scores_components=[
            ('s0', CurvatureVonMisesPredictor(ariadne)),
            ('s1', CurvatureSplinePredictor(ariadne)),
            ('s3', LengthPredictor(ariadne)),
        ],
        score_function=score_function
    )

    visual_predictor = None
    curvature_predictor = None
    distance_predictor = None

    if 'V' in predictors_flags:
        visual_predictor = VisualPredictorColor3DHistogram(ariadne, bins=[histogram_bins, histogram_bins, histogram_bins], distribution_parameters=[
            0.5001], enable_cache=True)
    if 'C' in predictors_flags:
        curvature_predictor = CurvatureVonMisesLastPredictor(
            ariadne, kappa=1)

    if 'M' in predictors_flags:
        curvature_predictor = CurvatureVonMisesPredictor(ariadne)

    if 'D' in predictors_flags:
        distance_predictor = DistancePredictorMaxDistance(
            ariadne, max_distance=100, bradford_coefficient=0.5)

    overall_predictor = OverallPredictor(
        predictors={
            'visual': visual_predictor,
            'curvature': curvature_predictor,
            'distance': distance_predictor
        },
        neighbourhood_levels=list(range(1, neigh_level + 1)),
        score_function=overall_score_function,
        start_depth=start_depth
    )
    return overall_predictor


def buildPredictor(ariadne, name="default"):

    if name == "default":
        overall_score_function = OverallScoreFunction(
            scores_components=[
                ('s0', CurvatureVonMisesPredictor(ariadne)),
                ('s1', CurvatureSplinePredictor(ariadne))
            ],
            score_function="s0*s1"
        )

        overall_predictor = OverallPredictor(
            predictors={
                # 'visual': VisualPredictorColor2DHistogram(ariadne, bins=[8, 8, 8], enable_cache=True),
                'visual': VisualPredictorColor3DHistogram(ariadne, bins=[2, 2, 2], distribution_parameters=[0.5001], enable_cache=True),
                'curvature': CurvatureVonMisesPredictor(ariadne),
                'distance': DistancePredictorMaxDistance(ariadne, max_distance=100, bradford_coefficient=0.5)
            },
            neighbourhood_levels=[1, 2, 3],
            score_function=overall_score_function
        )
        return overall_predictor

    elif name == "simplecurve":
        overall_score_function = OverallScoreFunction(
            scores_components=[
                ('s0', CurvatureVonMisesPredictor(ariadne)),
                ('s1', CurvatureSplinePredictor(ariadne))
            ],
            score_function="s0*s1"
        )

        overall_predictor = OverallPredictor(
            predictors={
                # 'visual': VisualPredictorColor2DHistogram(ariadne, bins=[8, 8, 8], enable_cache=True),
                'visual': VisualPredictorColor3DHistogram(ariadne, bins=[8, 8, 8], distribution_parameters=[0.5001], enable_cache=True),
                'curvature':  CurvatureVonMisesLastPredictor(ariadne, kappa=1),
                'distance': DistancePredictorMaxDistance(ariadne, max_distance=100, bradford_coefficient=0.5)
            },
            neighbourhood_levels=[1, 2, 3],
            score_function=overall_score_function,
            start_depth=2
        )
        return overall_predictor

    elif name == "simplecurvenod":
        overall_score_function = OverallScoreFunction(
            scores_components=[
                ('s0', CurvatureVonMisesPredictor(ariadne)),
                ('s1', CurvatureSplinePredictor(ariadne))
            ],
            score_function="s0*s1"
        )

        overall_predictor = OverallPredictor(
            predictors={
                # 'visual': VisualPredictorColor2DHistogram(ariadne, bins=[8, 8, 8], enable_cache=True),
                'visual': VisualPredictorColor3DHistogram(ariadne, bins=[8, 8, 8], distribution_parameters=[0.5001], enable_cache=True),
                'curvature':  CurvatureVonMisesLastPredictor(ariadne, kappa=1),
                'distance': None
            },
            neighbourhood_levels=[1, 2, 3],
            score_function=overall_score_function
        )
        return overall_predictor

    elif name.startswith("dyn_"):  # example: dyn_s0*s1@8@1@VCD@3
        try:
            parameters = name.split("_")[1]
            parameters = parameters.split("@")
            score_function = parameters[0]
            histogram_bins = int(parameters[1])
            start_depth = int(parameters[2])
            predictors_flags = parameters[3]
            neigh_level = int(parameters[4])
        except:
            print("Error parsing dynamic configuration: '{}'".format(name))
            sys.exit(0)

        overall_score_function = OverallScoreFunction(
            scores_components=[
                ('s0', CurvatureVonMisesPredictor(ariadne)),
                ('s1', CurvatureSplinePredictor(ariadne)),
                ('s3', LengthPredictor(ariadne)),
            ],
            score_function=score_function
        )

        visual_predictor = None
        curvature_predictor = None
        distance_predictor = None

        if 'V' in predictors_flags:
            visual_predictor = VisualPredictorColor3DHistogram(ariadne, bins=[histogram_bins, histogram_bins, histogram_bins], distribution_parameters=[
                0.5001], enable_cache=True)
        if 'C' in predictors_flags:
            curvature_predictor = CurvatureVonMisesLastPredictor(
                ariadne, kappa=1)

        if 'M' in predictors_flags:
            curvature_predictor = CurvatureVonMisesPredictor(ariadne)

        if 'D' in predictors_flags:
            distance_predictor = DistancePredictorMaxDistance(
                ariadne, max_distance=100, bradford_coefficient=0.5)

        overall_predictor = OverallPredictor(
            predictors={
                'visual': visual_predictor,
                'curvature': curvature_predictor,
                'distance': distance_predictor
            },
            neighbourhood_levels=list(range(1, neigh_level + 1)),
            score_function=overall_score_function,
            start_depth=start_depth
        )
        return overall_predictor

    elif name == "roads":
        overall_score_function = OverallScoreFunction(
            scores_components=[
                ('s0', CurvatureVonMisesPredictor(ariadne)),
                ('s1', CurvatureSplinePredictor(ariadne))
            ],
            score_function="s0*s1"
        )

        overall_predictor = OverallPredictor(
            predictors={
                # 'visual': VisualPredictorColor2DHistogram(ariadne, bins=[8, 8, 8], enable_cache=True),
                'visual': VisualPredictorColor3DHistogram(ariadne, bins=[8, 8, 8], distribution_parameters=[0.5001], enable_cache=True),
                'curvature':  CurvatureVonMisesLastPredictor(ariadne, kappa=0.1),
                'distance': DistancePredictorMaxDistance(ariadne, max_distance=100, bradford_coefficient=0.5)
            },
            neighbourhood_levels=[1, 2, 3],
            score_function=overall_score_function
        )
        return overall_predictor

    elif name == "roads2":
        overall_score_function = OverallScoreFunction(
            scores_components=[
                ('s0', CurvatureVonMisesPredictor(ariadne)),
                ('s1', CurvatureSplinePredictor(ariadne))
            ],
            score_function="s1"
        )

        overall_predictor = OverallPredictor(
            predictors={
                # 'visual': VisualPredictorColor2DHistogram(ariadne, bins=[8, 8, 8], enable_cache=True),
                'visual': VisualPredictorColor3DHistogram(ariadne, bins=[4, 4, 4], distribution_parameters=[0.5001], enable_cache=True),
                'curvature':  CurvatureVonMisesLastPredictor(ariadne, kappa=0.01),
                'distance': DistancePredictorMaxDistance(ariadne, max_distance=100, bradford_coefficient=0.5)
            },
            neighbourhood_levels=[1, 2, 4],
            score_function=overall_score_function
        )
        return overall_predictor

    elif name == "rivers":
        overall_score_function = OverallScoreFunction(
            scores_components=[
                ('s0', CurvatureVonMisesPredictor(ariadne)),
                ('s1', CurvatureSplinePredictor(ariadne))
            ],
            score_function="s1"
        )

        overall_predictor = OverallPredictor(
            predictors={
                # 'visual': VisualPredictorColor2DHistogram(ariadne, bins=[8, 8, 8], enable_cache=True),
                'visual': VisualPredictorColor3DHistogram(ariadne, bins=[8, 8, 8], distribution_parameters=[0.5001], enable_cache=True),
                'curvature':  CurvatureVonMisesLastPredictor(ariadne, kappa=0.1),
                'distance': DistancePredictorMaxDistance(ariadne, max_distance=100, bradford_coefficient=0.5)
            },
            neighbourhood_levels=[1, 2, 4],
            score_function=overall_score_function
        )
        return overall_predictor

    else:
        print("Predictor configuration '{}' does not exist!".format(name))
        sys.exit(0)
