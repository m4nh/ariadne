import argparse
import os
import cv2
import numpy as np
import sys
import time
import configparser

from ariadne.core import Ariadne, AriadnePathTip, AriadnePath, ImageSegmentator, AriadnePathFinder, AriadneMultiPathFinder
import ariadne.predictors_config as predictors_config
from ariadne.utils.interactivewindow import InteractiveWindow, InteractiveWindowKeys
import ariadne.utils.colors as cl

colors = cl.getPalette()

#######################################
# Arguments
#######################################
ap = argparse.ArgumentParser()
ap.add_argument("--image_file", required=True, help="Target image file.")
ap.add_argument("--configs_file", default='configs.ini', type=str, help='Configurations file.')
ap.add_argument("--config_name", default="default", type=str, help='Configuration name')
ap.add_argument("--debug", default=1, type=int, help='"1" if Debug Mode. "0" Otherwise.')
args = vars(ap.parse_args())

#######################################
# Input Image
#######################################
image = cv2.imread(args['image_file'])

#######################################
# Configuration
#######################################
config = configparser.ConfigParser()
config_name = args['config_name']
config.read(args['configs_file'])
print("ACTIVE CONFIG:", config_name)
if not config.has_section(config_name):
    print("Config {} not found!".format(config_name))
    sys.exit(0)

#######################################
# Segmentator
#######################################
segmentator = ImageSegmentator(segmentator_type='SLIC')
segmentator.options_map['n_segments'] = int(config.get(config_name, 'num_segments'))
segmentator.options_map['compactness'] = float(config.get(config_name, 'segmentator_compactness'))
segmentator.options_map['sigma'] = float(config.get(config_name, 'segmentator_sigma'))

#######################################
# Ariadne
#######################################
ariadne = Ariadne(image_file=args['image_file'], segmentator=segmentator)


clicked_points = []
clicked_nodes = []


#######################################
# Predictor
#######################################
overall_predictor = predictors_config.buildPredictorFromConfig(ariadne, config, config_name)

#######################################
# Multi Path Finder
#######################################
multi_path_finder = AriadneMultiPathFinder(
    ariadne, predictor=overall_predictor)


#######################################
# ACTIVATION
#######################################
active = False

#################################
# Options
#################################
boundary_color = np.array([243., 150., 33.])/255.

end_reached = False


def clickCallback(data):
    global output_image, clicked_points,  active, end_reached

    #######################################
    # Debug click to check Node Label
    #######################################
    if len(clicked_points) == 2:
        n = ariadne.graph.nearestNode(data[1])
        print("CLICKED NODE:", n)
        return

    #######################################
    # Click over tips
    #######################################
    clicked_points.append(data[1])
    print("CLICKED POINTS", clicked_points)
    if len(clicked_points) == 1:
        return

    print("CLICKED POINTS", clicked_points)
    #######################################
    # Get Current Clicked Node
    #######################################
    n = ariadne.graph.nearestNode(clicked_points[0])

    #######################################
    # Starts Multi-Path Finder search
    #######################################
    multi_path_finder.startSearchInNeighbourhood(
        n, depth=overall_predictor.start_depth)
    reaches_map = [0] * multi_path_finder.size()

    #######################################
    # Search main loop
    #######################################
    active = True
    while True:
        #######################################
        # Debug Draw
        #######################################
        if args['debug']:
            output_image = ariadne.graph.generateBoundaryImage(image, color=boundary_color)
            cv2.circle(output_image, tuple(
                clicked_points[0]), int(config.get(config_name, 'end_region_radius')), (0, 0, 1), 2)
            cv2.circle(output_image, tuple(
                clicked_points[1]), int(config.get(config_name, 'end_region_radius')), (0, 0, 1), 2)

        if active:
            print("ROUND", multi_path_finder.getIterations() + 1, "=" * 50)

            #######################################
            # Mult-Path Finder Next step
            #######################################
            multi_path_finder.nextStep()

            #######################################
            # Fetch scores for each Path
            #######################################
            scores_raw = multi_path_finder.getScores(single_components=True)
            scores = multi_path_finder.getScores(single_components=False)

            reaches_counter = 0
            for i, f in enumerate(scores):
                path_finder = multi_path_finder.path_finders[i]
                print("Path ", i, scores_raw[i], scores[i],
                      "REACHED" if reaches_map[i] > 0 else "")

                #######################################
                # Debug Draw of current Path
                #######################################
                if args['debug']:
                    color = colors[i % len(colors)]
                    if path_finder.isOpen():
                        path_finder.path.draw(
                            output_image, draw_numbers=False, color=color)

                    last_point = path_finder.path.as2DPoints()[-1]
                    cv2.putText(output_image, "{}".format(i), tuple(
                        last_point + np.array([5, 0])), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)

                #######################################
                # Close condition for current Path
                #######################################
                if scores[i] < float(config.get(config_name, 'min_score')):
                    path_finder.close()
                    # print("PAth", i, "Has Reached MIN SCORE!")
                if path_finder.path.endsInRegion(clicked_points[1], int(config.get(config_name, 'end_region_radius'))):
                    # print("PAth", i, "Has Reached Destiantion!")
                    n2 = ariadne.graph.nearestNode(clicked_points[1])
                    path_finder.path.addNode(n2)
                    path_finder.close()
                    reaches_map[i] = 1
                    reaches_counter += 1
                if multi_path_finder.getIterations() > int(config.get(config_name, 'max_length')):
                    multi_path_finder.close()

        c = 0
        step_time = 0
        if args['debug']:
            print("STOP")
            if not end_reached:
                c = window.showImg(output_image, step_time)

        if multi_path_finder.isFinished():
            end_reached = True
            print("END REACHED!")

            # output_image = ariadne.graph.generateBoundaryImage(image)
            # cv2.circle(output_image, tuple(
            #     clicked_points[0]), config.get(config_name, 'end_region_radius'), (0, 0, 1), 2)
            # cv2.circle(output_image, tuple(
            #     clicked_points[1]), config.get(config_name, 'end_region_radius'), (0, 0, 1), 2)

            # for i, f in enumerate(multi_path_finder.path_finders):
            #     f.path.draw(output_image, color=(0, 0, 1), draw_numbers=False)
            #     last_point = f.path.as2DPoints()[-1]
            #     cv2.putText(output_image, "{}".format(i), tuple(
            #         last_point + np.array([5, 0])), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)

            output_image = image.copy()  # ariadne.graph.generateBoundaryImage(image)

            best = multi_path_finder.getBestPathFinder()
            best.path.draw(output_image, color=(0, 255, 0), draw_numbers=False)
            window.showImg(output_image, 0)


window = InteractiveWindow("stereo_matching", autoexit=True)
window.registerCallback(clickCallback, event=InteractiveWindow.EVENT_MOUSEDOWN)

output_image = ariadne.graph.generateBoundaryImage(image, color=boundary_color)
window.showImg(image, time=0, disable_keys=False)

while True:
    for point in clicked_points:
        cv2.circle(output_image, tuple(point.astype(int)), 5, (255, 0, 0), -1)

    if not end_reached:
        window.showImg(output_image, time=1, disable_keys=True)
