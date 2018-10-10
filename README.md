# Ariadne

## Manual detection script

To try Ariadne you can launch:

```
python manual_path_finder.py --image_file images/image_1.jpg --config_name defaults
```

The ```config_name``` refers to the Section defined in the ```configs.ini``` configuration file. As an alternative you can try
the second configuration with:

```
python manual_path_finder.py --image_file images/image_2.jpg --config_name clutter
```

At the beginning, the script displays the image, then with the ```spacebar``` you can launch the superpixels segmentaion. Hence, 
with the mouse click on two random cable terminals and press ```spacebar``` again to start the iterative segmentation procedure.
