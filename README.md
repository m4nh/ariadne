# Ariadne

## Dependencies

The ```anaconda_environment.yml``` contains the mandatory dependencies for the Ariadne package. With anaconda installed you can directly create an environment from this manifest with:

```
conda env create -f anaconda_environment.yml
```

And then activate it with:

```
conda activate ariadne
```

## Manual detection script

To try Ariadne you can launch:

```
python manual_path_finder.py --image_file images/image_1.jpg --config_name default
```

The ```config_name``` refers to the Section defined in the ```configs.ini``` configuration file. As an alternative you can try
the second configuration with:

```
python manual_path_finder.py --image_file images/image_2.jpg --config_name clutter
```

At the beginning, the script displays the image, then with the ```spacebar``` you can launch the superpixels segmentaion. Hence, 
click with mouse on two random cable terminals and press ```spacebar``` again to start the iterative segmentation procedure.
