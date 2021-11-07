The [THINGS dataset]((https://osf.io/3fu6z/) is a great dataset for object images, containing 26,107 object images.  However, there are some categories of images that may interfere with our face detection results, if we are to use these images as negative training images.  Of these images, there are 1823 unique categories.  After filtering out [animals](https://gist.github.com/atduskgreg/3cf8ef48cb0d29cf151bedad81553a54) from this dataset, there are 1668 unique categories.

To download the THINGS dataset in its entirety, run
```shell
$ bash setup.sh
```

Now that you have the dataset, please run
```shell
$ julia object_categories.jl
```

This will create two text files; one will have all unique categories of images; the other will contain that list, removing categories that are:
  - Animals;
  - Hat or hair related objects;
  - Activities requiring humans.

The Julia script will filter these categories out of the downloaded images, as they contain too many faces.