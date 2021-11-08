The [THINGS dataset](https://osf.io/3fu6z/) is a great dataset for object images, containing 26,107 object images.  However, there are some categories of images that may interfere with our face detection results, if we are to use these images as negative training images.  Of these images, there are 1854 unique categories.  After filtering out [animals](https://gist.github.com/atduskgreg/3cf8ef48cb0d29cf151bedad81553a54) from this dataset, there are 1702 unique categories.  Further removing some categories (manually selected) that contained humans or facial features (see below), there are 1619 unique categories.

To download the THINGS dataset in its entirety, run
```shell
$ bash setup.sh
```

Now that you have the dataset, please run
```shell
$ julia object_categories.jl
```

This will create two text files; one will have all unique categories of images (`all_categories.txt`); the other will contain that list, removing categories that are (`all_categories_filtered.txt`):
  - Animals;
  - Hat or hair related objects;
  - Human-like objects;
  - Specific parts of faces;
  - Activities requiring humans.

The Julia script will filter these categories out of the downloaded images, as they contain too many faces/facial features.

After filtering all the potentially interfering images out of the THINGS dataset, we are left with 22,558 images.