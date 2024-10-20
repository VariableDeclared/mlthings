
reg_plates - v1 2024-10-19 6:46pm
==============================

This dataset was exported via roboflow.com on October 19, 2024 at 6:47 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 13 images.
Car-regs are annotated in COCO format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* Random shear of between -10° to +10° horizontally and -10° to +10° vertically

The following transformations were applied to the bounding boxes of each image:
* Random exposure adjustment of between -10 and +10 percent


