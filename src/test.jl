#!/usr/bin/env bash
    #=
    exec julia --project="~/FaceDetection.jl/src/" "${BASH_SOURCE[0]}" "$@" -e 'include(popfirst!(ARGS))' \
    "${BASH_SOURCE[0]}" "$@"
    =#


using ObjectDetector, FileIO

yolomod = YOLO.v3_608_COCO(batch=1, silent=true) # Load the YOLOv3-tiny model pretrained on COCO, with a batch size of 1

batch = emptybatch(yolomod) # Create a batch object. Automatically uses the GPU if available

img = load(joinpath(dirname(dirname(pathof(ObjectDetector))),"test","images","dog-cycle-car.png"))

batch[:,:,:,1], padding = prepareImage(img, yolomod) # Send resized image to the batch

res = yolomod(batch, detectThresh=0.5, overlapThresh=0.8) # Run the model on the length-1 batch

imgBoxes = drawBoxes(img, yolomod, padding, res)
save("result.png", imgBoxes)
