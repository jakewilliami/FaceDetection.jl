#! /usr/bin/env bash

trap "exit" INT

# if [[ $(uname -s) != "Darwin" ]]
# then
# 	echo "OpenCV has been ported to Julia, but has only been tested on OSX.  Please keep this in mind when using this implementation of Viola-Jone."
# fi

echo "Please ensure FaceDetection.jl is installed in your home directory..."
sleep 5

mkdir -p ~/FaceDetection.jl/test/images/testing/pos/
mkdir -p ~/FaceDetection.jl/test/images/testing/neg/
mkdir -p ~/FaceDetection.jl/test/images/testing/testing/pos/
mkdir -p ~/FaceDetection.jl/test/images/testing/testing/neg/

brew list > /tmp/brewlist
if ! grep "^imagemagick$" /tmp/brewlist > /dev/null 2>&1
then
	echo "Downloading ImageMagick"
	brew install imagemagick
fi
if ! grep "^julia$" /tmp/brewlist > /dev/null 2>&1
then
	echo "Downloading Julia"
	brew install julia
fi

echo "Downloading dependencies"
julia -E 'import Pkg; Pkg.activate(joinpath(homedir(), "FaceDetection.jl")); Pkg.instantiate()'

echo "Downloading face detection training data"
if [[ -d ~/FaceDetection.jl/test/images/pos/ ]]
then
	rm -rf ~/FaceDetection.jl/test/images/pos/
fi
if [[ -d ~/FaceDetection.jl/test/images/neg/ ]]
then
	rm -rf ~/FaceDetection.jl/test/images/neg/
fi
cd ~/FaceDetection.jl/ && \
	git clone https://github.com/OlegTheCat/face-detection-data && \
	mv face-detection-data/pos/ test/images/ && \
	mv face-detection-data/neg/ test/images/ && \
	rm -rf face-detection-data/
echo "Pruning the positive training images to have the same number as the negative images, or else there will be an array mismatch when constructing the image array in src/Adaboost.jl"
for i in $(seq $(ls ~/FaceDetection.jl/test/images/neg/ | wc -l) $(($(ls ~/FaceDetection.jl/test/images/pos/ | wc -l)-1)); do
	rm ~/FaceDetection.jl/test/images/pos/${i}.pgm
	# echo "${i}"
done

	
#### The following step was only for python testing code.  Netpbm.jl is powerful and fixed this.

# echo "Converting pgm files to png.  This will take a minute."
# find ../test/images/ -name "*.pgm" | \
# while IFS= read -r file; do
# 	magick "${file}" "${file}.png"
# done
	
