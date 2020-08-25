#! /usr/bin/env bash

trap "exit" INT

# if [[ $(uname -s) != "Darwin" ]]
# then
# 	echo "OpenCV has been ported to Julia, but has only been tested on OSX.  Please keep this in mind when using this implementation of Viola-Jone."
# fi

echo "Please ensure FaceDetection.jl is installed in your home directory..."
sleep 5

FD_HOME="${HOME}/FaceDetection.jl/"
MAIN="${FD_HOME}/data/images/"
ALT="${FD_HOME}/data/alt/"


setupWD() {
	mkdir -p ${MAIN}/testing/pos/
	mkdir -p ${MAIN}/testing/neg/
	mkdir -p ${MAIN}/testing/testing/pos/
	mkdir -p ${MAIN}/testing/testing/neg/
}


checkPackages() {
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
}


obtainDataset1() {
	echo "Downloading face detection training data"
	if [[ -d ${MAIN}/pos/ ]]
	then
		rm -rf ${MAIN}/pos/
	fi
	if [[ -d ${MAIN}/neg/ ]]
	then
		rm -rf ${MAIN}/neg/
	fi
	cd ${FD_HOME}/ && \
		git clone https://github.com/OlegTheCat/face-detection-data && \
		mv ${FD_HOME}/face-detection-data/pos/ ${MAIN}/ && \
		mv ${FD_HOME}/face-detection-data/neg/ ${MAIN}/ && \
		rm -rf ${FD_HOME}/face-detection-data/
	echo "Pruning the positive training images to have the same number as the negative images, or else there will be an array mismatch when constructing the image array in src/Adaboost.jl"
	for i in $(seq $(ls ${MAIN}/neg/ | wc -l) $(($(ls ${MAIN}/pos/ | wc -l)-1))); do
		rm ${MAIN}/pos/${i}.pgm
		# echo "${i}"
	done
}


obtainDataset2() {
	echo "Dowloading alternative training data"
	if [[ -d ${FD_HOME}/Viola-Jones/ ]]
	then
		rm -rf ${FD_HOME}/Viola-Jones/
	fi
	if [[ -d ${ALT}/testset/ ]]
	then
		rm -rf ${ALT}/testset/
	fi
	if [[ -d ${ALT}/trainset/ ]]
	then
		rm -rf ${ALT}/trainset/
	fi
	mkdir -p ${ALT}/
	cd ${FD_HOME}/ && \
		git clone https://github.com/INVASIS/Viola-Jones/ && \
		mv ${FD_HOME}/Viola-Jones/data/testset/ ${ALT}/ && \
		mv ${FD_HOME}/Viola-Jones/data/trainset/ ${ALT}/ && \
		rm -rf ${FD_HOME}/Viola-Jones
	echo "Pruning the positive training images to have the same number as the negative images, or else there will be an array mismatch when constructing the image array in src/Adaboost.jl"
	# for i in $(seq $(ls ${ALT}/trainset/faces | wc -l) $(($(ls ~/FaceDetection.jl/test/images/pos/ | wc -l)-1))); do
	# 	rm ${ALT}/trainset/non-faces/${i}.pgm
	# 	# echo "${i}"
	# done
	
	find ${ALT}/trainset/non-faces/ -maxdepth 1 -type f -name "*.png" -print | \
        head -n $(($(ls ${ALT}/trainset/non-faces | wc -l)-$(ls ${ALT}/trainset/faces | wc -l))) |\
		while IFS= read -r file
		do
			rm "${file}"
		done
}


convertPGM() {
#### The following step was only for python testing code.  Netpbm.jl is powerful and fixed this.

echo "Converting pgm files to png.  This will take a minute."
find ../test/images/ -name "*.pgm" | \
while IFS= read -r file; do
	magick "${file}" "${file}.png"
done
}




main() {
	# setupWD
	# checkPackages
	# obtainDataset1
	obtainDataset2
}


main
