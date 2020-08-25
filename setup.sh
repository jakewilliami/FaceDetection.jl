#! /usr/bin/env bash

trap "exit" INT

# if [[ $(uname -s) != "Darwin" ]]
# then
# 	echo "OpenCV has been ported to Julia, but has only been tested on OSX.  Please keep this in mind when using this implementation of Viola-Jone."
# fi

echo "Please ensure FaceDetection.jl is installed in your home directory..."
sleep 5

FD_HOME="${HOME}/FaceDetection.jl/"
MAIN="${FD_HOME}/data/main/"
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


obtainDatasetAlt() {
	echo "Downloading alternative face detection training data"
	if [[ -d ${ALT}/pos/ ]]
	then
		rm -rf ${ALT}/pos/
	fi
	if [[ -d ${ALT}/neg/ ]]
	then
		rm -rf ${ALT}/neg/
	fi
	cd ${FD_HOME}/ && \
		git clone https://github.com/OlegTheCat/face-detection-data && \
		mv ${FD_HOME}/face-detection-data/pos/ ${ALT}/ && \
		mv ${FD_HOME}/face-detection-data/neg/ ${ALT}/ && \
		rm -rf ${FD_HOME}/face-detection-data/
	echo "Pruning the positive training images to have the same number as the negative images, or else there will be an array mismatch when constructing the image array in src/Adaboost.jl"
	for i in $(seq $(ls ${ALT}/neg/ | wc -l) $(($(ls ${ALT}/pos/ | wc -l)-1))); do
		rm ${ALT}/pos/${i}.pgm
		# echo "${i}"
	done
}


obtainDatasetMain() {
	echo "Dowloading training data"
	if [[ -d ${FD_HOME}/Viola-Jones/ ]]
	then
		rm -rf ${FD_HOME}/Viola-Jones/
	fi
	if [[ -d ${MAIN}/testset/ ]]
	then
		rm -rf ${MAIN}/testset/
	fi
	if [[ -d ${MAIN}/trainset/ ]]
	then
		rm -rf ${MAIN}/trainset/
	fi
	mkdir -p ${MAIN}/
	cd ${FD_HOME}/ && \
		git clone https://github.com/INVASIS/Viola-Jones/ && \
		mv ${FD_HOME}/Viola-Jones/data/testset/ ${MAIN}/ && \
		mv ${FD_HOME}/Viola-Jones/data/trainset/ ${MAIN}/ && \
		rm -rf ${FD_HOME}/Viola-Jones
	echo "Pruning the positive training images to have the same number as the negative images, or else there will be an array mismatch when constructing the image array in src/Adaboost.jl"
	# for i in $(seq $(ls ${ALT}/trainset/faces | wc -l) $(($(ls ~/FaceDetection.jl/test/images/pos/ | wc -l)-1))); do
	# 	rm ${ALT}/trainset/non-faces/${i}.pgm
	# 	# echo "${i}"
	# done
	
	find ${MAIN}/trainset/non-faces/ -maxdepth 1 -type f -name "*.png" -print | \
        head -n $(($(ls ${MAIN}/trainset/non-faces | wc -l)-$(ls ${MAIN}/trainset/faces | wc -l))) |\
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
	obtainDatasetMain
	obtainDatasetAlt
}


main
