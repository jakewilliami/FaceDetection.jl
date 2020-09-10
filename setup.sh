#! /usr/bin/env bash

trap "exit" INT

# if [[ $(uname -s) != "Darwin" ]]
# then
# 	echo "OpenCV has been ported to Julia, but has only been tested on OSX.  Please keep this in mind when using this implementation of Viola-Jone."
# fi

# https://github.com/betars/Face-Resources

#echo "Please ensure FaceDetection.jl is installed in your home directory..."
#sleep 5
FD_HOME="$(dirname $0)"
# FD_HOME="${HOME}/FaceDetection.jl/"
MAIN="${FD_HOME}/data/main/"
ALT="${FD_HOME}/data/alt/"


setupWD() {
	mkdir -p ${ALT}/testing/testing/pos/
	mkdir -p ${ALT}/testing/testing/neg/
}


checkPackages() {
	if [[ $(uname -s) == "Darwin" ]]
	then
		brew list > /tmp/brewlist
		# if ! grep "^imagemagick$" /tmp/brewlist > /dev/null 2>&1
		# then
		# 	echo "Downloading ImageMagick"
		# 	brew install imagemagick
		# fi
		if ! grep "^julia$" /tmp/brewlist > /dev/null 2>&1
		then
			echo "Downloading Julia"
			brew cask install julia
		fi
	else
		echo "Please ensure Julia is downloaded and in your path."
	fi

	echo "Downloading dependencies"
	julia -E 'import Pkg; Pkg.activate(joinpath(homedir(), "FaceDetection.jl")); Pkg.instantiate()'
}


obtainDatasetAlt() {
	echo "Downloading alternative face detection training data"
	[[ -d ${FD_HOME}/face-detection-data/ ]] && rm -rf ${FD_HOME}/face-detection-data/
	[[ -d ${ALT}/pos/ ]] && rm -rf ${ALT}/pos/
	[[ -d ${ALT}/neg/ ]] && rm -rf ${ALT}/neg/
	
	cd ${FD_HOME}/ && \
		git clone https://github.com/OlegTheCat/face-detection-data && \
		mv ${FD_HOME}/face-detection-data/pos/ ${ALT}/ && \
		mv ${FD_HOME}/face-detection-data/neg/ ${ALT}/ && \
		rm -rf ${FD_HOME}/face-detection-data/
	# echo "Pruning the positive training images to have the same number as the negative images, or else there will be an array mismatch when constructing the image array in src/Adaboost.jl"
	# for i in $(seq $(ls ${ALT}/neg/ | wc -l) $(($(ls ${ALT}/pos/ | wc -l)-1))); do
	# 	rm ${ALT}/pos/${i}.pgm
	# 	# echo "${i}"
	# done
}


obtainDatasetMain() {
	echo "Dowloading training data"
	[[ -d ${FD_HOME}/Viola-Jones/ ]] && rm -rf ${FD_HOME}/Viola-Jones/
	[[ -d ${MAIN}/testset/ ]] && rm -rf ${MAIN}/testset/
	[[ -d ${MAIN}/trainset/ ]] && rm -rf ${MAIN}/trainset/
	
	mkdir -p ${MAIN}/
	cd ${FD_HOME}/ && \
		git clone https://github.com/INVASIS/Viola-Jones/ && \
		mv ${FD_HOME}/Viola-Jones/data/testset/ ${MAIN}/ && \
		mv ${FD_HOME}/Viola-Jones/data/trainset/ ${MAIN}/ && \
		rm -rf ${FD_HOME}/Viola-Jones
#	echo "Pruning the positive training images to have the same number as the negative images, or else there will be an array mismatch when constructing the image array in src/Adaboost.jl"
	# for i in $(seq $(ls ${ALT}/trainset/faces | wc -l) $(($(ls ~/FaceDetection.jl/test/images/pos/ | wc -l)-1))); do
	# 	rm ${ALT}/trainset/non-faces/${i}.pgm
	# 	# echo "${i}"
	# done
	
#	find ${MAIN}/trainset/non-faces/ -maxdepth 1 -type f -name "*.png" -print | \
#        head -n $(($(ls ${MAIN}/trainset/non-faces | wc -l)-$(ls ${MAIN}/trainset/faces | wc -l))) |\
#		while IFS= read -r file
#		do
#			rm "${file}"
#		done
}


obtainMITDataset() {
	URL="http://cbcl.mit.edu/projects/cbcl/software-datasets/faces.tar.gz"
	cd ${FD_HOME}/data/
    wget "${URL}" || echo -e "An error has occurred whilst trying to download the CMU/MIT dataset."
    DOWNLOADED_FILE="${URL##*/}"
    tar xvzf "${DOWNLOADED_FILE}"
    EXTRACTED_DIR="${DOWNLOADED_FILE%%.*}"
	rm "${DOWNLOADED_FILE}"
    cd - > /dev/null
}


obtainFDDBDataset() {
	URL="http://tamaraberg.com/faceDataset/originalPics.tar.gz"
	cd ${FD_HOME}/data/
    wget "${URL}" || echo -e "An error has occurred whilst trying to download the FDDB dataset."
    DOWNLOADED_FILE="${URL##*/}"
    tar xvzf "${DOWNLOADED_FILE}"
    EXTRACTED_DIR="${DOWNLOADED_FILE%%.*}"
	rm "${DOWNLOADED_FILE}"
    cd - > /dev/null
}


obtainLabelledFacesInTheWildDataset() {
	URL="http://vis-www.cs.umass.edu/lfw/lfw.tgz"
	cd ${FD_HOME}/data/
    wget "${URL}" || echo -e "An error has occurred whilst trying to download the FDDB dataset."
    DOWNLOADED_FILE="${URL##*/}"
    tar xvzf "${DOWNLOADED_FILE}"
    EXTRACTED_DIR="${DOWNLOADED_FILE%%.*}"
	rm "${DOWNLOADED_FILE}"
    cd - > /dev/null
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
	setupWD
	checkPackages
	obtainDatasetMain
	obtainDatasetAlt
}

if [[ "$(whoami)" == "jakeireland" && "$(uname -s)" == "Darwin" ]]; then
	obtainDatasetMain
else
	main
fi
