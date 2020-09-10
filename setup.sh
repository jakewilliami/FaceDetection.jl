#! /usr/bin/env bash

trap "exit" INT

# https://github.com/betars/Face-Resources
# https://github.com/polarisZhao/awesome-face#-datasets
# https://github.com/jian667/face-dataset

FD_HOME="$(realpath $(dirname $0))"
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
		if ! grep "^julia$" /tmp/brewlist > /dev/null 2>&1
		then
			echo "Downloading Julia"
			brew cask install julia
		fi
	else
		echo "Please ensure Julia is downloaded and in your path."
		sleep 10
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
    wget "${URL}" || echo -e "An error has occurred whilst trying to download the LFW dataset."
    DOWNLOADED_FILE="${URL##*/}"
    tar xvzf "${DOWNLOADED_FILE}"
    EXTRACTED_DIR="${DOWNLOADED_FILE%%.*}"
	rm "${DOWNLOADED_FILE}"
	mkdir "${FD_HOME}/data/lfw-all/"
	find "${FD_HOME}/data/lfw/" -type f -name "*.jpg" -print | \
	while IFS= read -r file
	do
		mv -v "${file}" "${FD_HOME}/data/lfw-all/"
	done
	rm -rf "${FD_HOME}/data/lfw/"
    cd - > /dev/null
}

collateAllNonFaces(){
	if [[ -d ${FD_HOME}/data/all-non-faces/ ]]
	then
		rm -rf ${FD_HOME}/data/all-non-faces/
	fi
	
	mkdir -p ${FD_HOME}/data/all-non-faces/

	cp -rv ${FD_HOME}/data/alt/neg/ ${FD_HOME}/data/all-non-faces/
	cp -rv ${FD_HOME}/data/main/testset/non-faces/ ${FD_HOME}/data/all-non-faces/
	cp -rv ${FD_HOME}/data/main/trainset/non-faces/ ${FD_HOME}/data/all-non-faces/
}


main() {
	setupWD
	checkPackages
	obtainDatasetMain
	obtainDatasetAlt
	obtainMITDataset
	obtainFDDBDataset
	obtainLabelledFacesInTheWildDataset
	collateAllNonFaces
}

if [[ "$(whoami)" == "jakeireland" && "$(uname -s)" == "Darwin" ]]
then
	obtainDatasetMain
	obtainDatasetAlt
	obtainLabelledFacesInTheWildDataset
	collateAllNonFaces
else
	main
fi
