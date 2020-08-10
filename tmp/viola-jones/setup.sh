#! /bin/bash

if [[ ! -d data/ ]]
then
	git clone https://github.com/opencv/opencv

	mv opencv/data ./
	rm -rf opencv/
fi

pip3 install numpy cv2-utils cvtools opencv-python
