#!/bin/bash
#wget -q 'https://files.osf.io/v1/resources/jum2f/providers/osfstorage/5d4d7ec80f488d0017907d30?action=download&direct&version=2' -O 'password.txt'
echo "Downloading object_images_A-C.zip"
#wget 'https://files.osf.io/v1/resources/jum2f/providers/osfstorage/5f89eef1d85b700286657a33?action=download&direct&version=1' -O 'object_images_A-C.zip'
echo "Downloading object_images_D-K.zip"
#wget 'https://files.osf.io/v1/resources/jum2f/providers/osfstorage/5f89f02b37b6bb0248309053?action=download&direct&version=1' -O 'object_images_D-K.zip'
echo "Downloading object_images_L-Q.zip"
#wget 'https://files.osf.io/v1/resources/jum2f/providers/osfstorage/5f89f10e37b6bb02483092bb?action=download&direct&version=2' -O 'object_images_L-Q.zip'
echo "Downloading object_images_R-S.zip"
#wget 'https://files.osf.io/v1/resources/jum2f/providers/osfstorage/5f89f218d85b700291656821?action=download&direct&version=1' -O 'object_images_R-S.zip'
echo "Downloading object_images_T-Z.zip"
#wget 'https://files.osf.io/v1/resources/jum2f/providers/osfstorage/5f89f30a37b6bb02483098c8?action=download&direct&version=1' -O 'object_images_T-Z.zip'

mkdir object_images
for z in ./*.zip; do
	unzip -P 'things4all' "$z" -d "object_images_tmp"
done

for d in ./object_images_tmp/*; do
    for i in "$d"/*; do
	mv -v "$i" ./object_images/"$(basename "$i")"
    done
    rm -d "$d"
done
rm -d "object_images_tmp"

#for d in ./object_images_*; do
#	[ -d "$d" ] || continue
#	for d2 in "$d"/*; do
#		for f in "$d2"/*; do
#			echo mv -v "$f" ./object_images/"$(basename "$f")"
#		done
 #  	done
#	sleep 3
#	# rm -d "$d"
#done
