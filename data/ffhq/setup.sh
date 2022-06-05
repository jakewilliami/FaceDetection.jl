echo "Please ensure you have followed the Google Drive API instructions listed here: https://docs.iterative.ai/PyDrive2/quickstart/"
sleep 5

pip3 install pydrive2
curl 'https://gist.githubusercontent.com/jakewilliami/6e361ca59df521c874a9021bde1d2c81/raw/2f277c36bcd725df71d30174e13f920d7bee7b97/download_ffhq_pydrive.py'  > download_ffhq_pydrive.py
echo "Downloading image thumbnails"
python3 download_ffhq.py -t --pydrive --cmd_auth

echo "Moving the images into one directory and deleting subdirectories."
# move images out of their subdirectories
for d in thumbnails128x128/*; do
	[ -d "$d" ] || continue
	for f in "$d"/*; do
		mv "$f" "thumbnails128x128/$(basename "$f")"
	done
done
# clean up the subdirectories
for d in thumbnails128x128/*; do
	if [ -d "$d" ]; then
		rm -d "$d"
	done
done
