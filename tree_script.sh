#DATASET_DIR='/home/pavel/projects/horses/soft/python/morphometry/datasets/2024/traits'
DATASET_DIR='/home/pavel/projects/horses/soft/python/morphometry/datasets/2025/new_photo_by_kalinkina'
CUR_DIR=`pwd`
cd $DATASET_DIR
tree -iFPf '*.tps' --prune | grep tps > filelist.txt
tree -iFPf '*.TPS' --prune | grep TPS >> filelist.txt
mv filelist.txt $CUR_DIR
cd $CUR_DIR


