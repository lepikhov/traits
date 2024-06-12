DATASET_DIR='/home/pavel/projects/horses/soft/python/morphometry/datasets/2024/traits'
CUR_DIR=`pwd`
cd $DATASET_DIR
tree -iFPf '*.tps' --prune | grep tps > filelist.txt
mv filelist.txt $CUR_DIR
cd $CUR_DIR


