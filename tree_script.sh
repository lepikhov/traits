DATASET_DIR=$1
CUR_DIR=`pwd`
cd $DATASET_DIR
tree -iFPf '*.tps' --prune | grep tps > filelist.txt
tree -iFPf '*.TPS' --prune | grep TPS >> filelist.txt
mv filelist.txt $CUR_DIR
cd $CUR_DIR


