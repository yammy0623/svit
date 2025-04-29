# previous svit finetune result
./dist_test.sh configs/mask_rcnn/svit-adapter-t-0.5x-ftune.py pretrained/svit-adapter-t-0.5x.pth 4 29600 --eval bbox segm

# apply tome
# source .envpy3.8/bin/activate
# ./dist_test.sh configs/mask_rcnn/tome-vit-adapter-t-3x.py pretrained/vit-adapter-t-3x.pth 4 29800 --eval bbox segm
