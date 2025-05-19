# previous svit finetune result
export CUDA_VISIBLE_DEVICES=2
# ./dist_test.sh configs/mask_rcnn/svit-adapter-t-0.5x-ftune.py pretrained/svit-adapter-t-0.5x.pth 1 29600 --eval bbox segm # >> org_test.txt

# apply tome
# source .envpy3.8/bin/activate
./dist_test.sh configs/mask_rcnn/tome-vit-adapter-t-3x.py pretrained/vit-adapter-t-3x.pth 1 29800 --eval bbox segm # >> org_test.txt

# test pure vit
# ./dist_test.sh configs/mask_rcnn/vit-adapter-t-3x.py pretrained/vit-adapter-t-3x.pth 1 29800 --eval bbox segm # >> org_pure_vit.txt
