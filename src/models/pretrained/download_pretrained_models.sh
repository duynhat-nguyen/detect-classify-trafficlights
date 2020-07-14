mkdir -p ./models/pretrained && cd ./models/pretrained
# SSD
wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
tar -xvf ssd_inception_v2_coco_2018_01_28.tar.gz
rm ssd_inception_v2_coco_2018_01_28.tar.gz
# EfficientDet
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d2_coco17_tpu-32.tar.gz
tar -xvf efficientdet_d2_coco17_tpu-32.tar.gz
rm efficientdet_d2_coco17_tpu-32.tar.gz
