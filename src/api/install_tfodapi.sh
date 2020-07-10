cd ./src/api/
git clone https://github.com/tensorflow/models.git
cd ./src/api/models/research/
protoc object_detection/protos/*.proto --python_out=.
