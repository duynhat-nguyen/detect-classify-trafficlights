cd ./src/api
git clone https://github.com/tensorflow/models.git 
cd models/research
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
cp object_detection/packages/tf2/setup.py .
python -m pip install .