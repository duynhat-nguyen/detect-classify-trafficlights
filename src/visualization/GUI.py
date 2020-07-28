import streamlit as st
import cv2
from PIL import Image
import pandas as pd
import numpy as np
import altair as alt
import os
import numpy as np
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from glob import glob

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


def main():
    st.title("Traffic lights detection")

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Info", "Train set","Validation set", "Inference 1 image"])

    if app_mode == "Train set":
        check_train_set()
    elif app_mode == "Info":
        info()
    elif app_mode == "Inference 1 image":
        inference_1_image()
    elif app_mode == "Validation set":
        check_valid_set()
        

def info():
    print("Hello from info")


@st.cache
def load_metadata(path):
    return pd.read_csv(path)

# This function uses some Pandas magic to summarize the metadata Dataframe.
@st.cache
def create_summary(metadata):
    one_hot_encoded = pd.get_dummies(metadata[["filename", "class"]], columns=["class"])
    summary = one_hot_encoded.groupby(["filename"]).sum().rename(columns={
        "class_green": "green",
        "class_red": "red",
    })
    return summary

def check_valid_set():
    train_label_file = "/content/detect-classify-trafficlights/data/tfod/valid/_annotations.csv"

    metadata = load_metadata(train_label_file)

    summary = create_summary(metadata)

    selected_frame_index, selected_frame = frame_selector_ui(summary)

    image_path = os.path.join("/content/detect-classify-trafficlights/data/tfod/valid/", selected_frame)

    image = load_image(image_path)

    st.text(image_path)
    
    boxes = metadata[metadata.filename == selected_frame].drop(columns=["filename", "width", "height"])
    
    st.write(boxes)

    draw_image_with_boxes(image, boxes, "Ground Truth")

    inference(image_path)

def check_train_set():
    train_label_file = "/content/detect-classify-trafficlights/data/tfod/train/_annotations.csv"

    metadata = load_metadata(train_label_file)

    summary = create_summary(metadata)

    selected_frame_index, selected_frame = frame_selector_ui(summary)

    image_path = os.path.join("/content/detect-classify-trafficlights/data/tfod/train/", selected_frame)

    image = load_image(image_path)

    st.text(image_path)
    
    boxes = metadata[metadata.filename == selected_frame].drop(columns=["filename", "width", "height"])
    
    st.write(boxes)

    draw_image_with_boxes(image, boxes, "Ground Truth")

    inference(image_path)

    # Uncomment these lines to peek at these DataFrames.
    # st.write('## Metadata', metadata[:], '## Summary', summary[:])

    # st.write("There are ", summary["green"][:].sum(), "boxes of green class")
    # st.write("There are ", summary["red"][:].sum(), "boxes of red class") 
    # st.write("Min(number of green boxes): ", summary["green"][:].min())
    # st.write("Max(number of green boxes): ", summary["green"][:].max())
    # st.write("Min(number of red boxes): ", summary["red"][:].min())
    # st.write("Min(number of red boxes): ", summary["red"][:].max())


def frame_selector_ui(summary):
    st.sidebar.markdown("# Frame")

    # The user can pick which type of object to search for.
    # object_type = st.sidebar.selectbox("Search for which objects?", summary.columns, 2)
    object_type = st.sidebar.selectbox("Search for which objects?", summary.columns)
    if object_type == "green":
        min_elts = 0
        max_elts = 5
    elif object_type == "red":
        min_elts = 0
        max_elts = 4

    # # The user can select a range for how many of the selected objecgt should be present.
    # min_elts, max_elts = st.sidebar.slider("How many %ss (select a range)?" % object_type, 0, 25, [10, 20])
    selected_frames = get_selected_frames(summary, object_type, min_elts, max_elts)

    # # Choose a frame out of the selected frames.
    # selected_frame_index = st.sidebar.slider("Choose a frame (index)", 0, len(selected_frames) - 1, 0)
    selected_frame_index = st.sidebar.slider("Choose a frame (index)", 0, 71, 0)

    # # Draw an altair chart in the sidebar with information on the frame.
    objects_per_frame = summary.loc[selected_frames, object_type].reset_index(drop=True).reset_index()
    chart = alt.Chart(objects_per_frame, height=120).mark_area().encode(
        alt.X("index:Q", scale=alt.Scale(nice=False)),
        alt.Y("%s:Q" % object_type))
    selected_frame_df = pd.DataFrame({"selected_frame": [selected_frame_index]})
    vline = alt.Chart(selected_frame_df).mark_rule(color="red").encode(
        alt.X("selected_frame:Q", axis=None)
    )
    st.sidebar.altair_chart(alt.layer(chart, vline, width=300))

    selected_frame = selected_frames[selected_frame_index]
    return selected_frame_index, selected_frame

def draw_image_with_boxes(image, boxes, header):
    # Superpose the semi-transparent object detection boxes.    # Colors for the boxes
    LABEL_COLORS = {
        "red": [255, 0, 0],
        "green": [0, 255, 0],
    }
    image_with_boxes = image.astype(np.float64)
    for _, (label, xmin, ymin, xmax, ymax) in boxes.iterrows():
        image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] += LABEL_COLORS[label]
        image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] /= 2

    # Draw the header and image.
    st.subheader(header)
    # st.markdown(description)
    st.image(image_with_boxes.astype(np.uint8), use_column_width=True)

@st.cache(hash_funcs={np.ufunc: str})
def get_selected_frames(summary, label, min_elts, max_elts):
    return summary[np.logical_and(summary[label] >= min_elts, summary[label] <= max_elts)].index

def inference(image_path):
    PATH_TO_FROZEN_GRAPH = "/content/drive/My Drive/detect-classify-trafficlights/tf1/exported/ssd/ssd_inception_v2_coco/frozen_inference_graph.pb"
    PATH_TO_LABELS = "/content/detect-classify-trafficlights/data/tfrecord/train/trafficlights_label_map.pbtxt"
    NUM_CLASSES = 2 #remember number of objects you are training? cool.

    ### Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    ###Loading label map
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    # PATH_TO_TEST_IMAGES_DIR = '/content/detect-classify-trafficlights/data/tfod/test/'
    # TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(0, 2) ]
    IMAGE_SIZE = (12, 8)

    image_tmp = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image = image_tmp.resize((1280, 720), Image.ANTIALIAS)
    
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=1)
    st.subheader("Inference")
    st.image(Image.fromarray(image_np), use_column_width=True)
    # print(type(image_np))

@st.cache(show_spinner=False)
def load_image(image_path):
    image = cv2.imread(image_path)
    image = image[:, :, [2, 1, 0]] # BGR -> RGB
    return image


### Load image into numpy function
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def inference_1_image():
    st.set_option('deprecation.showfileUploaderEncoding', False)
    image_path = st.file_uploader("Image for inference")
    if image_path is not None:
        image = Image.open(image_path)
        st.image(image, use_column_width=True)
        inference(image_path)

def model_ui():
    models_path = glob("/content/drive/My\ Drive/detect-classify-trafficlights/tf1/exported/ssd")
    models_list = []
    for i in models_path:
        models_list.append(os.path.basename(i))
    model_selection = st.sidebar.selectbox("Use which model?", models_list)
        
        
### Function to run inference on a single image which will later be used in an iteration
def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.int64)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

if __name__ == "__main__":
    main()
