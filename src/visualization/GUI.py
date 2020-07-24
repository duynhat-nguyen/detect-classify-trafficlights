import streamlit as st
import cv2
from PIL import Image
import pandas as pd
import numpy as np
import altair as alt
import os

def main():
    st.title("Traffic lights detection and classification")

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Info", "Check train set", "Inference 1 image"])

    if app_mode == "Check train set":
        check_train_set()
    elif app_mode == "Info":
        info()
    elif app_mode == "Inference 1 image":
        inference()

def info():
    print("Hello from info")

def check_train_set():
    # st.write("Hello from run_the_app")

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

    train_label_file = "/content/detect-classify-trafficlights/data/tfod/train/_annotations.csv"

    metadata = load_metadata(train_label_file)

    summary = create_summary(metadata)

    selected_frame_index, selected_frame = frame_selector_ui(summary)

    image_path = os.path.join("/content/detect-classify-trafficlights/data/tfod/train/", selected_frame)

    image = load_image(image_path)

    st.write(image_path)
    
    boxes = metadata[metadata.filename == selected_frame].drop(columns=["filename", "width", "height"])
    
    st.write(boxes)

    draw_image_with_boxes(image, boxes, "Ground Truth")

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
    # st.sidebar.altair_chart(alt.layer(chart, vline, width=300))

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

def inference():
    st.write("Hello from inference")

@st.cache(show_spinner=False)
def load_image(image_path):
    image = cv2.imread(image_path)
    image = image[:, :, [2, 1, 0]] # BGR -> RGB
    return image

if __name__ == "__main__":
    main()

