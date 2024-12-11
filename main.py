from ultralytics import YOLO
import random
import cv2
from tracker import Tracker
import streamlit as st

def main():

    #interface
    st.title('Catfish Counter')
    st.sidebar.title('Options')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > dif:first-child{width: 400px;}
    [data-testid="stSidebar"][aria-expanded="false"] > dif:first-child{width: 400px; margin-left: -400px;}
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown('---')

    conf = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.5)

    st.sidebar.markdown('---')

    camera = st.sidebar.radio("Pick a camera", [0, 1, 2, 3, 4, "Smartphone", "Upload (showcase)"],
                              captions= ["Camera 1", "Camera 2", "Camera 3", "Camera 4", "Camera 5", "Smartphone Camera", "Presentation only function"])

    if camera == "Smartphone":
        ip = st.sidebar.text_area("Camera IP", placeholder="Example: http://192.168.101.198:8080/video")
        st.sidebar.markdown('---')
        st.sidebar.markdown("Please get the IP Address of your camera from IP Webcam application in Playstore")
    elif camera == "Upload (showcase)":
        file = st.file_uploader("Choose a video", type=["mp4"])
        if file is not None:
            vid = file.name
            with open(vid, mode='wb') as f:
                f.write(file.read())
            st.video(file)
    else:
        ip = camera

    st.sidebar.markdown('---')

    live_button = st.sidebar.button('Preview')

    st.sidebar.markdown('---')

    with st.container(height=520, border=True):
        placeholder = st.empty()

    cola, colb = st.columns(2)
    with cola:
        st.markdown('Fish Count')
        count = st.markdown("0")
    with colb:
        st.markdown('Highest')
        maximum = st.markdown("0")

    col1, col2 = st.columns(2)
    with col1:
        start_button_pressed = st.button('Start')
    with col2:
        stop_button_pressed = st.button('Stop')

    if live_button:
        if camera == "Smartphone":
            cap = cv2.VideoCapture('{}'.format(ip))
            if cap is None or not cap.isOpened():
                placeholder.text("Camera is not available")
            else:
                while cap.isOpened() and not start_button_pressed:
                    ret, im = cap.read()
                    cv2.resize(im, (640, 480))
                    placeholder.image(im, channels="BGR")
        elif camera == "Upload (showcase)":
            cap = cv2.VideoCapture(vid)
            if cap is None or not cap.isOpened():
                placeholder.text("Video is not available")
            else:
                while cap.isOpened() and not start_button_pressed:
                    ret, im = cap.read()
                    cv2.resize(im, (640, 480))
                    placeholder.image(im, channels="BGR")
        else:
            cap = cv2.VideoCapture(ip)
            if cap is None or not cap.isOpened():
                placeholder.text("Camera is not available")
            else:
                while cap.isOpened() and not start_button_pressed:
                    ret, im = cap.read()
                    cv2.resize(im, (640,480))
                    placeholder.image(im, channels="BGR")
    # program
    if start_button_pressed:
        total = []
        if camera == "Smartphone":
            cap = cv2.VideoCapture('{}'.format(ip))
            if cap is None or not cap.isOpened():
                placeholder.text("Camera is not available")
        elif camera == "Upload (showcase)":
            cap = cv2.VideoCapture(vid)
            if cap is None or not cap.isOpened():
                placeholder.text("Video is not available")
        else:
            cap = cv2.VideoCapture(ip)
            if cap is None or not cap.isOpened():
                placeholder.text("Camera is not available")

        model = YOLO("nopretrain_best.onnx")

        tracker = Tracker()

        colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

        while cap.isOpened() and not stop_button_pressed:
            ret, im = cap.read()

            results = model.predict(im, conf=conf, device='cpu', augment=True, stream_buffer=False)

            for result in results:
                detections = []
                for r in result.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = r
                    x1 = int(x1)
                    x2 = int(x2)
                    y1 = int(y1)
                    y2 = int(y2)
                    class_id = int(class_id)
                    detections.append([x1, y1, x2, y2, score])

                tracker.update(im, detections)

                counter = len(detections)
                total.append(counter)
                maximums = max(total)
                if len(total) > 20:
                    total.clear()

                for track in tracker.tracks:
                    bbox = track.bbox
                    x1, y1, x2, y2 = bbox
                    track_id = track.track_id
                    cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 2)

                    img = cv2.resize(im, (640, 480))
                    placeholder.image(img, channels="BGR")
                    count.write(str(counter))
                    maximum.write(str(maximums))

            if cv2.waitKey(1) & 0xFF == stop_button_pressed:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        placeholder.markdown(
            """<h2 style='text-align: center; color: red;'> No Camera Input</h2>""", unsafe_allow_html=True,
        )

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass