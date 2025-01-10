import streamlit as st
import numpy as np
import cv2 as cv
from PIL import Image  ## used for open img

DEMO_IMAGE='image.jpg'

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                   "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                   "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                   "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                   ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                   ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                   ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                   ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

inwidth=368
inheight=368
net=cv.dnn.readNetFromTensorflow("graph_opt.pb") ## pretrained model using to pose detection


st.title("Human Pose Estimation opencv")
st.text("Make sure you have a clear image with all the parts clearly visible")

img_file_buffer=st.file_uploader("upload an image, Make sure you have a clear image",type=["jpg","jpeg","png"])
if img_file_buffer is not None:
   image=np.array(Image.open(img_file_buffer))
else:
   demo_image=DEMO_IMAGE
   image=np.array(Image.open(demo_image))
st.subheader('Original Image')
st.image(
    image, caption =f"Original Image",use_column_width=True
)
thres=st.slider("Threshold for detecting the key points",min_value=0,value=20,max_value=100,step=5)
thres=thres/100

@st.cache
def pose_estimation(frame):
    framewidth=frame.shape[1]
    frameheight=frame.shape[0]
    #  used to resize nd normalizing the img give clear version img
    net.setInput(cv.dnn.blobFromImage(frame,1.0,(inwidth,inheight),(127.5,127.5,127.5),swapRB=True,crop=False))
    
    out = net.forward() # giving actual prediction
    out=out[:,:19,:,:] # mobilenet output [1,57,-1,-1], we only need the firs 19 element
    assert(len(BODY_PARTS) <= out.shape[1])

    points = [] ## list
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponding body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (framewidth * point[0]) / out.shape[3]
        y = (frameheight * point[1]) / out.shape[2]

        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > thres else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    return frame


output=pose_estimation(image)
st.subheader("Position Estimated")
st.image(
    output,caption=f"Position Estimated",use_column_width=True)
st.markdown('''
             #
             ''')