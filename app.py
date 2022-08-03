from flask import Flask, render_template, request, redirect, url_for, flash
#import http response
from flask_http_response import success, result, error

import matplotlib.pyplot as plt 
import cv2
from pylab import rcParams
from IPython.display import Image
import cv2
import numpy as np
import mediapipe as mp
import argparse
import imutils
import math
from PIL import Image
import easyocr
from flask import jsonify

app = Flask(__name__)

reader=easyocr.Reader(['en'])



@app.route('/')
def form():
    return render_template('index.html')

@app.route('/', methods = ['POST']) #, 'GET'
def upload():
    if request.method == 'POST':
        f = request.files['file']
        reader=easyocr.Reader(['en'])    
        img = cv2.imread(f.filename)
        print("cv2 img",img)
        rcParams['figure.figsize']=8,16

        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils
        # Face Analisys

        #plt.title("Initial Image");plt.axis('on');plt.imshow(img);plt.show()

        img_height = img.shape[0]
        img_width = img.shape[1]
        face_detection_results = face_detection.process(img[:,:,::-1])
        if face_detection_results.detections:
            for face_no, face in enumerate(face_detection_results.detections):
                face_data = face.location_data

        #Declaring the variables for Calculating Bounding Box manually

        rxmin  = face_data.relative_bounding_box.xmin
        rymin  = face_data.relative_bounding_box.ymin
        rwidth  = face_data.relative_bounding_box.width
        rheight  = face_data.relative_bounding_box.height
        nor_width = rwidth*img_width
        nor_height = rheight*img_height
        box_width = 5*nor_width
        box_height = 8*nor_height
        nose_x = (face_data.relative_keypoints[2].x)*img_width
        nose_y = (face_data.relative_keypoints[2].y)*img_height
        ID_x = int((nose_x)-(box_width/2))
        ID_y = int((nose_y)-(box_height*0.45))
        rex = (face_data.relative_keypoints[0].x)
        rey = (face_data.relative_keypoints[0].y)
        lex = (face_data.relative_keypoints[1].x)
        ley = (face_data.relative_keypoints[1].y)

        #

        start_point = (ID_x, ID_y)
        end_point = (int(ID_x+box_width),int(ID_y+box_height))
        color = (255, 255, 255)
        thickness = 3

        #Calculating the angle for the rotation of the image

        angle = math.atan(rey-ley/rex-lex)*57.2958

        #Rotating the image

        im_pil = Image.fromarray(img)
        out = im_pil.rotate(angle)
        #img1 = img
        img1 = np.array(out)

        #plt.title("Resultant Image");plt.axis('on');plt.imshow(img1);plt.show()

        #Taking the newly rotated as an input

        img_height = img1.shape[0]
        img_width = img1.shape[1] 
        face_detection_results = face_detection.process(img1[:,:,::-1])
        if face_detection_results.detections:
            for face_no, face in enumerate(face_detection_results.detections):
                face_data = face.location_data

        #Declaring the variables for Calculating Bounding Box manually(for the new image i.e. rotated)

        rxmin  = face_data.relative_bounding_box.xmin
        rymin  = face_data.relative_bounding_box.ymin
        rwidth  = face_data.relative_bounding_box.width
        rheight  = face_data.relative_bounding_box.height
        nor_width = rwidth*img_width
        nor_height = rheight*img_height
        box_width = 5*nor_width
        box_height = 8*nor_height
        nose_x = (face_data.relative_keypoints[2].x)*img_width
        nose_y = (face_data.relative_keypoints[2].y)*img_height
        ID_x = int((nose_x)-(box_width/2))
        ID_y = int((nose_y)-(box_height*0.45))
        rex = (face_data.relative_keypoints[0].x)
        rey = (face_data.relative_keypoints[0].y)
        lex = (face_data.relative_keypoints[1].x)
        ley = (face_data.relative_keypoints[1].y)

        #Declaring the Bounding Box(for the new image i.e. rotated)

        start_point = (ID_x, ID_y)
        end_point = (int(ID_x+box_width),int(ID_y+box_height))
        color = (255, 255, 255)
        thickness = 3
        img_copy = img1[:,:,::-1].copy()
        if face_detection_results.detections:
            for face_no, face in enumerate(face_detection_results.detections):
                mp_drawing.draw_detection(image=img_copy, detection=face, keypoint_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0),thickness=5,circle_radius=2))

        # Plotting the image with bounding box
                                                                                    
        image1 = cv2.rectangle(img_copy, start_point, end_point, color, thickness)
        input_pts = np.float32([[ID_x,ID_y],[(ID_x+box_width),ID_y],
                                [(ID_x+box_width),(ID_y+box_height)],[ID_x,(ID_y+box_height)]])
        output_pts = np.float32([[0,0],[img_width,0],[img_width,img_height],[0,img_height]])
        M = cv2.getPerspectiveTransform(input_pts,output_pts)
        out = cv2.warpPerspective(image1,M,(image1.shape[1],image1.shape[0]))

        #Using OCR to find the text on the wrapped image

        output=reader.readtext(out)
        output[5:-1]
        print(output)
        result=""
        for i in output:
            for j in i:
                if(j == "VELLORE CAMPUS" or j == "CAMPUS"):
                    print("flag\n")
                    result = output[output.index(i) + 1 : -1]

        #printing the resultant text
        l = []
        for i in result:
            a,b,c = i
            #print(b)
            l.append(b)
        print("list printing",l)
    return render_template('index.html',l=l)

#call main
if __name__ == "__main__":
    app.run(debug=True)
