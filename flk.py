import re
import cv2
from tflite_runtime.interpreter import Interpreter
import numpy as np
import mediapipe as mp
from datetime import datetime 
import time
import json

from flask import Flask
from flask_cors import CORS
from flask_restful import Resource, Api


app = Flask(__name__)

api = Api(app)

CORS(app)

CAMERA_WIDTH = 320
CAMERA_HEIGHT = 320

@app.route('/')





def main():
    pillTime=''
    labels = load_labels()
    interpreter = Interpreter('detect.tflite')
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector(maxFaces=1)
    while True:
        success, img = cap.read()
        img, faces,checkMouth = detector.findFaceMesh(img)
        img=cv2.resize(img, (320,320))
        res = detect_objects(interpreter, img, 0.8)
        #print(res)
        #print(checkMouth)
        if res:
            for result in res:
                ymin, xmin, ymax, xmax = result['bounding_box']
                xmin = int(max(1,xmin * CAMERA_WIDTH))
                xmax = int(min(CAMERA_WIDTH, xmax * CAMERA_WIDTH))
                ymin = int(max(1, ymin * CAMERA_HEIGHT))
                ymax = int(min(CAMERA_HEIGHT, ymax * CAMERA_HEIGHT))
                
                cv2.rectangle(img,(xmin, ymin),(xmax, ymax),(0,255,0),3)
                cv2.putText(img,labels[int(result['class_id'])],(xmin, min(ymax, CAMERA_HEIGHT-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2,cv2.LINE_AA) 

        img =cv2.resize(img, (640,480))
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & checkMouth==1 and res:
            cv2.imwrite('image.png', img)
            t=time.time()
            pillTime=time.ctime(t)
            break
        pillJson=json.dumps({'PillTaken':"YES",'PillTakenDateTime':datetime.now().strftime("%Y/%m/%d %H:%M:%S")})
    return pillJson

def load_labels(path='labels.txt'):
  """Loads the labels file. Supports files with or without index numbers."""
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  return labels

def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = np.expand_dims((image-255)/255, axis=0)


def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  # Get all output details
  boxes = get_output_tensor(interpreter, 0)
  classes = get_output_tensor(interpreter, 1)
  scores = get_output_tensor(interpreter, 2)
  count = int(get_output_tensor(interpreter, 3))

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
          'bounding_box': boxes[i],
          'class_id': classes[i],
          'score': scores[i]
      }
      results.append(result)
  return results

class FaceMeshDetector():
    
    def __init__(self, staticMode=False, maxFaces=1, minDetectionCon=0.5, minTrackCon=0.5):

        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces,
                                                 self.minDetectionCon, self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)
        self.mpHolistic=mp.solutions.holistic
        self.holistic=self.mpHolistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def calculate_angle(self,a,b,c):
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End
    
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle >180.0:
            angle = 360-angle
            
        return angle

    def get_distance(self, a, b):
        a = np.array(a)
        b = np.array(b)
        return np.sqrt(np.sum((a - b) ** 2, axis=0))

    def findFaceMesh(self, img, draw=True):
        checkMouth=0
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        self.holisticResults= self.holistic.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    lipsOuter = [
                    # lipsUpperOuter
                    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                    # lipsLowerOuter in reverse order
                    375, 321, 405, 314, 17, 84, 181, 91, 146, 61]

                    lipsInner = [
                    # lipsUpperInner
                    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
                    # lipsLowerOuter in reverse order
                    324, 318, 402, 317, 14, 87, 178, 88, 95, 78]

                    lipsMiddle = [12,15]
                    # print("Facemesh: ", self.mpFaceMesh.FACE_CONNECTIONS)
                    # print(type(self.mpFaceMesh.FACE_CONNECTIONS))
                    tempLms = faceLms
                    for id, lms in enumerate(tempLms.landmark):
                        if id not in lipsOuter and id not in lipsInner and id not in lipsMiddle:
                            lms.x = 0
                            lms.y = 0
                            lms.z = 0
                    self.mpDraw.draw_landmarks(img, tempLms, self.mpFaceMesh.FACE_CONNECTIONS,
                                           self.drawSpec, self.drawSpec)
                    #self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACE_CONNECTIONS,
                    #                      self.drawSpec, self.drawSpec)
                    
                    # Right hand
                    self.mpDraw.draw_landmarks(img, self.holisticResults.right_hand_landmarks, self.mpHolistic.HAND_CONNECTIONS)

                    # Left Hand
                    self.mpDraw.draw_landmarks(img, self.holisticResults.left_hand_landmarks, self.mpHolistic.HAND_CONNECTIONS)

                    # Pose Detections
                   # self.mpDraw.draw_landmarks(img, self.holisticResults.pose_landmarks, self.mpHolistic.POSE_CONNECTIONS)
                face = []
                for id,lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x,y = int(lm.x*iw), int(lm.y*ih)
                    #cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                     #           0.7, (0, 255, 0), 1)
                    # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), 1)

                    face.append([x,y])
                # print('Distance: ', self.get_distance(face[15], face[12]))
                # print('Lower Lip distance', self.get_distance(face[15], face[17]))
                # print('OuterAngle: ', self.calculate_angle(face[0],face[61],face[17]))
                # print('InnerAngle: ', self.calculate_angle(face[13],face[61],face[14]))
                # print('Angle: ', self.calculate_angle(face[12],face[61],face[15]))
                data = {}
                data['omdist'] = self.get_distance(face[15], face[12])
                data['lldist'] = self.get_distance(face[15], face[17])
                data['angle'] = self.calculate_angle(face[12],face[61],face[15])
                data['outangle'] = self.calculate_angle(face[0],face[61],face[17])
                data['inangle'] = self.calculate_angle(face[13],face[61],face[14])
                cv2.putText(img, f'Open Mouth distance: {int(data["omdist"])}',
                    (450, 70), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255), 1)
                cv2.putText(img, f'Lower Lip distance: {int(data["lldist"])}', 
                    (450, 85), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
                cv2.putText(img, f'Angle: {int(data["angle"])}', 
                    (450, 100), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
                cv2.putText(img, f'Outer Angle: {int(data["outangle"])}', 
                    (450, 115), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
                cv2.putText(img, f'Inner Angle: {int(data["inangle"])}', 
                    (450, 130), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

                if int(data['omdist']) > 1.5 * int(data['lldist']):
                    checkMouth=1
                    cv2.putText(img, 'Mouth: Open', (20, 450), 
                        cv2.FONT_HERSHEY_PLAIN, 3,  (0, 0, 255), 5)
                else:
                    cv2.putText(img, 'Mouth: Closed', (20, 450), 
                        cv2.FONT_HERSHEY_PLAIN, 3,  (0, 0, 255), 5)
                    
                faces.append(face)
        return img, faces,checkMouth

if __name__ == '__main__':
    app.run()