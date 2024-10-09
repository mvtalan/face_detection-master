import face_recognition
import cv2
import numpy as np

#get a reference to the default webcam
video_capture = cv2.VideoCapture(0)

#load a sample picture and learn how to recognize it
your_image = face_recognition.load_image_file("markt.png")
your_face_encoding = face_recognition.face_encodings(your_image)[0]

#create an array of known face encodings
known_face_encodings = [
    your_face_encoding,
]

known_face_names = [
    'Mark'
]

print(known_face_encodings)

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    #grab a single frame of the video
    ret, frame = video_capture.read()
    
    #only process every other frame
    if process_this_frame:
        #resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0,0), fx=0.25, y=0.25)
        
        #convert image from BGR (OpenCV) to RGB (face_recognition uses)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
        
        #find all faces and facial encodings in the current frame of the video
        face_locations = face_recognition.face_locations(rgb_small_frame)
