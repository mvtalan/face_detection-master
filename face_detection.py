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

# print(known_face_encodings)

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
        small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
        
        #convert image from BGR (OpenCV) to RGB (face_recognition uses)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
        
        #find all faces and facial encodings in the current frame of the video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_names = []
        for face_encoding in face_encodings:
            #see if face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            
            name = "unknown"
            
            #if match found, just use the first one
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                
            #or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                
            face_names.append(name)

    process_this_frame = not process_this_frame
    
    #display the results
    for(top, right, bottom, left), nae in zip(face_locations, face_names):
        #scale back up face location
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        #draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
        #draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
        # if name != "Unknown":
        #     cv2.putText(frame, "Match.", (10, 30), font, 1.0 (0, 255, 0), 2, cv2.LINE_AA)
        
    #display the resulting image
    cv2.imshow('Video', frame)
    
    #hit 'q' on keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
#release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
        
