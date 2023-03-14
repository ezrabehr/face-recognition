import cv2
from simple_facerec import SimpleFacerec

# Load Camera
cap = cv2.VideoCapture(0) #if you have multiple webcams (normally you only have 1 cam, so index 0)

# Encode faces from a foler
simple_face_recognizer = SimpleFacerec() # using his code, we read all images from a folder
simple_face_recognizer.load_encoding_images(r"C:\Users\ezrab\python\Enigma Project\face_recognition\images") # endosing all said images

# will read the video frame after frame and diegnose each frame independently
while True:
    ret, frame = cap.read() # ret - returns true if we have the frame and false if not. frame - is the frame

    # Detect faces
    face_locations, face_names = simple_face_recognizer.detect_known_faces(frame) # loads the face and the names on the screen if it recognizes the face.
    for face_loc, name in zip(face_locations,face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3] # getting the face location, for example [184 500 328 356]

        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,200) , 2)
        cv2.rectangle(frame, (x1,y1), (x2, y2), (0,0,200), 4) # drawing the red box around the face

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) # wait 1 milisecond before going to the next frame, so that we can read the video in real time
    if key == 27: # 27 respondes to the 'esc' key on the keyboard
        break # exit loop

cap.release() # releasing the camera
cv2.destroyAllWindows() # close all windows