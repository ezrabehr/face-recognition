import cv2
import face_recognition


img = cv2.imread("Messi1.webp") # loading the image

rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # swapping the format from BGR to RGB 
# the reason why it's BGR and not RGB is because when OpenCV was first being developed, BGR color format was popular among camera manufacturers and image software providers

img_encoding = face_recognition.face_encodings(rgb_img)[0] # encoding the image so that the face recognition library can read it
# [0] - because this can load multipul images so we're using index 0.


# same code as before but just for a second image
img2 = cv2.imread(r"C:\Users\ezrab\python\Enigma Project\face_recognition\images\Messi.webp")
rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]


result = face_recognition.compare_faces([img_encoding],img_encoding2) # comaring between the images
print('Result: ', result)

cv2.imshow("Img", img) # displaying the image
cv2.imshow('Img 2',img2) # displaying second image
cv2.waitKey(0) # freezes the image(frame) until the user presses a key
