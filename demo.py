import cv2
import PIL.Image

# Load the cascade
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

# Read the input image
img = cv2.imread('Upload\download.jpg')

# Detect faces
faces = face_cascade.detectMultiScale(img, 1.1, 4)

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
# Display the output
cv2.imwrite('C:\Users\sagar\OneDrive\Desktop\NumberPlate\Output', img)

Real=PIL.Image.open("C:\Users\sagar\OneDrive\Desktop\NumberPlate\Output")