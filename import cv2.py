import cv2

# Load the pre-trained Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load the pre-trained face recognition model
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_recognition_model.xml")

# Load the pre-trained ANPR model
anpr_model = cv2.ml.ANN_MLP_load("anpr_model.xml")

# Load the test image
img = cv2.imread("test_image.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# For each detected face, perform facial recognition and ANPR
for (x, y, w, h) in faces:
    # Extract the face region from the image
    face = gray[y:y+h, x:x+w]
    
    # Resize the face region to a fixed size for the face recognition model
    face_resized = cv2.resize(face, (100, 100))
    
    # Predict the label of the face using the face recognition model
    label, confidence = face_recognizer.predict(face_resized)
    
    # Extract the license plate region from the image
    plate = gray[y+h:y+h+int(h/3), x:x+w]
    
    # Resize the license plate region to a fixed size for the ANPR model
    plate_resized = cv2.resize(plate, (200, 50))
    
    # Normalize the pixel values of the license plate region for the ANPR model
    plate_norm = plate_resized / 255.0
    
    # Flatten the license plate region to a 1D array for the ANPR model
    plate_flat = plate_norm.flatten()
    
    # Predict the license plate number using the ANPR model
    result = anpr_model.predict(plate_flat.reshape(1, -1))[1]
    plate_number = "".join([str(int(x)) for x in result[0]])
    
    # Draw a rectangle around the detected face and label it with the predicted name and confidence
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img, f"Name: {label}, Confidence: {confidence}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw a rectangle around the license plate and label it with the predicted number
    cv2.rectangle(img, (x, y+h), (x+w, y+h+int(h/3)), (0, 0, 255), 2)
    cv2.putText(img, f"Plate Number: {plate_number}", (x, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Display the resulting image
cv2.imshow("Facial Recognition and ANPR", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
