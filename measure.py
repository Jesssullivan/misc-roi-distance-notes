import math
from cv2 import cv2

DFOV_DEGREES = 50  # such as average laptop webcam horizontal field of view
KNOWN_ROI_MM = 240  # say, a convex rectangle around a human head

# image source:
cap = cv2.VideoCapture(0)

# detector:
cascade = cv2.CascadeClassifier('./haar/haarcascade_frontalface_alt2.xml')

while True:

    # Capture & resize a single image:
    _, image = cap.read()
    image = cv2.resize(image, (0, 0), fx=.7, fy=0.7, interpolation=cv2.INTER_NEAREST)

    # Convert to greyscale while processing:
    gray_conv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray_conv, (7, 7), 0)

    # get image dimensions:
    gray_width = gray.shape[1]
    gray_height = gray.shape[0]

    # calculate focal length based on the perimeter of image by diagonal field of view:
    focal_value = (gray_height / 2) / math.tan(math.radians(DFOV_DEGREES / 2))

    # run detector:
    result = cascade.detectMultiScale(gray)

    for x, y, h, w in result:

        # the thinking here is to calculate the distance just like one would
        # the measure difference of similar triangles....
        # ymmv, but yolo:
        dist = KNOWN_ROI_MM * focal_value / h
        dist_in = dist / 25.4

        # update display:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, 'Distance:' + str(round(dist_in)) + ' Inches',
                    (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('face detection', image)

        if cv2.waitKey(1) == ord('q'):
            break
