
*naive shenanigans measuring distance to ROI w/ opencv*

Knowing both the diagonal field of view of a camera's lens and the dimensions of the object we'd like to measure seems like more than enough to get a distance.

Note, [opencv has an extensive suite of actual calibration tools and utilities here.](https://docs.opencv.org/4.5.0/d9/db7/tutorial_py_table_of_contents_calib3d.html)

...But without calibration or much forethought, could rough measurements of known objects even be usable?  Some notes from a math challenged individual:

*also over here:*
```
# clone:
git clone https://github.com/Jesssullivan/misc-roi-distance-notes && cd misc-roi-distance-notes
```



Most webcams don't really provide a Field of View much greater than ~50 degrees- this is the value of a MacBook Pro's webcam for instance.  Here's the plan to get a Focal Length value from Field of View:


<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{Focal&space;Length&space;=&space;(\frac{ImageDimension}{2})&space;}{tan(\frac{FieldOfView}{2})}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{Focal&space;Length&space;=&space;(\frac{ImageDimension}{2})&space;}{tan(\frac{FieldOfView}{2})}" title="\frac{Focal Length = (\frac{ImageDimension}{2}) }{tan(\frac{FieldOfView}{2})}" /></a>



source a fresh venv to fiddle from:
```
# venv:
python3 -m venv distance_venv
source distance_venv/bin/activate

# depends are imutils & opencv-contrib-python:
pip3 install -r requirements.txt
```


The opencv people provide a bunch of [prebuilt Haar cascade models](https://github.com/opencv/opencv/tree/master/data/haarcascades), so let's just snag one of them to experiment. Here's one to detect human faces, we've all got one of those:


```
mkdir haar
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml  -O ./haar/haarcascade_frontalface_alt2.xml
```

Of course, an actual thing with fixed dimensions would be better, like a stop sign!

Let's try to calculate the *distance* as the *difference* between an actual dimension of the object with a detected dimension- here's the plan:


<a href="https://www.codecogs.com/eqnedit.php?latex=Distance&space;=&space;ActualDimension&space;*&space;\frac{FocalLength}{ROIDimension}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Distance&space;=&space;ActualDimension&space;*&space;\frac{FocalLength}{ROIDimension}" title="Distance = ActualDimension * \frac{FocalLength}{ROIDimension}" /></a>


#### *YMMV, but YOLO:*

```
# `python3 measure.py`
import math
from cv2 import cv2

DFOV_DEGREES = 50  # such as average laptop webcam horizontal field of view
KNOWN_ROI_MM = 240  # say, height of a human head  

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
        dist = KNOWN_ROI_MM * focal_value / h
        dist_in = dist / 25.4

        # update display:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, 'Distance:' + str(round(dist_in)) + ' Inches',
                    (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('face detection', image)

        if cv2.waitKey(1) == ord('q'):
            break

```

run test with:
```python3 measure.py```
