# Deep-Fakes
Basic implementation of Face Swapping using OpenCV and First Order Motion Model in Python.

## Installation and Setup

- After activating your virtual or conda env type the commands below to install dlib and other required packages. If the test.py file runs successfully then all requirements have been installed properly.
```
pip install -r requiremnts.txt
cd dlib
pip install dlib-19.20.0-cp37-cp37m-win_amd64.whl
cd ..
python test.py
```
- To see an example run the following command
```
python image_faceswap_custom.py
```
- To try out real-time faceswap with webcam run the following command
```
python realtime_faceswap_custom.py
```

## Custom Face Swap - Explanation

1. Variables
    1. source image - to get the face
    2. destination image - where the source face will be placed
2. First, we have to detect landmark points on source image and using that we make a convex hull.
3. Using the convex hull we create a mask to extract the source face.
4. Next, we make delaunay triangles using the face landmark points on the source face.
5. Repeat steps 2 and 3 for destination image.
6. Next we take delaunay triangles of source face one by one and try to warp them on triangle mask from destination face made using exact same landmark points as used for that source image delaunay triangle.
7. Then we place all the warped triangles on a destination image mask.
8. Finally, the mask is placed over the destination image and the result is shown.

## Deep Learning Implementation - Coming soon