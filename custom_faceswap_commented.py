import cv2
import numpy as np
import dlib

SOURCE_PATH = "media/images/jason.jpg"
DEST_PATH = "media/images/brucewills.jpg"

# dlib library's face detector and facial landmark predictor
frontal_face_detector = dlib.get_frontal_face_detector()
# detecting face
frontal_face_predictor = dlib.shape_predictor(
    "dataset/shape_predictor_68_face_landmarks.dat")
# detecting face landmark points

source_image = cv2.imread(SOURCE_PATH)
source_image_copy = source_image.copy()
source_image_grayscale = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)

destination_image = cv2.imread(DEST_PATH)
destination_image_copy = destination_image.copy()
destination_image_grayscale = cv2.cvtColor(
    destination_image, cv2.COLOR_BGR2GRAY)

# DEMO 1 - source image
cv2.imshow("1: Source image", source_image)
cv2.waitKey(0)

# zeros array canvas with same size as source_image
source_image_canvas = np.zeros_like(source_image_grayscale)
print(source_image_canvas.shape)

# DEMO 2 - source image canvas
cv2.imshow("2: Source image canvas", source_image_canvas)
cv2.waitKey(0)

# zeros array canvas with same size as destination_image
height, width, no_of_channels = destination_image.shape
destination_image_canvas = np.zeros((height, width, no_of_channels), np.uint8)
print(destination_image_canvas.shape)


def index_from_array(numpyarray):
    index = None
    for n in numpyarray[0]:
        index = n
        break
    return index


# finding faces in source_image - returns array containing pixels in the image where face was detected - [[(width(top left), height(top left)), (width(bottom right), height(bottom right))]]
source_faces = frontal_face_detector(source_image_grayscale)
print(source_faces)

# loop through all faces and landmark points
for source_face in source_faces:
    # predictor takes human face as input and returns the list of face landmarks
    source_face_landmarks = frontal_face_predictor(
        source_image_grayscale, source_face)
    source_face_landmark_points = []
    # store all 68 landmarks in list
    for landmark_no in range(68):
        x_point = source_face_landmarks.part(landmark_no).x
        y_point = source_face_landmarks.part(landmark_no).y
        source_face_landmark_points.append((x_point, y_point))
        # DEMO 3 - showing all landmark points on image
        cv2.circle(source_image, (x_point, y_point), 2, (255, 0, 0), -1)
        cv2.putText(source_image, str(landmark_no), (x_point, y_point),
                    cv2.FONT_HERSHEY_SIMPLEX, .2, (255, 255, 255))
    cv2.imshow("3: landmarks points in source", source_image)
    cv2.waitKey(0)
    source_image = cv2.imread(SOURCE_PATH)
    source_image_copy = source_image

    # finding convex hull of source image
    source_face_landmark_points_array = np.array(
        source_face_landmark_points, np.int32)
    source_face_convexhull = cv2.convexHull(source_face_landmark_points_array)
    # DEMO 4 - showing convex hull
    cv2.polylines(source_image, [source_face_convexhull], True, (255, 0, 0), 2)
    cv2.imshow("4: convex hull on image", source_image)
    cv2.waitKey(0)
    source_image = cv2.imread(SOURCE_PATH)
    source_image_copy = source_image

    # creating a mask - we replace the area inside convex hull on the canvas by white colour
    cv2.fillConvexPoly(source_image_canvas, source_face_convexhull, 255)
    # DEMO 5 - convex hull in canvas(masking)
    cv2.imshow("5: convex hull on canvas(mask)", source_image_canvas)
    cv2.waitKey(0)

    # place mask over source image
    source_face_image = cv2.bitwise_and(
        source_image, source_image, mask=source_image_canvas)
    # DEMO 6 - placing mask over source image(extracting source face)
    cv2.imshow(
        "6: extracting source face(placing mask over source image)", source_face_image)
    cv2.waitKey(0)

    # DELAUNAY TRIANGULATION

    # drawing a bounding rectangle around face convex hull because Subdiv2D needs a rectangle
    bounding_rectangle = cv2.boundingRect(source_face_convexhull)

    # empty delaunay subdivisions
    subdivisions = cv2.Subdiv2D(bounding_rectangle)
    # insert face landmark points which will be used to form subdivisions
    subdivisions.insert(source_face_landmark_points)
    # list of triangles - 6 point(x, y for all 3 vertices of triangle)
    triangles_vector = subdivisions.getTriangleList()
    triangles_array = np.array(triangles_vector, dtype=np.int32)
    print(triangles_array.shape)

    triangle_index_points_list = []
    source_face_image_copy = source_face_image.copy()

    for triangle in triangles_array:
        index_point_1 = (triangle[0], triangle[1])
        index_point_2 = (triangle[2], triangle[3])
        index_point_3 = (triangle[4], triangle[5])

        # getting facial landmark point number from triangle co-ordinate
        index_1 = np.where(
            (source_face_landmark_points_array == index_point_1).all(axis=1))
        index_1 = index_from_array(index_1)
        index_2 = np.where(
            (source_face_landmark_points_array == index_point_2).all(axis=1))
        index_2 = index_from_array(index_2)
        index_3 = np.where(
            (source_face_landmark_points_array == index_point_3).all(axis=1))
        index_3 = index_from_array(index_3)

        triangle = [index_1, index_2, index_3]
        triangle_index_points_list.append(triangle)
        # this list contains facial landmark point of each point of the triangle
        # eg. [1, 44, 20] - first point is landmark point 1, second point is landmark point 44, third point is landmark point 20

        # DEMO 7 - print face with delaunay triangles
        line_color = (255, 0, 0)
        cv2.line(source_face_image_copy, index_point_1,
                 index_point_2, line_color, 1)
        cv2.line(source_face_image_copy, index_point_2,
                 index_point_3, line_color, 1)
        cv2.line(source_face_image_copy, index_point_3,
                 index_point_1, line_color, 1)

    cv2.imshow("7: delaunay triangles", source_face_image_copy)
    cv2.waitKey(0)


# DESTINATION IMAGE
destination_faces = frontal_face_detector(destination_image_grayscale)

# loop through all faces and landmark points
for destination_face in destination_faces:
    # predictor takes human face as input and returns the list of face landmarks
    destination_face_landmarks = frontal_face_predictor(
        destination_image_grayscale, destination_face)
    destination_face_landmark_points = []
    # store all 68 landmarks in list
    for landmark_no in range(68):
        x_point = destination_face_landmarks.part(landmark_no).x
        y_point = destination_face_landmarks.part(landmark_no).y
        destination_face_landmark_points.append((x_point, y_point))
        cv2.circle(destination_image, (x_point, y_point), 2, (255, 0, 0), -1)
        cv2.putText(destination_image, str(landmark_no), (x_point,
                    y_point), cv2.FONT_HERSHEY_SIMPLEX, .2, (255, 255, 255))
    # destination_image = cv2.imread(destination_PATH)
    # destination_image_copy = destination_image

    # finding convex hull of destination image
    destination_face_landmark_points_array = np.array(
        destination_face_landmark_points, np.int32)
    destination_face_convexhull = cv2.convexHull(
        destination_face_landmark_points_array)
    # DEMO 8 - showing convex hull
    cv2.polylines(destination_image, [
                  destination_face_convexhull], True, (255, 0, 0), 1)
    cv2.imshow("8: convex hull on destination image", destination_image)
    cv2.waitKey(0)
    destination_image = cv2.imread(DEST_PATH)
    destination_image_copy = destination_image


# For every triangle from list of triangles in source image, crop bounding rectangle and extract only triangle points
for i, triangle_index_points in enumerate(triangle_index_points_list):
    # get x, y co-ordinates of vertices for source triangles
    source_triangle_point_1 = source_face_landmark_points[triangle_index_points[0]]
    source_triangle_point_2 = source_face_landmark_points[triangle_index_points[1]]
    source_triangle_point_3 = source_face_landmark_points[triangle_index_points[2]]
    # combining all 3 points
    source_triangle = np.array([source_triangle_point_1, source_triangle_point_2, source_triangle_point_3], np.int32)

    # draw bounding rectangle around triangle points and crop it for later use
    source_rectangle = cv2.boundingRect(source_triangle)
    (x, y, w, h) = source_rectangle
    cropped_source_rectangle = source_image[y:y+h, x:x+w]

    # remove rectangle points and only store triangle points
    source_triangle_points = np.array([[source_triangle_point_1[0]-x, source_triangle_point_1[1]-y], 
                                    [source_triangle_point_2[0]-x, source_triangle_point_2[1]-y], 
                                    [source_triangle_point_3[0]-x, source_triangle_point_3[1]-y]], np.int32)

    # DEMO 9 - Source Triangle
    if i==10:
        cv2.line(source_image, source_triangle_point_1, source_triangle_point_2, (255,255,255))
        cv2.line(source_image, source_triangle_point_2, source_triangle_point_3, (255,255,255))
        cv2.line(source_image, source_triangle_point_3, source_triangle_point_1, (255,255,255))
        cv2.rectangle(source_image, (x, y), (x+w, y+h), (0, 0, 255))
        cv2.imshow("9.1: Triangle + Bounding Rectangle", source_image)
        cv2.imshow("9.2: Cropped Rectangle", cropped_source_rectangle)
        cv2.waitKey(0)


    # Cropping destination triangle and creating a mask

    # get x, y co-ordinates of vertices for destination triangles
    destination_triangle_point_1 = destination_face_landmark_points[triangle_index_points[0]]
    destination_triangle_point_2 = destination_face_landmark_points[triangle_index_points[1]]
    destination_triangle_point_3 = destination_face_landmark_points[triangle_index_points[2]]
    # combining all 3 points
    destination_triangle = np.array([destination_triangle_point_1, destination_triangle_point_2, destination_triangle_point_3], np.int32)

    # draw bounding rectangle around triangle points and crop it for later use
    destination_rectangle = cv2.boundingRect(destination_triangle)
    (x, y, w, h) = destination_rectangle

    # crop destination rectangle and create a mask for later use
    cropped_destination_rectangle_mask = np.zeros((h, w), np.uint8)

    # remove rectangle points and only store triangle points
    destination_triangle_points = np.array([[destination_triangle_point_1[0]-x, destination_triangle_point_1[1]-y], 
                                    [destination_triangle_point_2[0]-x, destination_triangle_point_2[1]-y], 
                                    [destination_triangle_point_3[0]-x, destination_triangle_point_3[1]-y]], np.int32)

    # triangle points over cropped rectangle zero array mask
    cv2.fillConvexPoly(cropped_destination_rectangle_mask, destination_triangle_points, 255)

    # DEMO 10 - Destination Triangle Mask
    if i==10:
        print(destination_triangle_point_1, destination_triangle_point_2, destination_triangle_point_3)
        cv2.line(destination_image, destination_triangle_point_1, destination_triangle_point_2, (255, 255, 255))
        cv2.line(destination_image, destination_triangle_point_2, destination_triangle_point_3, (255, 255, 255))
        cv2.line(destination_image, destination_triangle_point_3, destination_triangle_point_1, (255, 255, 255))
        cv2.rectangle(destination_image, (x, y), (x+w, y+h), (0, 0, 255))
        cv2.imshow("10.1: Triangle + Rectangle", destination_image)
        cv2.imshow("10.2: Cropped Rectangle Mask", cropped_destination_rectangle_mask)
        cv2.waitKey(0)

    
    # Warp source triangle to match shape of destination triangle

    source_triangle_points = np.float32(source_triangle_points)
    destination_triangle_points = np.float32(destination_triangle_points)
    
    # creating transformation matrix for warp affine
    matrix = cv2.getAffineTransform(source_triangle_points, destination_triangle_points)
    # creating warped rectangle
    warped_rectangle = cv2.warpAffine(cropped_source_rectangle, matrix, (w, h))
    # w, h are destination triangle dimensions

    # placing destination rectangle mask over warped triangle
    warped_triangle = cv2.bitwise_and(warped_rectangle, warped_rectangle, mask=cropped_destination_rectangle_mask)

    # DEMO 11 - placing warped triangle inside mask
    if i==10:
        cv2.imshow("11.1: Warped Source Rectangle", warped_rectangle)
        cv2.imshow("11.2: Warped Source Triangle with mask", warped_triangle)
        cv2.waitKey(0)

    
    # Reconstructing destination face in empty canvas of destination image
    
    # removing white lines in triangle using masking
    # this is bounding rectangle for destination triangle(x, y, w, h)
    new_dest_face_canvas_area = destination_image_canvas[y:y+h, x:x+w]
    new_dest_face_canvas_area_gray = cv2.cvtColor(new_dest_face_canvas_area, cv2.COLOR_BGR2GRAY)
    # creating mask to cut pixels inside triangle excluding white lines
    _, mask_created_triangle = cv2.threshold(new_dest_face_canvas_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
    # try changing 1 to 200 to see how this makes a difference

    # placing created mask
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_created_triangle)
    # place masked triangle inside the small canvas area
    new_dest_face_canvas_area = cv2.add(new_dest_face_canvas_area, warped_triangle)
    # place new small canvas with triangle in it to large destination image canvas
    destination_image_canvas[y:y+h, x:x+w] = new_dest_face_canvas_area

    # DEMO 12 - reconstruction in destination image canvas
    if i==10:
        cv2.imshow("12: Pasting triangles on destination canvas", destination_image_canvas)
        cv2.waitKey(0)

# DEMO 13 - full reconstructed face in destination image canvas
cv2.imshow("13: Reconstructed face in destination image canvas", destination_image_canvas)
cv2.waitKey(0)


# DEMO 14 - Put reconstructed face on the destination image

# create new canvas of destination image
final_destination_canvas = np.zeros_like(destination_image_grayscale)

# create destination face mask
final_destination_face_mask = cv2.fillConvexPoly(final_destination_canvas, destination_face_convexhull, 255)
final_destination_canvas = cv2.bitwise_not(final_destination_face_mask)
cv2.imshow("14.1: Mask to remove destination face", final_destination_canvas)

# mask destination face
destination_face_masked = cv2.bitwise_and(destination_image, destination_image, mask=final_destination_canvas)
cv2.imshow("14.2: Masked destination image", destination_face_masked)

# add reconstructed face
destination_with_face = cv2.add(destination_face_masked, destination_image_canvas)
cv2.imshow("14.2: Destination image with source face", destination_with_face)
cv2.waitKey(0)


# DEMO 15 - Seamless cloning to make attachment blend with surrounding pixels

# we have to find center point of reconstructed convex hull
(x, y, w, h) = cv2.boundingRect(destination_face_convexhull)
destination_face_center_point = (int((x+x+w)/2), int((y+y+h)/2))

# seamless clone
seamless_cloned_face = cv2.seamlessClone(destination_with_face, destination_image, final_destination_face_mask, destination_face_center_point, cv2.NORMAL_CLONE)
cv2.imshow("15: Destination image with source face", seamless_cloned_face)
cv2.waitKey(0)
cv2.destroyAllWindows()