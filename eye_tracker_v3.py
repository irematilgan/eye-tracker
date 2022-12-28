import dlib
import numpy as np
import cv2
import math
import time
import json

button = [20,60,450,650]

LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]
THRESHOLD = 197

with open("configs/config.json", "r") as json_file:
    configs = json.load(json_file)

def midpoint(p1, p2):
    mid_x = int((p1.x+p2.x)/2)
    mid_y = int((p1.y+p2.y)/2)
    return (mid_x, mid_y)

def blinking(landmarks, points):
    right = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)
    left = (landmarks.part(points[3]).x, landmarks.part(points[0]).y)
    top = midpoint(landmarks.part(points[1]), landmarks.part(points[2]))
    bottom = midpoint(landmarks.part(points[4]), landmarks.part(points[5]))

    eye_width = math.hypot((left[0]-right[0]), (left[1]-right[1]))
    eye_height = math.hypot((top[0]-bottom[0]), (top[1]-bottom[1]))

    try:
        ratio = eye_width/eye_height
    except ZeroDivisionError:
        ratio = None

    return ratio

def isolate(frame, landmarks, points):
    region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])
    region = region.astype(np.int32)
    landmark_points = region

    ## Apply mask to get only the eye
    height, width = frame.shape[:2]
    black_frame = np.zeros((height, width), np.uint8)
    mask = np.full((height, width), 255, np.uint8)
    cv2.fillPoly(mask, [region], (0,0,0)) # fill eye region of mask with black
    eye = cv2.bitwise_not(black_frame, frame.copy(), mask = mask) # get bitwise not of mask

    ## Cropping on the eye
    margin = 5
    min_x = np.min(region[:,0]) - margin
    max_x = np.max(region[:,0]) + margin
    min_y = np.min(region[:, 1]) - margin
    max_y = np.max(region[:, 1]) + margin

    cropped_frame = eye[min_y:max_y, min_x:max_x]
    origin = (min_x, min_y)

    height, width = cropped_frame.shape[:2]
    center = (width/2, height/2)

    return (landmark_points, cropped_frame, height, width, origin, center)

def img_processor(eye_frame, threshold):
    ## Performs operations on the eye frame to isolate the iris

    kernel = np.ones((3,3), np.uint8)
    new_frame = cv2.bilateralFilter(eye_frame, 10, 15, 15) # smooths while keeping the edges sharp
    new_frame = cv2.erode(new_frame, kernel, iterations = 3)
    new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)[1]

    return new_frame

def iris_size(frame):
    """Returns the percentage of space that the iris takes up on
    the surface of the eye.
    Argument:
        frame (numpy.ndarray): Binarized iris frame
    """
    frame = frame[5:-5, 5:-5]
    height, width = frame.shape[:2]
    nb_pixels = height * width
    nb_blacks = nb_pixels - cv2.countNonZero(frame)
    return nb_blacks / nb_pixels

def detect_iris(eye_frame, threshold):
    iris_frame = img_processor(eye_frame, threshold)
    contours, _ = cv2.findContours(iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
    contours = sorted(contours, key = cv2.contourArea)
    try:
        moments = cv2.moments(contours[-2])
        x = int(moments['m10']/moments['m00'])
        y = int(moments['m01']/moments['m00'])
    except (IndexError, ZeroDivisionError):
        pass
    
    return (x,y)

def euc_dist(a,b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a-b)
def normalize(a):
    a = int(a)
    a = a/100
    return a
def detect_pupil(img, gray,left,right,top,bottom):
    gray = gray.astype(np.uint8)
    padding_x = 15
    padding_y = 10

    cropped_frame = gray[(left[1]-padding_x):(right[1]+padding_x), (left[0]-padding_y):(right[0]+padding_y)]
    # cv2.imshow('cropped',cropped_frame)

    kernel = np.ones((3,3), np.uint8)
    cropped_frame = cv2.bilateralFilter(cropped_frame, 10, 15, 15) # smooths while keeping the edges sharp
    cropped_frame = cv2.erode(cropped_frame, kernel, iterations = 3)
    # cv2.imshow('cropped', cropped_frame)

    # First binarize the image so that findContours can work correctly.
    _, thresh = cv2.threshold(cropped_frame, 70, 255, cv2.THRESH_BINARY_INV)
    # Now find the contours and then find the pupil in the contours.
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    cnt = max(contours, key = cv2.contourArea)
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    Fcx = cx + left[0]-padding_y
    Fcy = cy + left[1]-padding_x
    # cv2.circle(cnt_img,(cx,cy),3, (0,0,255),1)
    # cv2.circle(img,(Fcx, Fcy),3, (0,0,255),1)
    # cv2.imshow('contour',cnt_img)
    hor_ratio = (Fcx-left[0])/(right[0]-left[0])
    # print(hor_ratio)
    if hor_ratio < normalize(configs['EYE_LEFT']):
        print("Left")
        return 2
    elif hor_ratio >= normalize(configs['EYE_LEFT']) and hor_ratio <= normalize(configs['EYE_RIGHT']):
        
        ver_ratio = (Fcy-bottom[1])/(top[1]-bottom[1])
        print(ver_ratio)
        if ver_ratio >= normalize(configs['EYE_UP']):
            print("Forward")
            
            return 0, Fcx, Fcy
        elif ver_ratio <= normalize(configs['EYE_DOWN']):
            print("Backwards")
            return 1, Fcx, Fcy
        else:
            print("Center")
            return 4, Fcx, Fcy
    else:
        print("Right")
        return 3, Fcx, Fcy
    # return np.argmin([euc_dist([Fcx,Fcy], top), 100, euc_dist([Fcx,Fcy], left), euc_dist([Fcx,Fcy], right)])

def horizontal_ratio(pupil_left_coords, pupil_right_coords, centers, centers_R):
    """Returns a number between 0.0 and 1.0 that indicates the
    horizontal direction of the gaze. The extreme right is 0.0,
    the center is 0.5 and the extreme left is 1.0
    """
    print("Pupil Coord = " + str(pupil_left_coords[0]) + "\t Center = " + str(centers[0]*2))
    pupil_left = pupil_left_coords[0]/(centers[0]*2-10)
    pupil_right = pupil_right_coords[0]/(centers_R[0]*2-10)
    return (pupil_left+pupil_right)/2

def vertical_ratio(y, centers):
    """Returns a number between 0.0 and 1.0 that indicates the
    vertical direction of the gaze. The extreme top is 0.0,
    the center is 0.5 and the extreme bottom is 1.0
    """
    pupil_left = y/(centers[1]*2-10)
    return pupil_left

def is_blinking(landmarks, points):
    b_ratio = blinking(landmarks, LEFT_EYE_POINTS)
    return b_ratio > 4.5

def nothing(x):
    pass

# function that handles the mousclicks (It can be useful f we use a mechanism to start vehicle)
def process_click(event, x, y,flags, params):
    # check if the click is within the dimensions of the button
    if event == cv2.EVENT_LBUTTONDOWN:
        res = None
        if y > button[0] and y < button[1] and x > button[2] and x < button[3]: 
            val = cv2.getTrackbarPos('Capture','image')  
            newVal = (val+1)%2
            cv2.setTrackbarPos('Capture', 'image',newVal)
            print('Clicked on Button!')

            startCapture(newVal)
    
        return res

# function that handles the trackbar
def startCapture(val):
    # check if the value of the slider 
    if val == 1:
        res = 'STARTED'
    else:
        res = 'ENDED'  

    print(res)

def run_tracker(detector, predictor, frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    try:
        landmarks = predictor(gray, faces[0])

        ## Detect Blinking
        blinked = is_blinking(landmarks, LEFT_EYE_POINTS)
        cv2.putText(frame, 'Blinking = ' + str(blinked), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)

        ## LEFT EYE
        landmark_points, cropped_frame, cropped_h, cropped_w, origins, centers = isolate(gray, landmarks, LEFT_EYE_POINTS)

        iris_x,iris_y = detect_iris(cropped_frame, THRESHOLD)
        pupil_left_coords = (iris_x + origins[0], iris_y + origins[1])

        cv2.line(frame, (pupil_left_coords[0]-5, pupil_left_coords[1]), (pupil_left_coords[0] + 5, pupil_left_coords[1]), (0,255,0))
        cv2.line(frame, (pupil_left_coords[0], pupil_left_coords[1]-5), (pupil_left_coords[0], pupil_left_coords[1] + 5), (0,255,0))

        _, cropped_frame_R, cropped_h_R, cropped_w_R, origins_R, centers_R = isolate(gray, landmarks, RIGHT_EYE_POINTS)
        iris_x_R,iris_y_R = detect_iris(cropped_frame_R, THRESHOLD)
        pupil_right_coords = (iris_x_R + origins_R[0], iris_y_R + origins_R[1])
        
        cv2.line(frame, (pupil_right_coords[0]-5, pupil_right_coords[1]), (pupil_right_coords[0] + 5, pupil_right_coords[1]), (0,255,0))
        cv2.line(frame, (pupil_right_coords[0], pupil_right_coords[1]-5), (pupil_right_coords[0], pupil_right_coords[1] + 5), (0,255,0))

        left = (landmarks.part(LEFT_EYE_POINTS[0]).x, landmarks.part(LEFT_EYE_POINTS[0]).y)
        right = (landmarks.part(LEFT_EYE_POINTS[3]).x, landmarks.part(LEFT_EYE_POINTS[0]).y)
        top = midpoint(landmarks.part(LEFT_EYE_POINTS[1]), landmarks.part(LEFT_EYE_POINTS[2]))
        bottom = midpoint(landmarks.part(LEFT_EYE_POINTS[4]), landmarks.part(LEFT_EYE_POINTS[5]))
        
        dir_ind, Fcx, Fcy = detect_pupil(frame,gray,left,right,top,bottom)
        if blinked:
            dir_ind = 1

        return dir_ind, Fcx, Fcy
        # cv2.putText(frame, 'Looking ' + direction_names[dir_ind], (30,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        
    except Exception as e:
        print(e)
        pass


        # frame[button[0]:button[1],button[2]:button[3]] = 180
        # newVal = cv2.getTrackbarPos('Capture', 'image')
        # if newVal == 1:
        #     cv2.putText(frame, 'Capture: STARTED',(30,100),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)
        # else:
        #     cv2.putText(frame, 'Capture: ENDED',(30,100),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)
        # cv2.putText(frame, 'Button',(490,50),cv2.FONT_HERSHEY_SIMPLEX, 1,(0),2)

    # cv2.imshow('image',frame)



    

def main():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    direction_names = ['FORWARD', 'BACKWARD','LEFT', 'RIGHT', 'CENTER']


    cap = cv2.VideoCapture(0)
    cv2.namedWindow('image')
    # cv2.createTrackbar('threshold', 'image', 0, 255, nothing)
    # cv2.setTrackbarPos('threshold', 'image', 197)
    ret, frame = cap.read()

    ## BUTTON ON CAMERA
    # cv2.setMouseCallback('image',process_click)
    # cv2.createTrackbar("Capture", 'image', 0,1, startCapture)
    # cv2.putText(frame, 'Capture: ' + 'NOT STARTED',(30,100),cv2.FONT_HERSHEY_PLAIN, 1,(0,0,255),1)

    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)
        try:
            landmarks = predictor(gray, faces[0])

            ## Detect Blinking
            blinked = is_blinking(landmarks, LEFT_EYE_POINTS)
            cv2.putText(frame, 'Blinking = ' + str(blinked), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)

            ## LEFT EYE
            landmark_points, cropped_frame, cropped_h, cropped_w, origins, centers = isolate(gray, landmarks, LEFT_EYE_POINTS)

            iris_x,iris_y = detect_iris(cropped_frame, THRESHOLD)
            pupil_left_coords = (iris_x + origins[0], iris_y + origins[1])

            cv2.line(frame, (pupil_left_coords[0]-5, pupil_left_coords[1]), (pupil_left_coords[0] + 5, pupil_left_coords[1]), (0,255,0))
            cv2.line(frame, (pupil_left_coords[0], pupil_left_coords[1]-5), (pupil_left_coords[0], pupil_left_coords[1] + 5), (0,255,0))

            _, cropped_frame_R, cropped_h_R, cropped_w_R, origins_R, centers_R = isolate(gray, landmarks, RIGHT_EYE_POINTS)
            iris_x_R,iris_y_R = detect_iris(cropped_frame_R, THRESHOLD)
            pupil_right_coords = (iris_x_R + origins_R[0], iris_y_R + origins_R[1])
            
            cv2.line(frame, (pupil_right_coords[0]-5, pupil_right_coords[1]), (pupil_right_coords[0] + 5, pupil_right_coords[1]), (0,255,0))
            cv2.line(frame, (pupil_right_coords[0], pupil_right_coords[1]-5), (pupil_right_coords[0], pupil_right_coords[1] + 5), (0,255,0))

            left = (landmarks.part(LEFT_EYE_POINTS[0]).x, landmarks.part(LEFT_EYE_POINTS[0]).y)
            right = (landmarks.part(LEFT_EYE_POINTS[3]).x, landmarks.part(LEFT_EYE_POINTS[0]).y)
            top = midpoint(landmarks.part(LEFT_EYE_POINTS[1]), landmarks.part(LEFT_EYE_POINTS[2]))
            bottom = midpoint(landmarks.part(LEFT_EYE_POINTS[4]), landmarks.part(LEFT_EYE_POINTS[5]))
            
            dir_ind, fcx, fcy = detect_pupil(frame,gray,left,right,top,bottom)
            if blinked:
                dir_ind = 1
     
            cv2.putText(frame, 'Looking ' + direction_names[dir_ind], (30,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            cv2.circle(frame,(fcx, fcy),3, (0,0,255),1)
        except Exception as e:
            print(e)
            pass


        # frame[button[0]:button[1],button[2]:button[3]] = 180
        # newVal = cv2.getTrackbarPos('Capture', 'image')
        # if newVal == 1:
        #     cv2.putText(frame, 'Capture: STARTED',(30,100),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)
        # else:
        #     cv2.putText(frame, 'Capture: ENDED',(30,100),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)
        # cv2.putText(frame, 'Button',(490,50),cv2.FONT_HERSHEY_SIMPLEX, 1,(0),2)

        cv2.imshow('image',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()