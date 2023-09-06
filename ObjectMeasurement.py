import cv2
import utlis
import numpy as np

###################################
webcam = input("Do you want to use a webcam? (yes/no): ").strip().lower() == 'yes'
path = '1809.jpg'  # Change to the path of your jeans image
scale = 3
wP = 210 *scale
hP= 297 *scale
cap = cv2.VideoCapture(0) if webcam else None
if not webcam:
    img = cv2.imread(path)
    img = cv2.resize(img, (1920, 1080))  # Adjust the resolution if needed
###################################

while True:
    if webcam:
        success, img = cap.read()
    else: img = cv2.imread(path)

    imgContours, conts = utlis.getContours(img, minArea=50000, filter=4)
    if len(conts) != 0:
        biggest = conts[0][2]
        imgWarp = utlis.warpImg(img, biggest, wP, hP)
        imgContours2, conts2 = utlis.getContours(imgWarp, minArea=2000, filter=4, cThr=[50, 50], draw=False)

        if len(conts2) != 0:
            for obj in conts2:
                cv2.polylines(imgContours2, [obj[2]], True, (0, 255, 0), 2)
                nPoints = utlis.reorder(obj[2])
                nW = round((utlis.findDis(nPoints[0][0] // scale, nPoints[1][0] // scale) / 10), 1)
                nH = round((utlis.findDis(nPoints[1][0] // scale, nPoints[2][0] // scale) / 10), 1)

                # Add measurements here
                inseam_length = round((utlis.findDis(nPoints[0][0] // scale, nPoints[3][0] // scale) / 10), 1)
                knee_to_ankle = round((utlis.findDis(nPoints[1][0] // scale, nPoints[2][0] // scale) / 10), 1)
                front_rise = round((utlis.findDis(nPoints[0][0] // scale, nPoints[4][0] // scale) / 10), 1)
                seat_thigh_fit = round((utlis.findDis(nPoints[2][0] // scale, nPoints[5][0] // scale) / 10), 1)
                waist = round((utlis.findDis(nPoints[5][0] // scale, nPoints[6][0] // scale) / 10), 1)
                high_hip = round((utlis.findDis(nPoints[6][0] // scale, nPoints[7][0] // scale) / 10), 1)
                low_hip = round((utlis.findDis(nPoints[7][0] // scale, nPoints[8][0] // scale) / 10), 1)
                thigh = round((utlis.findDis(nPoints[8][0] // scale, nPoints[9][0] // scale) / 10), 1)
                knee = round((utlis.findDis(nPoints[9][0] // scale, nPoints[10][0] // scale) / 10), 1)
                leg_opening = round((utlis.findDis(nPoints[10][0] // scale, nPoints[1][0] // scale) / 10), 1)
                back_rise = round((utlis.findDis(nPoints[0][0] // scale, nPoints[11][0] // scale) / 10), 1)

                # Display measurements
                cv2.putText(imgContours2, 'Inseam: {} cm'.format(inseam_length), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(imgContours2, 'Knee to Ankle: {} cm'.format(knee_to_ankle), (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(imgContours2, 'Front Rise: {} cm'.format(front_rise), (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(imgContours2, 'Seat & Thigh Fit: {} cm'.format(seat_thigh_fit), (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(imgContours2, 'Waist: {} cm'.format(waist), (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(imgContours2, 'High Hip: {} cm'.format(high_hip), (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(imgContours2, 'Low Hip: {} cm'.format(low_hip), (10, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(imgContours2, 'Thigh: {} cm'.format(thigh), (10, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(imgContours2, 'Knee: {} cm'.format(knee), (10, 270),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(imgContours2, 'Leg Opening: {} cm'.format(leg_opening), (10, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(imgContours2, 'Back Rise: {} cm'.format(back_rise), (10, 330),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Jeans Measurement', imgContours2)

    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    cv2.imshow('Original', img)
    cv2.waitKey(1)