import numpy as np
import cv2
# from getdist import plots, MCSamples
# from PyQt5.QtWidgets import QApplication
# from dialogs.settings_dialog import Settings

width = 0
height = 0
eggCount = 0
exitCounter = 0
OffsetRefLines = 50  # Adjust ths value according to your usage
ReferenceFrame = None
distance_tresh = 200
radius_min = 0
radius_max = 0
area_min = 0
area_max = 0

# app = QApplication(sys.argv)
# set = Settings()
#sys.exit(app.exec_())

def reScaleFrame(frame, percent=75):
    width = int(frame.shape[1] * percent // 100)
    height = int(frame.shape[0] * percent // 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def CheckInTheArea(coordYContour, coordYEntranceLine, coordYExitLine):
    if ((coordYContour <= coordYEntranceLine) and (coordYContour >= coordYExitLine)):
        return 1
    else:
        return 0

def CheckEntranceLineCrossing(coordYContour, coordYEntranceLine):
    absDistance = abs(coordYContour - coordYEntranceLine)

    if ((coordYContour >= coordYEntranceLine) and (absDistance <= 3)):
        return 1
    else:
        return 0

def getDistance(coordYEgg1, coordYEgg2):
    dist = abs(coordYEgg1 - coordYEgg2)
    return dist

def check(ax1, ax2, orientation):
    if ax1 >= ax2:
        long = ax1
        theta = orientation
    else:
        long = ax2
        theta = 90 + orientation

    return theta, long

cap = cv2.VideoCapture('20180910_144521.mp4')
#cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('rtsp://admin:9ejq28Ez@172.16.1.65:554/Streaming/Channels/101?transportmode=unicast&profile=Profile_1')
fgbg = cv2.createBackgroundSubtractorMOG2()  # for mask

while True:

    (grabbed, frame) = cap.read()
    if not grabbed:
        print('Egg count: ' + str(eggCount))
        print('\n End of the video file...')
        break

    # get Settings radius/area values
    radius_min,radius_max = 10,30#set.getRadius()
    area_min,area_max = 370,1100#set.getArea()
    borderSize = 40#set.getBorderSizeValue()

    if radius_min == '':
        radius_min = 0
    if radius_max == '':
        radius_max = 0

    if area_min == '':
        area_min = 0
    if area_max == '':
        area_max = 0

    frame40 = reScaleFrame(frame, percent=40)#(432,768,3)

    height = np.size(frame40, 0)#432
    width = np.size(frame40, 1)#768

    fgmask = fgbg.apply(frame40)#(432,768)
    hsv = cv2.cvtColor(frame40, cv2.COLOR_BGR2HSV)#(432, 768, 3)
    th, bw = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)#th=137.0,bw=(432,768)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))#(3,3)
    morph = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)#(432,768)

    dist = cv2.distanceTransform(morph, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)#(432,768)
    distborder = cv2.copyMakeBorder(dist, borderSize, borderSize, borderSize, borderSize,
                                    cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)#(512,848)

    gap = 10
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (borderSize - gap) + 1, 2 * (borderSize - gap) + 1))#(61,61)
    kernel2 = cv2.copyMakeBorder(kernel2, gap, gap, gap, gap,
                                 cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)#(81,81)
    distTempl = cv2.distanceTransform(kernel2, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)#(81,81)
    nxcor = cv2.matchTemplate(distborder, distTempl, cv2.TM_CCOEFF_NORMED)#(432,768)

    mn, mx, _, _ = cv2.minMaxLoc(nxcor)#mn=-0.51,mx=0.945935
    th, peaks = cv2.threshold(nxcor, mx * 0.5, 255, cv2.THRESH_BINARY)#th=0.47,peaks=(432,768)
    peaks8u = cv2.convertScaleAbs(peaks)#(432,768)
    _, contours, hierarchy = cv2.findContours(peaks8u, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    peaks8u = cv2.convertScaleAbs(peaks)  # to use as mask

    # plot reference lines (entrance and exit lines)
    coordYEntranceLine = (height // 2) + OffsetRefLines
    coordYMiddleLine = (height // 2)
    coordYExitLine = (height // 2) - OffsetRefLines
    cv2.line(frame40, (0, coordYEntranceLine), (width, coordYEntranceLine), (255, 0, 250), 2)
    cv2.line(frame40, (0, coordYMiddleLine), (width, coordYMiddleLine), (10, 120, 10), 6)
    cv2.line(frame40, (0, coordYExitLine), (width, coordYExitLine), (255, 0, 250), 2)

    flag = False
    egg_list = []
    egg_index = 0

    for i in range(len(contours)):
        contour = contours[i]
        (x, y), radius = cv2.minEnclosingCircle(contour)
        radius = int(radius)
        (x, y, w, h) = cv2.boundingRect(contour)
        egg_index = i
        egg_list.append([x, y, flag])

        if len(contour) >= 5:
            if (radius <= int(radius_max) and radius >= int(radius_min)):
                ellipse = cv2.fitEllipse(contour)
                (center, axis, angle) = ellipse
                coordXContour, coordYContour = int(center[0]), int(center[1])
                coordXCentroid = (2 * coordXContour + w) // 2
                coordYCentroid = (2 * coordYContour + h) // 2
                ax1, ax2 = int(axis[0]) - 2, int(axis[1]) - 2
                orientation = int(angle)
                area = cv2.contourArea(contour)

                theta,long = check(ax1,ax2,orientation)
                if area >= int(area_min) and area <= int(area_max):
                    #print('egg list: ' + str(egg_list) + ' index: ' + str(egg_index))

                    if CheckInTheArea(coordYContour, coordYEntranceLine, coordYExitLine):
                        cv2.ellipse(frame40, (coordXContour, coordYContour), (ax1, ax2), orientation, 0, 360,(255, 0, 0), 10)  # blue
                        cv2.line(frame40,(coordXContour, coordYContour),(coordXContour+int(long*np.cos(theta/180*np.pi)), coordYContour+int(long*np.sin(theta/180*np.pi))),(0, 0, 255), 4)
                        cv2.line(frame40,(coordXContour, coordYContour),(coordXContour+int(long*np.cos(orientation/180*np.pi)), coordYContour+int(long*np.sin(orientation/180*np.pi))),(0, 0, 0), 4)
                        cv2.circle(frame40, (coordXContour, coordYContour), 1, (0, 255, 0), 6)  # green
                        cv2.putText(frame40, str(int(area)), (coordXContour, coordYContour), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, 0, 1, cv2.LINE_AA)
                    for k in range(len(egg_list)):
                        egg_new_X = x
                        egg_new_Y = y
                        dist = getDistance(egg_new_Y, egg_list[k][1])
                        if dist > distance_tresh:  # distance_tresh = 200
                            egg_list.append([egg_new_X, egg_new_Y, flag])
                    if CheckEntranceLineCrossing(egg_list[egg_index][1], coordYMiddleLine) and not egg_list[egg_index][
                        2]:
                        eggCount += 1
                        egg_list[egg_index][2] = True
                cv2.putText(frame40, "Entrance Eggs: {}".format(str(eggCount)), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (250, 0, 1), 2)
    cv2.imshow("Original Frame", frame40)
    key = cv2.waitKey(1)
    if key == 27:
        break

# cleanup the camera and close any open windows
cap.release()
cv2.destroyAllWindows()