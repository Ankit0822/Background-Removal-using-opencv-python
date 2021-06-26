import cv2
import cvzone
import os
from cvzone.SelfiSegmentationModule import SelfiSegmentation

# Creating camera object
cap = cv2.VideoCapture(0)
# Set the width
cap.set(3,640)
# Set the height
cap.set(4,480)
# Set the frame rate
cap.set(cv2.CAP_PROP_FPS,60)
# Create SelfiSegmentation object
segmentor = SelfiSegmentation()
# Frame Rate
fpsReader = cvzone.FPS()
#imgBg = cv2.imread('images/2.jpg')

listImg = os.listdir('Images')
print(listImg)
imgList = []
for imgPath in listImg:
    img = cv2.imread(f"Images/{imgPath}")
    imgList.append(img)
print(len(imgList))

indexImg = 0

while True:
    # get our image
    success, img = cap.read()
    # Run our segmentor
     
    imgOut = segmentor.removeBG(img,imgList[indexImg], threshold=0.5)
    

    imgStacked = cvzone.stackImages([img,imgOut],2,1)
    _, imgStacked = fpsReader.update(imgStacked,color=(0,0,255))
    print(indexImg)
    cv2.imshow("image",imgStacked)
    key = cv2.waitKey(1)
    if key == ord('a'):
        if indexImg > 0:
            indexImg -=1
    elif key == ord('d'):
        if indexImg < len(imgList)-1:
            indexImg +=1
    elif key == ord('q'):
        break
cv2.destroyAllWindows()
