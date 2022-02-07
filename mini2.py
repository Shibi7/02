import cv2
img = cv2.imread("img2.png")
greimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
contours, heirarchy = cv2.findContours(greimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
for i, cnt in enumerate(sorted_contours[:2],1):
    cv2.drawContours(img, cnt, -1, (0,255, 0), 3)
    cv2.putText(img, str(i), (cnt[0, 0, 0], cnt[0, 0, 1]-10),cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,255, 0),3)



cv2.imshow("sorted", img)
cv2.waitKey(0)