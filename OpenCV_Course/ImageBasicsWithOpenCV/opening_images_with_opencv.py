import cv2

img = cv2.imread('../newimg.jpg')

while True:
    cv2.imshow('Skater', img)

    # checking if we have waited 1ms and the esc key has been pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()

