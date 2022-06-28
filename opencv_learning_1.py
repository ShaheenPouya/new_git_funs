import cv2 as cv

# img = cv.imread('C:\photos\cat.png')
#
# cv.imshow('kitty', img)
#

capture = cv.VideoCapture('c:\photos\sample.mp4')


def Rescaleframe(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize (frame,dimensions, interpolation = cv.INTER_AREA)


while True:
    isTrue, frame = capture.read()
    big_frame = Rescaleframe(frame, 0.2)

    cv.imshow('alaki', frame)
    cv.imshow('alak', big_frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break



capture.release()
cv.destroyAllWindows()

