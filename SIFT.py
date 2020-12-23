import cv2

# read images
img1 = cv2.imread('a.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('b.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imshow('img1',img1)
cv2.imshow('img2',img2)

#sift
sift = cv2.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

print(len(keypoints_1), len(keypoints_2))

#feature matching
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)
cv2.imshow('img',img3)
cv2.waitKey(0)
