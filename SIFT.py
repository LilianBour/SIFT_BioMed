import cv2
from os import listdir
from os.path import isfile, join

#I. --SIFT on two images TEST--
#Read images
img1 = cv2.imread('CXR20_IM-0653-1001.png')
img2 = cv2.imread('CXR20_IM-0653-1001_mod.png')
cv2.imshow('img1',img1)
cv2.imshow('img2',img2)

#SIFT Test
sift = cv2.SIFT_create()
keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)
#print(len(keypoints_1), len(keypoints_2))
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)
#print(len(matches))
img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[0:30], img2, flags=2)
cv2.imshow('img',img3)
cv2.waitKey(0)


#II. IMAGE SEARCH
onlyfiles = [f for f in listdir("C:/Users/lilia/github/BioMed_SIFT/test") if isfile(join("C:/Users/lilia/github/BioMed_SIFT/test", f))] #List of pic names
max=0.02*len(onlyfiles) #Max images is 2% of the total number of images to avoid having too much images
image_matching=[]
img1 = cv2.imread('4369.png') #Base image

for i in range(len(onlyfiles)):
    print(round((i/len(onlyfiles))*100,2),'%') #Show progression
    #Open image with OpenCv
    name='C:/Users/lilia/github/BioMed_SIFT/test/'+onlyfiles[i]
    img2 = cv2.imread(name)

    #PART 1 --SIFT--
    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)

    #PART 2 --Ranking--
    #If the list has not enough element add anything
    if len(image_matching)<max:
        image_matching.append([matches,name,keypoints_1,keypoints_2])
        image_matching.sort(key=lambda x: len(x[0]),reverse=True)
        #for j in image_matching:
            #print(j[1],len(j[0]))
    #If the list has enough element, add only if there is more matching element than the minimum in the list
    if len(image_matching)>max and len(matches)>len(image_matching[-1][0]):
        image_matching.pop(-1)
        image_matching.append([matches,name,keypoints_1,keypoints_2])
        image_matching.sort(key=lambda x: len(x[0]),reverse=True)
        #for j in image_matching:
            #print(j[1],len(j[0]))

#PART 3 --Show Results--
for i in image_matching:
    img2 = cv2.imread(i[1])
    img3 = cv2.drawMatches(img1, i[2], img2, i[3], i[0][0:100], img2, flags=2)
    imS = cv2.resize(img3, (1500, 1000))
    cv2.imshow('img',imS)
    cv2.waitKey(0)