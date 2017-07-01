import numpy as np
import gzip
import struct

def read_data(label, image):
    with gzip.open(label, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(image, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return (label, image)

(train_lbl, train_img) = read_data(r'C:\Users\dinesh\Documents\exercise1\Dataset\train-labels-idx1-ubyte.gz', 
                                   r'C:\Users\dinesh\Documents\exercise1\Dataset\train-images-idx3-ubyte.gz')
(test_lbl, test_img) = read_data(r'C:\Users\dinesh\Documents\exercise1\Dataset\t10k-labels-idx1-ubyte.gz', 
                                   r'C:\Users\dinesh\Documents\exercise1\Dataset\t10k-images-idx3-ubyte.gz')
print(train_img)
print(len(train_img))

train_img[0]
train_img[0].shape
print(test_img)
print(train_lbl)
print(test_lbl)

%matplotlib inline
import matplotlib.pyplot as plt
for i in range(20):
    plt.subplot(2,10,i+1)
    plt.imshow(train_img[i], cmap='Greys_r')
    plt.axis('off')
plt.show()
print('label: %s' % (train_lbl[0:10],))
print('label: %s' % (train_lbl[10:20],))

for i in range(20):
    plt.subplot(2,10,i+1)
    plt.imshow(test_img[i], cmap='Greys_r')
    plt.axis('off')
plt.show()
print('label: %s' % (test_lbl[0:10],))
print('label: %s' % (test_lbl[10:20],))

# For deskwing the digits
import cv2
SZ=28
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

# We deskew image using its second order moments
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        # no deskewing needed. 
        return img.copy()
    # Calculate skew based on central momemts. 
    skew = m['mu11']/m['mu02']
    # Calculate affine transform to correct skewness. 
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    # Apply affine transform
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=affine_flags)
    return img

# to obtain the list of image after deskewing
train_deskew_images = list(map(deskew,train_img))

for i in range(50):
    plt.subplot(5,10,i+1)
    plt.imshow(train_deskew_images[i], cmap='Greys_r')
    plt.axis('off')
plt.show()
print('label: %s' % (train_lbl[0:10],))
print('label: %s' % (train_lbl[10:20],))
print('label: %s' % (train_lbl[20:30],))
print('label: %s' % (train_lbl[30:40],))
print('label: %s' % (train_lbl[40:50],))

zero_index = [1,21,37]

for i in range(3):
    plt.subplot(1,10,i+1)
    plt.imshow(train_deskew_images[zero_index[i]], cmap='Greys_r')
    plt.axis('off')
plt.show()

one_index = [3,6,8]
for i in range(3):
    plt.subplot(1,10,i+1)
    plt.imshow(train_deskew_images[one_index[i]], cmap='Greys_r')
    plt.axis('off')
plt.show()

two_index = [5,16,25]
for i in range(3):
    plt.subplot(1,10,i+1)
    plt.imshow(train_deskew_images[two_index[i]], cmap='Greys_r')
    plt.axis('off')
plt.show()

three_index = [7,10,12]
for i in range(3):
    plt.subplot(1,10,i+1)
    plt.imshow(train_deskew_images[three_index[i]], cmap='Greys_r')
    plt.axis('off')
plt.show()

four_index = [2,9,20]
for i in range(3):
    plt.subplot(1,10,i+1)
    plt.imshow(train_deskew_images[four_index[i]], cmap='Greys_r')
    plt.axis('off')
plt.show()

five_index = [0,11,35]
for i in range(3):
    plt.subplot(1,10,i+1)
    plt.imshow(train_deskew_images[five_index[i]], cmap='Greys_r')
    plt.axis('off')
plt.show()

six_index = [18,32,36]
for i in range(3):
    plt.subplot(1,10,i+1)
    plt.imshow(train_deskew_images[six_index[i]], cmap='Greys_r')
    plt.axis('off')
plt.show()

seven_index = [15,29,38]
for i in range(3):
    plt.subplot(1,10,i+1)
    plt.imshow(train_deskew_images[seven_index[i]], cmap='Greys_r')
    plt.axis('off')
plt.show()

eight_index = [17,31,41]
for i in range(3):
    plt.subplot(1,10,i+1)
    plt.imshow(train_deskew_images[eight_index[i]], cmap='Greys_r')
    plt.axis('off')
plt.show()

nine_index = [4,19,22]
for i in range(3):
    plt.subplot(1,10,i+1)
    plt.imshow(train_deskew_images[nine_index[i]], cmap='Greys_r')
    plt.axis('off')
plt.show()

# to obtain the list of image after deskewing
test_deskew_images = list(map(deskew,test_img))

for i in range(20):
    plt.subplot(2,10,i+1)
    plt.imshow(test_deskew_images[i], cmap='Greys_r')
    plt.axis('off')
plt.show()
print('label: %s' % (test_lbl[0:10],))
print('label: %s' % (test_lbl[10:20],))

# We vectorize image using flatten() function
def vector(img):
    img = img.flatten()
    return img

# to obtain the list of train image after vectorization
train_images_vector = list(map(vector,train_deskew_images))
train_images_vector[0].shape
train_images_vector[0]

plt.subplot(1,10,1)
plt.imshow(test_deskew_images[0], cmap='Greys_r')
plt.axis('off')
plt.show()

plt.plot(test_deskew_images[0])

# to obtain the list of train image after vectorization
test_images_vector = list(map(vector,test_deskew_images))
test_images_vector[0].shape
test_images_vector[0]
plt.plot(test_images_vector[0])

train_deskew_images[0]

plt.subplot(1,10,1)
plt.axis('off')
plt.imshow(train_deskew_images[0], cmap='Greys_r')
plt.show()

def dimReduction(img):
    train_14x14_image = []
    for image in img:
        pts1 = np.float32([[7,7],[21,7],[7,21],[21,21]])
        pts2 = np.float32([[0,0],[14,0],[0,14],[14,14]])

        M = cv2.getPerspectiveTransform(pts1,pts2)

        dst = cv2.warpPerspective(image,M,(14,14))
        train_14x14_image.append(dst)
    return train_14x14_image

reduced_train_images = dimReduction(np.asarray(train_deskew_images))

for i in range(20):
    plt.subplot(2,10,i+1)
    plt.imshow(reduced_train_images[i], cmap='Greys_r')
    plt.axis('off')
plt.show()
print('label: %s' % (train_lbl[0:10],))
print('label: %s' % (train_lbl[10:20],))

reduced_test_images = dimReduction(np.asarray(test_deskew_images))
for i in range(20):
    plt.subplot(2,10,i+1)
    plt.imshow(reduced_test_images[i], cmap='Greys_r')
    plt.axis('off')
plt.show()
print('label: %s' % (test_lbl[0:10],))
print('label: %s' % (test_lbl[10:20],))

reduced_train_images[0]

#Convert the list of images to array of images
train_image_array = np.asarray(reduced_train_images)

#To implement binary threshold of 0 and 255 for both train and test image
ret,threshold_train = cv2.threshold(train_image_array,127,255,cv2.THRESH_BINARY)

for i in range(20):
    plt.subplot(2,10,i+1)
    plt.imshow(threshold_train[i], cmap='Greys_r')
    plt.axis('off')
plt.show()
print('label: %s' % (train_lbl[0:10],))
print('label: %s' % (train_lbl[10:20],))

test_image_array = np.asarray(reduced_test_images)
ret1,threshold_test = cv2.threshold(test_image_array,127,255,cv2.THRESH_BINARY)
for i in range(20):
    plt.subplot(2,10,i+1)
    plt.imshow(threshold_test[i], cmap='Greys_r')
    plt.axis('off')
plt.show()
print('label: %s' % (test_lbl[0:10],))
print('label: %s' % (test_lbl[10:20],))

for i in range(3):
    plt.subplot(1,10,i+1)
    plt.imshow(threshold_train[zero_index[i]], cmap='Greys_r')
    plt.axis('off')
plt.show()

for i in range(3):
    plt.subplot(1,10,i+1)
    plt.imshow(threshold_train[one_index[i]], cmap='Greys_r')
    plt.axis('off')
plt.show()

for i in range(3):
    plt.subplot(1,10,i+1)
    plt.imshow(threshold_train[two_index[i]], cmap='Greys_r')
    plt.axis('off')
plt.show()

for i in range(3):
    plt.subplot(1,10,i+1)
    plt.imshow(threshold_train[three_index[i]], cmap='Greys_r')
    plt.axis('off')
plt.show()

for i in range(3):
    plt.subplot(1,10,i+1)
    plt.imshow(threshold_train[four_index[i]], cmap='Greys_r')
    plt.axis('off')
plt.show()

for i in range(3):
    plt.subplot(1,10,i+1)
    plt.imshow(threshold_train[five_index[i]], cmap='Greys_r')
    plt.axis('off')
plt.show()

for i in range(3):
    plt.subplot(1,10,i+1)
    plt.imshow(threshold_train[six_index[i]], cmap='Greys_r')
    plt.axis('off')
plt.show()

for i in range(3):
    plt.subplot(1,10,i+1)
    plt.imshow(threshold_train[seven_index[i]], cmap='Greys_r')
    plt.axis('off')
plt.show()

for i in range(3):
    plt.subplot(1,10,i+1)
    plt.imshow(threshold_train[eight_index[i]], cmap='Greys_r')
    plt.axis('off')
plt.show()

for i in range(3):
    plt.subplot(1,10,i+1)
    plt.imshow(threshold_train[nine_index[i]], cmap='Greys_r')
    plt.axis('off')
plt.show()

def templateMatching(train_img,train_temp, num):
    for j in range(3):
        img = train_img[num[j]]
        temp = train_temp[num[j]]
        w,h = temp.shape[::-1]
        # Apply template Matching
        res = cv2.matchTemplate(img,temp,cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = min_loc
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img,top_left, bottom_right, 255, 2)
        plt.subplot(121),plt.imshow(res,cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(img,cmap = 'gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        res = []
        plt.show()
        
templateMatching(train_deskew_images,threshold_train, zero_index)
templateMatching(train_deskew_images,threshold_train,one_index)
templateMatching(train_deskew_images,threshold_train,two_index)
templateMatching(train_deskew_images,threshold_train,three_index)
templateMatching(train_deskew_images,threshold_train,four_index)
templateMatching(train_deskew_images,threshold_train,five_index)
templateMatching(train_deskew_images,threshold_train,six_index)
templateMatching(train_deskew_images,threshold_train,seven_index)
templateMatching(train_deskew_images,threshold_train,eight_index)
templateMatching(train_deskew_images,threshold_train,nine_index)

winSize = (14,14)
blockSize = (7,7)
blockStride = (7,7)
cellSize = (7,7)
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradients = True
 
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradients)

# td = np.asarray(threshold_train[0])
# train_hog_data = [hog.compute(for row in threshold_train)]
train_hog_data = []
for row in threshold_train:
    descriptor = hog.compute(row)
    train_hog_data.append(descriptor)

test_hog_data = []
for row in threshold_test:
    descriptor = hog.compute(row)
    test_hog_data.append(descriptor)
    
train_hog_data[1]
train_hog_data[1]
test_hog_data[1]
trainData = np.float32(train_hog_data)
testData = np.float32(test_hog_data)

# trainData = np.array(train_hog_data)
responses = np.float32(np.repeat(np.arange(10),250)[:,np.newaxis])

# Set up SVM for OpenCV 3
svm = cv2.ml.SVM_create()
# Set SVM type
svm.setType(cv2.ml.SVM_C_SVC)
# Set SVM Kernel to Radial Basis Function (RBF) 
svm.setKernel(cv2.ml.SVM_RBF)
# Set parameter C
svm.setC(12.5)
# Set parameter Gamma
svm.setGamma(0.5)
 
# Train SVM on training data  
svm.train(trainData,cv2.ml.ROW_SAMPLE, train_lbl.astype(int))

svm.save("digits_svm_model.dat")
myarray = np.fromfile('digits_svm_model.dat',dtype=float)

myarray

# Test on a held out test set
testResponse = svm.predict(testData)[1].ravel()

testResponse
testResponse.size
testResponse.shape

mask = testResponse==test_lbl
correct = np.count_nonzero(mask)
print (correct*100.0/testResponse.size)

#predict
# output = SVM.predict(samples)[1].ravel()
reasult = testResponse.astype(int)

from pandas_ml import ConfusionMatrix
confusion_matrix = ConfusionMatrix(test_lbl, reasult)
print("Confusion matrix:\n%s" % confusion_matrix)
confusion_matrix.print_stats()
confusion_matrix.classification_report
confusion_matrix.pred
confusion_matrix.stats_class
confusion_matrix.plot()
plt.show()

confusion_matrix.FN
result = testResponse.astype(np.int32)
result
test_lbl.size

#Using test data to calculate hyper parameter C
ACC = confusion_matrix.ACC
ACC

C = confusion_matrix.ACC * (confusion_matrix.population)
C

confusion_matrix.TP
tuple_value = []
for i in range (test_lbl.size):
    if test_lbl[i]!= result[i]:
        print (i)
        # tuple_value = [(i,test_lbl[i],result[i])]
        tuple_value.append((i,test_lbl[i],result[i]))
        
tuple_value
len(tuple_value)

failed_classification = []
actual = []
predicted = []
for i in range (10):
    for x, y, z in tuple_value:
        if y  == i:           
            failed_classification.append((x,y,z))
            #print (failed_classification)
            plt.subplot(1,10,i+1)
            plt.imshow(threshold_test[x], cmap='Greys_r')
            #plt.suptitle('Actual -> Pridected')
            actual.append(y)
            predicted.append(z)
            plt.title(str(y)+' -> '+str(z)), plt.xticks([]), plt.yticks([])
            plt.axis('off')
            break
plt.show()
print('Actual: %s' % (actual[0:10],))
print('Predcted: %s' % (predicted[0:10],))

failed_classification




