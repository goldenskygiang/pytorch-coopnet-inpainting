from imageio import imwrite
from numpy.random import default_rng
import numpy as np
import os

def unpickle(file):
    import pickle
    fo = open(file, 'rb')
    dict = pickle.load(fo,encoding="latin1")
    fo.close()
    return dict

INPAINTING_GEN = True

rng = default_rng()

def random_missing(img):
    assert((img.shape == (32, 32, 3)))
    x = rng.integers(30)
    y = rng.integers(30)
    d = rng.integers(1, min(32 - x, 32 - y))

    mask = np.zeros(img.shape, dtype=np.uint8)

    img[x:x+d, y:y+d] = 0
    mask[x:x+d, y:y+d] = 255

    return img, mask

if INPAINTING_GEN:
    os.makedirs(name = "cifar-missing/img", exist_ok=True)
    os.makedirs(name = "cifar-missing/mask", exist_ok=True)

    os.makedirs(name = "cifar-missing-test/img", exist_ok=True)
    os.makedirs(name = "cifar-missing-test/mask", exist_ok=True)

else:
    os.makedirs(name = "cifar", exist_ok=True)
    os.makedirs(name = "cifar-test", exist_ok=True)

for j in range(1, 6):
    dataName = "cifar-10-batches-py/data_batch_" + str(j)
    Xtr = unpickle(dataName)
    print (dataName + " is loading...")

    for i in range(0, 10000):
        img = np.reshape(Xtr['data'][i], (3, 32, 32))  
        img = img.transpose(1, 2, 0)  

        if INPAINTING_GEN:
            trainPicName = 'cifar-missing/img/' + str(i + (j - 1)*10000) + '.png'
            trainMaskName = 'cifar-missing/mask/' + str(i + (j - 1)*10000) + '.png'
            miss, mask = random_missing(img)

            imwrite(trainPicName, img)
            imwrite(trainMaskName, mask)
        else:
            picName = 'cifar/' + str(i + (j - 1)*10000) + '.png'
            imwrite(picName, img)

    print (dataName + " loaded.")

print ("test_batch is loading...")


testXtr = unpickle("cifar-10-batches-py/test_batch")
for i in range(0, 10000):
    img = np.reshape(testXtr['data'][i], (3, 32, 32))
    img = img.transpose(1, 2, 0)

    if INPAINTING_GEN:
        trainPicName = 'cifar-missing-test/img' + str(i +  50000) + '.png'
        trainMaskName = 'cifar-missing-test/mask' + str(i + 50000) + '.png'
        miss, mask = random_missing(img)

        imwrite(trainPicName, miss)
        imwrite(trainMaskName, mask)
    else:
        picName = 'cifar-test/' + str(i+50000) + '.png'
        imwrite(picName, img)
print ("test_batch loaded.")
