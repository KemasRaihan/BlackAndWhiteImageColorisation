import numpy as np
import cv2
from cv2 import dnn

# Models: https://github.com/richzhang/colorization/tree/caffe/colorization/models
# Points: https://github.com/richzhang/colorization/blob/caffe/colorization/resources/pts_in_hull.npy
# Inspired by: https://www.geeksforgeeks.org/black-and-white-image-colorization-with-opencv-and-deep-learning/

#-------------------Model File Paths----------------------#
prototxt_path = 'models/colorization_deploy_v2.prototxt'
model_path = 'models/colorization_release_v2.caffemodel'
kernel_path = 'models/pts_in_hull.npy'
img_path = 'images/cat_bw.jpeg'
#-----------------------------------------------------------#

def import_model(prototxt_path, model_path, kernel_path):
    #---------------Reading Model Params------------------------#
    try:
        net = dnn.readNetFromCaffe(prototxt_path, model_path)
    except cv2.error as e:
        print(f"Error loading model: {e}")

    try:
        points = np.load(kernel_path)
    except FileNotFoundError:
        print("Kernel file not found. Ensure 'pts_in_hull.npy' is in the correct path.")
    #-----------------------------------------------------------#
    return [net, points]

def colorise_image(img_path, net, points, img_size):
    # reading and preprocessing Image
    bw_img = cv2.imread(img_path)
    if bw_img is None:
        raise FileNotFoundError(f"Image file not found: {img_path}")

    scaled = bw_img.astype("float32") / 255.0
    
    # open-cv reads image in BGR
    lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    # add the cluster centers as 1x1 convolutions to the model
    points = points.transpose().reshape(2,313,1,1)
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    net.getLayer(class8).blobs = [points.astype(np.float32)]
    net.getLayer(conv8).blobs = [np.full([1,313], 2.606, dtype="float32")]

    # resize the image for the network
    # need to train the image in 224x224 dimension
    resized = cv2.resize(lab_img, (224,224))

    # Split the L channel
    L = cv2.split(resized)[0]

    # Mean subtraction
    L -= 50

    # predicting the ab channels from the input L channel
    net.setInput(cv2.dnn.blobFromImage(L))
    ab_channel = net.forward()[0,:,:,:].transpose((1,2,0))

    # resize the predicted 'ab' volume to the same dimension as the input image
    ab_channel =cv2.resize(ab_channel, (bw_img.shape[1], bw_img.shape[0]))

    # take the L channel from the image
    L = cv2.split(lab_img)[0]

    # join the L channel with predicted ab channel
    colorized = np.concatenate((L[:,:,np.newaxis], ab_channel), axis=2)

    # convert the image from Lab to BGR
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    # change the image to 0-255 range and convert it from float32 to int
    colorized = (255 * colorized).astype("uint8")

    # resize the images and show them together
    bw_img = cv2.resize(bw_img, (img_size[0], img_size[1]))
    colorized = cv2.resize(colorized, (img_size[0], img_size[1]))

    # display the black-and-white image with the colorized image
    cv2.imshow("BW Image", bw_img)
    cv2.imshow("Colorized Image", colorized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


model, points = import_model(prototxt_path, model_path, kernel_path)
colorise_image(img_path, model, points, [640,480])