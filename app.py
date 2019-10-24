from flask import Flask, render_template, url_for, request, redirect
import base64
import datetime
import os
import io
 
from scipy.misc import imsave, imread, imresize
import glob
import cv2
import dlib
import numpy as np
from scipy import ndimage
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from keras.preprocessing.image import array_to_img
from math import sqrt
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True





#sys.path.append(os.path.abspath("./model"))
#from load import * 


app = Flask(__name__)

#global model, graph
#initialize these variables
#model, graph = init()
TEMPLATE = np.float32([(383, 591), (386, 636), (394, 681), (403, 723), (414, 765), (436, 803), (464, 835), (498, 859), (540, 866), (579, 861), (612, 839), (639, 809), (661, 773), (673, 734), (682, 693), (691, 650), (696, 608), (415, 591), (435, 571), (463, 568), (490, 575), (517, 583), (586, 582), (612, 573), (637, 569), (662, 575), (676, 597), (549, 606), (549, 639), (548, 670), (548, 701), (515, 711), (530, 718), (547, 724), (563, 720), (577, 715), (450, 611), (467, 600), (488, 601), (502, 613), (485, 617), (466, 617), (591, 617), (607, 606), (626, 606), (640, 618), (627, 624), (607, 623), (486, 761), (506, 754), (526, 750), (545, 757), (565, 752), (583, 757), (598, 766), (582, 784), (563, 795), (543, 797), (523, 794), (503, 783), (495, 763), (525, 766), (545, 769), (565, 766), (591, 767), (564, 767), (544, 770), (524, 766)])

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = TEMPLATE
facePredictor = "shape_predictor_68_face_landmarks.dat"
imgDim = 96

class AlignDlib:

    #: Landmark indices corresponding to the inner eyes and bottom lip.
    INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]

    #: Landmark indices corresponding to the outer eyes and nose.
    OUTER_EYES_AND_NOSE = [36, 45, 33]

    def __init__(self, facePredictor):
        """
        Instantiate an 'AlignDlib' object.
        :param facePredictor: The path to dlib's
        :type facePredictor: str
        """
        assert facePredictor is not None

        #pylint: disable=no-member
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(facePredictor)

    def getAllFaceBoundingBoxes(self, rgbImg):
        """
        Find all face bounding boxes in an image.
        :param rgbImg: RGB image to process. Shape: (height, width, 3)
        :type rgbImg: numpy.ndarray
        :return: All face bounding boxes in an image.
        :rtype: dlib.rectangles
        """
        assert rgbImg is not None

        try:
            return self.detector(rgbImg, 1)
        except Exception as e: #pylint: disable=broad-except
            print("Warning: {}".format(e))
            # In rare cases, exceptions are thrown.
            return []

    def getLargestFaceBoundingBox(self, rgbImg, skipMulti=False):
        """
        Find the largest face bounding box in an image.
        :param rgbImg: RGB image to process. Shape: (height, width, 3)
        :type rgbImg: numpy.ndarray
        :param skipMulti: Skip image if more than one face detected.
        :type skipMulti: bool
        :return: The largest face bounding box in an image, or None.
        :rtype: dlib.rectangle
        """
        assert rgbImg is not None

        faces = self.getAllFaceBoundingBoxes(rgbImg)
        if (not skipMulti and len(faces) > 0) or len(faces) == 1:
            return max(faces, key=lambda rect: rect.width() * rect.height())
        else:
            return None

    def findLandmarks(self, rgbImg, bb):
        """
        Find the landmarks of a face.
        :param rgbImg: RGB image to process. Shape: (height, width, 3)
        :type rgbImg: numpy.ndarray
        :param bb: Bounding box around the face to find landmarks for.
        :type bb: dlib.rectangle
        :return: Detected landmark locations.
        :rtype: list of (x,y) tuples
        """
        assert rgbImg is not None
        assert bb is not None

        points = self.predictor(rgbImg, bb)
        #return list(map(lambda p: (p.x, p.y), points.parts()))
        return [(p.x, p.y) for p in points.parts()]




    #pylint: disable=dangerous-default-value
    def align(self, imgDim, rgbImg, bb=None,
              landmarks=None, landmarkIndices=INNER_EYES_AND_BOTTOM_LIP,
              skipMulti=False, scale=1.0):
        """align(imgDim, rgbImg, bb=None, landmarks=None, landmarkIndices=INNER_EYES_AND_BOTTOM_LIP)
        Transform and align a face in an image.
        :param imgDim: The edge length in pixels of the square the image is resized to.
        :type imgDim: int
        :param rgbImg: RGB image to process. Shape: (height, width, 3)
        :type rgbImg: numpy.ndarray
        :param bb: Bounding box around the face to align. \
                   Defaults to the largest face.
        :type bb: dlib.rectangle
        :param landmarks: Detected landmark locations. \
                          Landmarks found on `bb` if not provided.
        :type landmarks: list of (x,y) tuples
        :param landmarkIndices: The indices to transform to.
        :type landmarkIndices: list of ints
        :param skipMulti: Skip image if more than one face detected.
        :type skipMulti: bool
        :param scale: Scale image before cropping to the size given by imgDim.
        :type scale: float
        :return: The aligned RGB image. Shape: (imgDim, imgDim, 3)
        :rtype: numpy.ndarray
        """
        assert imgDim is not None
        assert rgbImg is not None
        assert landmarkIndices is not None

        if bb is None:
            bb = self.getLargestFaceBoundingBox(rgbImg, skipMulti)
            if bb is None:
                return

        if landmarks is None:
            landmarks = self.findLandmarks(rgbImg, bb)

        row,col,= rgbImg.shape[:2]
        print(row, col)
        bottom= rgbImg[row-2:row, 0:col]
        mean= cv2.mean(bottom)[0]
        bordersize=0
        border=cv2.copyMakeBorder(rgbImg, top=bordersize+200, bottom=bordersize+400, left=bordersize+100, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[mean,mean,mean] )
        # plt.subplot(131),plt.imshow(rgbImg),plt.title('Input')
        # plt.subplot(132),plt.imshow(border),plt.title('Output')
        npLandmarks = np.float32(landmarks)
        npLandmarkIndices = np.array(landmarkIndices)

        #pylint: disable=maybe-no-member
        H = cv2.getAffineTransform(npLandmarks[npLandmarkIndices],
                                    MINMAX_TEMPLATE[npLandmarkIndices])
        thumbnail = cv2.warpAffine(rgbImg, H, (1400, 1400))
        print('Affline transformation',H)
        print('Thumbnail',thumbnail)

        
        return thumbnail, H


def align_face(imgData):
    alignment = AlignDlib(facePredictor)
    # Detect face and return bounding box
    jc_orig= imgData
    bb = alignment.getLargestFaceBoundingBox(jc_orig)
    print(jc_orig.shape)
    row,col,= jc_orig.shape[:2]
    print(row, col)
    # Transform image using specified face landmark indices and crop image to 96x96
    jc_aligned, M = alignment.align(96, jc_orig, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
    rgb_image = cv2.cvtColor(jc_aligned, cv2.COLOR_BGR2RGB)
    y=0
    x=0
    h=0
    w=0
    crop_img = jc_aligned[y+510:y+h-540, x+440:x+w-750]
    res = cv2.resize(crop_img,(imgDim, imgDim), interpolation = cv2.INTER_CUBIC)
    #plt.imshow(res)
    #plt.show()

    return res

def emotion_analysis(emotions):
    objects = ('surprised', 'fearful', 'disgusted', 'happy', 'sad', 'angry', 'neutral')
    y_pos = np.arange(len(objects))
    plt.barh(y_pos, emotions, align='center', alpha=0.5)
    plt.yticks(y_pos, objects)
    plt.xlabel('percentage')

    plt.savefig("emotion.jpg")
    #plt.show()

# imgData as path to newly uploaded image
# @app.route('/predict/',methods=['GET','POST'])
def predict(img_read):

    # imgData = request.get_data()
    # img_load = "upload/" # Enter Directory of all images 
    # data_path = os.path.join(img_load,'*.png')
    # files = glob.glob(data_path)
    # for f1 in files:
    #img_read = imread(f1)
    print(img_read)
    crop_and_align_img= align_face(img_read)

    model= load_model('DetectedImg/weights_best3 0.h5') 
    x = image.img_to_array(crop_and_align_img)
    x = np.expand_dims(x, axis = 0)

    x /= 255
    x = np.array(x, 'float32')
    tf.image.per_image_standardization(x)

    custom = model.predict(x)
    print(custom[0])
    print(np.argmax(custom,axis=1))
    
    
    #emotion_analysis(custom[0])
    load_img= cv2.imread('emotion.jpg')
    return load_img

# Jakaria: function for saving image
@app.route('/save-image', methods=['GET', 'POST'])
def save_image():
    data_url = request.values['imgBase64']
    content = data_url.split(';')[1]
    image_encoded = content.split(',')[1]
    body = base64.b64decode(bytes(image_encoded.encode('utf-8')))
    new_file_path_and_name = datetime.datetime.now().strftime("upload/"+"%y%m%d_%H%M%S")+"image.png"
    with open(new_file_path_and_name, "wb") as fh:
        fh.write(body)
        read_img= ndimage.imread(new_file_path_and_name) 
        im_rgb = cv2.cvtColor(read_img, cv2.COLOR_BGR2RGB)
        #img= np.asarray(read_img)
        #rgb = img.convert('rgb')
        #img= np.array(read_img)
        #img=cv2.imread(read_img)
        
        #print('PNG',im_rgb)
        #print(im_rgb.shape)
        #print('IMG',img)
        #print(read_img.dtype)
        predict(im_rgb)
   

    return 'ok'


# Jakaria: Loading homepage
@app.route('/')
def app_index():
    return render_template('app_index.html')

if __name__ == "__main__":
	#decide what port to run the app in
	# port = int(os.environ.get('PORT', 5000))
	#run the app locally on the givn port
	#app.run(host='0.0.0.0', port=port)
	#optional if we want to run in debugging mode
	#app.run(debug=True

     app.run(debug=True) # http://127.0.0.0:5000