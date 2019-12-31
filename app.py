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
        #print(row, col)
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
        #print('Affline transformation',H)
        #print('Thumbnail',thumbnail)

        
        return thumbnail, H


def align_face(imgData):
    alignment = AlignDlib(facePredictor)
    # Detect face and return bounding box
    jc_orig= imgData
    bb = alignment.getLargestFaceBoundingBox(jc_orig)
    row,col,= jc_orig.shape[:2]
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

def normalize_image(img_read):
    img_y_cr_cb = cv2.cvtColor(img_read, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_y_cr_cb)
    img_y_np = np.asarray(y).astype(float)

    img_y_np /= 255
    img_y_np -= img_y_np.mean()
    img_y_np /= img_y_np.std()
    scale = np.max([np.abs(np.percentile(img_y_np, 1.0)),
                        np.abs(np.percentile(img_y_np, 99.0))])
    img_y_np = img_y_np / scale
    img_y_np = np.clip(img_y_np, -1.0, 1.0)
    img_y_np = (img_y_np + 1.0) / 2.0
        
    img_y_np = (img_y_np * 255 + 0.5).astype(np.uint8)

    img_y_cr_cb_eq = cv2.merge((img_y_np, cr, cb))
        
    img_nrm = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCR_CB2BGR)
    

    return img_nrm

def reverse_image(img_read, new_img):
    alignment = AlignDlib(facePredictor)
    jc_orig = img_read
    bb = alignment.getLargestFaceBoundingBox(jc_orig)
    row,col,= jc_orig.shape[:2]
    jc_aligned, M = alignment.align(96, jc_orig, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
    y=0 
    x=0
    h=0
    w=0
    crop_img = jc_aligned[y+510:y+h-540, x+440:x+w-750]

    width = int(crop_img.shape[1])
    height = int(crop_img.shape[0])

    new_img = cv2.resize(new_img*255,(width, height))
    jc_aligned[y+510:y+h-540, x+440:x+w-750] = new_img
    M1 = cv2.invertAffineTransform(M)
    new= cv2.warpAffine(jc_aligned,M1,(1400,1400))
    new1 = new[y:y+h-(1400-row), x:x+w-(1400-col)]
    back = cv2.resize(new1,(col, row))

    kernel = np.ones((5, 5), np.float32)/25
    dstination = cv2.filter2D(back, -1, kernel)

    return dstination

    


def load_morph():
    model= load_model('all_tune_wt.h5') 
    return model

def norm_img_list(img_nrm):

    crop_and_align_img= align_face(img_nrm)
    
    images_list = list()
    images_list.append(crop_and_align_img)
    image = np.array(images_list, 'float32')
    image /= 255

    return image


def happy(img_read):
    image = normalize_image(img_read)
    print(image.shape)
    img_nrm = norm_img_list(image)
    print(img_nrm.shape)
    happy_model= load_morph()
    context=["3"]
    num_classes=6
    lines= np.array(context)
    emotion = keras.utils.to_categorical(lines, num_classes)
    labels = np.array(emotion).astype('float32')

    custom = happy_model.predict([img_nrm, labels]) 

    happy_img = reverse_image(image, custom[0])

    return happy_img

def surprised(img_read):
    image = normalize_image(img_read)
    print(image.shape)
    img_nrm = norm_img_list(image)
    print(img_nrm.shape)
    surprised_model= load_morph()
    context=["0"]
    num_classes=6
    lines= np.array(context)
    emotion = keras.utils.to_categorical(lines, num_classes)
    labels = np.array(emotion).astype('float32')

    custom = surprised_model.predict([img_nrm, labels]) 

    surprised_img = reverse_image(image, custom[0])

    return surprised_img

def fear(img_read):
    image = normalize_image(img_read)
    print(image.shape)
    img_nrm = norm_img_list(image)
    print(img_nrm.shape)
    fear_model= load_morph()
    context=["1"]
    num_classes=6
    lines= np.array(context)
    emotion = keras.utils.to_categorical(lines, num_classes)
    labels = np.array(emotion).astype('float32')

    custom = fear_model.predict([img_nrm, labels]) 

    fear_img = reverse_image(image, custom[0])

    return fear_img

def disgust(img_read):
    image = normalize_image(img_read)
    print(image.shape)
    img_nrm = norm_img_list(image)
    print(img_nrm.shape)
    disgust_model= load_morph()
    context=["2"]
    num_classes=6
    lines= np.array(context)
    emotion = keras.utils.to_categorical(lines, num_classes)
    labels = np.array(emotion).astype('float32')

    custom = disgust_model.predict([img_nrm, labels]) 

    disgust_img = reverse_image(image, custom[0])

    return disgust_img

def sad(img_read):
    image = normalize_image(img_read)
    print(image.shape)
    img_nrm = norm_img_list(image)
    print(img_nrm.shape)
    sad_model= load_morph()
    context=["4"]
    num_classes=6
    lines= np.array(context)
    emotion = keras.utils.to_categorical(lines, num_classes)
    labels = np.array(emotion).astype('float32')

    custom = sad_model.predict([img_nrm, labels]) 

    sad_img = reverse_image(image, custom[0])

    return sad_img

def angry(img_read):
    image = normalize_image(img_read)
    print(image.shape)
    img_nrm = norm_img_list(image)
    print(img_nrm.shape)
    angry_model= load_morph()
    context=["5"]
    num_classes=6
    lines= np.array(context)
    emotion = keras.utils.to_categorical(lines, num_classes)
    labels = np.array(emotion).astype('float32')

    custom = angry_model.predict([img_nrm, labels]) 

    angry_img = reverse_image(image, custom[0])

    return angry_img

def model_classification(img_read):


    crop_and_align_img= align_face(img_read)

    model= load_model('weights_best3 0.h5') 
    x = image.img_to_array(crop_and_align_img)
    x = np.expand_dims(x, axis = 0)

    x /= 255
    x = np.array(x, 'float32')
    tf.image.per_image_standardization(x)

    custom = model.predict(x)

    return custom

# Jakaria: function for classify image
@app.route('/save-image', methods=['GET', 'POST'])
def save_image():
    data_url = request.values['imgBase64']
    legend = 'Emotion Percentage'
    labels = ['surprised', 'fearful', 'disgusted', 'happy', 'sad', 'angry', 'neutral']
    content = data_url.split(';')[1]
    image_encoded = content.split(',')[1]
    body = base64.b64decode(bytes(image_encoded.encode('utf-8')))
    new_file_path_and_name = datetime.datetime.now().strftime("upload/"+"%y%m%d_%H%M%S")+"image.png"
    with open(new_file_path_and_name, "wb") as fh:
        fh.write(body)
        read_img= ndimage.imread(new_file_path_and_name) 
        im_rgb = cv2.cvtColor(read_img, cv2.COLOR_BGR2RGB)
        custom= model_classification(im_rgb)
        custom = [x * 100 for x in custom]
        
        pred_result= custom[0]
        #print(pred_result)
        #pred_result= pred_result.astype(np.str)
        sting_result= np.array2string(pred_result, precision=2, separator=",", suppress_small= True)
        print(sting_result)
        classification_result = sting_result.strip()
        classification_result = classification_result.replace(" ", "")
        #print(classification_result)

        
        # res_array= np.array(custom)
        # list(res_array)
        # arr= res_array.tolist()
        # tup= tuple(arr)

    return classification_result

@app.route('/happy-image', methods=['GET', 'POST'])
def happy_image():
    data_url = request.values['imgBase64']
    content = data_url.split(';')[1]
    image_encoded = content.split(',')[1]
    body = base64.b64decode(bytes(image_encoded.encode('utf-8')))
    new_file_path_and_name = datetime.datetime.now().strftime("upload/"+"%y%m%d_%H%M%S")+"image.png"
    with open(new_file_path_and_name, "wb") as fh:
        fh.write(body)

    read_img= ndimage.imread(new_file_path_and_name) 
    im_rgb = cv2.cvtColor(read_img, cv2.COLOR_BGR2RGB)

    custom= happy(im_rgb)
    happy_morphed= np.array2string(custom)
    base64_str = base64.b64decode(str(custom.encode('utf-8')))

    print(custom)
    print(base64_str)
    cv2.imwrite("upload/happy.jpg", custom)
    #custom = [x * 100 for x in custom]
        
    return base64_str

@app.route('/surprised-image', methods=['GET', 'POST'])
def surprised_image():
    data_url = request.values['imgBase64']
    content = data_url.split(';')[1]
    image_encoded = content.split(',')[1]
    body = base64.b64decode(bytes(image_encoded.encode('utf-8')))
    new_file_path_and_name = datetime.datetime.now().strftime("upload/"+"%y%m%d_%H%M%S")+"image.png"
    with open(new_file_path_and_name, "wb") as fh:
        fh.write(body)

    read_img= ndimage.imread(new_file_path_and_name) 
    im_rgb = cv2.cvtColor(read_img, cv2.COLOR_BGR2RGB)

    custom= surprised(im_rgb)
    print(custom)
    cv2.imwrite("upload/surprised.jpg", custom)
    #custom = [x * 100 for x in custom]
        
    return "0"

@app.route('/fear-image', methods=['GET', 'POST'])
def fear_image():
    data_url = request.values['imgBase64']
    content = data_url.split(';')[1]
    image_encoded = content.split(',')[1]
    body = base64.b64decode(bytes(image_encoded.encode('utf-8')))
    new_file_path_and_name = datetime.datetime.now().strftime("upload/"+"%y%m%d_%H%M%S")+"image.png"
    with open(new_file_path_and_name, "wb") as fh:
        fh.write(body)

    read_img= ndimage.imread(new_file_path_and_name) 
    im_rgb = cv2.cvtColor(read_img, cv2.COLOR_BGR2RGB)

    custom= fear(im_rgb)
    print(custom)
    cv2.imwrite("upload/fear.jpg", custom)
    #custom = [x * 100 for x in custom]
        
    return "0"

@app.route('/disgust-image', methods=['GET', 'POST'])
def disgust_image():
    data_url = request.values['imgBase64']
    content = data_url.split(';')[1]
    image_encoded = content.split(',')[1]
    body = base64.b64decode(bytes(image_encoded.encode('utf-8')))
    new_file_path_and_name = datetime.datetime.now().strftime("upload/"+"%y%m%d_%H%M%S")+"image.png"
    with open(new_file_path_and_name, "wb") as fh:
        fh.write(body)

    read_img= ndimage.imread(new_file_path_and_name) 
    im_rgb = cv2.cvtColor(read_img, cv2.COLOR_BGR2RGB)

    custom= disgust(im_rgb)
    print(custom)
    cv2.imwrite("upload/disgust.jpg", custom)
    #custom = [x * 100 for x in custom]
        
    return "0"

@app.route('/sad-image', methods=['GET', 'POST'])
def sad_image():
    data_url = request.values['imgBase64']
    content = data_url.split(';')[1]
    image_encoded = content.split(',')[1]
    body = base64.b64decode(bytes(image_encoded.encode('utf-8')))
    new_file_path_and_name = datetime.datetime.now().strftime("upload/"+"%y%m%d_%H%M%S")+"image.png"
    with open(new_file_path_and_name, "wb") as fh:
        fh.write(body)

    read_img= ndimage.imread(new_file_path_and_name) 
    im_rgb = cv2.cvtColor(read_img, cv2.COLOR_BGR2RGB)

    custom= sad(im_rgb)
    print(custom)
    cv2.imwrite("upload/sad.jpg", custom)
    #custom = [x * 100 for x in custom]
        
    return "0"

@app.route('/angry-image', methods=['GET', 'POST'])
def angry_image():
    data_url = request.values['imgBase64']
    content = data_url.split(';')[1]
    image_encoded = content.split(',')[1]
    body = base64.b64decode(bytes(image_encoded.encode('utf-8')))
    new_file_path_and_name = datetime.datetime.now().strftime("upload/"+"%y%m%d_%H%M%S")+"image.png"
    with open(new_file_path_and_name, "wb") as fh:
        fh.write(body)

    read_img= ndimage.imread(new_file_path_and_name) 
    im_rgb = cv2.cvtColor(read_img, cv2.COLOR_BGR2RGB)

    custom= angry(im_rgb)
    print(custom)
    cv2.imwrite("upload/angry.jpg", custom)
    #custom = [x * 100 for x in custom]
        
    return "0"

        



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