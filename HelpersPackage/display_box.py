from xml.etree import cElementTree 
import numpy as np
import matplotlib.pyplot as plt
import cv2
from google.colab.patches import cv2_imshow
from copy import deepcopy
import os
from sys import stdout
from PIL import Image

class dataset():

   def __init__(self,folder_path,model=None,read_xml=True,resized=True,target_shape=(228,228),classes_dict=None):
    self.model =model
    self.shape =target_shape
    self.folder = folder_path
    self.list_dir_files = []
    self.box_coordinate =[]
    self.name_classes = []
    self.classes_dict = classes_dict
    if read_xml: 
      self.read_xml_files()     
    if resized and read_xml :
      self.resized_coordinate = np.array(self.resize_boxes())

   def show_boxes(self,number_random_image,model,subplot,callbacks_list=None):
      self.subplot = subplot
      self.callbacks =callbacks_list
      self.model =model
      self.read_plot_img(
                    data_size= len(self.list_dir_files),
                    nb_image=number_random_image,
                    predict=True)     
      
   def read_xml_files(self):
    cnt=1
    file_names = os.listdir(self.folder)
    histime = {'tree':[],'root':[],'xmldict':[]}
    for path in file_names:
      if path.split('.')[1]=='xml' and 'mixed' not in path:
        print('\rlecture fichier xml {}'.format(cnt),end ='')
        cnt+=1 
        file_abs_path = os.path.join(self.folder,path)
        tree = cElementTree.parse(file_abs_path)
        root = tree.getroot()
        xmldict = XmlDictConfig(root)
        self.list_dir_files = self.list_dir_files + [file_abs_path.split('.')[0]+'.jpg'] 
        self.box_coordinate.append(np.array( list(xmldict['object']['bndbox'].values()) ).astype('int') ),
        self.name_classes = self.name_classes + [xmldict['object']['name']]

    print('  Found {} ground thruth boxes '.format(len(self.box_coordinate)) )
    if not self.classes_dict:
      #self.classe_indices = {0:'Arduino_Nano', 1:'ESP8266',2: 'Heltec_ESP32_Lora',3: 'Raspberry_Pi_3'}
      pass
    else:
      self.classe_indices = {val:k for k,val in classes_dict.items()}

    self.classe_indices = { 0: 'apple', 1: 'banana', 2: 'orange'} 
    # self.classe_indices = {idx : name for idx,name in enumerate(list(set(self.name_classes))) } 
    # to adapt in the order prediction of model

   def read_plot_img(self,data_size,nb_image=5,predict=False): 
    per = np.random.permutation(data_size)
    ind_choices = per[:nb_image]
    rand_file = [self.list_dir_files[rnd] for rnd in ind_choices ]
    subplot = (1,nb_image,1)
    img_file_name = []
    images =[]
    self.fig = plt.figure(figsize= (self.subplot[1]*5,self.subplot[0]*5))
    for idx,path in enumerate(rand_file):
      img_array = cv2.imread(path) # DATA ON THE FLY ...
      img_original_shape = img_array.shape

      img_resized =cv2.resize(img_array,self.shape)
      self.img_resized_shape =img_resized.shape

      if predict : # DISPLAYING IMAGES AND BOXES 
        print('\rprediction image {}'.format(idx +1) ,end = '')
        boxe_predicted,cls_one_hot_predi=self.model.predict(np.array(img_resized.reshape(1,self.shape[0],self.shape[1],3)/self.shape[0]),
                                                            callbacks = self.callbacks)
        self.display(img_resized,
                true_boxes = self.resized_coordinate[ind_choices[idx]]/self.shape[0],                   
                box_predi = boxe_predicted,
                label=[self.name_classes[ind_choices[idx]],
                      self.classe_indices[np.argmax(cls_one_hot_predi)] ],                                                       
                )

        
   def display(self,image,true_boxes,box_predi,label):
    width = self.img_resized_shape[0]
    heigth = self.img_resized_shape[1]

    if (true_boxes is not None ) and (box_predi is not None):
      imh = self.plot_one_box(image,true_boxes,(0,255,0),w=width,h=heigth,titre=label[0])
      self.plot_one_box(imh,box_predi,(255,0,0),w=width,h=heigth, showe=True,titre=label[1])

    elif true_boxes is not None:
      self.plot_one_box(image,true_boxes,(0,255,0),w=width,h=heigth,showe=True,titre=label[0])
    elif box_predi is not None:
      self.plot_one_box(image,box_predi,(0,0,255),w=width,h=heigth,showe=True,titre=label[1])

    if None:
        subplot = display_one_image(img_array, name_classes[test.index(True)], subplot, False)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.show()

   def plot_one_box(self,img,box,color,w,h,showe=False,titre=None):
    box = box* np.array([w, h, w, h])
    if titre:
      (startX, startY, endX, endY) = box.astype("int").ravel()
      cv2.rectangle(img,(startX,startY),(endX,endY),color,2)
      y=startY - 15 if startY - 15 > 15 else startY + 15
      cv2.putText(img, titre, (startX, y),	cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # Displaying the image 
    if showe:
      self.display_one_image(img)
      #cv2_imshow(img)
    else:return img

   def display_one_image(self,image,red=False):
      self.fig.add_subplot(self.subplot[0],self.subplot[1],self.subplot[2])
      plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      #plt.imshow(image) for microcontroller
      # plt.show()
      self.subplot = (self.subplot[0],self.subplot[1],self.subplot[2]+1)

   def resize_one_boxe(self,path,coordinate):
      coordinate =deepcopy(coordinate)
      #imageToPredict = cv2.imread(path, 3)
      #y_,x_ = imageToPredict.shape[0],imageToPredict.shape[1]
      x_ , y_ = Image.open(path).size #get true shape without loading image
      x_scale = self.shape[0] / x_
      y_scale = self.shape[1] / y_
      (origLeft, origTop, origRight, origBottom) = coordinate
      x = int(np.round(origLeft * x_scale)); y = int(np.round(origTop * y_scale))   
      xmax = int(np.round(origRight * x_scale)); ymax = int(np.round(origBottom * y_scale))      
      return np.array([x,y,xmax,ymax])
   
   def resize_boxes(self):
    resized_coordinate=[]
    cnt=1
    n_files =len(self.list_dir_files)
    for cord,src_path in zip(self.box_coordinate,self.list_dir_files):
      print( '\rresizing box {}/{}'.format(cnt,n_files),end='')
      resized_coordinate.append(self.resize_one_boxe(src_path,cord))
      cnt +=1
    return resized_coordinate


class XmlListConfig(list):
    def __init__(self, aList):
        for element in aList:
            if element:
                # treat like dict
                if len(element) == 1 or element[0].tag != element[1].tag:
                    self.append(XmlDictConfig(element))
                # treat like list
                elif element[0].tag == element[1].tag:
                    self.append(XmlListConfig(element))
            elif element.text:
                text = element.text.strip()
                if text:
                    self.append(text)


class XmlDictConfig(dict):
    '''
    Example usage:

    >>> tree = ElementTree.parse('your_file.xml')
    >>> root = tree.getroot()
    >>> xmldict = XmlDictConfig(root)

    Or, if you want to use an XML string:

    >>> root = ElementTree.XML(xml_string)
    >>> xmldict = XmlDictConfig(root)

    And then use xmldict for what it is... a dict.
    '''
    def __init__(self, parent_element):
        if parent_element.items():
            self.update(dict(parent_element.items()))
        for element in parent_element:
            if element:
                # treat like dict - we assume that if the first two tags
                # in a series are different, then they are all different.
                if len(element) == 1 or element[0].tag != element[1].tag:
                    aDict = XmlDictConfig(element)
                # treat like list - we assume that if the first two tags
                # in a series are the same, then the rest are the same.
                else:
                    # here, we put the list in dictionary; the key is the
                    # tag name the list elements all share in common, and
                    # the value is the list itself 
                    aDict = {element[0].tag: XmlListConfig(element)}
                # if the tag has attributes, add those to the dict
                if element.items():
                    aDict.update(dict(element.items()))
                self.update({element.tag: aDict})
            # this assumes that if you've got an attribute in a tag,
            # you won't be having any text. This may or may not be a 
            # good idea -- time will tell. It works for the way we are
            # currently doing XML configuration files...
            elif element.items():
                self.update({element.tag: dict(element.items())})
            # finally, if there are no child tags and no attributes, extract
            # the text
            else:
                self.update({element.tag: element.text})

class custom_metrics:

  def calculate_iou( target_boxes, pred_boxes ):
      xA = K.maximum( target_boxes[ ... , 0], pred_boxes[ ... , 0] )
      yA = K.maximum( target_boxes[ ... , 1], pred_boxes[ ... , 1] )
      xB = K.minimum( target_boxes[ ... , 2], pred_boxes[ ... , 2] )
      yB = K.minimum( target_boxes[ ... , 3], pred_boxes[ ... , 3] )
      
      interArea = K.maximum( 0.0, xB - xA ) * K.maximum( 0.0, yB - yA )
      boxA_Area = (target_boxes[ ... , 2] - target_boxes[ ... , 0]) * (target_boxes[ ... , 3] - target_boxes[ ... , 1])
      boxB_Area = (pred_boxes[ ... , 2] - pred_boxes[ ... , 0]) * (pred_boxes[ ... , 3] - pred_boxes[ ... , 1])
      
      iou = interArea / ( boxA_Area + boxB_Area - interArea)
      return iou

  def iou_metric( y_true, y_pred):
      return calculate_iou(y_true, y_pred)


      
