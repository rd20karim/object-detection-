#@title Image Data Generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
def data_generator(TRAINING_DIR,TEST_DIR,TARGET_SIZE,COLOR_MODE,BATCH_SIZE=32,
                     CLASSE_MODE='categorical', SHUFFLE=True,SHUFFLE_TEST = False):
                   
  training_datagen = ImageDataGenerator(rescale=1. / 255)                              
  testing_datagen = ImageDataGenerator(rescale=1. / 255)
  "Takes the path to a directory & generates batches of augmented data."
  train_generator = training_datagen.flow_from_directory(
                                                          TRAINING_DIR,
                                                          target_size=TARGET_SIZE,
                                                          class_mode=CLASSE_MODE,
                                                          color_mode=COLOR_MODE,
                                                          batch_size=BATCH_SIZE,
                                                          shuffle=SHUFFLE
                                                        )
  "Takes the path to a directory & generates batches of augmented data."
  test_generator = testing_datagen.flow_from_directory(
                                                          TEST_DIR,
                                                          target_size=TARGET_SIZE,
                                                          class_mode=CLASSE_MODE,
                                                          color_mode=COLOR_MODE,
                                                          batch_size= BATCH_SIZE,
                                                          shuffle=SHUFFLE_TEST
                                                      )
  return train_generator,test_generator

class folderPreparations:
  
  def adapt_data_to_image_generator(new_path,name_classes=None,xml_objet=None,old_path=None):
    if xml_objet:
      os.mkdir(new_path)
      name_classes=list(set(xml_objet.name_classes))
      for cls in name_classes:
        os.mkdir(os.path.join(new_path,cls))
      for id_cls,filedir in enumerate(xml_objet.list_dir_files):
        if filedir.split('.')[1]=='jpg':
          print(filedir.split('/')[-2])
          new = os.path.join(new_path,xml_objet.name_classes[id_cls],filedir.split('/')[-1])
          print(new)
          shutil.copy(filedir,new)

    elif old_path :
      os.mkdir(new_path)
      for cls in name_classes:
        os.mkdir(os.path.join(new_path,cls))
        for filename in os.listdir(old_path):
          if filename.split('.')[1]=='jpg' and cls in filename:
            shutil.copy(os.path.join(old_path,filename),os.path.join(new_path,cls,filename))
    else:
      print('You sholud give the old path or an xml_object')
