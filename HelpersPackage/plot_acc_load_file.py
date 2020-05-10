# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 17:37:50 2020

@author:R.karim 
"""
import os
import matplotlib.pylab as plt
from tensorflow.keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py
from tensorflow.keras.models import model_from_json
import pandas as pd 
import pickle 
import  matplotlib.image as mpimg
import seaborn as sn
import pandas as pd
from sklearn import metrics

def show_confusion_matrix(loaded_model,data_gen):
  
  print('prediction and plot . . .')
  test_predi  = loaded_model.predict_generator(data_gen)
  test_predi_class = np.argmax(test_predi,1)
  'Matrix confusion'
  conf  =  metrics.confusion_matrix(y_true=data_gen.classes,
                                    y_pred=test_predi_class)
  name_classes = [name for name in tuple(data_gen.classe_indices.keys())]
  df = pd.DataFrame(conf, name_classes , name_classes)
  sn.set(font_scale=1.4)  # for label size
  plt.figure(figsize=(8,6))
  sn.heatmap(df, annot=True, annot_kws={"size": 14}, cmap='gray',fmt='g')  # font size
  plt.xlabel('Predicted label')
  plt.ylabel('True label')
  plt.show()

class acc_loss_plot(object):
  
      """
      # Arguments : 
      historique_logs : can be the history returned by the model in case of plot in function with epochs
                        or the dictionnary of metrics for every batch returned using a callback object
      train : boolean set by default to True
              it should set False if no train metrics is available
      test  :boolean set by default to False
              it should be set to True if test metrics is available
      ens   :list of index of metrix to plot in the same figure,by default it's set to [0,1,2,3]
      same  :boolean set by default to False
             it can be set to True for plot the metrics of a given index in the same figure
      xlabel : the x labels can be epochs or batches 
      """
      def __init__(self,historique_logs,train=True,test=False,same=False,xlabel='epochs',ens=[0,1,2,3]):
        if 'batch' in xlabel:
          self.h = historique_logs #dict 
        else : 
            self.h = historique_logs.history  #dict 
        self.tr = train
        self.ts = test
        self.ky_val = list(self.h.items())
        self.xlab = xlabel
        self.same = same
        self.ens  = ens # index of metrics to plot in the same figure


      def show(self):
              lg=list()
              for idx,met in enumerate(self.ky_val):
                    self.plot_metrics(met,idx,self.same,lg)


      def plot_metrics(self,met,idx,s,lg):      
            if not s:  #new figure for every metrics
              plt.figure(idx+1)
              plt.plot(met[1])
              plt.xlabel(self.xlab)
              plt.ylabel(met[0]) 
              plt.legend([str(met[0])])
              plt.show()
            
            elif idx in self.ens and self.ts and self.tr and self.same: #same figure for loss,acc train and/or test
              plt.figure(1)
              plt.plot(met[1])
              plt.xlabel(self.xlab)
              plt.ylabel(met[0]) 
              lg += [str(met[0])]
              if idx == self.ens[-1]:
                  plt.legend(lg)
                  
            else:
              plt.figure(idx)
              plt.plot(met[1])
              plt.xlabel(self.xlab)
              plt.ylabel(met[0]) 
              plt.legend([str(met[0])])
              plt.show()

"""Compute confusion matrix and normalize."""
              
class intermediat_layer:

 
    def get_intermediate_outputs(name_layer,architecture,data_generator):  
      intermediate_layer_model = tf.compat.v1.keras.Model(inputs=architecture.input,
                                      outputs=architecture.get_layer(name_layer).output)
      return intermediate_layer_model.predict_generator(data_generator)

        
    def show_filtre_images(image_outputs_layer,cmap='gray',start=0,end=1,step=1):
        #image_outputs_layer Output of the get_intermediate_outputs function
      shape = image_outputs_layer.shape
      for im in range(start,end,step):
        for filt in range(shape[-1]):
            print('image filter ',filt)
            plt.imshow(image_outputs_layer[im,:,:,filt],cmap=cmap)
            plt.show()
            
            
class file:
    
        def save_model(architecture,name_file,save_weight = True,dictn=True):

            # serialize model(Architecture) to JSON
            model_json = architecture.to_json()
            with open(name_file+'_model'+".json", "w") as json_file:
                print('Save the architecture of model to disk ... ')
                json_file.write(model_json)
            
            # serialize weights to HDF5                
            if save_weight:
                
              print("Save weights to disk ... ")
              architecture.save_weights(name_file+'_weights'+".h5")
        
        
        def load_model(path_model,path_weight=None):

            # load json and create model
            
            json_file = open(path_model, 'r')
            print("Load architecture model from disk ... ")
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = tf.keras.models.model_from_json(loaded_model_json)
            
            # load weights into new model
  
            if path_weight:
                print("Loaded weights model from disk ...")
                loaded_model.load_weights(path_weight)
            
            return loaded_model
        
        def save_model_history(hist_mod,name_file,num_file ='',csv=False,dictn = False):
            
            '''
            assuming you stored your model.fit results in a 'his_mod' variable: 
            convert the history.history dict to a pandas DataFrame or csv     
            save by default to json file
            save to csv  if csv set to True
            save in the form of dictn if dictn set to True
                
            '''
            if csv:            
                if num_file:
                    hist_csv_file = 'history'+'_'+str(num_file)+'.csv'
                else :
                    hist_csv_file = name_file+'hist'+'.csv'
                
                hist_df = pd.DataFrame(hist_mod.history) 
                
                with open(hist_csv_file, mode='w') as f:
                    hist_df .to_csv(f)                     

            elif dictn:
                
                with open(name_file, 'wb') as file_pi:
                    pickle.dump(hist_mod.history, file_pi)
                                     
            # or save to json:  
            else:
                if num_file:
                    hist_json_file = 'history'+'_'+str(num_file)+'.json'
                else :
                    hist_json_file = name_file + 'hist' + '.json'
                
                hist_df = pd.DataFrame(hist_mod.history) 
                                  
                with open(hist_json_file, mode='w') as f:
                    hist_df.to_json(f)
        
        def load_history(name_file):
            
            if 'csv' in name_file:
                return pd.read_csv(name_file)
            
            elif 'pickle' in name_file:
                return pd.read_pickle(name_file)
            
            else:
                return pd.read_json(name_file)
            


            
                        


            













