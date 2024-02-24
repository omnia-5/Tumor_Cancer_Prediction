from sklearn import svm
from ModelEvl import ModelEvaluation as evl
import pickle as pk

class SVMModel:
    
#-----------------------SVM-----------------------------

   def __init__(self):
       pass
     
   def trainModelLinear(self, x_train, y_train):
       self.SVMmodel = svm.SVC(kernel = 'linear')
       self.SVMmodel.fit(x_train,y_train)
       #save model
       with open("./trainedModels/SVMModel.pickle", "wb") as file:
           pk.dump(self.SVMmodel,file)
         
           
   def loadSavedModel(self):
       self.SVML = pk.load(open("./trainedModels/SVMModel.pickle", "rb"))
       return self.SVML
   def loadSavedModelExt(self):
     self.SVMLEx = pk.load(open("./trainedModels/SVMModelwithExt.pickle", "rb"))
     return self.SVMLEx

#----------------evaluation of SVM----------------------

   def ModelAccuracy(self, y_test, y_pred):
      evl.modelEvaluation(self.SVML, y_test, y_pred, "SVM")
       

   def ModelAccuracyEx(self, y_test, y_pred):
      evl.modelEvaluation(self.SVMLEx, y_test, y_pred, "SVM")





