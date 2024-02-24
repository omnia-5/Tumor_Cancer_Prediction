from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class DataPre:
    
 def __init__(self, df):
     self.df = df
#---------------------Data Preprocessing--------------------\
 def Preprocessing(self):      
# function to return column adddress F1 To F30
# print(df.isna().sum()) there is no nan values in any column
  def fun(x):
      F = 'F' + str(x)
      return F

# function to replace nan in each column with its mean
  for x in range(1, 31):
        F = fun(x)
        self.df.replace(self.df[F].mean(), np.nan, True)

# print(df.duplicated()) =false ->there is no duplicated rows
# drop duplicate code -> Keep the last occurance of the duplicated row and remove others in the set
  self.df.drop_duplicates(keep="last")

#drow outlierss
  #sns.boxplot(data=self.df)
  
 

# B-> negative      M->positive
# replace diagnosis column with 0->B  1->M
  self.df.loc[self.df["diagnosis"] == "B", "diagnosis"] = 0
  self.df.loc[self.df["diagnosis"] == "M", "diagnosis"] = 1
  
  #-----data Scaling
  impData = self.df.drop(['diagnosis','Index'], axis = 1)
  scaler = StandardScaler()
  scaler.fit(impData)
  self.scaled_data = scaler.transform(impData)
  
 
  return self.df["diagnosis"],self.scaled_data

#----------------------Feature Extraction

 def featureExtraction(self):
      

         #-----Feature Extraction
   pca = PCA(n_components=3)
   pca.fit(self.scaled_data)
   pca_data = pca.transform(self.scaled_data)
   return pca_data