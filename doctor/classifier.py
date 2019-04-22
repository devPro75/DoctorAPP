
# coding: utf-8

# ----------
# **Breast Cancer Analysis and Prediction**
# =====================================

import numpy as np 
import pandas as pd 
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# For Plotting and Visualizations 
#import seaborn as sns
#import matplotlib.pyplot as plt
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (16, 6)}
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# plt.rcParams.update(params)
# plt.style.use('fivethirtyeight')
# plt.rcParams['font.size'] = 12


#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import accuracy_score
#import plotly.figure_factory as ff
#import scikitplot as skplt

import pickle
from imblearn.over_sampling import SMOTE

from sklearn.naive_bayes import BernoulliNB

import warnings
warnings.filterwarnings('ignore')
import os

#import plotly.offline as py
# py.init_notebook_mode(connected=True)
#import plotly.graph_objs as go
#import plotly.tools as tls
#import plotly.figure_factory as ff

# Any results you write to the current directory are saved as output.


# #### Define helper function

# In[2]:


# helper Functions
import pandas_profiling as pp
def fit_model(model, model_name, xtrain, xtest, ytrain, ytest):
    print('====================================')
    print('\033[1m' + 'Building Model '  +'\033[0m')
    print('====================================')

    model = model
    trained_model = model.fit(xtrain, ytrain)

    pred_test = trained_model.predict(xtest)
    pred_train = trained_model.predict(xtrain)

    # Accuracy is the fraction of predictions correctly predicted by the classifier
    # Calculate train and test accuracy
    train_acc = accuracy_score(ytrain, pred_train)
    test_acc  = accuracy_score(ytest, pred_test)
    print ('\033[1m' + "\nTrain Accuracy :: "  +'\033[0m' + str(np.round(train_acc*100, 2)) + ' %')
    print ('\033[1m' +"\nTest Accuracy :: "  +'\033[0m'  + str(np.round(test_acc*100, 2)) + ' %')

    print ('\033[1m' + "\n Train : Classification Report: \n" +'\033[0m')
    print(classification_report(ytrain, model.predict(xtrain)))


    print ('\033[1m' + "\n Test : Classification Report: \n" +'\033[0m')
    print(classification_report(ytest, model.predict(xtest)))

    print ('\033[1m' + "\n Train : Confusion matrix: \n" +'\033[0m')
    #skplt.metrics.plot_confusion_matrix(ytrain, pred_train, title="Confusion Matrix for Training set",text_fontsize='large')
    # plt.show()
    
    print ('\033[1m' + "\n Test : Confusion matrix: \n" +'\033[0m')
    #skplt.metrics.plot_confusion_matrix(ytest, pred_test, title="Confusion Matrix for Testing set",text_fontsize='large')
    # plt.show()
    
def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    '''
    print the accuracy score, classification report and confusion matrix of classifier
    '''
    if train:
        '''
        training performance
        '''
        print("\nTrain Result:\n")
        print('-------------')
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_train, clf.predict(X_train))))
        print("Classification Report: \n {}\n".format(classification_report(y_train, clf.predict(X_train))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_train, clf.predict(X_train))))

        res = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
        print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
        print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))
        
    elif train==False:
        '''
        test performance
        '''
        print("\nTest Result:\n")        
        print('------------')
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, clf.predict(X_test))))
        print("Classification Report: \n {}\n".format(classification_report(y_test, clf.predict(X_test))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, clf.predict(X_test))))    


# ### Load the data

# In[3]:


data = pd.read_csv('wpbc.csv')


# Next, we check the data by looking for the features, its shape . 

# In[4]:


data.head()


# In[5]:


data.shape


# Next, we check the distribution and statistics of data. As, can be observed many features have large varying scales.

# In[6]:


data.describe()


# Surprisingly, this dataset doesn't have missing data. Cool! 

# In[7]:


data.isnull().any()


# ### Data Profiling

# In[8]:


# pp.ProfileReport(data)


# #### Diagnosis of Target Variable

# In[9]:


# plt.style.use('fivethirtyeight')
M, B = data['Outcome'].value_counts()
# sns.countplot(x='Outcome',data=data, palette="husl")
# plt.show()


# In[10]:


data['Lymph_node_status'] = data['Lymph_node_status'].replace('?',0)

M = data.loc[data['Outcome']=='N',:]
M.head()


# In[11]:


B = data.loc[data['Outcome'] == 'R', :]
B.head()


# In[12]:


M = M.drop(['Outcome'], axis=1)
B = B.drop(['Outcome'], axis=1)


# ### Plots
# Next we make some Kernel Density Estimation (KDE) plots to check the distribution of malignant and benign cases for various features.

# In[13]:


# plt.subplots(figsize=(15,45))
# sns.set_style('darkgrid')
# # plt.subplots_adjust (hspace=0.4, wspace=0.2)
# i=0
# for col in M.columns:
#     i+=1
#     plt.subplot(7,5,i)
#     # first (0th) column of M is diagnosis, non-numerical
#     sns.distplot(M[col],color='m',label='N',hist=False, rug=True)
#     sns.distplot(B[col],color='b',label='R',hist=False, rug=True)
    # plt.legend(loc='right-upper')
    # plt.title(col)     


# From kde plots, it appears that radius_mean, perimeter_mean, area_mean, concavity_mean, concave_points_mean, compactness_mean, radius_worst, texture_worst, perimeter_worst, area_worst, concavity_worst, concave points_worst, compactness_worst show more difference between malignant and benign populations compared to others. Another way to check these differences can be by using boxplots.

# In[14]:


def plot_distribution(data_select) :  
    tmp1 = M[data_select]
    tmp2 = B[data_select]
    hist_data = [tmp1, tmp2]
    
    group_labels = ['N', 'R']
    colors = ['#FFD700', '#7EC0EE']

    fig = ff.create_distplot(hist_data, group_labels, colors = colors, show_hist = True, curve_type='kde')
    
    fig['layout'].update(title = data_select)

    # py.iplot(fig, filename = 'Density plot')


# In[15]:


for col in M.columns[2:33]:
    M[col] = M[col].astype('float64')
    # plot_distribution(col)


# ### Conclusion
# Using feature selection and voting classifier we could achieve a high accuracy of 98.24%.  In addition, we achieved high Precision and Recall of 99.29% and 95.89%, respectively.
# #### Thank you for reading. As always, all your comments and suggestions are very welcome!

# Next, we explore if and how these features are corrleted to one another. This is done using heatmap (as shown below). As here we have 30 features to look and compare.

# In[16]:


# sns.set(style="white")
# fig,ax=plt.subplots(figsize=(16,16))
# corr=data.corr()
# mask = np.zeros_like(corr)
# mask[np.triu_indices_from(mask)] = True
# cmap = sns.diverging_palette(220, 10, as_cmap=True)
# sns.heatmap(corr,vmin=-1,vmax=1,fmt = ".1f",annot=True,cmap="coolwarm", mask=mask, square=True)
# plt.show()


# As expected some of the features show complete correlation such as radius_mean, area_mean, perimeter_mean, radius_worst, area_worst and perimeter_worst. Also, texture_mean and texture_worst are correlated. concave points_mean and concavity_mean are very strongly correlated. radius_se, perimeter_se and Area_Se are strongly correlated. compactness_mean is correlated to concavity_mean, compactness_worst.

# In order to avoid high variance that can appear due to many correlated features, we drop some of the very highly correlated features. For example- we have dropped here  area_mean, perimeter_mean, radius_worst, area_worst and perimeter_worst while we kept radius_mean. The drop in shape is also checked here.  Also, we code Malignant and Benign cases with numbers- 1 and 0 respectively.

# In[17]:


data['Outcome']=data['Outcome'].map({'N':1,'R':0})
# All data being used here
#data = data.drop(['Mean_Area', 'Mean_Perimeter', 'Worst_Radius', 'Worst_Area', 'Worst_Perimeter','Worst_Texture','Mean_Concavity','Perimeter_SE'],axis=1)
print("Shape: "+str(data.shape))
print(data.describe())


# # Building model
# Before starting with bulding our model, we make test and train sets.

# In[43]:


import pickle
from sklearn.decomposition import PCA

X = data.drop(['Outcome'],axis=1).values
print("X shape:"+str(X.shape))
y = data['Outcome'].values
# Split into trai and test
X_train,X_test,y_train, y_test=train_test_split(X,y,test_size=0.1, random_state=8)

print("Before applying PCA X Shape : ", X_train.shape)
pca = PCA(n_components=25)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# In[44]:


# Save and load PCA
pickle.dump(pca, open('ml_models/pca', 'wb'))
pca = pickle.load(open('ml_models/pca', 'rb'))
# loaded pca model but where we are using it ?

# its not have to be used here, thhis us backjend training 
# yes i get it its saving updated pca in pickle format and loading 

# ## Performing Oversampling

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

# Used PCA and Oversampling

# In[47]:
from xgboost import XGBClassifier

model = XGBClassifier()
#model = BernoulliNB()
model.fit(X_train_res, y_train_res)
# testing on test data again.
probas = model.predict_proba(X_test)
print("saving model ")
pickle.dump(model, open('ml_models/model.pkl', 'wb'))
# skplt.metrics.plot_precision_recall_curve(y_test, probas,title = 'Precision Recall Curve', figsize = (7,5))
# plt.show()
# skplt.metrics.plot_roc_curve(y_test, probas,title = 'ROC Curve', figsize = (7,5))
# plt.show()


# #### Save the model to disk

# In[48]:


from sklearn.externals import joblib

# Save to file in the current working directory
joblib_file = "ml_models/model.pkl"  
joblib.dump(model, joblib_file)

# Load from file
model = joblib.load(joblib_file)

# Calculate the accuracy and predictions
score = model.score(X_test, y_test)  
print("Test score: {0:.2f} %".format(100 * score))  
Ypredict = model.predict(X_test)
print("Prediction {}".format(Ypredict))


# ---