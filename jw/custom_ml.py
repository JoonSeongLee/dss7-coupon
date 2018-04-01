import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import scipy as sp
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import sklearn as sk
import matplotlib as mpl
import matplotlib.pylab as plt
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set()
sns.set_style("whitegrid")
sns.set_color_codes()

# Model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB

# preprocss with etc
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp


def df_to_nparray():
    return 0

def cls_conf(X, y, model, test_ratio=0.5):
    '''
    a, b = cls_conf(X,y,GaussianNB(),0.5)
    a, b = cls_conf(X,y,QuadraticDiscriminantAnalysis(store_covariance=True),0.5)
    a, b = cls_conf(X,y,LinearDiscriminantAnalysis(n_components=3, 
                                 solver="svd", store_covariance=True),0.5)
    a, b = cls_conf(X,y,LogisticRegression(), 0.5)
    a, b = cls_conf(X,y,SVC(gamma=0.0026, C=10), 0.5)
    '''

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio,
                                                        random_state=0)
    model = model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return confusion_matrix(y_test, y_pred), classification_report(y_test, y_pred)


def bin_roc(X,y,model,test_ratio=0.5):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio,
                                                        random_state=0)
    classifier = model.fit(X_train, y_train)
    
    fpr, tpr, threshold = roc_curve(y_test, classifier.decision_function(X_test))
    
    plt.figure(figsize=(10, 10))
    lw = 2
    plt.plot(fpr, tpr, label = 'auc area = {}'.format(auc(fpr,tpr)))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.1, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=20)
    plt.ylabel('True Positive Rate',fontsize=20)
    plt.title('ROC',fontsize=20)
    plt.legend(loc='lower right',fontsize=20)
    plt.show()

def multi_roc(X, y, model, test_ratio=0.5):

    y = label_binarize(y, classes=np.unique(y))
    n_classes = y.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio,
                                                        random_state=0)

    ovr = OneVsRestClassifier(model)
    y_score = ovr.fit(X_train, y_train).predict_proba(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(10, 10))
    lw = 2
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=lw,
                 label='class {} (auc area = {})'.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.1, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=20)
    plt.ylabel('True Positive Rate',fontsize=20)
    plt.title('ROC',fontsize=20)
    plt.legend(loc='lower right',fontsize=20)
    plt.show()
    
