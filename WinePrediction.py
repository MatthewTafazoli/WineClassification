import numpy as np 
import pandas as pd 
import os
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

from collections import Counter

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

#sklearn methods for prediction
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier

#libraries for model evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve


# Ignoring Unnecessary warnings
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('winequality-red.csv')

# Simple checks of the data#
# print(df.head(20)) #check first 20 rows of the csv file#
# print(df.shape) #check shape of data#
# print(df.info()) #get further details of data in each column#
# print(sorted(df['quality'].unique())) #Check the range of quality values

# Bar graph of quality value count
plt.figure(1, figsize=(5,5))
df['quality'].value_counts().sort_index().plot.bar(title = "quality values", rot=0)
# plt.show()

# Pie graph representing percent of quality count
plt.figure(1, figsize=(5,5))
df['quality'].value_counts().plot.pie(autopct="%1.1f%%")
# plt.show()

#check amount of each quality value and if any value in the dataframe is null
# print(df.quality.value_counts().sort_index())
# print(df.isnull().sum()) 

## Function to create a histogram, and a boxplot and scatter plot.
def diagnostic_plots(df, variable,target):

    plt.figure(figsize=(15, 5))

    # histogram
    plt.subplot(1, 4, 1)
    sns.histplot(df[variable], bins=30,color = 'red')
    plt.title('Histogram')

    # boxplot
    plt.subplot(1, 4, 3)
    sns.boxplot(y=df[variable],color = 'green')
    plt.title('Boxplot')

    # scatterplot
    plt.subplot(1, 4, 2)
    plt.scatter(df[variable],df[target],color = 'blue')
    plt.title('Scatterplot')
    
    
    # barplot
    plt.subplot(1, 4, 4)
    sns.barplot(x = target, y = variable, data = df)   
    plt.title('Barplot')
    
    
    # plt.show()

# Take each variable in the df and create plots for them using the above
for variable in df:
    diagnostic_plots(df,variable,'quality')

# Use pandas built in feature to compute pairwise correlation of columns, and then use sns to generate heatmap
corr = df.corr() 
plt.figure(figsize=(10, 5))
k = 12 #number of variables for heatmap
cols = corr.nlargest(k, 'quality')['quality'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},  
                 yticklabels=cols.values, xticklabels=cols.values,cmap="Reds")
# plt.show()

#Binary Classification of the wine into either "good" or "bad", with a threshold quality of 5 based on the dataset
bins = (2, 5, 8)
group_names = ['bad', 'good']
df['quality'] = pd.cut(df['quality'], bins = bins, labels = group_names)

#sklearn's transformer for encoding quality values to binary classes
encoder = LabelEncoder()
df['quality'] = encoder.fit_transform(df['quality'])
# print(df['quality'].value_counts()) #check the balance of the classes
# print(df.head()) #check that the quality values have been changed to binary classes

#split into training/test datasets
x_train, x_test, y_train, y_test = train_test_split(df.drop('quality', axis=1), df['quality'], test_size=0.3, random_state=0)
# print(x_train.shape, x_test.shape) #check the size of training/test datasets

# Use sklearn's method for standardizing feature values of the training set and applying transform to train/test set
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#define the various classifiers we will use
lr = LogisticRegression(max_iter=20000,penalty='l2')
lda = LinearDiscriminantAnalysis()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
bg = BaggingClassifier()
gb = GradientBoostingClassifier()
ada = AdaBoostClassifier()
xgb = XGBClassifier()
extr = ExtraTreesClassifier()
clf1 = ExtraTreesClassifier() #This and clf2 are used for the Voting Classifier
clf2 = RandomForestClassifier()
vc = VotingClassifier(estimators=[('ext', clf1),('rf', clf2)], voting='soft')

#map classifiers and names to list
classifiers = [
    ("Logistic Regression" , lr),
    ("Linear Discriminant Analysis" , lda),
    ("Decision Tree" , dt),
    ("Random Forest", rf),
    ("Bagging", bg),
    ("Gradient Boosting", gb),
    ("Ada Boost", ada),
    ("XGB", xgb),
    ("Extra Trees", extr),
    ("Voting Classifier", vc)
]

for classifier, func in classifiers:
    # Fit clf to the training set
    model = func.fit(x_train, y_train)
    print(10*"=", f"{classifier}",10*"=", 
          "\n",'\033[1m' + "train accuracy:" + '\033[0m', model.score(x_train, y_train),"\n", 
          '\033[1m' + "test accuracy:" + '\033[0m', model.score(x_test, y_test))
    model_pred = func.predict(x_test)
    print('\033[1m' + "classification report: " + '\033[0m', "\n", 
          classification_report(model_pred, y_test))
    cm = confusion_matrix(y_test, model_pred)

    df1 = pd.DataFrame(columns=["Bad","Good"], index= ["Bad","Good"], data= cm )

    f,ax = plt.subplots(figsize=(4,4))

    sns.heatmap(df1, annot=True,cmap="Greens", fmt= '.0f',ax=ax,linewidths = 5, cbar = False)
    plt.xlabel("Predicted Label")
    plt.xticks(size = 12)
    plt.yticks(size = 12, rotation = 0)
    plt.ylabel("True Label")
    plt.title(f"{classifier} Confusion Matrix", size = 12)
    
    roc_auc = roc_auc_score(y_test, model_pred)

    fpr, tpr, thresholds = roc_curve(y_test, func.predict_proba(x_test)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{classifier} ROC')
    
    # plt.show()