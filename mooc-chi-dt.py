# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 19:06:06 2021

@author: LENOVO
"""
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import chi2
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import mean_squared_error
#from xgboost import XGBClassifier
import seaborn as sns




#READ DATASET OF MOOC DATASET
df=pd.read_csv('cs_mitx.csv',encoding='latin-1')
#print(df)

#Data Describe
#print(df.shape)
#print(df.isnull().any())

df=df.dropna()
#print(" \nCount total missing values at each column in a DataFrame : \n\n",df.isnull().sum())

#drop irrelevant column 
df=df.drop(['composite','e_value','v_value','n','c_value','Pa','normalized_Pa','Motivation_Label','Completed_or_Not'], axis = 1) 
#print(df.shape)
#print(df.columns)
#print(df.info())
#print(df.describe())

#------------------------Exploratory Data Analysis--------------------


# =============================================================================
# print(df['final_cc_cname_DI'].value_counts().plot(kind='bar'))
# plt.title("Value counts Country ")
# plt.xlabel("Country")
# plt.xticks(rotation=90)
# plt.ylabel("Count")
# plt.show()
# 
# 
# series = pd.Series(df['gender'])
# lab=['Male','Female']
# value_counts = series.value_counts()
# plt.pie(value_counts, labels=lab, autopct='%1.1f%%', shadow=True)
# 
# # Set aspect ratio to be equal so that pie is drawn as a circle
# plt.axis('auto')
# 
# # Set title
# plt.title('Gender Ratio')
# 
# # Display the chart
# plt.show()
# =============================================================================



# =============================================================================
# value_counts = pd.Series(df['LoE_DI']).value_counts()
# 
# # Plotting the bar chart
# plt.bar(value_counts.index, value_counts.values, color=['red', 'green', 'blue', 'yellow'])
# plt.xlabel('Education')
# plt.ylabel('Counts')
# plt.title('Level of Education of Student')
# plt.xticks(rotation=90)
# plt.show()
# 
# =============================================================================



print(df['age'].describe())
sns.histplot(data=df, x='age', kde=True)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Distribution of Age')
plt.show()







#--------------------DATA Pre-Processing----------------------------------------------------
grade=[]
a1=0
a=0
b1=0
b=0
c1=0
c=0
d=0
d1=0
f=0
for row in df['grade']:
    if row >=0.91 and row<=1.0:
        grade.append('A+')
        a1=a1+1
    if row >=0.81 and row<=0.9:
        grade.append('A')
        a=a+1
    if row >=0.71 and row<=0.8:
        grade.append('B+')
        b1=b1+1
    if row >=0.61 and row <=0.7:
        grade.append('B')
        b=b+1
    if row >= 0.51 and row<=0.6:
        grade.append('C+')
        c1=c1+1
    if row >=0.41 and row<=0.5:
        grade.append('C')
        c=c+1
    if row >=0.31 and row<=0.4:
        grade.append('D+')
        d1=d1+1
    
    if row >=0.21 and row<=0.3:
        grade.append('D')
        d=d+1
    if row >=0.0 and row <=0.2:
        grade.append('F')
        f=f+1

#print(a1,a,b1,b,c1,c,d1,d,f)
df['grade_class']=grade
#print(df['grade_class'])
#print(df['grade_class'].value_counts().plot(kind='bar'))

# =============================================================================
# sns.boxplot(data=df, x='grade_class', y='nevents')
# plt.xlabel('Category')
# plt.ylabel('Age')
# plt.title('Age Distribution by Category')
# plt.show()
# =============================================================================

#--------------------DATA ENCODING-----------------------------
#print(df.final_cc_cname_DI.nunique())
cc={'United Kingdom':1, 'Brazil':2, 'United States':3, 'India':4, 'Russian Federation':5,
 'Pakistan':6, 'Other Middle East/Central Asia':7, 'Other South Asia':8,
 'Other Europe':9, 'Greece':10, 'Other South America':11, 'Portugal':12, 'Canada':13,
 'Indonesia':14, 'Other Africa':15, 'Ukraine':16, 'Other East Asia':17, 'Egypt':18, 'Spain':19,
 'Nigeria':20, 'Morocco':21, 'France':22, 'Australia':23, 'Mexico':24, 'Philippines':25, 'Japan':26,
 'Germany':27, 'Poland':28, 'China':29, 'Other North & Central Amer., Caribbean':30,
 'Unknown/Other':31, 'Colombia':32, 'Bangladesh':33, 'Other Oceania':34}
df['final_cc_cname_DI']=df['final_cc_cname_DI'].map(cc)
#print(df['final_cc_cname_DI'])

#print(df.LoE_DI.unique())
loe={"Bachelor's":2, 'Secondary':1, "Master's":3, 'Doctorate':4, 'Less than Secondary':0}
df['LoE_DI']=df['LoE_DI'].map(loe)
#print(df['LoE_DI'])

#print("GENDER VALUE",df.gender.unique())
gen={'f':1,'m':0}
df['gender']=df['gender'].map(gen)
#print(df['gender'])


uni={'MITx':1}
df['institute']=df['institute'].map(uni)

course={'6.00x':1}
df['course_id']=df['course_id'].map(course)

gr={'A+':1,'A':2,'B+':3,'B':4,'C+':5,'C':6,'D+':7,'D':8,'F':9}
df['grade_class']=df['grade_class'].map(gr)

#NO OF DAYS USER HAVE INTERACTED WITH COURSE GET BY STARTTING DATE AND ENDING DATE

df['last_event_DI']=pd.to_datetime(df['last_event_DI'],format='%d-%m-%Y')
df['start_time_DI']=pd.to_datetime(df['start_time_DI'],format='%d-%m-%Y')
#df['start_time_DI'] = df['start_time_DI'].dt.strftime('%m/%d/%Y')
#df['last_event_DI'] = df['last_event_DI'].dt.strftime('%m/%d/%Y')
df['days']=  df['last_event_DI'] - df['start_time_DI'] 

#print(df['days']) 
df['days_act']=df['days'].dt.days
#print(df['days_act'])

"""print(df['last_event_DI'])
print(df['start_time_DI'])
"""

df=df.drop(['userid_DI','year','semester','last_event_DI','start_time_DI','row_id','days'], axis = 1) 


#print(df.columns)
"""print((df['days_act']<0).values.any())

c=0
for x in df['days_act']:
    if(x<0): 
        c=c+1
        print(x)"""
#print(df.shape)

# Get names of indexes for which column have negative value
indexNames = df[ df['days_act'] < 0 ].index
# Delete these row indexes from dataFrame
df.drop(indexNames , inplace=True)

#print(df.shape)
#print((df['days_act']<0).values.any())
#print(df.dtypes)
df['grade'] = df['grade'].astype(int) 
#print(df.dtypes)

student_features = df.columns.tolist()
student_features.remove('grade') 
student_features.remove('grade_class') 

#print(student_features)


# correlation_matrix = df.corr()

# # Print the correlation between the target variable and input variables
# target_correlation = correlation_matrix['grade_class']
# print(target_correlation)
# sns.heatmap(correlation_matrix, cmap='coolwarm')
# plt.title("Correlation Matrix")
# plt.show()

# =============================================================================
# #days active
# value_counts = pd.Series(df['days_act']).value_counts() 
# # # Plotting the bar chart
# plt.bar(value_counts.index, value_counts.values, color=['red', 'green', 'blue', 'yellow'])
# plt.xlabel('Days active')
# plt.ylabel('Count')
# plt.title('Active days of student')
# plt.xticks(rotation=90)
# plt.show()
# =============================================================================

# =============================================================================
# sns.barplot(x=df['grade_class'], y=df['days_act'])
# plt.title('How Active days of student related with grades')
# plt.show()
# =============================================================================

grouped = df.groupby('grade_class')['nevents'].sum()
print(grouped)

#----------------------------SAMPLING FOR IMBALANCED CLASS DATASET--------------------------#
target = df['grade_class']
counter = Counter(target)
for k,v in counter.items():
	per = v / len(target) * 100
	#print('Class=%d, Count=%d, Percentage=%.3f%%' % (k, v, per))
X = df[student_features].copy()
y=df['grade_class'] 
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
# summarize distribution
counter = Counter(y)
for k,v in counter.items():
	per = v / len(y) * 100
	#print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
    
#print(student_features)
#-------------------------------CHI SQUARE------------------------------------#
#X = df[student_features].copy()
#y=df['grade_class']
#print(X)
#print(y.shape)

chi_scores = chi2(X,y)
#print(chi_scores)


p_values = pd.Series(chi_scores[1],index = X.columns)
p_values.sort_values(ascending = True , inplace = True)
#p_values.plot.bar()

#---------------------------------REMOVED FEATURES BASED ON CHI SQUARE--------------------#
student_features.remove('course_id') 
student_features.remove('institute') 
#student_features.remove('YOB') 
#student_features.remove('gender')
student_features.remove('age')




#-------------------------------------------PREPARE DATA FOR TRAINING
"""X = df[student_features].copy()
y=df['grade_class']  

sns.heatmap(X, annot=True, cmap="YlGnBu")

# Display the heatmap
plt.show()"""

#------------------------------------------SPLIT DATASET 80%train 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
print("Training split input- ", X_train.shape)
print("Testing split input- ", X_test.shape)
print("Training split OUTPUT- ", y_train.shape)
print("Testing split OUTPUT- ", y_test.shape)

#-----------------------------------TRAIN MODEL
#------------------------------------------------MLP ----------------------------#
"""clf = MLPClassifier(hidden_layer_sizes=(256,128,64,32),activation="relu",random_state=1).fit(X_train, y_train)
#all_accuracies = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10)
pred=clf.predict(X_test)"""

#-----------DECISION TREE-----------------#
"""dtree=DecisionTreeClassifier()
all_accuracies = cross_val_score(estimator=dtree, X=X_train, y=y_train, cv=10)
dtree.fit(X_train,y_train)
pred=dtree.predict(X_test)"""

#-----------------------SVM---------------------#
"""classifier = SVC(kernel = 'linear')
#classifier=OneVsRestClassifier(SVC(kernel='linear', probability=True, class_weight='balanced'))
all_accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)"""

#--------------------------------KNN--------------------------#
"""classifier = KNeighborsClassifier(n_neighbors=5)
all_accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)"""

#-------------------------------------XGBOOST----------------------------
"""model = XGBClassifier()
#cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
all_accuracies = cross_val_score(model, X, y, scoring='accuracy', cv=10, n_jobs=-1, error_score='raise')
model.fit(X_train, y_train)
pred = model.predict(X_test)"""

#-----------------------------------EVALUATION--------------------------------------#

"""print("CONFUSION MATRIX---\n",confusion_matrix(y_test,pred))
print("CLASSIFICATION REPORT---\n",classification_report(y_test,pred))
print("ACCURACY--\n",accuracy_score(y_test, pred))

print(all_accuracies.mean())

MSE = np.square(np.subtract(y_test,pred)).mean() 
print("RMSE-->",MSE)"""


#-----------------------------------------GNB----------------------------------------#
"""classifier = GaussianNB()
all_accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)"""

#-------------------------------------------Gradient Boosting--------------------------#
"""model_GB = GradientBoostingClassifier(n_estimators=500)
model_GB.fit(X_train , y_train)
y_pred = model_GB.predict(X_test)

print("CONFUSION MATRIX---\n",confusion_matrix(y_test,y_pred))
print("CLASSIFICATION REPORT---\n",classification_report(y_test,y_pred))
print("ACCURACY--\n",accuracy_score(y_test, y_pred))

#MSE = np.square(np.subtract(y_test,y_pred)).mean() 
MSE = mean_squared_error(y_test, y_pred)

# Calculate the root mean square error (RMSE)
rmse = np.sqrt(MSE)

print("RMSE-->",MSE)"""

#----------------------------------------------ADA BOOST-----------------------------------"
"""model_ad = AdaBoostClassifier()
model_ad.fit(X_train , y_train)
pred = model_ad.predict(X_test)"""




