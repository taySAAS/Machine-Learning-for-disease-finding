import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter
# pd.ArrowDtype 
# np.dtype 

#%matplotlib inline

#Reading the train.csv by removing the last colum since it is empty
DATA_PATH = "C:/Users/SAAS_User/Documents/Code/Machine-Learning-for-disease-finding/Training.csv"
data = pd.read_csv(DATA_PATH).dropna(axis = 1)

#check if the data set is balanced
disease_counts = data["prognosis"].value_counts()
temp_df = pd.DataFrame({
    "Disease": disease_counts.index,
    "Counts": disease_counts.values
})

#target value into numerical value "Prognosis"
encoder = LabelEncoder()
data["prognosis"]
encoder.fit_transform(data["prognosis"])

#Splitting data
X = data.iloc[:,:-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test =train_test_split(
  X, y, test_size = 0.2, random_state = 24)

print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Test: {X_test.shape}, {y_test.shape}")

#defining a scoring matric for K-fold cross validation
def cv_scoring(estimator, X, y):
    return accuracy_score(y,
estimator.predict(X))

# initializing models
models = {
    "SVC": SVC(),
    "Gaussian NB": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=18)
}

#training and testing SVM classifier
svm_model = SVC()
svm_model.fit(X_train, y_train)
preds = svm_model.predict(X_test)

print(f"Accuracy on train data by SVM Classifier\
    : {accuracy_score(y_train, svm_model.predict(X_train))*100}")

print(f"Accuracy on test data by SVM Classifier\
    : {accuracy_score(y_test, preds)*100}")

cf_matrix = confusion_matrix(y_test, preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for SVM Classifier on Test Data")
plt.show()

#Training and testing Naive Bayes Classifier
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
preds = nb_model.predict(X_test)
print(f"Accuracy on train data Naive Bayes Classifier\
    : {accuracy_score(y_train, nb_model.predict(X_train))*100}")

print(f"Accuracy on test data by Naive Bayes Classifier\
    : {accuracy_score(y_test, preds)*100}")
cf_matrix = confusion_matrix(y_test, preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot = True)
plt.title("Confusion Matrix for Naive Bayes Classifier on Test Data")
plt.show()

#Training and testing random forest classifier
rf_model = RandomForestClassifier(random_state=18)
rf_model.fit(X_train, y_train)
preds = rf_model.predict(X_test)
print(f"Accuracy on train data on Random Forest Classifier\
    : {accuracy_score(y_train, rf_model.predict(X_train))*100}")

cf_matrix = confusion_matrix(y_test, preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot = True)
plt.title("Confusion Matrix for Random Forest Classifier on Test Data")
plt.show()

#Training the models on a whole data set
final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)
final_svm_model.fit(X, y)
final_nb_model.fit(X, y)
final_rf_model.fit(X, y)

#Reading the test data
test_data = pd.read_csv("/Users/SAAS_User/Documents/Code/Machine-Learning-for-disease-finding/Testing.csv").dropna(axis=1)

test_X = test_data.iloc[:, :-1]
test_Y = encoder.transform(test_data.iloc[:, -1])

#Making prediction by the mode of predictions made by all classifiers
svm_preds = final_svm_model.predict(test_X)
nb_preds = final_nb_model.predict(test_X)
rf_preds = final_rf_model.predict(test_X)

from scipy import stats

# data = zip(svm_preds, nb_preds, rf_preds)

# df = pd.DataFrame(data, columns=['values'])

# [print([i, j, k]) for i,k,j in ]


final_preds = [Counter([i,j,k]).most_common(1)[0][0] for i,k,j in zip(svm_preds, nb_preds, rf_preds)]
final_preds = np.array(final_preds)

actual_answers = test_data.iloc[:, -1]

actual_answers = [str(elem) for elem in actual_answers.tolist()]
final_preds = [str(elem) for elem in final_preds.tolist()]

accuracy = accuracy_score(actual_answers, final_preds)*100
print(f"Accuracy on Test dataset by the combined model\
       : {accuracy}")
cf_matrix = confusion_matrix(actual_answers, final_preds)
plt.figure(figsize=(12,8))

sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Combined Model on Test Dataset")
plt.show()

#Producing cross validation score for the models
for model_name in models:
    model = models[model_name]
    scores = cross_val_score(model, X, y, cv = 10, 
                             n_jobs = -1, 
                             scoring = cv_scoring)
    print("=="*30)
    print(model_name)
    print(f"Scores: {scores}")
    print(f"Mean Score: {np.mean(scores)}")


#plt.figure(figsize = (18,8))
#sns.barplot(x = "Disease", y = "Counts", data = temp_df)
#plt.xticks(rotation=90)
#plt.show()
