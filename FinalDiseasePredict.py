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
DATA_PATH = "/Users/taylorsmith/Coding/Intermediate programing grade 10/Data science project/Training.csv"
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

#print(f"Train: {X_train.shape}, {y_train.shape}")
#print(f"Test: {X_test.shape}, {y_test.shape}")

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

#print(f"Accuracy on train data by SVM Classifier\
    #: {accuracy_score(y_train, svm_model.predict(X_train))*100}")

#print(f"Accuracy on test data by SVM Classifier\
    #: {accuracy_score(y_test, preds)*100}")
cf_matrix = confusion_matrix(y_test, preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for SVM Classifier on Test Data")
#plt.show()

#Training and testing Naive Bayes Classifier
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
preds = nb_model.predict(X_test)
#print(f"Accuracy on train data Naive Bayes Classifier\
    #: {accuracy_score(y_train, nb_model.predict(X_train))*100}")

#print(f"Accuracy on test data by Naive Bayes Classifier\
    #: {accuracy_score(y_test, preds)*100}")
cf_matrix = confusion_matrix(y_test, preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot = True)
plt.title("Confusion Matrix for Naive Bayes Classifier on Test Data")
#plt.show()

#Training and testing random forest classifier
rf_model = RandomForestClassifier(random_state=18)
rf_model.fit(X_train, y_train)
preds = rf_model.predict(X_test)
#print(f"Accuracy on train data on Random Forest Classifier\
    #: {accuracy_score(y_train, rf_model.predict(X_train))*100}")

cf_matrix = confusion_matrix(y_test, preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot = True)
plt.title("Confusion Matrix for Random Forest Classifier on Test Data")
#plt.show()

#Training the models on a whole data set
final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)
final_svm_model.fit(X, y)
final_nb_model.fit(X, y)
final_rf_model.fit(X, y)

#Reading the test data
test_data = pd.read_csv("/Users/taylorsmith/Coding/Intermediate programing grade 10/Data science project/Testing.csv").dropna(axis=1)
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
#print(f"Accuracy on Test dataset by the combined model\
       #: {accuracy}")
cf_matrix = confusion_matrix(actual_answers, final_preds)

plt.figure(figsize=(12,8))

sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Combined Model on Test Dataset")
#plt.show()

#Producing cross validation score for the models
for model_name in models:
    model = models[model_name]
    scores = cross_val_score(model, X, y, cv = 10, 
                             n_jobs = -1, 
                             scoring = cv_scoring)
    #print("=="*30)
    #print(model_name)
    #print(f"Scores: {scores}")
    #print(f"Mean Score: {np.mean(scores)}")


#plt.figure(figsize = (18,8))
#sns.barplot(x = "Disease", y = "Counts", data = temp_df)
#plt.xticks(rotation=90)
#plt.show()


symptoms = X.columns.values

#creating a symptom index dictionary to encode the imput symptoms into numerical form
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index

data_dict = {
    "symptom_index":symptom_index,
    "predictions_classes": encoder.classes_
}

#defining the function
#Input: string containing symptoms seperated by commas
#Output: generated predictions by models
def predictDisease(symptoms):
    #creating input data 
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1

    #reshape imput data into format for model predictions
    input_data = np.array(input_data).reshape(1,-1)

    #generating individual outputs based on model

    bug_causer = final_rf_model.predict(input_data)[0]
    rf_prediction = bug_causer
    nb_prediction = final_nb_model.predict(input_data)[0]
    svm_prediction = final_svm_model.predict(input_data)[0]


    #making final prediction based off model's predictions f
    import statistics
    final_prediction = statistics.mode([rf_prediction, nb_prediction, svm_prediction])
    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction":final_prediction
    }
    return predictions



#Testing the function

def main():
    playing = True
    while playing:
        symptom_index = {}
        for index, value in enumerate(symptoms):
            symptom = " ".join([i.capitalize() for i in value.split("_")])
            symptom_index[symptom] = index
        print("\nHere are the possible symptoms that can be inputed in this program:\n")
        print(", ".join(symptom_index.keys()))

        choice = input("\nHow many symptoms would you like to input: two, three, or four?  ")

        symptom_count = {
            "two": 2,
            "three": 3,
            "four": 4
        }

        if choice in symptom_count:
            symptomsimput = [input(f"What is your symptom #{i + 1}?").strip() for i in range(symptom_count[choice])]
            my_predictions = predictDisease(symptomsimput)
            print()
            print("You have:")
            print(my_predictions['final_prediction'])
            print("\nResults from each model\n")
            print(my_predictions)
        else:
            print("Invalid choice: enter two, three, or four")

        other = input ("\nWould you like to enter another symptom?")
        
        if other == "no":
            playing = False
main()

