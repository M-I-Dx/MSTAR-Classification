from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image
from imutils import paths
import numpy as np
import os


def extract_color_stats(image_):
    # split the input image into its respective RGB color channels
    # and then create a feature vector with 6 values: the mean and
    # standard deviation for each of the 3 channels, respectively
    (R, G, B) = image_.split()
    features_ = [np.mean(R), np.mean(G), np.mean(B), np.std(R),
                 np.std(G), np.std(B)]

    # return our set of features
    return features_


models = {
    "knn": KNeighborsClassifier(n_neighbors=1),
    "naive_bayes": GaussianNB(),
    "logit": LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=3000,class_weight="balanced"),
    "svm": SVC(kernel="linear", class_weight="balanced"),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "mlp": MLPClassifier()
}

imagePaths = paths.list_images("PreprocessedData")
data = []
labels = []

# loop over our input images
for imagePath in imagePaths:
    # load the input image from disk, compute color channel
    # statistics, and then update our data list
    image = Image.open(imagePath)
    features = extract_color_stats(image)
    data.append(features)

    # extract the class label from the file path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# encode the labels, converting them from strings to integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# perform a training and testing split, using 75% of the data for
# training and 25% for evaluation
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.25)


print("---------------------------------------------k-Nearest Neighbor------------------------------------------------")
model = models["knn"]
model.fit(trainX, trainY)

predictions = model.predict(testX)
print(classification_report(testY, predictions,
                            target_names=le.classes_))

print("-------------------------------------------------Naïve Bayes---------------------------------------------------")
model = models["naive_bayes"]
model.fit(trainX, trainY)

predictions = model.predict(testX)
print(classification_report(testY, predictions,
                            target_names=le.classes_))

print("---------------------------------------------Logistic Regression-----------------------------------------------")
model = models["logit"]
model.fit(trainX, trainY)

predictions = model.predict(testX)
print(classification_report(testY, predictions,
                            target_names=le.classes_))

print("---------------------------------------------Support Vector Machines-------------------------------------------")
model = models["svm"]
model.fit(trainX, trainY)

predictions = model.predict(testX)
print(classification_report(testY, predictions,
                            target_names=le.classes_))

print("--------------------------------------------------Decision Trees-----------------------------------------------")
model = models["decision_tree"]
model.fit(trainX, trainY)

predictions = model.predict(testX)
print(classification_report(testY, predictions,
                            target_names=le.classes_))

print("--------------------------------------------------Random Forests-----------------------------------------------")
model = models["random_forest"]
model.fit(trainX, trainY)

predictions = model.predict(testX)
print(classification_report(testY, predictions,
                            target_names=le.classes_))

print("----------------------------------------------multi-layer Perceptrons------------------------------------------")
model = models["mlp"]
model.fit(trainX, trainY)

predictions = model.predict(testX)
print(classification_report(testY, predictions,
                            target_names=le.classes_))
