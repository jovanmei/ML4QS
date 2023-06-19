import numpy as np
import pandas as pd

from Python3Code.util import util
from Python3Code.util.VisualizeDataset import VisualizeDataset
from Python3Code.Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
from Python3Code.Chapter7.FeatureSelection import FeatureSelectionClassification
from Python3Code.Chapter7.LearningAlgorithms import ClassificationAlgorithms
from Python3Code.Chapter7.Evaluation import ClassificationEvaluation


def lstm(train_X, train_y, test_X, hidden_size=64, epochs=30, batch_size=32, print_model_details=False):
    # Reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.values.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.values.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape[2])
    # Create the model
    model = Sequential()
    model.add(LSTM(hidden_size, input_shape=(1, train_X.shape[2])))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=2)

    from sklearn.metrics import classification_report
    # Evaluate the model on the test set
    y_pred = model.predict(test_X)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(test_y, axis=1)

    report = classification_report(y_true_classes, y_pred_classes)
    print(report)

    # # Apply the model
    # pred_prob_train_y = model.predict(train_X)
    # pred_prob_test_y = model.predict(test_X)
    # pred_train_y = model.predict(train_X)
    # pred_test_y = model.predict(test_X)
    #
    # classes = ['labelwalk', 'labelsit', 'labelride', 'labelrun']
    # frame_prob_training_y = pd.DataFrame(pred_prob_train_y, columns=classes)
    # frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=classes)

    if print_model_details:
        model.summary()

    y_pred = model.predict(test_X)

    return y_pred
    # return pred_train_y, pred_test_y, frame_prob_training_y, frame_prob_test_y


DataViz = VisualizeDataset(__file__)

dataset = pd.read_csv("res/ch5_res.csv", index_col=0)
# Convert the index to datetime if it's not already in the correct format
dataset.index = pd.to_datetime(dataset.index)

prepare = PrepareDatasetForLearning()
# Split the dataset into train and test sets
train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(dataset, ['label'], 'like', 0.7,
                                                                               filter=True, temporal=False)

print('Training set length is: ', len(train_X.index))
print('Test set length is: ', len(test_X.index))


# based on python2 features, slightly different.
selected_features = ['gyr_phone_y_temp_std_ws_120', 'pca_5_temp_std_ws_120', 'acc_phone_x_temp_std_ws_120',
                     'mag_phone_y', 'pca_2_temp_std_ws_120', 'acc_phone_y_temp_mean_ws_120',
                     'loc_phone_speed_temp_mean_ws_120',
                     'mag_phone_z_max_freq', 'loc_phone_longitude_temp_mean_ws_120', 'pca_5_temp_mean_ws_120']

# selected_features = ['acc_phone_y_freq_0.0_Hz_ws_40', 'press_phone_pressure_temp_mean_ws_120',
# 'gyr_phone_x_temp_std_ws_120', 'mag_watch_y_pse', 'mag_phone_z_max_freq', 'gyr_watch_y_freq_weighted',
# 'gyr_phone_y_freq_1.0_Hz_ws_40', 'acc_phone_x_freq_1.9_Hz_ws_40', 'mag_watch_z_freq_0.9_Hz_ws_40',
# 'acc_watch_y_freq_0.5_Hz_ws_40']

learner = ClassificationAlgorithms()
eval = ClassificationEvaluation()

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import Adam

labels = train_y[['class']]
labels = pd.get_dummies(labels)
train_y = labels
print(train_y)

y_pred = lstm(train_X[selected_features], train_y, test_X[selected_features], print_model_details=True)

# Evaluate model
# Make predictions
y_pred = y_pred > 0.5

# print(y_pred)
print(test_X.shape)
print(y_pred.shape)


test_y = labels
print(test_y)
test_y = test_y > 0.5
chosen_rows = np.array(test_y)
# Choose the first 410 rows
chosen_rows = chosen_rows[:410, :]
# Print the shape of the chosen rows
print(chosen_rows.shape)

# test_cm = eval.confusion_matrix(test_y, y_pred, labels)

# DataViz.plot_confusion_matrix(test_cm, labels, normalize=False)


import matplotlib.pyplot as plt
import seaborn as sns

data = np.array(y_pred)

# Extract true labels and predicted labels
true_labels = chosen_rows[:, :4]  # Assuming all four columns represent the true labels
predicted_labels = data[:, :4]  # Assuming all four columns represent the predicted labels

# Compute the confusion matrix
confusion_matrix = np.zeros((4, 4), dtype=int)  # Assuming four labels
for true, pred in zip(true_labels, predicted_labels):
    for i in range(4):
        if true[i] and pred[i]:
            confusion_matrix[i, i] += 1  # True positive

# Compute the predicted label distribution
predicted_distribution = np.sum(predicted_labels, axis=0)

# Visualize the predicted label distribution
class_names = ['Ride', 'Run', 'Sit', 'Walk']  # Modify class names accordingly
plt.figure(figsize=(8, 6))
sns.barplot(x=class_names, y=predicted_distribution, color='blue')
plt.xlabel('Predicted Label')
plt.ylabel('Count')
plt.title('Predicted Label Distribution')
plt.show()

# Visualize the confusion matrix with only predicted labels
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Predicted Labels Only)')
plt.show()

# from sklearn.metrics import confusion_matrix, classification_report
#
# print(confusion_matrix(chosen_rows, data))
# print(classification_report(test_y, y_pred))
