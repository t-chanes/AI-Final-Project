import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers, datasets, models
from tensorflow.keras import Sequential
import os
import cv2
from PIL import Image
import json
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout, Dense, Flatten, BatchNormalization, LeakyReLU, GlobalAveragePooling2D, Conv2D, MaxPool2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.xception import Xception
from tqdm import tqdm
import gc

# Filter out annoying TensorFlow Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

class animalTrainer:
    def __init__(self, model_nm, img_dims=(224, 224), epochs=5):
        self.model_nm = model_nm
        self.img_dims = img_dims
        self.root_dir = "./animals/animals"
        self.experiment_dir = "./Testing_Images"
        self.labels = []
        self.processed_animal_pics = None
        self.train_images = None
        self.test_images = None
        self.train_labels = None
        self.test_labels = None
        self.num_epochs = epochs
        self.class_names = None


    def processAnimalPics(self):
        images = []
        print("\n\n\n--- Preprocessing Animal Pictures ---\n\n")
        animal_dirs = os.listdir(self.root_dir)

        for animal_dir in tqdm(animal_dirs):

            animal_imgs = os.listdir(os.path.join(self.root_dir, animal_dir))

            for i in range(len(animal_imgs)):

                img = cv2.imread(
                    os.path.join(self.root_dir, animal_dir, animal_imgs[i])
                )
                resized_img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
                resized_img = resized_img / 255.0
                images.append(resized_img)
                self.labels.append(animal_dir)

        images = np.array(images, dtype="float32")
        self.processed_animal_pics = images
        print("Done!")


    def processExperimentPics(self):
        experiment_images = []

        if len(os.listdir(self.experiment_dir)) < 1:
            print("\n****NO IMAGES IN EXPERIMENTAL DIRECTORY****\n")
            exit()

        else:

            dir = os.listdir(self.experiment_dir)

            print("\n\n\n--- Preprocessing Experimental Images ---\n\n\n")
            for img in tqdm(dir):

                img = Image.open(os.path.join(self.experiment_dir, img)).convert('RGB').resize((224,224))
                resized_img = np.array(img) / 255.0
                experiment_images.append(resized_img)

            experiment_images = np.array(experiment_images, dtype="float32")
            return experiment_images



    def createLabels(self):
        print("\n\n\n--- Creating Labels ---\n\n")
        le = preprocessing.LabelEncoder()
        le.fit(self.labels)
        class_names = le.classes_
        self.class_names = class_names
        self.labels = le.transform(self.labels)
        self.labels = np.array(self.labels, dtype="uint8")
        self.labels = np.resize(self.labels, (len(self.labels), 1))
        print("Done!")


    def gen_datasets(self):
        print("\n\n\n--- Generating Datasets ---\n\n")
        (
            self.train_images,
            self.test_images,
            self.train_labels,
            self.test_labels) = train_test_split(
            self.processed_animal_pics,
            self.labels,
            test_size=0.33,
            stratify=self.labels)
        print("Done!")


    def train_model(self):
        print("\n\n\n--- Training Model ---\n\n")
        model = Sequential()

        base_model = Xception( include_top=False, weights="imagenet", input_shape=(224, 224, 3) )
        print(f"Number of layers: {len(base_model.layers)}")

        for layer in base_model.layers[:]:
            layer.trainable = False

        for layer in base_model.layers[90:]:
            layer.trainable = True

        model.add(base_model)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(units=90, activation="softmax"))
        model.summary()

        early_stopping = EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            min_delta=1e-5,
            patience=20,
            restore_best_weights=True,
            verbose=0)

        model.compile(
            optimizer=tensorflow.keras.optimizers.RMSprop(learning_rate=0.0001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        fit_model = model.fit(
            self.train_images,
            self.train_labels,
            batch_size=32,
            epochs=self.num_epochs,
            callbacks=[early_stopping],
            validation_split=0.2)

        model.save(self.model_nm)
        print("Done!")

        print("\n\n\n--- Exporting History ---\n\n")

        with open("history.txt", "w") as file:
            json.dump(fit_model.history, file)
        
        print("Done!")



    def predict(self):
        print("\n\n\n--- Attempting Prediction ---\n\n")
        experiment_images = self.processExperimentPics()

        model = models.load_model(self.model_nm)
        result = model.predict(experiment_images)
        result = np.argmax(result, axis=1)

        with open("animal_names.txt") as f:
            animal_names = [line.strip() for line in f]

        le = preprocessing.LabelEncoder()
        le.fit(animal_names)
        plt.imshow(experiment_images[0])
        plt.title(f"Prediction: {str(le.inverse_transform([result]))}")
        plt.show()


    def quick_predict(self):
        print("\n\n\n--- Attempting Prediction ---\n\n")

        model = models.load_model(self.model_nm)
        result = model.predict(self.test_images)
        result = np.argmax(result, axis=1)

        xception_cm = confusion_matrix(self.test_labels, result)
        plt.figure(figsize = (30,10))
        plt.subplot(2,2,1)
        sns.heatmap(xception_cm,cmap = 'Blues',annot = True, xticklabels = self.class_names, yticklabels = self.class_names)
        plt.show()


    def full_run(self):
        self.processAnimalPics()
        self.createLabels()
        self.gen_datasets()
        # self.train_model()

        results = self.quick_predict()
        



runner = animalTrainer("0448P-0422-model")
# runner.predict()
runner.full_run()


