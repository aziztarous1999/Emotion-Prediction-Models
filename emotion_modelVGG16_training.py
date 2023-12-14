from utils import *


emotion_modelVGG16= Sequential()

# Block 1
emotion_modelVGG16.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)))
emotion_modelVGG16.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
emotion_modelVGG16.add(MaxPooling2D((2, 2), strides=(2, 2)))

# Block 2
emotion_modelVGG16.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
emotion_modelVGG16.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
emotion_modelVGG16.add(MaxPooling2D((2, 2), strides=(2, 2)))

# Block 3
emotion_modelVGG16.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
emotion_modelVGG16.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
emotion_modelVGG16.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
emotion_modelVGG16.add(MaxPooling2D((2, 2), strides=(2, 2)))

# Block 4
emotion_modelVGG16.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
emotion_modelVGG16.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
emotion_modelVGG16.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
emotion_modelVGG16.add(MaxPooling2D((2, 2), strides=(2, 2)))

# Block 5
emotion_modelVGG16.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
emotion_modelVGG16.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
emotion_modelVGG16.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
emotion_modelVGG16.add(MaxPooling2D((2, 2), strides=(2, 2)))

emotion_modelVGG16.add(Flatten())
emotion_modelVGG16.add(Dense(4096, activation='relu'))
emotion_modelVGG16.add(Dropout(0.5))
emotion_modelVGG16.add(Dense(4096, activation='relu'))
emotion_modelVGG16.add(Dropout(0.5))
emotion_modelVGG16.add(Dense(7, activation='softmax'))
emotion_modelVGG16.summary()

# Compile the model
emotion_modelVGG16.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])


# Configure Model Checkpoint
file_s = r'TestTraining/emotion_modelVGG16.h5'
checkpointer = ModelCheckpoint(file_s, monitor='loss',verbose=1,save_best_only=True,
                               save_weights_only=False,save_freq='epoch')


emotion_modelVGG16.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

callback_list=[checkpointer]
historyVGG16 = emotion_modelVGG16.fit(
        train_generator,
    steps_per_epoch=len(train_generator)/16,
    batch_size=128,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)/16,
epochs=20,
    shuffle=True,
    callbacks=[callback_list])



test_generator = val_datagen.flow_from_directory(
    'test',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical',
    shuffle=False
)

y_pred = emotion_modelVGG16.predict(test_generator)

y_true = test_generator.classes

y_pred_classes = np.argmax(y_pred, axis=1)

accuracy = accuracy_score(y_true, y_pred_classes)
print(f'Test Accuracy: {accuracy * 100:.2f}%')



emotion_modelVGG16.save(file_s)
print("Dense Model Training Completed!!!")