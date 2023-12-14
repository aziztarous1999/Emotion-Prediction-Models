from utils import *

emotion_modelVGG16= Sequential()

# Build the model
Googlenet_model = Sequential()

# Convolutional Block 1
Googlenet_model.add(Conv2D(32, (3, 3), padding='same', input_shape=(48, 48, 1)))
Googlenet_model.add(BatchNormalization())
Googlenet_model.add(Activation('relu'))
Googlenet_model.add(Conv2D(32, (3, 3), padding='same'))
Googlenet_model.add(BatchNormalization())
Googlenet_model.add(Activation('relu'))
Googlenet_model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional Block 2
Googlenet_model.add(Conv2D(64, (3, 3), padding='same'))
Googlenet_model.add(BatchNormalization())
Googlenet_model.add(Activation('relu'))
Googlenet_model.add(Conv2D(64, (3, 3), padding='same'))
Googlenet_model.add(BatchNormalization())
Googlenet_model.add(Activation('relu'))
Googlenet_model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional Block 3
Googlenet_model.add(Conv2D(128, (3, 3), padding='same'))
Googlenet_model.add(BatchNormalization())
Googlenet_model.add(Activation('relu'))
Googlenet_model.add(Conv2D(128, (3, 3), padding='same'))
Googlenet_model.add(BatchNormalization())
Googlenet_model.add(Activation('relu'))
Googlenet_model.add(MaxPooling2D(pool_size=(2, 2)))

# Fully Connected Layers
Googlenet_model.add(Flatten())
Googlenet_model.add(Dense(512))
Googlenet_model.add(BatchNormalization())
Googlenet_model.add(Activation('relu'))
Googlenet_model.add(Dropout(0.5))
Googlenet_model.add(Dense(7, activation='softmax'))
Googlenet_model.summary()

# Compile the model
Googlenet_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

fle_s=r'TestTraining/Googlenet_Model.h5'
checkpointer = ModelCheckpoint(fle_s, monitor='loss',verbose=1,save_best_only=True,
                               save_weights_only=False, mode='auto',save_freq='epoch')

callback_list=[checkpointer]


historyGooglenet = Googlenet_model.fit(
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

y_pred = Googlenet_model.predict(test_generator)

y_true = test_generator.classes

y_pred_classes = np.argmax(y_pred, axis=1)

accuracy = accuracy_score(y_true, y_pred_classes)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

Googlenet_model.save(fle_s)
print("Googlenet Model Training Completed!!!")