from utils import *

#model construction

Custom_Model = Sequential()

Custom_Model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
Custom_Model.add(BatchNormalization())
Custom_Model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
Custom_Model.add(BatchNormalization())
Custom_Model.add(MaxPooling2D(pool_size=(2, 2)))
Custom_Model.add(Dropout(0.25))

Custom_Model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
Custom_Model.add(BatchNormalization())
Custom_Model.add(MaxPooling2D(pool_size=(2, 2)))
Custom_Model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
Custom_Model.add(BatchNormalization())
Custom_Model.add(MaxPooling2D(pool_size=(2, 2)))
Custom_Model.add(Dropout(0.25))

Custom_Model.add(Flatten())
Custom_Model.add(Dense(512, activation='relu'))
Custom_Model.add(BatchNormalization())
Custom_Model.add(Dropout(0.5))
Custom_Model.add(Dense(7, activation='softmax'))
Custom_Model.summary()

# Model Compilation
Custom_Model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

fle_s=r'models/Custom_Model.h5'
checkpointer = ModelCheckpoint(fle_s, monitor='loss',verbose=1,save_best_only=True,
                               save_weights_only=False, mode='auto',save_freq='epoch')
callback_list=[checkpointer]

# Model Training
history = Custom_Model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator)/8
    ,epochs=20,shuffle=True,callbacks=[callback_list],
    validation_data=validation_generator,
    validation_steps=len(validation_generator)/64
)

y_pred = Custom_Model.predict(test_generator)

y_true = test_generator.classes

y_pred_classes = np.argmax(y_pred, axis=1)

accuracy = accuracy_score(y_true, y_pred_classes)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

Custom_Model.save(fle_s)
print("Custom Model Training Completed!!!")