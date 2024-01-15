from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import subprocess
import os
import cv2 

app = Flask(__name__, template_folder='templates')

training_status = {'status': 'idle'}
live_status= {'status': 'idle'}
def preprocess_image(img):
    img = img.convert("L")
    img = img.resize((48, 48))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array

def decode_predictions(predictions):
    classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    decoded_preds = {}
    for i, pred in enumerate(predictions[0]):
        decoded_preds[classes[i]] = float((pred)*100)
    return decoded_preds

def train_model():
    global training_status
    training_status['status'] = 'training'
    try:
        model_choice = request.json.get('model_choice', 'model')
        
        print(f'training : {model_choice}')
        subprocess.run(['python', f'{model_choice}_training.py'], check=True)
    except subprocess.CalledProcessError as e:
        print('Error during training:', e)
    training_status['status'] = 'idle'


# live streaming 

def predict_emotion(model_path, img_path, confidence_threshold=0.5):
    if not os.path.isfile(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return None

    loaded_model = load_model(model_path)

    if not os.path.isfile(img_path):
        print(f"Error: Image file '{img_path}' not found.")
        return None

    input_size = 48  

    # Load the image in color (RGB)
    img_color = cv2.imread(img_path)
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    # Resize the image to the model's input size
    img_gray = cv2.resize(img_gray, (input_size, input_size))
    # Expand dimensions to match the expected input shape
    img_array = np.expand_dims(img_gray, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values

    # Make predictions
    predictions = loaded_model.predict(img_array)

    # Interpret the predictions
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    predicted_emotion_index = np.argmax(predictions)
    predicted_emotion = emotion_labels[predicted_emotion_index]
    confidence=predictions[0][predicted_emotion_index]
    
    # Check if the prediction confidence is above the threshold
    if confidence >= confidence_threshold:
        print(f"Predicted Emotion: {predicted_emotion} with confidence: {confidence:.2%}")
    else:
        print(f"Prediction confidence below threshold ({confidence_threshold:.2%}). Prediction result may not be reliable.")

    return predicted_emotion,confidence



def mainInstructions(image_path,model_path):
    # Open the default camera
    camera = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not camera.isOpened():
        print("Failed to open the camera")
        exit()
    # Capture a frame from the camera
    text = "When your face is detected, click \"SPACE\" to save or \"ESC\" to cancel!"  # The text to be written
    while True:
        ret, frame = camera.read()
        
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Load the pre-trained face detection model
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        font = cv2.FONT_HERSHEY_COMPLEX  # Font type
        font_scale = 0.5  # Font scale
        thickness = 1  # Thickness of the text
        color = (0, 255, 0)  # Text color in BGR format (Red in this case)
        text_color_bg=(0, 0, 0)
        # Get the dimensions of the image
        image_height, image_width, _ = frame.shape

        # Calculate the width and height of the text box
        text_box_width, text_box_height = cv2.getTextSize(text, font, font_scale, thickness)[0]

        # Calculate the position to align the text horizontally in the middle
        text_x = (image_width - text_box_width) // 2
        position =(text_x, 40)  # Position of the text (top-left corner)
        x, y = position
        cv2.rectangle(frame, (text_x-40, 15), (x + text_box_width+40, y + text_box_height), text_color_bg, -1)
        # Write the text on the image
        cv2.putText(frame, text, position, font, font_scale, color, thickness)


        cv2.imshow("My Emotion", frame)
        cv2.setWindowProperty("My Emotion", cv2.WND_PROP_TOPMOST, 1)
        
        # Save the captured frame as an image file
        cv2.imwrite(image_path, frame)

        predicted_emotion,confidence = predict_emotion(model_path, image_path,0.8)
        if(confidence>0.6):
            text = f"Predicted Emotion: {predicted_emotion} with confidence: {confidence:.2%}"
        else:
            text= "Not Sure!"
        
        key = cv2.waitKey(1)
        if key == ord(' '):
            

            
            # Check if the frame is captured successfully
            if not ret:
                print("Failed to capture the frame")
                exit()
            
        # 27 = ECHAP UNICODE
        if key == 27:
            # Close the window
            cv2.destroyAllWindows()
            # Release the camera and close the window
            camera.release()
            return "Operation Canceled!"








def live_model():
    model_choice = request.form.get('model_choice', 'Custom_model')
    model_path = f'models\{model_choice}.h5'  # Fix the concatenation here
    #path of the image that will be saved and entred as input to the model to detect the emotion
    imagePath=r"./compare.jpg"
    mainInstructions(imagePath,model_path)


@app.route('/')
def main_page():
    return render_template('presentation.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        img = Image.open(request.files['image'].stream).convert("RGB")
        model_choice = request.form.get('model_choice', 'Custom_model')
        model_path = f'datasets\Models\{model_choice}.h5'  # Fix the concatenation here
        model = load_model(model_path)
        preprocessed_img = preprocess_image(img)
        predictions = model.predict(preprocessed_img)
        decoded_predictions = decode_predictions(predictions)
        return jsonify({'predictions': decoded_predictions})
    except Exception as e:
        return jsonify({'error': str(e)})



@app.route('/train', methods=['POST'])
def train():
    global training_status
    if training_status['status'] == 'idle':
        train_model()
        return jsonify({'success': True, 'message': 'Training completed.'})
    else:
        return jsonify({'success': False, 'message': 'Training in progress.'})

@app.route('/live', methods=['POST'])
def live():
    global live_status
    if live_status['status'] == 'idle':
        live_model()
        return jsonify({'success': True, 'message': 'Live on.'})
    else:
        return jsonify({'success': False, 'message': 'Live in progress.'})


@app.route('/training_status', methods=['GET'])
def get_training_status():
    global training_status
    return jsonify(training_status)

if __name__ == '__main__':
    app.run(debug=True)
