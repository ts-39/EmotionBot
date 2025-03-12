# EmotionBot

Emotion Detection and Response

Overview

This project aims to detect emotions from facial expressions and respond with an appropriate phrase. While many models exist for facial emotion recognition, I needed a way to extract the detected emotions. To achieve this, I utilized the open-source API paz and made slight modifications to enable emotion extraction.

Modifications to PAZ

In the original PAZ pipeline, the detected class name was not being stored externally. To overcome this, I modified the following code in paz/pipelines/detection.py (line 540):

Original Code:

def call(self, image):
    boxes2D = self.detect(image.copy())['boxes2D']
    boxes2D = self.square(boxes2D)
    boxes2D = self.clip(image, boxes2D)
    cropped_images = self.crop(image, boxes2D)
    for cropped_image, box2D in zip(cropped_images, boxes2D):
        predictions = self.classify(cropped_image)
        box2D.class_name = predictions['class_name']
        box2D.score = np.amax(predictions['scores'])
    image = self.draw(image, boxes2D)
    return self.wrap(image, boxes2D)

Modified Code:

def call(self, image):
    boxes2D = self.detect(image.copy())['boxes2D']
    boxes2D = self.square(boxes2D)
    boxes2D = self.clip(image, boxes2D)
    cropped_images = self.crop(image, boxes2D)
    with open('classnames.txt', mode='a') as f:  # Added file logging
        for cropped_image, box2D in zip(cropped_images, boxes2D):
            predictions = self.classify(cropped_image)
            box2D.class_name = predictions['class_name']
            box2D.score = np.amax(predictions['scores'])
            f.write(box2D.class_name + '\n')  # Log the detected class
    image = self.draw(image, boxes2D)
    return self.wrap(image, boxes2D)

This modification logs detected emotions into classnames.txt, allowing the system to read and generate appropriate verbal responses.

Facial Emotion Recognition Model

The facial emotion recognition pipeline used in this project is based on MiniXception, a lightweight deep learning model trained for real-time facial expression recognition. This model is part of the PAZ framework and is capable of detecting common emotions such as:

Neutral

Happy

Surprise

Sad

Angry

Disgust

Fear

Once an emotion is detected, the system generates a spoken response using Google Text-to-Speech (gTTS) and plays it back via Pygame.

Requirements

A requirements.txt file is provided to install all necessary dependencies. You can install them using:

pip install -r requirements.txt

Usage

Ensure your webcam is connected.

Run the main script to start real-time emotion detection.

The system will analyze facial expressions, store detected emotions, and respond with an appropriate phrase.

Future Improvements

Enhancing response variations based on confidence levels.

Supporting multi-language responses.

Improving detection accuracy with additional training data.

This project showcases the integration of computer vision, deep learning, and text-to-speech synthesis to create an interactive emotion-aware system. Feedback and contributions are welcome!

