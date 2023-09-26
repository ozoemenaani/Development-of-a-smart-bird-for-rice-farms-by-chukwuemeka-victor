import cv2
import pygame as pg
import pygame
import time
import os
import random
import base64
import io
import requests
from config import *
from PIL import Image
from threading import Thread
import numpy as np
import copy
import sys
import json

import random

from twilio.rest import Client

# Twilio credentials
account_sid = 'ACdb36a66910f9a64e39747991373d396d'
auth_token = 'f5ac69122337df5c2c2ee842fc5c5068'
client = Client(account_sid, auth_token)


# Initialize Pygame mixer
pygame.mixer.init()


class Camera:
    
    def __init__(self, source):
        # Set up the camera
        self.video = cv2.VideoCapture(source)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Set up the sound files
        sound_folder = 'sounds'
        self.sound_files = [os.path.join(sound_folder, file) for file in os.listdir(sound_folder)]
        
        # Video capture
        self.soundCondition = False
        self.uploadCondition = False
        

        # Time tracking variables
        self.start_time = time.time()
        self.message_interval = 20 * 60  # 5 minutes (in seconds)
        self.last_message_time = 0

        # Initialize instance variables to store prediction and resp
        self.prediction = None
        self.resp = None
        
    def getRawFrame(self):
        # Returns the raw frame
        _, frameToReturn = self.video.read()
        return frameToReturn
        
    # Frame with annotations
    def getFrameAnnotations(self):
        success, img = self.video.read()

        if success:

            # Rotate Camera Upside down if needed
            # img = cv2.rotate(img, cv2.ROTATE_180)
            # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
            height, width, channels = img.shape
            scale = ROBOFLOW_SIZE / max(height, width)
            img = cv2.resize(img, (round(scale * width), round(scale * height)))

            # Encode image to base64 string
            retval, buffer = cv2.imencode('.jpg', img)
            img_str = base64.b64encode(buffer)

            # Get predictions from Roboflow Infer API
            resp = self.getRoboflowPredictions(img_str)

            rawImg = copy.deepcopy(img)
            # Draw all predictions
            respCount = 0
            for prediction in resp:
                if prediction["confidence"] > CONFIDENCE_THRESHOLD:
                    respCount += 1
                    self.writeOnStream(prediction['x'], prediction['y'], prediction['width'], prediction['height'],
                                    prediction['class'],
                                    img)
            
            # Store prediction and resp as instance variables
            self.prediction = prediction['class'] if respCount > 0 else None
            self.resp = resp
            
            return respCount > 0, img, rawImg, resp


        def getRoboflowPredictions(self, img_str):
            max_retries = 3
            retries = 0
            while retries < max_retries:
                try:
                    resp = requests.post(infer_url, data=img_str, headers={
                        "Content-Type": "application/x-www-form-urlencoded"
                    }, stream=True).json()['predictions']
                    return resp
                except requests.exceptions.ConnectionError as e:
                    print(f"Connection error occurred: {e}. Retrying...")
                    retries += 1
                    time.sleep(1)  # Wait for 1 second before retrying

            # If max retries exceeded, handle the failure gracefully
            print("Max retries exceeded. Unable to connect to the server.")
        
            return resp

    def getFrame(self):
        getFrameAnnotationpredicition, sound, img, rawImg, apiResponse = self.getFrameAnnotations()
        # Multithread sound
        if self.prediction == 'birds':  # Use self.prediction to check the last predicted class
            if not self.soundCondition:
                self.soundCondition = True
                soundThread = Thread(target=self.playSound)
                soundThread.start()
        else:
            if self.soundCondition:
                self.stopSound()
                self.soundCondition = False

        # Multithread Active Learning
        if not self.uploadCondition and sound:
            # Do not add blurry images to dataset
            if cv2.Laplacian(rawImg, cv2.CV_64F).var() > LAPLACIAN_THRESHOLD:
                self.uploadCondition = True
                uploadThread = Thread(target=self.activeLearning, args=[rawImg, apiResponse])
                uploadThread.start()

        # Check if enough time has passed since the last message
        current_time = time.time()
        if current_time - self.last_message_time >= self.message_interval:
            # Send a message when an object is detected
            if self.prediction == 'bird':
                self.send_message("Object detected!")
                self.last_message_time = current_time

        return img

    def activeLearning(self, image, apiResponse):
        success, imageId = self.uploadImage(image)
        if success:
            self.uploadAnnotation(imageId, apiResponse)

        self.uploadCondition = False

    def playSound(self):
        while self.soundCondition:
            # Choose a random sound file
            sound_file = random.choice(self.sound_files)

            # Load the sound file
            sound = pygame.mixer.Sound(sound_file)
            
            # Set the volume level to 200% (2.0 times the default volume)
            sound.set_volume(2.0)

            # Play the sound
            sound.play()

            # Apply echo effect by playing the same sound with a delay and reduced volume
            echo_delay = 0.9  # Adjust the delay (in seconds) for the echo effect
            echo_volume = 0.9  # Adjust the volume for the echo effect
            
            pygame.time.delay(int(echo_delay * 1000))  # Convert to milliseconds
            sound.set_volume(echo_volume)
            sound.play()

            # Wait for the sound to finish playing
            while pygame.mixer.get_busy():
                time.sleep(0.1)
            
    def stopSound(self):
        # Stop the sound
        pygame.mixer.music.stop()

    def uploadImage(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pilImage = Image.fromarray(frame)
        # Convert to JPEG Buffer
        buffered = io.BytesIO()
        pilImage.save(buffered, quality=90, format="JPEG")

        # Base 64 Encode
        img_str = base64.b64encode(buffered.getvalue())
        img_str = img_str.decode("ascii")

        r = requests.post(image_upload_url, data=img_str, headers={
            "Content-Type": "application/x-www-form-urlencoded"
        })
        return r.json()['success'], r.json()['id']
        
    def uploadAnnotation(self, imageId, apiResponse):
        # CreateML Dataset Format
        data = []
        annotations = []
        for prediction in apiResponse:
            if prediction["confidence"] < CONFIDENCE_THRESHOLD:
                continue
            annotations.append({"label": prediction['class'],
                                "coordinates": {
                                    "x": prediction['x'],
                                    "y": prediction['y'],
                                    "width": prediction['width'],
                                    "height": prediction['height']
                                }})
        data.append({
            "image": "bird.jpg",
            "annotations": annotations
        })

        # Save to Json File
        with open('activeLearning.json', 'w') as outfile:
            json.dump(data, outfile)

        annotationFilename = "activeLearning.json"

        # Read Annotation as String
        annotationStr = open(annotationFilename, "r").read()

        # Construct the URL
        annotation_upload_url = "".join([
            "https://api.roboflow.com/dataset/", DATASET_NAME, "/annotate/", imageId,
            "?api_key=", ROBOFLOW_API_KEY,
            "&name=", annotationFilename
        ])

        # POST to the API
        r = requests.post(annotation_upload_url, data=annotationStr, headers={
            "Content-Type": "text/plain"
        })
        # return r.json()['success']
        return True
    ''''
    def writeOnStream(self, x, y, width, height, className, frame):
        # Draw a Rectangle around detected image
        cv2.rectangle(frame, (int(x - width / 2), int(y + height / 2)), (int(x + width / 2), int(y - height / 2)),
                      (255, 0, 0), 2)

        # Draw filled box for class name
        cv2.rectangle(frame, (int(x - width / 2), int(y + height / 2)), (int(x + width / 2), int(y + height / 2) + 35),
                      (255, 0, 0), cv2.FILLED)

        # Set label font + draw Text
        font = cv2.FONT_HERSHEY_DUPLEX

        cv2.putText(frame, className, (int(x - width / 2 + 6), int(y + height / 2 + 26)), font, 0.5, (255, 255, 255), 1)

    '''
    def writeOnStream(self, x, y, width, height, className, frame):
        # Draw a Rectangle around detected image
        cv2.rectangle(frame, (int(x - width / 2), int(y + height / 2)), (int(x + width / 2), int(y - height / 2)),
                      (255, 0, 0), 2)

        # Draw filled box for class name
        cv2.rectangle(frame, (int(x - width / 2), int(y + height / 2)), (int(x + width / 2), int(y + height / 2) + 35),
                      (255, 0, 0), cv2.FILLED)

        # Set label font + draw Text
        font = cv2.FONT_HERSHEY_DUPLEX

        cv2.putText(frame, className, (int(x - width / 2 + 6), int(y + height / 2 + 26)), font, 0.5, (255, 255, 255), 1)

    def send_message(self, message):
        twilio_number = '+19282850742'
        recipient_number = '+2347064571331'

        # Send the message using Twilio API
        client.messages.create(
            body=message,
            from_=twilio_number,
            to=recipient_number
        )

if __name__ == '__main__':
    source = 0  # Change this to your specific camera source if needed
    camera = Camera(source)

    while True:
        frame = camera.getFrame()
        cv2.imshow('Camera', frame)
        
        
        if cv2.waitKey(1) == ord('q'):
            break

    camera.video.release()
    cv2.destroyAllWindows()

