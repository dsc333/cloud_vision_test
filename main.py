'''
Google sample test code for vision API:
https://github.com/GoogleCloudPlatform/python-docs-samples/blob/main/vision/snippets/detect/detect.py

GCP requirement:
Before running, go to the Google Vision API menu in the GCP console
and enable the API

Install Python modules from requirements.txt:
    pip3 install -r requirements.txt

Define an environment variable API_KEY in a file named .env (dot is needed) in the current
folder.
'''
from google.cloud import vision
from dotenv import load_dotenv
import os

load_dotenv()

# Read API KEY
API_KEY=os.environ.get('API_KEY')

if not API_KEY:
    print('No API key found.')

# Authenticate and connect to client 
client = vision.ImageAnnotatorClient(client_options={"api_key": API_KEY})

path='image.jpg'
with open(path, "rb") as image_file:
    content = image_file.read()
image = vision.Image(content=content)
response={}

# Object detection
objects = client.object_localization(image=image).localized_object_annotations
print(f"Number of objects found: {len(objects)}")
response['objects'] = {}
for idx, object_ in enumerate(objects):
    response['objects'][f"{object_.name}_{idx}"] = f"{object_.score:0.2f}"
    print(f"{object_.name} (confidence: {object_.score:0.2f})")

# Text detection
text_detected = client.text_detection(image=image)
texts = text_detected.text_annotations

# Detected text copied to a list that's keyed by 'texts' in
# response dictionary
print('\n\nRecognized text')
texts_list = []
for text in texts:
    texts_list.append(text.description)
    print(f'"{text.description}"')
response['texts'] = texts_list
