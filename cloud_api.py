import json
import time
import os
import cv2
import base64
import httplib2

import dir_helper

from apiclient.discovery import build
from oauth2client.client import GoogleCredentials

def query_google_ocr(image_content):
  '''Run a label request on a single image'''

  API_DISCOVERY_FILE = 'https://vision.googleapis.com/$discovery/rest?version=v1'
  http = httplib2.Http()
  creds_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'google_credentials.json')
  with open(creds_path) as f:
      creds_json = f.read()

  credentials = GoogleCredentials.from_json(creds_json).create_scoped(
      ['https://www.googleapis.com/auth/cloud-platform'])


  credentials.authorize(http)

  service = build('vision', 'v1', http=http, discoveryServiceUrl=API_DISCOVERY_FILE)

  service_request = service.images().annotate(
    body={
      'requests': [{
        'image': {
          'content': image_content
         },
        'features': [{
          'type': 'TEXT_DETECTION',
          'maxResults': 1
         }]
       }]
    })

  return service_request.execute()

def get_labels(response, combine=False):
  if 'textAnnotations' not in response['responses'][0]:
    return '' if combine else []

  detections = response['responses'][0]['textAnnotations']

  if combine:
    return detections[0]['description'].replace('\n', ' ').strip()
  else:
    return label_boxes(detections[1:])


def label_boxes(detections):
  boxes = []
  for det in detections:
    xs = [x['x'] for x in det['boundingPoly']]
    ys = [x['y'] for x in det['boundingPoly']]

    min_x = min(xs)
    min_y = min(xs)

    boxes.append((min_x, min_y, max(xs) - min_x, max(ys) - min_y, det['description']))

  return boxes

def get_cell_label(cache_base, img_base, photo_file, box, zoom, sleep_delay):
  cache_path = cache_base + photo_file + '_' + '_'.join([str(x) for x in box[:4]]) + '.json'

  if os.path.isfile(cache_path):
    with open(cache_path, 'r') as cache_file:
      response = json.loads(cache_file.read())
  else:
    img = cv2.imread(img_base + photo_file)
    x1 = zoom * box[0]
    x2 = x1 + (zoom * box[2])
    y1 = zoom * box[1]
    y2 = y1 + (zoom * box[3])

    cell = img[y1:y2, x1:x2]

    retval, cell_buffer = cv2.imencode('.jpg', cell)

    image_content = base64.b64encode(cell_buffer).decode()

    response = query_google_ocr(image_content)

    time.sleep(sleep_delay)

    if 'responses' in response:
      dir_helper.ensure(cache_path)
      with open(cache_path, 'w') as cache_file:
        json.dump(response, cache_file)
    else:
      return ''

  return get_labels(response, combine=True)

def add_labels(boxes, image_base, image_path, cache_path, zoom, sleep_delay):
  labeled = []
  for box in boxes:
    label = get_cell_label(cache_path, image_base, image_path, box, zoom, sleep_delay)
    labeled.append((box[0], box[1], box[2], box[3], [label]))

  return labeled
