import json
import time
import os
import cv2
import base64
import httplib2
import ocr_cache as cache

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

  request_body = {
      'requests': [{
        'image': {
          'content': image_content
         },
        'features': [{
          'type': 'TEXT_DETECTION',
          'maxResults': 1
         }]
       }]
    }

  version = 'v1'

  cache_prefix = 'google/vision/%s' % version
  request_body_blob = json.dumps(request_body).encode('utf8')

  response_json = cache.get(cache_prefix, request_body_blob)

  if response_json:
    return json.loads(response_json.decode('utf8'))

  service = build('vision', version, http=http, discoveryServiceUrl=API_DISCOVERY_FILE)
  service_request = service.images().annotate(body=request_body)


  responses = service_request.execute()['responses']


  if 'error' in responses[0]:
      raise Exception("Received error from Google: %s" % responses[0]['error'])

  cache.put(cache_prefix, request_body_blob, json.dumps(responses).encode('utf8'))

  return responses

def get_labels(responses, combine=False):
  if 'textAnnotations' not in responses[0]:
    return '' if combine else []

  detections = responses[0]['textAnnotations']

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

  img = cv2.imread(img_base + photo_file)
  x1 = int(round(zoom * box[0]))
  x2 = x1 + int(round(zoom * box[2]))
  y1 = int(round(zoom * box[1]))
  y2 = y1 + int(round(zoom * box[3]))

  cell = img[y1:y2, x1:x2]

  retval, cell_buffer = cv2.imencode('.jpg', cell)

  image_content = base64.b64encode(cell_buffer).decode()

  responses = query_google_ocr(image_content)

  time.sleep(sleep_delay)

  return get_labels(responses, combine=True)

def add_labels(boxes, image_base, image_path, cache_path, zoom, sleep_delay):
  labeled = []
  for box in boxes:
    label = get_cell_label(cache_path, image_base, image_path, box, zoom, sleep_delay)
    labeled.append((box[0], box[1], box[2], box[3], [label]))

  return labeled
