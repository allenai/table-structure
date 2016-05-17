import json
import time
import os
import http.client, urllib.request, urllib.parse, urllib.error, base64
import sub_key
import dir_helper

json_cache_path = 'json_cache'

# API vars
headers = {
    # Request headers
    'Content-Type': 'application/octet-stream',
    'Ocp-Apim-Subscription-Key': sub_key.get_key(),
}

params = urllib.parse.urlencode({
    # Request parameters
    'language': 'en',
    'detectOrientation ': 'true',
})

def get_json_data(image, base_path, zoom_level, pref, sleep_delay):
  zoom_prefix = str(zoom_level) + 'x/' if zoom_level > 1 else ''
  json_cache_file = pref + json_cache_path + '/' + zoom_prefix + image + '.json'

  if os.path.isfile(json_cache_file):
    with open(json_cache_file, 'r') as j_file:
      data = json.loads(j_file.read())

    if 'statusCode' not in data or data['statusCode'] != 429:
      return data

  with open(base_path + '/' + zoom_prefix + image, 'rb') as img_file:
    img_data = img_file.read()

  data = None

  while data is None:
    conn = None
    try:
      conn = http.client.HTTPSConnection('api.projectoxford.ai', timeout=10)
      conn.request("POST", "/vision/v1/ocr?%s" % params, body=img_data, headers=headers)
      response = conn.getresponse()
      data = response.read()
      conn.close()
    except Exception as e:
      print("[Errno {0}] {1}".format(e.errno, e.strerror))
      data = None
    else:
      if conn is not None:
        conn.close()
        conn = None

  json_data = json.loads(data.decode('utf-8')) # Need to double-check if utf-8 is correct

  dir_helper.ensure(json_cache_file)
  with open(json_cache_file, 'w') as json_file:
    json.dump(json_data, json_file)

  time.sleep(sleep_delay)

  return json_data
