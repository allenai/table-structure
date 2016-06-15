import json
import time
import http.client, urllib.request, urllib.parse, urllib.error
import ocr_cache as cache
import sub_key

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

  url = "/vision/v1/ocr?%s" % params

  with open(base_path + '/' + zoom_prefix + image, 'rb') as img_file:
    img_data = img_file.read()

  cache_prefix = 'oxford' + url
  data = cache.get(cache_prefix, img_data)

  if data:
      return json.loads(data.decode('utf8'))

  conn = None
  try:
    conn = http.client.HTTPSConnection('api.projectoxford.ai', timeout=10)
    conn.request("POST", url, body=img_data, headers=headers)
    response = conn.getresponse()
    if response.status == 200:
        data = response.read()
        cache.put(cache_prefix, img_data, data)

    conn.close()
  finally:
    if conn is not None:
      conn.close()
      conn = None

  time.sleep(sleep_delay)

  return json.loads(data.decode('utf-8')) # Need to double-check if utf-8 is correct
