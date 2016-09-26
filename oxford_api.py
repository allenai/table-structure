import json
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

def get_json_data(image, base_path, zoom_level, pref):
    
  zoom_prefix = str(zoom_level) + 'x/' if zoom_level > 1 else ''

  url = "/vision/v1.0/ocr?%s" % params


  full_path = base_path + '/' + zoom_prefix + image 
  with open(full_path, 'rb') as img_file:
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
    body = response.read()
    if response.status == 200:
        data = body
        cache.put(cache_prefix, img_data, data)
    else:
        raise Exception("Error with retrieving Oxford OCR results for %s: %s" % (full_path, body))

  finally:
    if conn is not None:
      conn.close()
      conn = None

  return json.loads(data.decode('utf-8')) # Need to double-check if utf-8 is correct
