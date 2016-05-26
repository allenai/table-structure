import cv2
import os

def get_edge_image(base_path, img_name):
  img = cv2.imread(base_path + '/' + img_name)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(gray, 50, 150, apertureSize = 3)

  return edges

def process_image(base_path, img_name, out_path):
  dist_img = get_transform(base_path, img_name)[1]

  cv2.imwrite(out_path + '/' + img_name, dist_img)

def get_transform(base_path, img_name):
  edges = get_edge_image(base_path, img_name)
#   cv2.imwrite(out_path + '/' + img_name + '_edges.jpg', edges)

  inverted = get_inverted(edges)
#   cv2.imwrite(out_path + '/' + img_name + '_inverted.jpg', inverted)

  dist_img = cv2.distanceTransform(inverted, cv2.DIST_L2, 3)

  return (dist_img, scale_output(dist_img.copy()))

def scale_output(img):
  img_area = len(img) * len(img[0])
  for r in range(len(img)):
    for c in range(len(img[r])):
      img[r][c] = img[r][c] * 1.0 / img_area

  return img

def get_inverted(img):
  for r in range(len(img)):
    for c in range(len(img[r])):
      img[r][c] = 255 - img[r][c]

  return img

def process():
  base_dir = 'regents_table'
  out_path = 'temp_dt'

  images = [img for img in os.listdir(base_dir) if img.endswith('.jpg')]

  for image in images:
    print('Processing: ' + image)
    process_image(base_dir, image, out_path)
