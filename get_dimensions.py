import os
import sys
import cv2
import time

import score_rows
import sub_key
import oxford_api
import boxer
import liner
import hallucinator
import spreadsheeter
import cloud_api

zoom_level = 3
oxford_delay = 5
cloud_delay = 0

def run_full_test(image_dir, info_dir):
  images = [img for img in os.listdir(image_dir) if img.endswith('.jpg')]
  run_test(images, image_dir, info_dir)


def run_test_image(image, img_dir, info_dir, zoom_prefix):
    # Get OCR data from the oxford API
    data = oxford_api.get_json_data(image, img_dir, zoom_level, info_dir, oxford_delay)

    # Extract lines from the image
    lines = liner.get_lines(image, img_dir)

    # Extract hierarchical contours
    h_boxes, hierarchy = hallucinator.get_contours(image, img_dir)

    child_boxes, base_box = get_child_boxes(h_boxes, hierarchy, image, img_dir)

    ocr_boxes, raw_boxes = boxer.get_boxes(data, zoom_level, lines, child_boxes, info_dir + 'combos/features/' + image + '.txt')

    merged_boxes = boxer.merge_box_groups(child_boxes, ocr_boxes, 0.9, base_box)

    merged_labels = boxer.merge_ocr_boxes(raw_boxes, [])

    boxes = cloud_api.add_labels(merged_boxes, img_dir + '/' + zoom_prefix, image, info_dir + 'google_cache/' + zoom_prefix, zoom_level, cloud_delay)

    scores = liner.rate_lines(lines, boxes)

    filtered_lines = liner.filter_lines(lines, boxes, scores);

    new_lines = liner.remove_lines(lines, filtered_lines, scores)

    rows, cols = score_rows.get_structure(boxes, new_lines)

    return (rows, cols, boxes)

def run_test(images, img_dir, info_dir):
  zoom_prefix = str(zoom_level) + 'x/' if zoom_level > 1 else ''

  for image in images:
    print('Processing: ' + image)

    # Write to xlsx and json
    rows, cols, boxes = run_test_image(image, img_dir, info_dir, zoom_prefix)

    spreadsheeter.output(rows, cols, boxes, info_dir + 'xlsx' + '/' + zoom_prefix + image + '.xlsx', info_dir + 'json_out' + '/' + zoom_prefix + image + '.json')

    print('Complete')

def get_full_box(image, base_dir):
 height, width, channels = cv2.imread(base_dir + '/' + image).shape 

 return (0, 0, width, height, '')

def get_child_boxes(h_boxes, hierarchy, image, base_dir):
  root_boxes = hallucinator.get_root_contours(h_boxes, hierarchy)
  best_root = hallucinator.get_most_nested(root_boxes, hierarchy, h_boxes)

  if best_root is None:
    best_rects = h_boxes
    base_box = get_full_box(image, base_dir)
  else:
    best_rects = hallucinator.get_rects(best_root[1], h_boxes)
    base_box = hallucinator.contour_to_box(best_root[0][1])

  return (hallucinator.contours_to_boxes(hallucinator.get_child_contours(best_rects, hierarchy)), base_box)

if __name__ == '__main__':
  if len(sys.argv) != 3:
    exit('Usage: ' + sys.argv[0] + ' src_dir out_dir')

  image_dir = sys.argv[1].rstrip('/')
  info_dir = sys.argv[2].rstrip('/') + '/'

  run_full_test(image_dir, info_dir)
