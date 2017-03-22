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
import hough_boxes
MAX_BOXES = 150

class TableStructureException(Exception):
    pass

def run_full_test(image_dir, info_dir):
  images = [img for img in os.listdir(image_dir) if img.endswith('.jpg')]
  run_test(images, image_dir, info_dir)


def run_image_hough(image, zoom_level, img_dir, info_dir, zoom_prefix):

    h_boxes = hough_boxes.get_boxes(image, img_dir)

    if len(h_boxes) > MAX_BOXES:
        raise TableStructureException("total number of boxes exceeds limit of %s" % MAX_BOXES)

    boxes = cloud_api.add_labels(h_boxes, img_dir + '/' + zoom_prefix, image, zoom_level)

    new_lines = ([],[])

    rows, cols = score_rows.get_structure(boxes, new_lines)

    return (rows, cols, boxes)


def run_test_image(image, zoom_level, img_dir, info_dir, zoom_prefix):
    # Get OCR data from the oxford API
    data = oxford_api.get_json_data(image, img_dir, zoom_level, info_dir)

    # Extract lines from the image
    lines = liner.get_lines(image, img_dir)

    # Extract hierarchical contours
    h_boxes, hierarchy = hallucinator.get_contours(image, img_dir)

    child_boxes, base_box = get_child_boxes(h_boxes, hierarchy, image, img_dir)

    ocr_boxes, raw_boxes = boxer.get_boxes(data, zoom_level, lines, child_boxes, info_dir + 'combos/features/' + image + '.txt', img_dir + '/' + zoom_prefix, image)

    merged_boxes = boxer.merge_box_groups(child_boxes, ocr_boxes, 0.9, base_box)

    boxes = cloud_api.add_labels(merged_boxes, img_dir + '/' + zoom_prefix, image, zoom_level)

    scores = liner.rate_lines(lines, boxes)

    filtered_lines = liner.filter_lines(lines, boxes, scores);

    new_lines = liner.remove_lines(lines, filtered_lines, scores)

    rows, cols = score_rows.get_structure(boxes, new_lines)

    return (rows, cols, boxes)

def run_test(images, zoom_level, img_dir, info_dir):
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
  best_rects = h_boxes
  base_box = get_full_box(image, base_dir)

  return (hallucinator.contours_to_boxes(hallucinator.get_child_contours(best_rects, hierarchy)), base_box)

if __name__ == '__main__':
  if len(sys.argv) != 3:
    exit('Usage: ' + sys.argv[0] + ' src_dir out_dir')

  image_dir = sys.argv[1].rstrip('/')
  info_dir = sys.argv[2].rstrip('/') + '/'

  run_full_test(image_dir, info_dir)
