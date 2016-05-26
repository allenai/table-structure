import clusterer
import dir_helper
import distance_transform
import os
import os.path
import operator
import pickle
from itertools import combinations
from functools import cmp_to_key

should_record_features = False

def get_boxes(data, zoom_level, lines, contour_boxes, feature_file, img_base, image):
  raw_boxes = get_boxes_from_json(data, zoom_level)

  dist_imgs = distance_transform.get_transform(img_base, image)

  combined = combine_boxes(raw_boxes, lines, contour_boxes, feature_file, dist_imgs)

  return combined, raw_boxes

def get_boxes_from_json(data, zoom_level = 1):
  boxes = []
  if 'regions' in data:
    for i, region in enumerate(data['regions']):
      if 'lines' in region:
        for j, line in enumerate(region['lines']):
          if 'words' in line:
            for k, word in enumerate(line['words']):
              if 'boundingBox' in word:
                bbox = [int(int(x) / zoom_level) for x in word['boundingBox'].split(',')]

                text = word['text'] if 'text' in word else ''

                bbox.append(text)
                bbox.append({'region': i, 'line': j, 'word': k})
                boxes.append(bbox)

  return boxes

def combine_boxes(boxes, lines, contour_boxes, feature_file, dist_imgs):
  # To use the classifier, do this:
  box_classifier_scores = score_boxes(boxes, lines, feature_file, dist_imgs)

  box_scores = modify_box_scores(boxes, contour_boxes, box_classifier_scores)

  score_clusters = clusterer.cluster_scores(box_scores, 0.99)

  # Now need to translate score_cluster indices into boxes,
  # And then combine the boxes in each cluster

  clustered = [[boxes[idx] for idx in cluster] for cluster in score_clusters]

  combined = combine_clustered_boxes(clustered)

  return combined

# This method lowers scores for boxes to prevent merging across
# cell boundaries
def modify_box_scores(boxes, c_boxes, scores):
  threshold = 0.5
  # First calculate the cells which each box overlaps
  # This should be only one, but we're using a set to be
  # careful for now, in case thresholds are lowered
  overlaps = [set() for i in range(len(boxes))]
  for i, box_1 in enumerate(boxes):
    for j, box_2 in enumerate(c_boxes):
      horiz_over = max(0, min(box_1[0] + box_1[2], box_2[0] + box_2[2]) - max(box_1[0], box_2[0]))
      vert_over = max(0, min(box_1[1] + box_1[3], box_2[1] + box_2[3]) - max(box_1[1], box_2[1]))

      overlap_area = horiz_over * vert_over
      min_area = min(box_1[2] * box_1[3], box_2[2] * box_2[3])

      if overlap_area * 1.0 / min_area > threshold:
        overlaps[i].add(j)

  # Now, for each combination of boxes, we want
  # to penalize if they do not overlap a shared cell
  # (if their overlap sets are disjoint)

  for comb in combinations(enumerate(boxes), 2):
    i = comb[0][0]
    j = comb[1][0]

    box_1 = comb[0][1]
    box_2 = comb[1][1]

    if overlaps[i] != overlaps[j] and overlaps[i].isdisjoint(overlaps[j]):
      # This means they overlap different boxes
      # For now this will be an absolute penalty
      scores[i][j] = 0.0
      scores[j][i] = 0.0

  return scores

def combine_clustered_boxes(clusters):
  new_boxes = []

  for cluster in clusters:
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    labels = []

    # Sort the boxes in the cluster by y and then x
    # to preserve label ordering
    cluster.sort(key = lambda x: (x[1], x[0]))

    for box in cluster:
      min_x = min(min_x, box[0])
      min_y = min(min_y, box[1])
      max_x = max(max_x, box[0] + box[2])
      max_y = max(max_y, box[1] + box[3])
      labels.append(box[4])

    # We keep it x, y, width, height, labels for now
    # Although we later change it to be x1, y1, x2, y2, labels
    # For now we want to keep this paradigm
    new_boxes.append((min_x, min_y, max_x - min_x, max_y - min_y, labels))

  return new_boxes

def score_boxes(boxes, lines, feature_file, dist_imgs):
  box_scores = [[0.0 for box in boxes] for box in boxes]

  # Remove since we're appending for now
  if should_record_features:
    dir_helper.ensure(feature_file)

    open(feature_file, 'w').close()

  classifier_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'classifier.pkl')

  with open(classifier_path, 'rb') as f:
    classifier = pickle.load(f)

  features = {}

  for comb in combinations(enumerate(boxes), 2):
    i = comb[0][0]
    j = comb[1][0]

    box_1 = comb[0][1]
    box_2 = comb[1][1]

    scores = {}
    features = {}

    # 1. Higher score the closer together they are
    # horiz and vert (percentage and flat)
    horiz_dist = max(0, max(box_1[0], box_2[0]) - min(box_1[0] + box_1[2], box_2[0] + box_2[2]))
    vert_dist = max(0, max(box_1[1], box_2[1]) - min(box_1[1] + box_1[3], box_2[1] + box_2[3]))

    min_horiz_range = min(box_1[2], box_2[2])
    min_vert_range = min(box_1[3], box_2[3])

    dist = (horiz_dist ** 2 + vert_dist ** 2) ** 0.5
    min_range = (min_horiz_range ** 2 + min_vert_range ** 2) ** 0.5

    scores['dist_pix'] = 1.0 / (1.0 + dist)
    scores['dist_perc'] = 1.0 / (1.0 + (dist * 1.0 / min_range))
    features['dist_pix'] = dist
    features['dist_perc'] = dist * 1.0 / min_range

    # 2. Higher score if they overlap (horiz and vert)
    # both percentage and flat
    horiz_over = max(0, min(box_1[0] + box_1[2], box_2[0] + box_2[2]) - max(box_1[0], box_2[0]))
    vert_over = max(0, min(box_1[1] + box_1[3], box_2[1] + box_2[3]) - max(box_1[1], box_2[1]))

    scores['overlap_pix'] = 1.0 * horiz_over * vert_over
    scores['overlap_perc'] = 1.0 * horiz_over * vert_over / (min_horiz_range * min_vert_range)
    features['overlap_pix'] = horiz_over * vert_over
    features['overlap_perc'] = scores['overlap_perc']

    # Check for horizontal line b/w boxes
    scores['no_div_horiz_line'] = 0.0 if line_between(box_1, box_2, lines, 1) else 1.0
    features['no_div_horiz_line'] = 0 if line_between(box_1, box_2, lines, 1) else 1

    # Check for vertical line b/w boxes
    scores['no_div_vert_line'] = 0.0 if line_between(box_1, box_2, lines, 0) else 1.0
    features['no_div_vert_line'] = 0 if line_between(box_1, box_2, lines, 0) else 1

    # Additional features?
    # Ratio of areas
    box_1_area = box_1[2] * box_1[3]
    box_2_area = box_2[2] * box_2[3]

    features['ratio_of_areas_1'] = box_1_area * 1.0 / box_2_area
    features['ratio_of_areas_2'] = box_2_area * 1.0 / box_1_area

    # Area of box 1
    features['box_1_area'] = box_1_area

    # Area of box 2
    features['box_2_area'] = box_2_area

    # Left to right distance, and right to left distance
    features['ltr_dist'] = min(abs(box_1[0] - (box_2[0] + box_2[2])), box_2[0] - (box_1[0] + box_1[2]))

    # and top to bottom and bottom to top
    features['ttb_dist'] = min(abs(box_1[1] - (box_2[1] + box_2[3])), box_2[1] - (box_1[1] + box_1[3]))

    # Line in between ignoring overlap
    features['no_horiz_line_in_middle'] = 0 if line_in_middle(box_1, box_2, lines, 1) else 1
    features['no_vert_line_in_middle'] = 0 if line_in_middle(box_1, box_2, lines, 0) else 1

    # Distance between centers
    features['dist_bw_vert_centers'] = abs((box_1[0] + box_1[0] + box_1[2]) - (box_2[0] + box_2[0] + box_2[2])) / 2.0
    features['dist_bw_horiz_centers'] = abs((box_1[1] + box_1[1] + box_1[3]) - (box_2[1] + box_2[1] + box_2[3])) / 2.0

    # Distance between tops
    features['dist_bw_lefts'] = abs(box_1[0] - box_2[0])
    features['dist_bw_tops'] = abs(box_1[1] - box_2[1])

    # Distance between bottoms
    features['dist_bw_rights'] = abs((box_1[0] + box_1[2]) - (box_2[0] + box_2[2]))
    features['dist_bw_bottoms'] = abs((box_1[1] + box_1[3]) - (box_2[1] + box_2[3]))

    # Distance transform features
    features['dist_trans_tb_scaled'] = get_tb_dt(dist_imgs[1], box_1, box_2)
    features['dist_trans_lr_scaled'] = get_lr_dt(dist_imgs[1], box_1, box_2)

    score = get_classifier_score(features, classifier)
    record_features(box_1, box_2, features, feature_file)

    box_scores[i][j] = score
    box_scores[j][i] = score

  return box_scores

def record_features(box_1, box_2, features, feature_file):
  if should_record_features:
    with open(feature_file, 'a') as f:
      f.write(','.join([str(x) for x in (box_1[0:4] + box_2[0:4])]))

      for x in sorted(features.keys()):
        f.write(',' + str(features[x]))

      f.write('\n')

def get_classifier_score(features, classifier):
  features = [features[x] for x in sorted(features.keys())]

  pred = classifier.predict([features])

  return pred[0] * 1.0

# Line in between without intersection
def line_in_middle(box1, box2, lines, offset):
  for line in lines[(offset + 1) % 2]:
    # Should this be a comparison only b/w the innermost values?

    # Check if box1, line, box2
    is_bw_1 = box1[offset] + box1[offset + 2] <= line['border'] <= box2[offset]

    # Check if box2, line, box1
    is_bw_2 = box2[offset] + box2[offset + 2] <= line['border'] <= box1[offset]

    if is_bw_1 or is_bw_2:
      return True
  return False

def line_between(box1, box2, lines, offset):
  for line in lines[(offset + 1) % 2]:
    # Check if box1, line, box2
    is_bw_1 = box1[offset] + box1[offset + 2] <= line['border'] <= box2[offset]

    # Check if box2, line, box1
    is_bw_2 = box2[offset] + box2[offset + 2] <= line['border'] <= box1[offset]

    alt_offset = (offset + 1) % 2

    # Check if the line overlaps with box1
    intersects_1 = max(0, min(line['end'], box1[alt_offset] + box1[alt_offset + 2]) - max(line['start'], box1[alt_offset]))

    # Check if the line overlaps with box2
    intersects_2 = max(0, min(line['end'], box2[alt_offset] + box2[alt_offset + 2]) - max(line['start'], box2[alt_offset]))
    if (is_bw_1 or is_bw_2) and (intersects_1 or intersects_2):
      return True

  return False

def combine_scores(scores):
  total = 0.0

  # Naive combination for now
  for score_type in scores:
    total += scores[score_type]

  return total

def add_labels(boxes, label_boxes, threshold):
  labeled = []
  for box in boxes:
    labels = []
    for label_box in label_boxes:
      horiz_over = max(0, min(box[0] + box[2], label_box[0] + label_box[2]) - max(box[0], label_box[0]))
      vert_over = max(0, min(box[1] + box[3], label_box[1] + label_box[3]) - max(box[1], label_box[1]))
      overlap_area = horiz_over * vert_over
      min_area = min(box[2] * box[3], label_box[2] * label_box[3])
      if overlap_area * 1.0 / min_area > threshold:
        labels.append(label_box)

    # Double-check this sorting order, but this is y, then x
    label = [x[4] for x in sorted(labels, key=cmp_to_key(label_comparator))]

    labeled.append((box[0], box[1], box[2], box[3], label))

  return labeled

# Comparator for labels with some variance allowed for lines with variable toplines
def label_comparator(a, b):
  top_dist = abs(a[1] - b[1])
  min_height = min(a[3], b[3])

  if top_dist * 1.0 / min_height > 0.25:
    # This means the difference of the tops
    # is significant, so sort by it
    return a[1] - b[1]

  # Otherwise we want to sort by the left edges
  return a[0] - b[0]

# This method merges two groups of boxes by calculating overlap
# Overlapping boxes are connected
# Then any X-many relationships are resolved into the 'many' side
def merge_box_groups(group_1, group_2, threshold, bbox):
  conns_1 = [[] for i in range(len(group_1))]
  conns_2 = [[] for i in range(len(group_2))]

  for i, box_1 in enumerate(group_1):
    # Should these check off threshold, instead?
    for j, box_2 in enumerate(group_2):
      horiz_over = max(0, min(box_1[0] + box_1[2], box_2[0] + box_2[2]) - max(box_1[0], box_2[0]))
      vert_over = max(0, min(box_1[1] + box_1[3], box_2[1] + box_2[3]) - max(box_1[1], box_2[1]))

      overlap_area = horiz_over * vert_over
      min_area = min(box_1[2] * box_1[3], box_2[2] * box_2[3])

      if overlap_area * 1.0 / min_area > threshold:
        conns_1[i].append(j)
        conns_2[j].append(i)

  # We have two ways we could approach this filtering:
  # 1. Find ones with 1-1 or 1-0, and add the largest of them
  # 2. For all of them, add the ones in a 1-many, add the many
  # and then filter with a set. I think this should be equivalent
  # given that there are no intragroup overlaps
  # and thus am going to do option 1 for now.

  # We only want boxes that overlap the bounding box, but
  # If it isn't a 1-0, we know the overlap with a cell box is
  # above the threshold, and don't need to check this case
  # Thus we only check 1-0 ones for now (for which we could get
  # away with only checking the second group, technically)

  merged = []
  for i, conns in enumerate(conns_1):
    if len(conns) == 0:
      if box_overlap(group_1[i], bbox) > threshold:
        merged.append(group_1[i])
    elif len(conns) == 1:
      selector = None
      if len(conns_2[conns[0]]) == 1:
        # This is a 1-1 correspondence, and we want the biggest
        # box, as it is likely more accurate
        selector = get_largest
      else:
        # This is a many-1 correspondence, and we want to take
        # the 1, being the smaller box
        # We may be able to make a correct assumption about
        # which is smallest, but we'll check for now
        selector = get_smallest
        
      merged.append(selector(group_1[i], group_2[conns[0]]))

  for i, conns in enumerate(conns_2):
    if len(conns) == 0:
      if box_overlap(group_2[i], bbox) > threshold:
        merged.append(group_2[i])
    # This needs to check if we already processed, to avoid dedups
    # We will have processed it only if it is a 1-1
    elif len(conns) == 1 and len(conns_1[conns[0]]) != 1:
      # This is guaranteed to be a many-1, thus we
      # can simply take the smallest
      merged.append(get_smallest(group_2[i], group_1[conns[0]]))

  return merged

# TODO: Refactor so that everything uses this
# Maybe move it to a more centralized location
def box_overlap(box_1, box_2):
  horiz_over = max(0, horiz_overlap(box_1, box_2))
  vert_over = max(0, vert_overlap(box_1, box_2))

  overlap_area = horiz_over * vert_over
  min_area = min(box_1[2] * box_1[3], box_2[2] * box_2[3])

  return overlap_area * 1.0 / min_area

# Return the larger of the two boxes
def get_largest(box_1, box_2):
  area_1 = box_1[2] * box_1[3]
  area_2 = box_2[2] * box_2[3]

  if area_1 > area_2:
    return box_1
  
  return box_2

def get_smallest(box_1, box_2):
  area_1 = box_1[2] * box_1[3]
  area_2 = box_2[2] * box_2[3]

  if area_1 <= area_2:
    return box_1
  
  return box_2

def predict_missing_boxes(rows, cols, boxes):
  indices = {}
  box_map = {}

  for i, box in enumerate(boxes):
    box_map[btt(box)] = i

  # We want to mark the row and column to which
  # each cell belongs, and also calculate the dimensions of
  # each row and column
  for i, row in enumerate(rows):
    for box in row[5]:
      idx = box_map[btt(box)]
      indices[idx] = {}

      # This could be made more efficient with a default dict
      if 'rows' not in indices[idx]:
        indices[idx]['rows'] = []

      indices[idx]['rows'].append(i)

  for i, col in enumerate(cols):
    for box in col[5]:
      idx = box_map[btt(box)]

      # See note about efficiency above
      if 'cols' not in indices[idx]:
        indices[idx]['cols'] = []

      indices[idx]['cols'].append(i)

  cells = [[False for x in range(len(cols))] for y in range(len(rows))]

  for i,box in enumerate(boxes):
    for k,row_idx in enumerate(indices[i]['rows']):
      for j,col_idx in enumerate(indices[i]['cols']):
        cells[row_idx][col_idx] = True

  # Calculate bbox for predicted missing boxes
  missing_boxes = []
  for i, row in enumerate(cells):
    for j, cell in enumerate(row):
      if not cell:
        try:
          box_left = min([box[0] for box in cols[j][5] if min(indices[box_map[btt(box)]]['cols']) == j])
          box_right = max([box[0] + box[2] for box in cols[j][5] if max(indices[box_map[btt(box)]]['cols']) == j])

          box_top = min([box[1] for box in rows[i][5] if min(indices[box_map[btt(box)]]['rows']) == i])
          box_bottom = max([box[1] + box[3] for box in rows[i][5] if max(indices[box_map[btt(box)]]['rows']) == i])

          # Mark the missing box
          missing_boxes.append([box_left, box_top, box_right - box_left, box_bottom - box_top, []])
        except ValueError as e:
          print('Problem with adding boxes: ' + str(e))

  return missing_boxes

def btt(box):
  return tuple(box[:4])

def merge_ocr_boxes(raw_boxes, ai2_boxes, combo=False):
  threshold = 0.5
  r_to_a = [[] for i in range(len(raw_boxes))]
  a_to_r = [[] for i in range(len(ai2_boxes))]

  for r, r_box in enumerate(raw_boxes):
    for a, a_box in enumerate(ai2_boxes):

      if box_overlap(r_box, a_box) > threshold:
        r_to_a[r].append(a)
        a_to_r[a].append(r)

  # For now we'll take the raw box if there's a conflict
  merged = []

  for i, conns in enumerate(a_to_r):
    if len(conns) < 1:
      merged.append(ai2_boxes[i])

  return raw_boxes + merged

def get_tb_dt(img, box_1, box_2):
  if vert_overlap(box_1, box_2) > 0:
    return -1

  left = min(box_1[0], box_2[0])
  right = max(box_1[0] + box_1[2], box_2[0] + box_2[2])
  top = min(box_1[1] + box_1[3], box_2[1], box_2[3])
  bot = max(box_1[1], box_2[1])

  return max_val(img, left, top, right, bot)

def get_lr_dt(img, box_1, box_2):
  if horiz_overlap(box_1, box_2) > 0:
    return -1

  left = min(box_1[0] + box_1[2], box_2[0] + box_2[2])
  right = max(box_1[0], box_2[0])
  top = min(box_1[1], box_2[1])
  bot = max(box_1[1] + box_1[3], box_2[1], box_2[3])

  return max_val(img, left, top, right, bot)

def max_val(img, left, top, right, bot):
  return img[top:(bot+1), left:(right+1)].max()

def horiz_overlap(box_1, box_2):
  return min(box_1[0] + box_1[2], box_2[0] + box_2[2]) - max(box_1[0], box_2[0])

def vert_overlap(box_1, box_2):
  return min(box_1[1] + box_1[3], box_2[1] + box_2[3]) - max(box_1[1], box_2[1])
