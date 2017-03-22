import itertools
from collections import deque

def cluster_scores(score_matrix, threshold):
  clusters = []
  for i in range(len(score_matrix)):
    curr_cluster = set([i])
    for j in range(len(score_matrix)):
      curr_score = score_matrix[i][j]
      if curr_score > threshold:
        curr_cluster.add(j)

      clusters.append(curr_cluster)

  # Now we need to merge any clusters with shared elements
  # Based roughly on the algorithm from Niklas at:
  # http://stackoverflow.com/questions/9110837/python-simple-list-merging-based-on-intersections
  have_merged = True
  while have_merged:
    have_merged = False
    new_clusters = []
    while len(clusters) > 0:
      first = clusters[0]
      remaining = clusters[1:]
      clusters = []

      for cluster in remaining:
        if cluster.isdisjoint(first):
          clusters.append(cluster)
        else:
          have_merged = True
          first |= cluster

      new_clusters.append(first)

    clusters = new_clusters

  return clusters


def newer_cluster_scores(score_matrix, threshold):
  # Start with a set of all boxes
  # Then, for any given pair in that set that's not likely to
  # be in a (col|row) together, split the set. Continue
  # this until we get a basis? Then need to reconcile
  # the halves, and reduce, based on composition

  # Make a set with all boxes
  to_check = deque([frozenset(range(len(score_matrix)))])
  finalized = set()
  cache = set()


  while len(to_check) > 0:
    cluster = to_check.pop()
    valid = True
    for comb in itertools.combinations(cluster, 2):
      # We want to split if there are any two elements
      # that should not be in a row together
      if score_matrix[comb[0]][comb[1]] < threshold:
        valid = False

        split_1 = cluster.difference([comb[0]])
        split_2 = cluster.difference([comb[1]])

        if hash(split_1) not in cache:
          to_check.append(split_1)
          cache.add(hash(split_1))

        if hash(split_2) not in cache:
          to_check.append(split_2)
          cache.add(hash(split_2))

        break

    # If we made it to here, we checked all the combinations
    # and they all belong in a column together
    if valid and cluster not in finalized:
      finalized.add(cluster)

  # Check if any of the sets are proper subsets of another
  # set in the final group. If so, we do not need the
  # subset, as the info is elsewhere
  basis = set()
  for x in finalized:
    keep = True
    for y in finalized:
      if x < y:
        keep = False
        break

    if keep:
      basis.add(x)

  return basis

def new_cluster_scores(score_matrix, threshold):
  clusters = []
  for i in range(len(score_matrix)):
    curr_cluster = set([i])
    for j in range(len(score_matrix)):
      curr_score = score_matrix[i][j]
      if curr_score > threshold:
        curr_cluster.add(j)

    if curr_cluster not in clusters:
      clusters.append(curr_cluster)

  # Remove duplicate sets
  # new_clusters = [k for k,_ in itertools.groupby(clusters)]
  new_clusters = clusters

  basis = []
  for cluster in new_clusters:
    # Get other sets that are proper subsets of this cluster
    related = [x for x in new_clusters if x < cluster]

    # Then get the union of all the related, and check if
    # they union to be equal to this one. If so, ignore it
    rel_union = set().union(*related)

    if cluster != rel_union:
      basis.append(cluster)

  return basis
