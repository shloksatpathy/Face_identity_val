from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=100):
        self.next_object_id = 0
        
        # Maps object ID to its current centroid and bounding box
        self.objects = OrderedDict()
        
        # Maps object ID to the number of consecutive frames it has been missing
        self.disappeared = OrderedDict()
        
        # Maximum number of consecutive frames a given object is allowed to be marked as
        # "disappeared" until we deregister it from tracking
        self.max_disappeared = max_disappeared
        
        # Maximum distance between centroids to associate an object
        self.max_distance = max_distance

    def register(self, centroid, rect):
        self.objects[self.next_object_id] = (centroid, rect)
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        return self.next_object_id - 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        """
        rects is a list of bounding box tuples: (startX, startY, endX, endY)
        Returns a dictionary of tracked objects: {object_id: (centroid, rect)}
        """
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i], rects[i])
        else:
            object_ids = list(self.objects.keys())
            object_values = list(self.objects.values())
            object_centroids = [val[0] for val in object_values]

            # Compute the distance between each pair of object centroids and input centroids
            D = dist.cdist(np.array(object_centroids), input_centroids)

            # In order to perform this matching we must (1) find the smallest value in each row 
            # and then (2) sort the row indexes based on their minimum values
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = (input_centroids[col], rects[col])
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col], rects[col])

        return self.objects
