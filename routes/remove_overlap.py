# This script was written by Ray Wang
# raywang0328@gmail.com

from flask import render_template, request, session, jsonify
import pandas as pd
import cv2
from scipy.spatial import KDTree
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image
from python.config import application as app
import io
import python.config 
import math



def is_same_image( des0, des1):

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)


    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des0, des1, k=2)

  
    mask_matches = [[0, 0] for _ in range(len(matches))]

   
    for t, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            mask_matches[t]=[1, 0]

    if np.unique(mask_matches, return_counts = True)[1].shape == (2,) and np.unique(mask_matches, return_counts = True)[1][1] > 50  :
        return np.unique(mask_matches, return_counts = True)
    else:
        return None

def check_pair_existence(pair, list_of_pairs):

    pair_set = set(pair)
    
 
    for existing_pair in list_of_pairs:
       
        if pair_set == set(existing_pair):
            return True
            
 
    return False



@app.route('/remove_overlap', methods=['POST'])
def remove_overlap():
    if 'img_directory' not in request.files:
        return jsonify({'message': 'No file part in the request.'}), 400
    files = request.files.getlist('img_directory')

    

    df = python.config.csv_file
    df = df.dropna()

    image_names = []
    image_numbers = []
    image_crop = []
    image_kp_and_des = []
    longitudes= []
    latitudes = []
    

    sift = cv2.SIFT_create()

  

    for file in files:

        name = secure_filename(file.filename)
        im = Image.open(io.BytesIO(file.read()))

        for index, row in df.iterrows():
            if name in str(row['image_name']):
                if row['x1'] != "x1":
                    image_names.append(name)
                    image_numbers.append(row['number'])
                    img_np = np.array(im.crop((float(row['x1']),float(row['y1']),float(row['x2']),float(row['y2']))))
                    image_crop.append(img_np)
                    kp, des = sift.detectAndCompute(img_np, None)
                    image_kp_and_des.append((kp, des))
                    longitudes.append(row['longitude'])
                    latitudes.append(row['latitude'])


    tree = KDTree(list(zip(longitudes, latitudes)))


    R_earth_meters = 6371e3  # meters

 
    radius_meters = 10 # meters


    radius_degrees = radius_meters / R_earth_meters * (180. / np.pi)


    def process_point(i):
        pairs = {} 
  
        indices = tree.query_ball_point([longitudes[i], latitudes[i]], radius_degrees)
        for j in indices:
            if j != i: 
              
                image_file_1, image_file_2 = image_names[i], image_names[j]
                image_num_1, image_num_2 = image_numbers[i], image_numbers[j]
                if image_file_1 != image_file_2:

                    path1 = image_file_1  + "||" + str(image_num_1) 
                    path2 = image_file_2  + "||" + str(image_num_2) 

                    kp0, des0 = image_kp_and_des[i]
                    kp1, des1 = image_kp_and_des[j]

                    pair_key = frozenset([path1, path2])

                    if pair_key not in pairs:
                        happ = is_same_image(des0, des1)
                        if happ is not None:
                        
                            pairs[pair_key] = [path1,path2]
        return pairs

    all_pairs = []
    for i in range(len(image_names)):
        pairs = process_point(i)
        all_pairs.append(pairs)


    pairs = {}
    for pair in all_pairs:
        pairs.update(pair)

    if len(pairs) == 0:
        return "Nothing"
    

    class DisjointSet:
        def __init__(self):
            self.leader = {} 
            self.group = {}  

        def add(self, a, b):
            leader_a = self.leader.get(a)
            leader_b = self.leader.get(b)
            if leader_a is not None:
                if leader_b is not None:
                    if leader_a == leader_b:
                        return
                    group_a = self.group[leader_a]
                    group_b = self.group[leader_b]
                    if len(group_a) < len(group_b):
                        a, leader_a, group_a, b, leader_b, group_b = b, leader_b, group_b, a, leader_a, group_a
                    group_a |= group_b
                    del self.group[leader_b]
                    for i in group_b:
                        self.leader[i] = leader_a
                else:
                    self.group[leader_a].add(b)
                    self.leader[b] = leader_a
            else:
                if leader_b is not None:
                    self.group[leader_b].add(a)
                    self.leader[a] = leader_b
                else:
                    self.group[a] = {a, b}
                    self.leader[a] = self.leader[b] = a

        def get(self, a):
            leader = self.leader.get(a)
            if leader is not None:
                return self.group[leader]
            else:
                return None

    ds = DisjointSet()
    for pair in pairs.keys():
        a, b = tuple(pair)
        ds.add(a, b)
        
        
    Overlapped = [] 
    for group in ds.group.values():
       
        group_list = list(group)
        for item in group_list[1:]:
            Overlapped.append(item)
            

    to_remove_tuples = [tuple(x.split('||')) for x in Overlapped]


  
    def should_remove(row):
        return (row['image_name'], str(row['number'])) in to_remove_tuples

    
    df = df[~df.apply(should_remove, axis=1)]
    df = df.reset_index(drop=True)



    df.to_csv('detections.csv', index=False,mode = 'w')

    python.config.csv_file = df
   
    return str("Deleted: " + str(len(Overlapped)) + " item(s) from the csv. Specific Images: " + str(Overlapped))
            
    





        