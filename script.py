# This script was written by Ray Wang
# raywang0328@gmail.com

import numpy as np
import supervision as sv
from PIL import Image
import piexif
from PIL.ExifTags import TAGS, GPSTAGS
import math

def get_exif(filename):
    image = Image.open(filename)
    image.verify()
    return image._getexif()

def get_geotagging(exif):
    if not exif:
        raise ValueError("No EXIF metadata found")

    geotagging = {}
    for (idx, tag) in TAGS.items():
        if tag == 'GPSInfo':
            if idx not in exif:
                raise ValueError("No EXIF geotagging found")

            for (t, value) in GPSTAGS.items():
                if t in exif[idx]:
                    geotagging[value] = exif[idx][t]

    return geotagging

def get_decimal_from_dms(dms, ref):

    degrees = dms[0]
    minutes = dms[1] / 60.0
    seconds = dms[2] / 3600.0

    if ref in ['S', 'W']:
        degrees = -degrees
        minutes = -minutes
        seconds = -seconds

    return round(degrees + minutes + seconds, 8)

def get_lat_lon(exif_data):
    geotagging = get_geotagging(exif_data)
    lat = get_decimal_from_dms(geotagging['GPSLatitude'], geotagging['GPSLatitudeRef'])
    lon = get_decimal_from_dms(geotagging['GPSLongitude'], geotagging['GPSLongitudeRef'])

    return (lat, lon)

def get_altitude(image_path):
    img = Image.open(image_path)
    exif_data = piexif.load(img.info['exif'])
    
    # Exif tag for GPS information
    gps_info = exif_data['GPS']
    
    # Exif tag for altitude is 6 in the GPS IFD
    if piexif.GPSIFD.GPSAltitude in gps_info:
        # Altitude is stored as a ratio, so we need to divide to get the actual value
        altitude = gps_info[piexif.GPSIFD.GPSAltitude][0] / gps_info[piexif.GPSIFD.GPSAltitude][1]
        return altitude
    else:
        return None

def calculate_GSD(alt, sensor_width, focal_length, image_width):
    
    return (alt * sensor_width ) / (focal_length * image_width)



def latlong_to_cartesian(lat, long):
    R = 6371  # kilometers
    lat = math.radians(lat)
    long = math.radians(long)
    x = R * long * math.cos(lat)
    y = R * lat
    return x, y

def area_of_box(box):
    (min_lat, min_long, max_lat, max_long) = box
    min_x, min_y = latlong_to_cartesian(min_lat, min_long)
    max_x, max_y = latlong_to_cartesian(max_lat, max_long)
    return (max_x - min_x) * (max_y - min_y)

def iou_of_boxes(box1, box2):
    (min_lat1, min_long1, max_lat1, max_long1) = box1
    (min_lat2, min_long2, max_lat2, max_long2) = box2

    intersection_box = (
        max(min_lat1, min_lat2), max(min_long1, min_long2),
        min(max_lat1, max_lat2), min(max_long1, max_long2)
    )

    if (intersection_box[2] < intersection_box[0] or
        intersection_box[3] < intersection_box[1]):
        # No intersection
        intersection_area = 0
    else:
        intersection_area = area_of_box(intersection_box)

    union_area = area_of_box(box1) + area_of_box(box2) - intersection_area

    return intersection_area / union_area if union_area != 0 else 0
    


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def delete_big(detections: sv.Detections , image):
    num = []
    xyxy = detections.xyxy
    mask = detections.mask
    confidence = detections.confidence
    class_id = detections.class_id
    tracker_id = detections.tracker_id

    for i in range(len(detections.xyxy)):
        x1, y1, x2, y2 = detections.xyxy[i]
        area = (x2 - x1) * (y2 - y1)
        limit = (image.shape[0] * image.shape[1]) / 8
        if area > limit:
          num.append(i)

    xyxy = np.delete(xyxy,num,0)
    if confidence is not None:
        confidence = np.delete(confidence,num)
    if class_id is not None:
        class_id = np.delete(class_id,num)


    return sv.Detections(xyxy, mask, confidence,class_id,tracker_id )

def delete_rock(detections):
    rocks = []
    xyxy = detections.xyxy
    mask = detections.mask
    confidence = detections.confidence
    class_id = detections.class_id
    tracker_id = detections.tracker_id

    for i in range(len(detections.class_id)):
        if detections.class_id[i] == 1:
            for f in range(len(detections.class_id)):
                if i != f:
                    iou = bb_intersection_over_union(detections.xyxy[i], detections.xyxy[f])
                    if iou >= 0.8:
                        rocks.append(f)
            rocks.append(i)

    xyxy = np.delete(xyxy,rocks,0)
    confidence = np.delete(confidence,rocks)
    class_id = np.delete(class_id,rocks)                  

    return sv.Detections(xyxy,mask, confidence,class_id,tracker_id )

def delete_overlap(detections):
    rocks = []
    xyxy = detections.xyxy
    mask = detections.mask
    confidence = detections.confidence
    class_id = detections.class_id
    tracker_id = detections.tracker_id

    for i in range(len(detections.xyxy)):
        for f in range(len(detections.xyxy)):
            if i != f:
                if i  not in rocks:
                    iou = bb_intersection_over_union(detections.xyxy[i], detections.xyxy[f])
                    if iou >= 0.4:
                        rocks.append(f)
                    
    xyxy = np.delete(xyxy,rocks,0)
    if confidence is not None:
        confidence = np.delete(confidence,rocks)
    if class_id is not None:
        class_id = np.delete(class_id,rocks)
               

    return sv.Detections(xyxy, mask, confidence,class_id,tracker_id )

def does_box1_cover_box2(box1, box2):
    return box1[0] <= box2[0] and box1[1] <= box2[1] and box1[2] >= box2[2] and box1[3] >= box2[3]


def delete_box(detections): 
    rocks = []
    xyxy = detections.xyxy
    mask = detections.mask
    confidence = detections.confidence
    class_id = detections.class_id
    tracker_id = detections.tracker_id

    for i in range(len(detections.xyxy)):
        for f in range(len(detections.xyxy)):
            if i != f:
                if i  not in rocks:
                    if does_box1_cover_box2(detections.xyxy[i],  detections.xyxy[f]):
                        rocks.append(i)
                    
    xyxy = np.delete(xyxy,rocks,0)
    if confidence is not None:
        confidence = np.delete(confidence,rocks)
    if class_id is not None:
        class_id = np.delete(class_id,rocks)
               

    return sv.Detections(xyxy,mask, confidence,class_id,tracker_id )
    

def isolate(detections):
    temp = []
    largest = 0
    largest_num = 0
    xyxy = detections.xyxy
    mask = detections.mask
    confidence = detections.confidence
    class_id = detections.class_id
    tracker_id = detections.tracker_id

    for i in range(len(detections.xyxy)):
        temp.append(i)
        if confidence[i] > largest_num:
            largest = i
            largest_num = confidence[i]
            
    temp = np.array(temp)
    
    if len(temp) > 1:
        temp = np.delete(temp, largest)   
        xyxy = np.delete(xyxy,temp,0)
        if confidence is not None:
            confidence = np.delete(confidence,temp)
        if class_id is not None:
            class_id = np.delete(class_id,temp)
               

    return sv.Detections(xyxy,mask,confidence,class_id,tracker_id )