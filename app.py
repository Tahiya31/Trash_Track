# This script was written by Ray Wang
# raywang0328@gmail.com

from flask import Flask, request, jsonify, render_template, send_file, Response
import numpy as np
import os
from io import BytesIO
import csv
import io
from werkzeug.utils import secure_filename
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import supervision as sv
import pandas as pd
import base64
import math
import torch
from io import StringIO
from script import delete_big, delete_rock, delete_overlap, does_box1_cover_box2, delete_box, isolate, bb_intersection_over_union, get_lat_lon, get_geotagging, get_altitude, get_exif
from routes.remove_overlap import remove_overlap
from routes.mapping import mapping, show_map
from routes.plots import plots 
from python.config import application as app

from GroundingDINO.groundingdino.util.inference import Model


current_dir = os.getcwd()  
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate

GROUNDING_DINO_CONFIG_PATH = current_dir  + "/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = current_dir  + "/weights/groundingdino_swint_ogc.pth"


#grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
if torch.cuda.is_available():
    grounding_dino_model = Model(GROUNDING_DINO_CONFIG_PATH, GROUNDING_DINO_CHECKPOINT_PATH, device = "cuda")
    print("gpu")
else:
    grounding_dino_model = Model(GROUNDING_DINO_CONFIG_PATH, GROUNDING_DINO_CHECKPOINT_PATH, device='cpu')
    print("cpu")

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


# app = Flask(__name__)
try:
    df = pd.read_csv('detections.csv')  
except pd.errors.EmptyDataError:
    df = pd.DataFrame()

import python.config
python.config.csv_file = df




def predict_class(img):
    prompt = ['wood' , 'cage',  'fishing gear',  'nature', 'plastic', 'metal', 'wheel']

    image = img

    inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
    max_index = torch.argmax(probs) 
    
    
    return(prompt[max_index])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/csv/')
def check_csv():
    return render_template('csv.html')

@app.route('/overlap/')
def overlap():
    return render_template('overlap.html')

@app.route('/map/')
def maps():
    return render_template('map.html')

@app.route('/plots/')
def plot():
    return render_template('plots.html')

@app.route('/about/')
def about():
    return render_template('about.html')


@app.route('/process_images', methods=['POST'])
def process_images():
    if 'img_directory' not in request.files:
     return jsonify({'message': 'No file part in the request.'}), 400

    files = request.files.getlist('img_directory')
  
    csv_file_path = 'detections.csv'

    if not os.path.exists(csv_file_path):

        # File does not exist, create it   
        with open(csv_file_path, 'w') as f:
            pass
        
    # Clear the content of the CSV and write the column titles
    column_titles = ['x1','y1','x2','y2','longitude','latitude','altitude','image_name','number','type']
   
    with open(csv_file_path, 'r') as read_file:
        first_line = read_file.readline()

    if not first_line:
        # File is empty
        with open(csv_file_path, 'w', newline='') as write_file:   
            writer = csv.writer(write_file)
            writer.writerow(column_titles)

    try:
        df = pd.read_csv('detections.csv')  
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(column_titles)

    python.config.csv_file = df
    

    CLASSES = ['all trashes', 'rocks']
    BOX_TRESHOLD = 0.3
    TEXT_TRESHOLD = 0.25

    crops = []
    total = []
    
    for file in files:
        # read the image file
        filestr = file.read()
        # convert string data to numpy arra

        image_name = secure_filename(file.filename)
        
           
        seen = df.isin([image_name]).any().any() 

        if seen == False:
            im = Image.open(io.BytesIO(filestr))
            im = np.asarray(im)

            detections1 = grounding_dino_model.predict_with_classes(
                image=im,
                classes=CLASSES, #enhance_class_name(class_names=CLASSES),
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD
            )
            
            detections2 = grounding_dino_model.predict_with_classes(
            image=im,
            classes=CLASSES, #enhance_class_name(class_names=CLASSES),
            box_threshold= BOX_TRESHOLD - 0.15,
            text_threshold=TEXT_TRESHOLD - 0.15
            )

            detections1 = delete_big(detections1, im)
            detections1 = delete_rock(detections1)

            detections2 = delete_big(detections2, im)
            detections2 = delete_rock(detections2)

        
            xyxy = np.vstack((detections1.xyxy, detections2.xyxy))
            mask = detections1.mask
            confidence = np.concatenate((detections1.confidence,detections2.confidence))
            class_id = np.concatenate((detections1.class_id, detections2.class_id))
            tracker_id = detections1.tracker_id


            detections = sv.Detections(xyxy,mask,confidence,class_id,tracker_id)

            
            detections = delete_overlap(detections)
            detections = delete_box(detections)
            
            detections = detections[detections.class_id != None]
            
            
            lon, lat = get_lat_lon(get_exif(file))
            alt = get_altitude(file)


            if len(detections.xyxy) != 0:
                for i in range(len(detections.xyxy)):
                    im = Image.fromarray(np.uint8(im)).convert('RGB')
                    img_np = np.array(im.crop(detections.xyxy[i]))
                    output_image = Image.fromarray(img_np.astype('uint8'))
                    buffered = BytesIO()
                    output_image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue())
                    img_str = img_str.decode('utf-8')

                
                    crops.append(img_str)
                    predicted_class = predict_class(img_np)

                    custom_values = [str(detections.xyxy[i][0]), str(detections.xyxy[i][1]), str(detections.xyxy[i][2]), str(detections.xyxy[i][3]), lon, lat, alt, image_name, i, predicted_class]

                    df.loc[len(df)] = custom_values  
                
                    with open(csv_file_path, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(custom_values)

                    total.append(custom_values)
            else: 
                custom_values = ["NA", "NA", "NA","NA", lon, lat, alt, image_name, "NA", "NA"]
                df.loc[len(df)] = custom_values
                with open(csv_file_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(custom_values)


            
        else:
            im = Image.open(io.BytesIO(filestr))
            for index, row in df.iterrows():
                if image_name in str(row['image_name']):
                    if math.isnan(row[0]):
                       break
                    img_np = np.array(im.crop((row[0],row[1],row[2],row[3])))
                    output_image = Image.fromarray(img_np.astype('uint8'))
                    buffered = BytesIO()
                    output_image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue())
                    img_str = img_str.decode('utf-8')
                    crops.append(img_str)
                    total.append([str(row[0]),str(row[1]),str(row[2]),str(row[3]),str(row[4]),str(row[5]),str(row[6]), str(row[7]),str(row[8]),str(row[9])])
               
    df.to_csv('detections.csv', index=False, mode='w') 
    python.config.csv_file = df

    return jsonify({'predicted_classes': [row[-3:] for row in total], 'crops': crops}), 200


@app.route('/clear_results')
def clear_results():
  # Open detections.csv in write mode
#  global csv_file
  python.config.csv_file = pd.DataFrame()
  
  
  with open('detections.csv', 'w') as f:  
    pass # Truncate the file
 

  return jsonify({'message': 'Results cleared'})

@app.route('/get_results')
def get_results():
#  global csv_file
  return python.config.csv_file.to_csv()


  with open('detections.csv') as f:
    return f.read()
  
@app.route('/download_results')
def download_results():
  #  global csv_file
    str_io = StringIO()
    python.config.csv_file.to_csv(str_io, index=False)
    csv_data = str_io.getvalue()

    # Create a Flask response with the CSV data and send it
    response = Response(csv_data, mimetype='text/csv')
    response.headers.set("Content-Disposition", "attachment", filename="detections.csv")
    return response

@app.route('/download_map')
def download_map():
    html_map_str = python.config.map.get_root().render()

    response = Response(html_map_str, mimetype='text/html')

    # Set Content-Disposition header
    response.headers["Content-Disposition"] = "attachment; filename=map.html"

    return response

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
 # global csv_file
  csv_f = request.files['csv']

  python.config.csv_file = pd.read_csv(csv_f)
  
  python.config.csv_file.to_csv('detections.csv', index=False, mode='w') 
  
  return jsonify({'message': 'CSV uploaded successfully'})

@app.route('/update_class')
def update_class():
  index = request.args.get('index', type=int)
  new_class = request.args.get('new_class')

  df = python.config.csv_file
  if new_class == "repeated trash":
    df = df.drop(index)
  else:
    df.loc[index, 'type'] = new_class
  python.config.csv_file = df
  
  # Update CSV file
  df.to_csv('detections.csv', index=False)
  
  # Return updated row as a JSON
  if new_class == "repeated trash":
    return jsonify({"message": "Row deleted"}), 200
  else:
    # Return updated row as a JSON
    return jsonify(df.loc[index].to_dict()), 200



if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8000)

