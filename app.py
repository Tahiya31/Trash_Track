import os
import io
import csv
import math
import base64
import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_file, Response
from werkzeug.utils import secure_filename
from sklearn.cluster import AgglomerativeClustering # ADDED FOR CLUSTERING

# Project specific imports
from python.config import application as app
import python.config
from script import get_lat_lon, get_altitude, get_exif

# --- IMPORT OUR NEW ML PIPELINE ---
from ml_pipeline import init_models, run_full_pipeline

# Initialize the machine learning models when the app starts
init_models()

column_titles = ['x1','y1','x2','y2','longitude','latitude','altitude','image_name','number','type']

try:
    df = pd.read_csv('detections.csv')
    if df.empty and len(df.columns) == 0:
        df = pd.DataFrame(columns=column_titles)
except (pd.errors.EmptyDataError, FileNotFoundError):
    df = pd.DataFrame(columns=column_titles)

python.config.csv_file = df

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

@app.route('/add/')
def add_page():
    return render_template('add.html')

@app.route('/add_manual_detection', methods=['POST'])
def add_manual_detection():
    """Receives JSON from the Add page and appends it to the dataset."""
    data = request.get_json()
    
    lat = data.get('latitude')
    lon = data.get('longitude')
    d_type = data.get('type')
    desc = data.get('description', 'MANUAL_REPORT')
    
    custom_values = ["NA", "NA", "NA", "NA", lon, lat, 0, f"Manual: {desc}", "NA", d_type]
    
    df = python.config.csv_file
    df.loc[len(df)] = custom_values
    python.config.csv_file = df
    
    csv_file_path = 'detections.csv'
    with open(csv_file_path, 'a', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(custom_values)
        
    return jsonify({'message': 'Manual detection added successfully', 'data': custom_values}), 200

@app.route('/process_images', methods=['POST'])
def process_images():
    if 'img_directory' not in request.files:
        return jsonify({'message': 'No file part in the request.'}), 400

    files = request.files.getlist('img_directory')
    n_clusters = int(request.form.get('n_clusters', 4))
    csv_file_path = 'detections.csv'

    if not os.path.exists(csv_file_path):
        with open(csv_file_path, 'w', newline='') as write_file:   
            writer = csv.writer(write_file)
            writer.writerow(column_titles)

    try:
        df = pd.read_csv('detections.csv')  
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=column_titles)

    python.config.csv_file = df
    
    all_embeddings = [] 
    all_image_detections = [] 
    
    for file in files:
        filestr = file.read()
        image_name = secure_filename(file.filename)
        seen = df.isin([image_name]).any().any() 

        if not seen:
            im_pil = Image.open(io.BytesIO(filestr)).convert('RGB')
            im_numpy = np.asarray(im_pil)
            
            lon, lat = get_lat_lon(get_exif(file))
            alt = get_altitude(file)

            detections = run_full_pipeline(im_numpy)
            
            if len(detections) > 0:
                for det in detections:
                    det['lon'] = lon
                    det['lat'] = lat
                    det['alt'] = alt
                    det['image_name'] = image_name
                    all_image_detections.append(det)
                    all_embeddings.append(det['embedding'])
            else:
                custom_values = ["NA", "NA", "NA", "NA", lon, lat, alt, image_name, "NA", "NA"]
                df.loc[len(df)] = custom_values
                with open(csv_file_path, 'a', newline='') as f_out:
                    writer = csv.writer(f_out)
                    writer.writerow(custom_values)
        else:
            im_pil = Image.open(io.BytesIO(filestr)).convert('RGB')
            for index, row in df.iterrows():
                if image_name in str(row['image_name']):
                    if pd.isna(row['x1']) or row['x1'] == "NA":
                        break
                    
                    crop_box = (float(row['x1']), float(row['y1']), float(row['x2']), float(row['y2']))
                    crop_pil = im_pil.crop(crop_box)
                    
                    buffered = io.BytesIO()
                    crop_pil.save(buffered, format="JPEG", quality=85)
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                    all_image_detections.append({
                        'bbox': [row['x1'], row['y1'], row['x2'], row['y2']],
                        'crop_base64': img_str,
                        'embedding': None,
                        'image_name': image_name
                    })

    # Agglomerative Clustering across detections
    if len(all_embeddings) > 0:
        actual_k = min(n_clusters, len(all_embeddings)) 
        if actual_k > 1:
            cluster_labels = AgglomerativeClustering(n_clusters=actual_k).fit_predict(all_embeddings)
        else:
            cluster_labels = [0] * len(all_embeddings)
            
        for idx, det in enumerate(all_image_detections):
            if 'embedding' in det and det['embedding'] is not None:
                x1, y1, x2, y2 = det['bbox']
                cluster_name = f"Cluster {cluster_labels[idx]}"
                det['cluster_name'] = cluster_name
                
                custom_values = [str(x1), str(y1), str(x2), str(y2), det['lon'], det['lat'], det['alt'], det['image_name'], idx, cluster_name]
                df.loc[len(df)] = custom_values  
                
                with open(csv_file_path, 'a', newline='') as f_out:
                    writer = csv.writer(f_out)
                    writer.writerow(custom_values)
    else:
        for det in all_image_detections:
            det['cluster_name'] = "Cluster 0"

    df.to_csv('detections.csv', index=False) 
    python.config.csv_file = df

    return jsonify({'detections': all_image_detections}), 200

@app.route('/clear_results')
def clear_results():
    python.config.csv_file = pd.DataFrame(columns=column_titles)
    with open('detections.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(column_titles)
    return jsonify({'message': 'Results cleared'})

@app.route('/get_results')
def get_results():
    return python.config.csv_file.to_csv()

@app.route('/download_results')
def download_results():
    str_io = io.StringIO()
    python.config.csv_file.to_csv(str_io, index=False)
    csv_data = str_io.getvalue()
    response = Response(csv_data, mimetype='text/csv')
    response.headers.set("Content-Disposition", "attachment", filename="detections.csv")
    return response

@app.route('/download_map')
def download_map():
    html_map_str = python.config.map.get_root().render()
    response = Response(html_map_str, mimetype='text/html')
    response.headers["Content-Disposition"] = "attachment; filename=map.html"
    return response

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
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
    
    df.to_csv('detections.csv', index=False)
    
    if new_class == "repeated trash":
        return jsonify({"message": "Row deleted"}), 200
    else:
        return jsonify(df.loc[index].to_dict()), 200

@app.route('/update_cluster_class')
def update_cluster_class():
    cluster_name = request.args.get('cluster_name')
    new_class = request.args.get('new_class')

    df = python.config.csv_file
    
    # If it's a false positive, delete the whole cluster from the dataset
    if new_class == "false positive (noise)":
        df = df[df['type'] != cluster_name].reset_index(drop=True)
    else:
        # Otherwise, update the type for all rows in this cluster
        df.loc[df['type'] == cluster_name, 'type'] = new_class
        
    python.config.csv_file = df
    df.to_csv('detections.csv', index=False)
    
    return jsonify({"message": f"Updated entire {cluster_name} to {new_class}"}), 200

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8000)