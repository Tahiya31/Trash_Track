# This script was written by Ray Wang
# raywang0328@gmail.com

from flask import Flask, request, jsonify
import pandas as pd
import folium
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from python.config import application as app
import python.config

@app.route('/plotting', methods=['POST'])
def plots():
    data = request.get_json()
    num_clusters = int(data['num_clusters'])
    df = python.config.csv_file
    if df.empty:
        return jsonify({'message': 'Nothing in the file!'}), 400
    
    df = df.dropna()

    scaler = StandardScaler()
    df[['longitude', 'latitude']] = scaler.fit_transform(df[['longitude', 'latitude']])

    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    df['cluster'] = kmeans.fit_predict(df[['longitude', 'latitude']])

    item_types = df['type'].unique()
    palette = sns.color_palette('hls', len(item_types))
    color_dict = dict(zip(item_types, palette))

    images = []

    for i in range(num_clusters):
        sns.countplot(x='type', data=df[df['cluster']==i], palette=color_dict)
        plt.title(f'Frequency Plot for Cluster {i}')
        sio = BytesIO()
        plt.savefig(sio, format='png')
        plt.close()
        sio.seek(0)
        image = base64.b64encode(sio.read()).decode('utf-8')
        images.append(image)

    # reverse scale transformation to get the original coordinates
    df[['longitude', 'latitude']] = scaler.inverse_transform(df[['longitude', 'latitude']])

    # calculate cluster centers
    cluster_centers = df.groupby('cluster')[['longitude', 'latitude']].mean().reset_index()

    # initialize the map with the first cluster's coordinates

    cluster_map = python.config.map
    # add a marker for each cluster center
    for _, row in cluster_centers.iterrows():
        folium.Marker(
            location=[row['longitude'], row['latitude']],
            popup=f"Cluster {row['cluster']}"
        ).add_to(cluster_map)

    # save the map as HTML
    python.config.map = cluster_map

    return jsonify({'images': images}), 200
