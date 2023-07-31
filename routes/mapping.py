# This script was written by Ray Wang
# raywang0328@gmail.com

from flask import redirect, render_template, url_for,request
import pandas as pd
from python.config import application as app
import csv
from pyproj import Transformer
from script import calculate_GSD
import statistics
import folium
import folium.raster_layers
import python.config




@app.route('/mapping', methods=['POST'])
def mapping(fast,skip=False):

    boxes = []
    lats = []
    longs = []
    predictions = []

    if skip == False:
        if python.config.map != '':
        
            return python.config.map.get_root().render()

    df = python.config.csv_file 
    df = df.dropna()

    for index, row1 in df.iterrows():

            long, lat = float(row1[4]), float(row1[5])
            alt = float(row1[6])
            gsd = calculate_GSD(alt, 13.2 ,8.8 , 5472)
            bbox_pixels = [float(row1[0]),float(row1[1]),float(row1[2]),float(row1[3])]
            predictions.append(row1[9])


            transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

            # Calculate the top left corner of the bounding box
            tl_lon, tl_lat = transformer.transform(gsd * bbox_pixels[0], gsd * bbox_pixels[1])

            long_temp, lat_temp = transformer.transform(gsd * 2736, gsd * 1824)

            ref_lon = long - long_temp
            ref_lat = lat - lat_temp


            tl_lon += ref_lon
            tl_lat += ref_lat

            # Calculate the bottom right corner of the bounding box
            br_lon, br_lat = transformer.transform(gsd * bbox_pixels[2], gsd * bbox_pixels[3])
            br_lon += ref_lon
            br_lat += ref_lat


            box1 = [tl_lon,tl_lat,br_lon,br_lat]
            lats.append(tl_lat)
            longs.append(tl_lon)
            lats.append(br_lat)
            longs.append(br_lon)
            boxes.append(box1)

      
    



    tile = folium.TileLayer(
            tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr = 'Esri',
            name = 'Esri Satellite',
            overlay = False,
            control = True
        )


    if len(longs) != 0:
        average_lat = statistics.mean(longs)
        average_long = statistics.mean(lats)
        m = folium.Map(location=[average_lat, average_long], zoom_start=15, max_zoom = 40)

    else: 
        m = folium.Map(zoom_start=15, max_zoom = 40, tiles=tile)

    
    
    tile.add_to(m)

  

    for i, box1 in enumerate(boxes):

        if predictions[i] == "plastic": 
            color = '#dee619'
            temp = 'Images/plastic.png'
        elif predictions[i] == "cage":
            color = '#D900EB'
            temp = 'Images/cage.png'
        elif predictions[i] == "wood": 
            color = '#FC6F15'
            temp = 'Images/wood.png'
        elif predictions[i] == "fishing gear":
            color = '#e75480'
            temp = 'Images/fishing_gear.png'
        elif predictions[i] == "nature":
            color = '#00FF00' 
            temp = 'Images/nature.png'
        elif predictions[i] == "metal":
            color = '#46473E' 
            temp = 'Images/metal.png'
        elif predictions[i] == "wheel":
            color = '#110C0A' 
            temp = 'Images/wheel.png'

        if fast == False: 
            folium.raster_layers.ImageOverlay(
                image = temp,
                bounds = [[box1[0], box1[1]], [box1[2], box1[3]]],
                overlay = False,
                pixelated= False

            ).add_to(m)
            


        '''
        folium.Marker(
            location=[(box1[0] + box1[2]) / 2, (box1[1]+ box1[3])/2],
            icon = mark,
            color=color,
        ).add_to(m)
        '''
        folium.Rectangle(
            bounds=[[box1[0], box1[1]], [box1[2], box1[3]]],
            color=color,
            fill=True,
            fill_color=color
        ).add_to(m)
        


        
        
    # Color mapping for the legend
    color_dict = {"metal": "#46473E", "wheel": "#110C0A", "plastic": "#dee619", "cage": "#D900EB", "wood": "#FC6F15", "nature": "#00FF00", "fishing gear": "#e75480"}

    # Add custom legend
    legend_html = '''
    <div style="position: fixed; bottom: 75px; left: 50px; width: 120px; height: 180px; border:2px solid grey; z-index:9999; font-size:14px; color: white; background-color: rgba(0, 0, 0, 0.5);">
    &nbsp;<b>Legend:</b><br>
    '''

    for key, value in color_dict.items():
        legend_html += '&nbsp;<i class="fa fa-square fa-1x" style="color:{}"></i> {}<br>'.format(value, key)
    

    legend_html += '</div>'

    m.get_root().html.add_child(folium.Element(legend_html))
    python.config.map = m 

    
    return m.get_root().render()
   
   
@app.route('/get_cur', methods=['GET'])
def get_cur_map():

  if python.config.map != '':
      return python.config.map.get_root().render()
  else: 
      return mapping(fast = False, skip = True)
  

@app.route('/show_map', methods=['GET'])
def show_map():
  f = mapping(fast = False, skip = True)
 
  return f

@app.route('/fast_show_map', methods=['GET'])
def faster_show_map():
  f = mapping(fast = True, skip = True)
 
  return f