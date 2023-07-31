
# Colby Trash Deletion App 🗑️ 

Marine debris presents substantial ecological challenge to the ecosystem of Maine's islands where volunteer groups annually undertake cleanup initiatives on islands. These cleanup efforts hindered by unpredictable challenges in trash volume and placement. In this web app, we created a pipeline that leverages aerial drones and machine learning to automatically detect, classify, and map marine trash.

## Project overview 🔎
* 🗃️ Dataset: Drone footage from Tim Stonesifer & Dr. Whitney King
* 🕵🏻 Trash Detection: [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
* 🖐 Trash Classification: [OpenAI CLIP](https://github.com/openai/CLIP)
* 🚮 Remove Duplicate Trash: [SIFT](https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html)
* 🗺️ Trash Visualization: [Folium Library](https://python-visualization.github.io/folium/)
* 🌐 Web App: HTML, CSS, Javascript, Flask 

For more information, refer to [Presentation](https://docs.google.com/presentation/d/1rI7PSf180x29OTPDD2gRQmT9HyvPoCecKxQgDgZoxtc/edit?usp=sharing). Paper is coming soon! 


## Paper 📄
Still working on it ⚙️


## Demo 

https://github.com/RayWang0328/Colby-Trash-App/assets/19866871/f9c72a17-f664-4ac9-9043-45be41a96a5e

## Local Instillation 🛠️ 

### 0. CPU or GPU
It is advisable to run this program on your local machine only if you have a GPU that is CUDA compatible. The reason for this is because the models used in this app are pretty heavy, computationally. 

If you do have a CUDA compatible GPU, make sure its activated by running this after installing everything below:
```python
>>> python
>>> import torch
>>> torch.cuda.is_available()
```

If CUDA is properly set up, the above should print `True`

### 1. Clone Repository onto a folder local machine: Open Terminal on your machine
```
git clone https://github.com/RayWang0328/Colby-Trash-App.git
```
### 2. Change Directory into folder and activate virtual environment

**For Windows Users:**

```
cd Colby-Trash-App
py -3 -m venv .venv
.venv\Scripts\activate
```

**For Mac Users:**

```
cd Colby-Trash-App
python3 -m venv .venv
. .venv/bin/activate
```

### 3. Install Required Libraries 
```
pip install -r requirements.txt
```
```
python -m pip install scipy
```
### 4. Install GroundingDINO libraries
```
git clone https://github.com/IDEA-Research/GroundingDINO.git
```
```
cd GroundingDINO/
```
Install dependencies:
```
pip install -e .
cd ..
```
### 5. Download GroundingDINO weights

For this step, you will need to have wget installed.
```
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```
### 6. Run Application: 
```
python app.py
```

Go to http://localhost:8000/ to see web app working
    
## Deployment

Alternatively, the app can be deployed as a web app. This will cost money (0.51$/hr), but it can be accessed anywhere. 

Docker Image: 🐋
```javascript
  rayw03/trash-app
```
This is easily achieved through [Paperspace](https://www.paperspace.com). Make an account, create a project, then create a Deployment. 

In the deployment menu, select one of the GPUs as your machine type. Have Image be `rayw03/trash-app` and port be `8000`. Click `Deploy` and you're done! 

If you want to shut it down, change Enabled from `True` to `False`.  





