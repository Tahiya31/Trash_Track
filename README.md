
# Sustainable Marine Debris Clean-Up App 🗑️ 

Marine debris presents substantial ecological challenge to the ecosystem of coastal islands where volunteer groups annually undertake cleanup initiatives on islands. These cleanup efforts hindered by unpredictable challenges in trash volume and position that requires monitoring,
coordination, collection, and removal efforts. In this web app, we created a pipeline that leverages aerial drones and machine learning to automatically detect, classify, and map marine trash.

## Project overview 🔎
* 🗃️ Dataset: Drone footage from Tim Stonesifer & Dr. Whitney King
* 🕵🏻 Trash Detection: [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
* 🖐 Trash Classification: [OpenAI CLIP](https://github.com/openai/CLIP)
* 🚮 Remove Duplicate Trash: [SIFT](https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html)
* 🗺️ Trash Visualization: [Folium Library](https://python-visualization.github.io/folium/)
* 🌐 Web App: HTML, CSS, Javascript, Flask 

For more information, refer to our paper which will be available soon.

### [Press Highlights][https://colbyecho.news/2024/04/19/ai-on-the-island/]


## Paper 📄
Raymond Wang, Tahiya Chowdhury, Nicholas R. Record, and D. Whitney King. 2018. Designing A Sustainable Marine Debris Clean-up Framework without Human Labels. In Proceedings of ACM SIGCAS/SIGCHI
Conference on Computing and Sustainable Societies (ACM COMPASS ’24), ACM, New York, NY, USA.

### Abstract

Marine debris poses a significant ecological threat to birds, fish, andother animal life. Traditional methods for assessing debris accumulation involve labor-intensive and costly manual surveys. This study introduces a framework that utilizes aerial imagery captured
by drones to conduct remote trash surveys. Leveraging computer vision techniques, our approach detects, classifies, and maps marine debris distributions. The framework uses Grounding DINO, a transformer-based zero-shot object detector, and CLIP, a vision-language model for zero-shot object classification, enabling the detection and classification of debris objects based on material type without the need for training labels. To mitigate over-counting due to different views of the same object, Scale-Invariant Feature Transform (SIFT) is employed for duplicate matching using local object features. Additionally, we have developed a user-friendly web application that facilitates end-to-end analysis of drone images, including object detection, classification, and visualization on a map to support cleanup efforts. Our method achieves competitive performance in detection (0.69 mean IoU) and classification (0.74 F1 score) across seven debris object classes without labeled data, comparable to state-of-the-art supervised methods. This framework has the potential to streamline automated trash sampling surveys, fostering efficient and sustainable community-led cleanup initiatives.

## Demo 

https://github.com/RayWang0328/Colby-Trash-App/assets/19866871/f9c72a17-f664-4ac9-9043-45be41a96a5e

## Local Installation 🛠️ 

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

In the deployment menu, select one of the GPUs as your machine type. For Image, select Public container register, and have the Image be `rayw03/trash-app` and port be `8000`. Click `Deploy` and you're done! 

Paperspace charges by the hour, so remember to shut it down when you're not using it!





