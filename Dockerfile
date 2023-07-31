FROM python:3.8.5

COPY . .

WORKDIR /app

ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX;8.9;9.0"
WORKDIR /GroundingDINO
RUN pip install -r requirements.txt


RUN pip install -e .

# Go back to app directory
WORKDIR /app

# Install any needed packages specified in requirements.txt
RUN python -m pip install scipy
COPY requirements.txt .
RUN pip install -r requirements.txt

# Installing ffmpeg libsm6 libxext6
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


# Create and enter the weights folder
WORKDIR /app/weights
RUN wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Go back to app directory
WORKDIR /app

COPY . .

# Run the command when the container launches
CMD ["python", "app.py"]