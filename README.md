# GCP Handwritten Digits Recognizer

A lightweight and scalable handwritten digit recognition service built with FastAPI, designed for real-time inference. This service leverages a trained deep learning model to recognize handwritten digits from input images and is efficiently deployed on **Google Cloud Run**, ensuring seamless scalability and low-latency predictions.

## Installation

Install dependencies:
```sh
pip3 install -r requirements.txt
```
For development:
```sh
pip3 install -r requirements-dev.txt
```

---

## Model Training

Train the model using the EMNIST dataset:
```sh
cd ./ml && python3 train.py \
    --data ./data \
    --checkpoint_dir ./work_dirs/exp1 \
    --max_epochs 50 \
    --batch_size 256 \
    --num_workers 16
```
Note: This command will **automatically download the EMNIST dataset** if it is not already present in the specified --data directory.

## Model Testing & Inference

### Run a Prediction
Predict on a single image:
```sh
cd ./ml && python3 predict.py \
    --checkpoint emnist_digits_recognizer.ckpt \
    --image ./concat_img/7V900Ii.png
```

### Test on EMNIST Test Set
Evaluate the trained model:
```sh
cd ./ml && python3 test.py \
    --checkpoint emnist_digits_recognizer.ckpt \
    --data ./data
```

---

## Model Performance

**Checkpoint:** `ml/emnist_digits_recognizer.ckpt`

| Metric               | Value  |
|----------------------|--------|
| **Test Accuracy**    | 0.871  |
| **Model Size (MB)**  | 4.914  |
| **Number of Params** | 428M   |

---

## Deployment

### Build Docker Image
Build an amd64-compatible Docker image:
```sh
docker buildx build --platform linux/amd64 --no-cache -t digits-fastapi-app-amd64 .
```
> Google Cloud Run currently supports **Linux/amd64** only.

### Push to Google Artifact Registry
Tag and push the image:
```sh
docker tag digits-fastapi-app-amd64 \
    us-central1-docker.pkg.dev/digits-454618/digits-fastapi-app-amd64/digits-fastapi-app-amd64:latest

docker push us-central1-docker.pkg.dev/digits-454618/digits-fastapi-app-amd64/digits-fastapi-app-amd64
```
> Customize the **GCP location**, **project ID**, and **Docker image name** as needed.

### Deploy on Google Cloud Run
Deploy the service:
```sh
gcloud run services replace ./gcp/service.yaml --region us-central1
gcloud run services set-iam-policy digits-fastapi-app-amd64 ./gcp/gcr-service-policy.yaml --region us-central1
```
> Ensure the **region name** matches your settings.

Check the deployment at [Google Cloud Run Console](https://console.cloud.google.com/run?project=digits-454618).

---

## Usage

Send a digit image for recognition:
```sh
curl -X 'POST' 'https://digits-fastapi-app-amd64-124569006945.us-central1.run.app/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@./ml/concat_img/15aI4952.png'
```

For more details, visit the [Wiki Page](https://github.com/hsjimwang/GCP-Handwritten-Digits-Recognizer/wiki).

