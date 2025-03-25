# GCP-Handwritten-Digits-Recognizer
A FastAPI-based handwritten digit recognition service deployed on Google Cloud Run.

## Install dependency
```
pip3 install -r requirements.txt

# For developers
pip3 install -r requirements-dev.txt
```

## Train the model
```
cd ./ml && python3 train.py --data ./data --checkpoint_dir ./work_dirs/exp1 --max_epochs 50 --batch_size 256 --num_workers 16
```

## Predict the model
```
cd ./ml && python3 predict.py --checkpoint emnist_digits_recognizer.ckpt --image ./concat_img/7V900Ii.png
```

## Build docker image
```
docker buildx build --platform linux/amd64 --no-cache -t digits-fastapi-app-amd64 .
```
* To deploy on Google Cloud Run, it only supports on Linux/amd64 currently.

## Tag and push docker image to Google Artifact Registry
```
docker tag digits-fastapi-app-amd64 us-central1-docker.pkg.dev/digits-454618/digits-fastapi-app-amd64/digits-fastapi-app-amd64:latest
docker push us-central1-docker.pkg.dev/digits-454618/digits-fastapi-app-amd64/digits-fastapi-app-amd64
```
* Note: You can customize the GCP location, project id, and docker image name.

## Deploy service on Google Cloud Run
```
gcloud run services replace ./gcp/service.yaml --region us-central1
gcloud run services set-iam-policy digits-fastapi-app-amd64 ./gcp/gcr-service-policy.yaml --region us-central1
```
Note: The location name must be identical to the above item.

Go to https://console.cloud.google.com/run?inv=1&invt=Abs2ew&project=digits-454618 to check the running service and also check the url.

## Usage
```
curl -X 'POST' 'https://digits-fastapi-app-amd64-124569006945.us-central1.run.app/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@./ml/concat_img/15aI4952.png' 
```

For more details, please refer to: [Wiki Page](https://github.com/hsjimwang/GCP-Handwritten-Digits-Recognizer/wiki).
