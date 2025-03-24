import requests


def test_predict():
    url = "http://127.0.0.1:8080/predict"
    file_path = "./ml/concat_img/15aI4952.png"

    with open(file_path, "rb") as file:
        files = {"file": ("15aI4952.png", file, "image/png")}
        headers = {"accept": "application/json"}

        response = requests.post(url, headers=headers, files=files)
        print(response.json())

        assert response.status_code == 200
        assert "prediction" in response.json()


test_predict()
