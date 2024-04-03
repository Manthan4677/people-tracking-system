import os
import gdown

if not os.path.exists("model/"):
    print("[INFO] making directory ./model")
    os.mkdir("model")

print("[INFO] downloading model/gender_detection.model")
gdown.download(
    "https://drive.google.com/uc?id=15bs1HHfbVnimIWqy7wuGuZ7aJD6kUKZ6",
    output="model/gender_detection.model",
)
