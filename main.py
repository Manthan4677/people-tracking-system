import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import argparse
import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification
import cvlib as cv
from PIL import Image
from keras.models import load_model
from imgbeddings import imgbeddings
import psycopg2
import uuid
from dotenv import load_dotenv

load_dotenv(override=True)

model = load_model("model/gender_detection.model")
ibed = imgbeddings()
classes = ["male", "female"]


FACE_THRESHOLD = 0.06

# Load the ViT Age Detection Model
age_model = ViTForImageClassification.from_pretrained("nateraw/vit-age-classifier")
age_transforms = ViTImageProcessor.from_pretrained("nateraw/vit-age-classifier")

# connecting with postgres database
conn = psycopg2.connect(os.environ["POSTGRES_URI"])
cur = conn.cursor()

age_classes = [
    "0-2",
    "3-9",
    "10-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "more than 70",
]


def gender_detection(face, frame):
    startX, startY, endX, endY = face
    face_crop = np.copy(frame[startY:endY, startX:endX])

    if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
        return "NA"

    # preprocessing
    face_crop = cv2.resize(face_crop, (96, 96))
    face_crop = face_crop.astype("float") / 255.0
    face_crop = np.expand_dims(face_crop, axis=0)

    # apply gender detection on face
    conf = model.predict(face_crop)[0]

    # get label with maximum accuracy
    idx = np.argmax(conf)
    return classes[idx]


def age_detection(face, frame):
    startX, startY, endX, endY = face
    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
    face_crop = np.copy(frame[startY:endY, startX:endX])
    # Using https://huggingface.co/nateraw/vit-age-classifier
    inputs = age_transforms(face_crop, return_tensors="pt")
    output = age_model(**inputs)
    proba = output.logits.softmax(1)
    preds = proba.argmax(1)
    age_group = age_classes[preds[0]]
    return age_group

def createImgbeddings(face, frame, section_name):
    for start_x, start_y, end_x, end_y in face:
        if end_y < start_y + 10 or end_x < start_x + 10:
            continue
        embedding = ibed.to_embeddings(
            Image.fromarray(frame[start_y:end_y, start_x:end_x])
        )[0]
        string_representation = "[" + ",".join(str(x) for x in embedding.tolist()) + "]"
        cur.execute(
            "SELECT id, embedding <=> %s AS distance, gender, age FROM visitors ORDER BY embedding <=> %s LIMIT 1;",
            (string_representation, string_representation),
        )
        rows = cur.fetchall()
        uid = "unknown"
        age = "unknown"
        gender = "unknown"
        if len(rows) == 0:
            uid = str(uuid.uuid4())
            gender = gender_detection((start_x, start_y, end_x, end_y), frame)
            age = age_detection((start_x, start_y, end_x, end_y), frame)
            cur.execute(
                "INSERT INTO visitors (id, embedding, section , gender , age) values (%s,%s,%s,%s,%s)",
                (uid, string_representation, section_name, gender, age),
            )
            print("First Person Inserted:")

        else:
            if rows[0][1] > FACE_THRESHOLD:
                uid = str(uuid.uuid4())
                gender = gender_detection((start_x, start_y, end_x, end_y), frame)
                age = age_detection((start_x, start_y, end_x, end_y), frame)
                cur.execute(
                    "INSERT INTO visitors (id, embedding, section , gender , age) values (%s,%s,%s,%s,%s)",
                    (uid, string_representation, section_name, gender, age),
                )
                print("Unknown Person Detected")
            else:
                uid = rows[0][0]
                gender = rows[0][2]
                age = rows[0][3]
        if uid != "unknown":
            cur.execute(
                "INSERT INTO visit_times (id, section) VALUES (%s, %s)",
                (uid, section_name),
            )
        cv2.putText(
            frame,
            f"Person: {uid[:6]} / {gender} / {age}",
            (start_x, start_y - 20),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (0, 0, 255),
            thickness=2,
        )
        cv2.rectangle(
            frame,
            (start_x, start_y),
            (end_x, end_y),
            (255, 51, 28),
            thickness=1,
        )
    conn.commit()


def process_frame(frame, section_name):
    face_detection_result = cv.detect_face(frame)

    # Check if faces were detected
    if face_detection_result is not None:
        faces, _ = face_detection_result
        createImgbeddings(faces, frame, section_name)


def main():
    parser = argparse.ArgumentParser(
        description="People analyzer"
    )  # Setting up Web-Cam Resolution
    parser.add_argument("section", type=str, help="Section where the camera is located")
    parser.add_argument("--webcam-resolution", default=[640, 480], nargs=2, type=int)
    args = parser.parse_args()
    section_name = args.section

    frame_width, frame_height = args.webcam_resolution

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Initializing object tracker
    while True:
        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            continue
        process_frame(frame, section_name)
        cv2.imshow("People analytics", frame)
        # Detect faces
        if cv2.waitKey(30) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
