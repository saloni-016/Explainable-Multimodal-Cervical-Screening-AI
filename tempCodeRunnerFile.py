from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import sqlite3
from datetime import datetime
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from torchvision.models import vit_b_16

from explain_xgboost import (
    explain_clinical_input,
    create_clinical_explanation
)

app = Flask(__name__)
app.secret_key = "secret123"

UPLOAD_FOLDER = "static/uploads"
EXPLANATION_FOLDER = "static/explanations"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXPLANATION_FOLDER, exist_ok=True)

model = joblib.load("model.pkl")
imputer = joblib.load("imputer.pkl")

MODEL_PATH = "vit_sipakmed.pth"
CLASSES_PATH = "sipakmed_classes.json"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

with open(CLASSES_PATH, "r") as f:
    classes = json.load(f)

vit_model = vit_b_16()

vit_model.heads.head = torch.nn.Linear(
    vit_model.heads.head.in_features,
    len(classes)
)

vit_model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

vit_model = vit_model.to(device)
vit_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def init_db():

    conn = sqlite3.connect("database.db")
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        email TEXT UNIQUE,
        password TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT,
        risk_level TEXT,
        probability REAL,
        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()


init_db()

def generate_vit_explanation(image_path, explanation_filename):

    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    original_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        original_output = vit_model(original_tensor)

        original_probabilities = torch.softmax(
            original_output,
            dim=1
        )

        predicted_index = (
            original_probabilities
            .argmax(dim=1)
            .item()
        )

        original_confidence = (
            original_probabilities[
                0,
                predicted_index
            ].item()
        )

    grid_size = 14
    patch_size = 16

    occluded_images = []

    for row in range(grid_size):

        for col in range(grid_size):

            occluded_image = image.copy()

            pixels = occluded_image.load()

            start_x = col * patch_size
            start_y = row * patch_size

            for y in range(
                start_y,
                min(start_y + patch_size, 224)
            ):

                for x in range(
                    start_x,
                    min(start_x + patch_size, 224)
                ):

                    pixels[x, y] = (0, 0, 0)

            occluded_images.append(
                transform(occluded_image)
            )

    importance_map = np.zeros(
        (grid_size, grid_size)
    )

    batch_size = 16

    for start in range(
        0,
        len(occluded_images),
        batch_size
    ):

        batch = torch.stack(
            occluded_images[
                start:start + batch_size
            ]
        ).to(device)

        with torch.no_grad():

            outputs = vit_model(batch)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            new_confidences = probabilities[
                :,
                predicted_index
            ].cpu().numpy()

        for i, new_confidence in enumerate(
            new_confidences
        ):

            patch_index = start + i

            row = patch_index // grid_size
            col = patch_index % grid_size

            importance_map[
                row,
                col
            ] = (
                original_confidence -
                new_confidence
            )

    positive_map = np.maximum(
        importance_map,
        0
    )

    total_contribution = positive_map.sum()

    if total_contribution > 0:

        contribution_map = (
            positive_map /
            total_contribution
        ) * 100

    else:

        contribution_map = np.zeros_like(
            positive_map
        )

    if positive_map.max() > 0:

        normalized_map = (
            positive_map /
            positive_map.max()
        )

    else:

        normalized_map = positive_map

    heatmap = Image.fromarray(
        np.uint8(
            normalized_map * 255
        )
    )

    heatmap = heatmap.resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    heatmap_array = np.array(
        heatmap
    )

    class_name = classes[
        predicted_index
    ].replace(
        "im_",
        ""
    )

    plt.figure(
        figsize=(8, 8)
    )

    plt.imshow(image)

    plt.imshow(
        heatmap_array,
        alpha=0.5,
        cmap="jet"
    )

    plt.axis("off")

    plt.title(
        f"ViT Explanation: "
        f"{class_name} "
        f"({original_confidence * 100:.2f}%)"
    )

    explanation_path = os.path.join(
        EXPLANATION_FOLDER,
        explanation_filename
    )

    plt.savefig(
        explanation_path,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()

    regions = {

        "Upper-left region":
        contribution_map[0:5, 0:5].sum(),

        "Upper-central region":
        contribution_map[0:5, 5:10].sum(),

        "Upper-right region":
        contribution_map[0:5, 10:14].sum(),

        "Central region":
        contribution_map[5:10, 5:10].sum(),

        "Lower-left region":
        contribution_map[10:14, 0:5].sum(),

        "Lower-central region":
        contribution_map[10:14, 5:10].sum(),

        "Lower-right region":
        contribution_map[10:14, 10:14].sum()
    }

    sorted_regions = sorted(
        regions.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_regions = [
        (name, value)
        for name, value in sorted_regions[:3]
        if value > 0
    ]

    explanation = [

        f"The model predicted the cell as "
        f"{class_name} with "
        f"{original_confidence * 100:.2f}% confidence.",

        "The heatmap shows which image regions "
        "contributed most to this prediction."
    ]

    for region, contribution in top_regions:

        explanation.append(
            f"{region} — "
            f"{contribution:.2f}% contribution "
            f"toward the prediction."
        )

    explanation.append(
        "Red regions represent the strongest "
        "positive contribution, yellow regions "
        "represent moderate contribution, and "
        "blue regions represent lower contribution."
    )

    if top_regions:

        explanation.append(
            f"Therefore, the "
            f"{top_regions[0][0].lower()} "
            f"had the greatest influence on the "
            f"model's {class_name} prediction."
        )

    return {

        "class_name": class_name,

        "confidence": round(
            original_confidence * 100,
            2
        ),

        "explanation": explanation,

        "explanation_image":
        explanation_filename
    }


@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route(
    "/signup",
    methods=["GET", "POST"]
)
def signup():

    if request.method == "POST":

        username = request.form.get(
            "username"
        )

        email = request.form.get(
            "email"
        )

        password = request.form.get(
            "password"
        )

        confirm_password = request.form.get(
            "confirm_password"
        )

        if not username or not email or not password:

            return render_template(
                "signup.html",
                error="Please fill all required fields."
            )

        if password != confirm_password:

            return render_template(
                "signup.html",
                error="Passwords do not match."
            )

        conn = sqlite3.connect(
            "database.db"
        )

        cur = conn.cursor()

        cur.execute(
            "SELECT * FROM users WHERE email=?",
            (email,)
        )

        existing_user = cur.fetchone()

        if existing_user:

            conn.close()

            return render_template(
                "signup.html",
                error="Email already registered!"
            )

        cur.execute("""
        INSERT INTO users (
            username,
            email,
            password
        )
        VALUES (?, ?, ?)
        """, (
            username,
            email,
            password
        ))

        conn.commit()
        conn.close()

        session["email"] = email
        session["username"] = username

        return redirect(
            url_for("choose")
        )

    return render_template(
        "signup.html"
    )


@app.route(
    "/signin",
    methods=["GET", "POST"]
)
def signin():

    if request.method == "POST":

        email = request.form.get(
            "email"
        )

        password = request.form.get(
            "password"
        )

        conn = sqlite3.connect(
            "database.db"
        )

        cur = conn.cursor()

        cur.execute("""
        SELECT *
        FROM users
        WHERE email=? AND password=?
        """, (
            email,
            password
        ))

        user = cur.fetchone()

        conn.close()

        if user:

            session["email"] = email
            session["username"] = user[1]

            return redirect(
                url_for("choose")
            )

        return render_template(
            "signin.html",
            error="Invalid Email or Password!"
        )

    return render_template(
        "signin.html"
    )


@app.route("/choose")
def choose():

    if "email" not in session:

        return redirect(
            url_for("signin")
        )

    return render_template(
        "choose.html"
    )


@app.route("/details")
def details():

    if "email" not in session:

        return redirect(
            url_for("signin")
        )

    return render_template(
        "details.html"
    )


@app.route("/image_upload")
def image_upload():

    if "email" not in session:

        return redirect(
            url_for("signin")
        )

    return render_template(
        "image_upload.html"
    )


@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    if "email" not in session:

        return redirect(
            url_for("signin")
        )

    values = [

        float(request.form.get(
            "Age",
            0
        )),

        float(request.form.get(
            "Number_of_sexual_partners",
            0
        )),

        float(request.form.get(
            "First_sexual_intercourse",
            0
        )),

        float(request.form.get(
            "Smokes",
            0
        )),

        float(request.form.get(
            "Smokes_years",
            0
        )),

        float(request.form.get(
            "Hormonal_Contraceptives",
            0
        )),

        float(request.form.get(
            "IUD",
            0
        )),

        float(request.form.get(
            "STDs",
            0
        )),

        float(request.form.get(
            "STDs_number",
            0
        )),

        float(request.form.get(
            "Dx_Cancer",
            0
        )),

        float(request.form.get(
            "Dx_CIN",
            0
        )),

        float(request.form.get(
            "Dx_HPV",
            0
        ))
    ]

    result = explain_clinical_input(
        values
    )

    explanation_text = create_clinical_explanation(
        result
    )

    probability = result[
        "probability"
    ]

    if probability >= 70:

        risk_level = "High Risk"

        advice = [
            {
                "text":
                "Immediate consultation required"
            },
            {
                "text":
                "Advanced screening needed"
            },
            {
                "text":
                "Do not delay medical care"
            }
        ]

    elif probability >= 40:

        risk_level = "Medium Risk"

        advice = [
            {
                "text":
                "Schedule doctor visit"
            },
            {
                "text":
                "Regular checkups needed"
            },
            {
                "text":
                "Discuss the result with a healthcare professional"
            }
        ]

    else:

        risk_level = "Low Risk"

        advice = [
            {
                "text":
                "Maintain healthy lifestyle"
            },
            {
                "text":
                "Routine screening"
            },
            {
                "text":
                "Continue preventive care"
            }
        ]

    session["risk_level"] = risk_level
    session["prob_percent"] = probability

    session["last_result"] = {

        "risk_level": risk_level,
        "prob_percent": probability,
        "prediction": result["prediction"],
        "advice": advice,
        "explanations": result["explanations"],
        "explanation_text": explanation_text
    }

    return redirect(
        url_for("result")
    )


@app.route("/result")
def result():

    if "last_result" not in session:

        return redirect(
            url_for("details")
        )

    data = session[
        "last_result"
    ]

    return render_template(
        "result.html",
        risk_level=data["risk_level"],
        prob_percent=data["prob_percent"],
        prediction=data["prediction"],
        advice=data["advice"],
        explanations=data["explanations"],
        explanation_text=data["explanation_text"]
    )


@app.route(
    "/predict_image",
    methods=["POST"]
)
def predict_image():

    if "email" not in session:

        return redirect(
            url_for("signin")
        )

    image = request.files.get(
        "scan_image"
    )

    if image is None:

        return "No image uploaded"

    if image.filename == "":

        return "No image selected"

    allowed_extensions = {
        ".bmp",
        ".png",
        ".jpg",
        ".jpeg"
    }

    extension = os.path.splitext(
        image.filename
    )[1].lower()

    if extension not in allowed_extensions:

        return (
            "Please upload a BMP, PNG, JPG "
            "or JPEG cervical-cell image."
        )

    timestamp = datetime.now().strftime(
        "%Y%m%d%H%M%S%f"
    )

    image_filename = (
        timestamp +
        extension
    )

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        image_filename
    )

    image.save(
        image_path
    )

    explanation_filename = (
        "explanation_" +
        timestamp +
        ".png"
    )

    result = generate_vit_explanation(
        image_path,
        explanation_filename
    )

    session["last_image_result"] = {

        "image_file":
        image_filename,

        "class_name":
        result["class_name"],

        "confidence":
        result["confidence"],

        "explanation":
        result["explanation"],

        "explanation_image":
        result["explanation_image"]
    }

    return redirect(
        url_for("image_result")
    )


@app.route("/image_result")
def image_result():

    if "last_image_result" not in session:

        return redirect(
            url_for("image_upload")
        )

    data = session[
        "last_image_result"
    ]

    return render_template(
        "image_result.html",
        image_file=data["image_file"],
        class_name=data["class_name"],
        confidence=data["confidence"],
        explanation=data["explanation"],
        explanation_image=data["explanation_image"]
    )


@app.route(
    "/save_result",
    methods=["GET", "POST"]
)
def save_result():

    if "email" not in session:

        return redirect(
            url_for("signin")
        )

    if "risk_level" not in session:

        return redirect(
            url_for("details")
        )

    conn = sqlite3.connect(
        "database.db"
    )

    cur = conn.cursor()

    cur.execute("""
    INSERT INTO results (
        email,
        risk_level,
        probability,
        created_at
    )
    VALUES (?, ?, ?, ?)
    """, (
        session["email"],
        session["risk_level"],
        session["prob_percent"],
        datetime.now().strftime(
            "%d-%m-%Y %H:%M"
        )
    ))

    conn.commit()
    conn.close()

    return redirect(
        request.referrer or
        url_for("result")
    )


@app.route("/history")
def history():

    if "email" not in session:

        return redirect(
            url_for("signin")
        )

    conn = sqlite3.connect(
        "database.db"
    )

    cur = conn.cursor()

    cur.execute("""
    SELECT
        risk_level,
        probability,
        created_at
    FROM results
    WHERE email=?
    ORDER BY id DESC
    """, (
        session["email"],
    ))

    rows = cur.fetchall()

    conn.close()

    return render_template(
        "history.html",
        rows=rows
    )


@app.route("/compare")
def compare():

    if "email" not in session:

        return redirect(
            url_for("signin")
        )

    conn = sqlite3.connect(
        "database.db"
    )

    cur = conn.cursor()

    cur.execute("""
    SELECT
        risk_level,
        probability,
        created_at
    FROM results
    WHERE email=?
    ORDER BY id DESC
    LIMIT 2
    """, (
        session["email"],
    ))

    rows = cur.fetchall()

    conn.close()

    if len(rows) < 2:

        return render_template(
            "compare.html",
            error="Not enough data to compare"
        )

    latest = rows[0]
    previous = rows[1]

    comparison = {

        "risk_change":
        f"{previous[0]} -> {latest[0]}",

        "prob_change":
        float(latest[1])
        -
        float(previous[1])
    }

    return render_template(
        "compare.html",
        latest=latest,
        previous=previous,
        comparison=comparison
    )


@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("home")
    )


if __name__ == "__main__":

    app.run(
        debug=True
    )