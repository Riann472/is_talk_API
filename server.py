from flask import Flask, jsonify, request
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import matplotlib.pyplot as plt
import csv
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = Flask(__name__)
# 🔊 Carregar YAMNet
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

# 📄 Baixar labels
labels_url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
label_path = tf.keras.utils.get_file("yamnet_class_map.csv", labels_url)

with open(label_path) as f:
    reader = csv.reader(f)
    next(reader)
    class_names = [row[2] for row in reader]


@app.route("/", methods=["POST"])
def is_talk():
    audio = request.files["audio"]
    waveform, sr = librosa.load(audio, sr=16000)

    scores, embeddings, spectrogram = yamnet_model(waveform)

    mean_scores = tf.reduce_mean(scores, axis=0)
    top_class = tf.argmax(mean_scores)
    top_score = mean_scores[top_class].numpy()
    top_label = class_names[top_class]

    print(f"\n🔍 Detecção: {top_label}")
    print(f"📈 Confiança: {top_score:.2f}")

    # VER GRAFICO DO AUDIO
    # plt.figure(figsize=(10, 4))
    # plt.plot(waveform)
    # plt.title(f"Áudio: {audio.filename}")
    # plt.xlabel("Amostras")
    # plt.ylabel("Amplitude")
    # plt.show()

    # 🧠 Verificar se é voz/fala/canto e transcrever
    speech_related = [
        "Speech",
        "Child speech, kid speaking",
        "Conversation",
        "Narration, monologue",
        "Babbling",
        "Speech synthesizer",
        "Chatter",
        "Hubbub, speech noise, speech babble",
        "Singing",
        "Child singing",
        "Synthetic singing",
        "Rapping",
        "Whispering",
    ]

    if top_label in speech_related:
        print("\n🗣️ Detecção é de fala/canto, chamando a api de transcrição...")
        # CHAMAR A API DE TRANSCRIÇÃO A PARTIR DAQUI
        return jsonify({"categoria": top_label, "message": "transcrição"})
    else:
        return jsonify(
            {
                "categoria": top_label,
                "error": "🔇 Áudio não classificado como fala, não será transcrito.",
            }
        )


app.run(debug=True)
