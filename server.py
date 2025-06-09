import httpx
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import matplotlib.pyplot as plt
import csv
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = FastAPI()
# 🔊 Carregar YAMNet
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

# 📄 Baixar labels
# labels_url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
# label_path = tf.keras.utils.get_file("yamnet_class_map.csv", labels_url)

# with open(label_path) as f:
#     reader = csv.reader(f)
#     next(reader)
#     class_names = [row[2] for row in reader]

traducao = [
    'Fala de criança, criança falando', 'Conversa', 'Narração, monólogo', 'Balbucio', 'Sintetizador de voz', 
    'Grito', 'Rugido', 'Grito agudo', 'Berro', 'Crianças gritando', 'Grito alto', 'Sussurro', 'Risada', 
    'Riso de bebê', 'Gargalhada', 'Risadinha', 'Gargalhada alta', 'Riso contido', 'Choro, soluço', 
    'Choro de bebê', 'Choramingar', 'Lamento, gemido', 'Suspiro', 'Canto', 'Coral', 'Iodel', 
    'Canto ritual', 'Mantra', 'Criança cantando', 'Canto sintético', 'Rap', 'Zumbido', 'Gemido', 
    'Grunhido', 'Assobio', 'Respiração', 'Chiado', 'Ronco', 'Suspiro ofegante', 'Ofegar', 'Fungar', 
    'Tosse', 'Limpar a garganta', 'Espirro', 'Fungar', 'Correr', 'Arrastar os pés', 'Caminhar, passos', 
    'Mastigação', 'Mordida', 'Gargarejo', 'Ronco de estômago', 'Arroto', 'Soluço', 'Flatulência', 
    'Mãos', 'Estalar de dedos', 'Palmas', 'Batimentos cardíacos', 'Sopro cardíaco', 'Aplausos', 
    'Palmas', 'Conversa animada', 'Multidão', 'Barulho de fundo, murmúrio', 'Crianças brincando', 
    'Animal', 'Animais domésticos', 'Cão', 'Latido', 'Latido agudo', 'Uivo', 'Latido (onomatopeia)', 
    'Rosnar', 'Choramingar (cão)', 'Gato', 'Ronronar', 'Miar', 'Sibilo', 'Miado alto', 
    'Animais de fazenda', 'Cavalo', 'Tropear', 'Relinchar', 'Gado', 'Mugir', 'Sino de vaca', 
    'Porco', 'Oinc', 'Cabra', 'Bale', 'Ovelha', 'Aves domésticas', 'Galinha, galo', 'Cacarejo', 
    'Cantar do galo', 'Peru', 'Gorgolejo', 'Pato', 'Grasnar', 'Ganso', 'Grasnar', 'Animais selvagens', 
    'Felinos rugindo', 'Rugido', 'Pássaro', 'Canto de pássaro', 'Piar', 'Grasnar', 'Pombo, rolinha', 
    'Arrulhar', 'Corvo', 'Grasnar', 'Coruja', 'Piar', 'Bater de asas', 'Canídeos', 'Roedores', 
    'Rato', 'Passos leves', 'Inseto', 'Grilo', 'Mosquito', 'Mosca', 'Zumbido', 'Abelha, vespa', 
    'Sapo', 'Coaxar', 'Cobra', 'Chocalhar', 'Canto de baleia', 'Música', 'Instrumento musical', 
    'Instrumento de cordas dedilhadas', 'Violão', 'Guitarra elétrica', 'Baixo', 'Violão acústico', 
    'Guitarra slide', 'Tapping', 'Dedilhar', 'Banjo', 'Sitar', 'Bandolim', 'Cítara', 'Ukulele', 
    'Teclado', 'Piano', 'Piano elétrico', 'Órgão', 'Órgão eletrônico', 'Órgão Hammond', 
    'Sintetizador', 'Sampler', 'Cravo', 'Percussão', 'Bateria', 'Caixa de ritmos', 'Tambor', 
    'Caixa', 'Rimshot', 'Rufar', 'Bumbo', 'Tímpano', 'Tabla', 'Prato', 'Chimbal', 'Bloco de madeira', 
    'Pandeiro', 'Chocalho', 'Maraca', 'Gongo', 'Sinos tubulares', 'Percussão de baquetas', 
    'Marimba, xilofone', 'Glockenspiel', 'Vibrafone', 'Pan de aço', 'Orquestra', 'Metais', 
    'Trompa', 'Trompete', 'Trombone', 'Cordas friccionadas', 'Seção de cordas', 'Violino', 
    'Pizzicato', 'Violoncelo', 'Contrabaixo', 'Instrumentos de sopro', 'Flauta', 'Saxofone', 
    'Clarinete', 'Harpa', 'Sino', 'Sino de igreja', 'Sino de natal', 'Sino de bicicleta', 
    'Diapasão', 'Carilhão', 'Sino de vento', 'Repique de sinos', 'Gaita', 'Sanfona', 'Gaita de foles', 
    'Didgeridoo', 'Shofar', 'Theremin', 'Tigela cantante', 'Scratching', 'Música pop', 'Hip hop', 
    'Beatbox', 'Rock', 'Heavy metal', 'Punk rock', 'Grunge', 'Rock progressivo', 'Rock and roll', 
    'Rock psicodélico', 'Rhythm and blues', 'Soul', 'Reggae', 'Música country', 'Swing', 
    'Bluegrass', 'Funk', 'Música folclórica', 'Música do Oriente Médio', 'Jazz', 'Disco', 
    'Música clássica', 'Ópera', 'Música eletrônica', 'House', 'Techno', 'Dubstep', 'Drum and bass', 
    'Eletrônica', 'Música eletrônica dançante', 'Música ambiente', 'Trance', 'Música latina', 
    'Salsa', 'Flamenco', 'Blues', 'Música infantil', 'New age', 'Música vocal', 'A capella', 
    'Música africana', 'Afrobeat', 'Música cristã', 'Gospel', 'Música asiática', 'Música carnática', 
    'Música de Bollywood', 'Ska', 'Música tradicional', 'Música independente', 'Canção', 
    'Música de fundo', 'Música tema', 'Jingle', 'Trilha sonora', 'Canção de ninar', 
    'Música de videogame', 'Música natalina', 'Música dançante', 'Música de casamento', 
    'Música alegre', 'Música triste', 'Música suave', 'Música animada', 'Música irritada', 
    'Música assustadora', 'Vento', 'Farfalhar de folhas', 'Ruído de vento (microfone)', 
    'Tempestade', 'Trovão', 'Água', 'Chuva', 'Gota de chuva', 'Chuva na superfície', 'Riacho', 
    'Cachoeira', 'Oceano', 'Ondas', 'Vapor', 'Borbulhar', 'Fogo', 'Estalar', 'Veículo', 
    'Barco', 'Veleiro', 'Canoa', 'Lancha', 'Navio', 'Veículo motorizado', 'Carro', 'Buzina', 
    'Tut', 'Alarme de carro', 'Vidro elétrico', 'Derrapagem', 'Pneu cantando', 'Carro passando', 
    'Carro de corrida', 'Caminhão', 'Freio a ar', 'Buzina de caminhão', 'Bipe de ré', 
    'Carro de sorvete', 'Ônibus', 'Veículo de emergência', 'Carro de polícia', 'Ambulância', 
    'Caminhão de bombeiros', 'Moto', 'Ruído de trânsito', 'Transporte ferroviário', 'Trem', 
    'Apito de trem', 'Buzina de trem', 'Vagão', 'Ranger de trem', 'Metrô', 'Aeronave', 
    'Motor de avião', 'Motor a jato', 'Hélice', 'Helicóptero', 'Avião', 'Bicicleta', 'Skate', 
    'Motor', 'Motor pequeno', 'Motor de dentista', 'Cortador de grama', 'Motosserra', 
    'Motor médio', 'Motor grande', 'Batida de motor', 'Motor ligando', 'Marcha lenta', 
    'Acelerar', 'Porta', 'Campainha', 'Ding-dong', 'Porta deslizante', 'Porta batendo', 
    'Bater na porta', 'Bater levemente', 'Ranger', 'Abrir/fechar armário', 'Abrir/fechar gaveta', 
    'Pratos, panelas', 'Talheres', 'Cortar comida', 'Fritar', 'Micro-ondas', 'Liquidificador', 
    'Torneira', 'Encher pia', 'Encher banheira', 'Secador de cabelo', 'Descarga', 'Escova de dente', 
    'Escova elétrica', 'Aspirador', 'Zíper', 'Chaveiro', 'Moeda caindo', 'Tesoura', 'Barbeador', 
    'Embaralhar cartas', 'Digitação', 'Máquina de escrever', 'Teclado', 'Escrever', 'Alarme', 
    'Telefone', 'Telefone tocando', 'Toque', 'Discagem', 'Tom de discagem', 'Ocupado', 
    'Despertador', 'Sirene', 'Sirene de alerta', 'Buzina', 'Detector de fumaça', 'Alarme de incêndio', 
    'Sirene de nevoeiro', 'Apito', 'Apito de vapor', 'Mecanismos', 'Catraca', 'Relógio', 
    'Tique', 'Tique-taque', 'Engrenagens', 'Polias', 'Máquina de costura', 'Ventilador', 
    'Ar-condicionado', 'Caixa registradora', 'Impressora', 'Câmera', 'Câmera DSLR', 'Ferramentas', 
    'Martelo', 'Britadeira', 'Serra', 'Lixa', 'Lixar', 'Ferramenta elétrica', 'Furadeira', 
    'Explosão', 'Tiro', 'Metralhadora', 'Rajada', 'Artilharia', 'Arma de brinquedo', 
    'Fogos de artifício', 'Foguete', 'Estouro', 'Erupção', 'Estrondo', 'Madeira', 'Talhar', 
    'Estilhaço', 'Rachadura', 'Vidro', 'Tinir', 'Estilhaçar', 'Líquido', 'Respingar', 
    'Agitar', 'Esmagar', 'Pingar', 'Despejar', 'Gotejar', 'Jorrar', 'Encher', 'Spray', 
    'Bomba', 'Mexer', 'Ferver', 'Sonar', 'Flecha', 'Zunir', 'Baque', 'Tum', 'Sintonizador', 
    'Pedal de efeito', 'Efeito chorus', 'Quicar bola', 'Explodir', 'Tapa', 'Golpe', 'Esmagar', 
    'Quebrar', 'Quicar', 'Chicotear', 'Bater asa', 'Arranhar', 'Raspar', 'Esfregar', 'Rolar', 
    'Esmagar', 'Amassar', 'Rasgar', 'Bip', 'Ping', 'Tinido', 'Clang', 'Guincho', 'Ranger', 
    'Farfalhar', 'Zumbido', 'Barulho', 'Chiado', 'Clicar', 'Click-clack', 'Estrondo', 
    'Pluft', 'Tinir', 'Zumbido', 'Zunir', 'Boing', 'Triturar', 'Silêncio', 'Onda senoidal', 
    'Harmônico', 'Tom de chirp', 'Efeito sonoro', 'Pulso', 'Ambiente interno pequeno', 
    'Ambiente interno grande', 'Espaço público', 'Externo urbano', 'Externo rural', 
    'Reverberação', 'Eco', 'Ruído', 'Ruído ambiental', 'Ruído estático', 'Zumbido elétrico', 
    'Distorção', 'Sidetone', 'Cacofonia', 'Ruído branco', 'Ruído rosa', 'Latejar', 'Vibração', 
    'Televisão', 'Rádio', 'Gravação de campo'
]


@app.post("/")
async def is_talk(files: List[UploadFile] = File(...)):
    if not files:
        return {"error": "Nenhum arquivo enviado."}

    audio = files[0]
    audio_bytes = await audio.read()
    audio_buffer = io.BytesIO(audio_bytes)

    print(audio)
    try:
        waveform, sr = librosa.load(audio_buffer, sr=16000)
    except Exception as e:
        return {"error": f"Erro ao carregar áudio: {str(e)}"}
    
    scores, embeddings, spectrogram = yamnet_model(waveform)

    mean_scores = tf.reduce_mean(scores, axis=0)
    top_class = tf.argmax(mean_scores)
    top_score = mean_scores[top_class].numpy()
    top_label = traducao[top_class]

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
    "Fala", 
    "Fala de criança, criança falando", 
    "Conversa", 
    "Narração, monólogo",
    "Balbucio", 
    "Sintetizador de voz", 
    "Conversa animada", 
    "Barulho de fundo, murmúrio",
    "Canto", 
    "Criança cantando", 
    "Canto sintético",
    "Rap", 
    "Sussurro"
    ]

    if top_label in speech_related:
        print("\n🗣️ Detecção é de fala/canto, chamando a api de transcrição...")
        async with httpx.AsyncClient() as client:
            res = await client.post('https://transcription.zafiras.com.br/transcribe',
            files={'files': (audio.filename, io.BytesIO(audio_bytes), audio.content_type)},
            timeout=30.0
        )
            
        data = res.json()["results"][0]
        print(data)
        return {
            "categoria": top_label,
            "message": data["transcript"],  
            "filename": data["filename"]   
        }
    else:
        return {
            "categoria": top_label,
            "error": "🔇 Áudio não classificado como fala, não será transcrito.",
            }