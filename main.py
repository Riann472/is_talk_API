# 🚀 Instalar dependências
# !pip install tensorflow tensorflow-hub librosa matplotlib openai-whisper ffmpeg-python -q

# 📥 Importar pacotes
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import matplotlib.pyplot as plt
import whisper
import csv
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# 🔊 Carregar YAMNet
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

# 📄 Baixar labels
# labels_url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
# label_path = tf.keras.utils.get_file("yamnet_class_map.csv", labels_url)

# Carregar labels
# with open(label_path) as f:
#     reader = csv.reader(f)
#     next(reader)
#     for row in reader:
#         print(row)
#     class_names = [row[2] for row in reader]

traducao = ['Fala', 'Fala infantil, criança falando', 'Conversa', 'Narração, monólogo', 'Balbucio', 'Sintetizador de fala', 'Grito', 'Berro', 'Grito forte', 'Gritar', 'Crianças gritando', 'Gritando', 'Sussurro', 'Riso', 'Riso de bebê', 'Risada', 'Riso contido', 'Riso alto', 'Risada abafada', 'Choro, soluço', 'Choro de bebê, choro infantil', 'Choramingar', 'Lamentar, gemer', 'Suspiro', 'Canto', 'Coral', 'Yodel', 'Cântico', 'Mantra', 'Criança cantando', 'Canto sintético', 'Rap', 'Cantarolar', 'Gemido', 'Grunhido', 'Assobio', 'Respiração', 'Chiado', 'Ronco', 'Ofegar', 'Ofegar (arfar)', 'Resmungo', 'Limpar a garganta', 'Espirro', 'Cheirar', 'Correr', 'Arrastar os pés', 'Andar, passos', 'Mastigar', 'Morder', 'Gargarejo', 'Barulho do estômago', 'Arrotar', 'Soltar pum', 'Mãos', 'Estalar os dedos', 'Palmas', 'Sons do coração, batimento cardíaco', 'Sopro cardíaco', 'Torcer', 'Aplausos', 'Tagarelice', 'Multidão', 'Barulho confuso, barulho de fala', 'Crianças brincando', 'Animal', 'Animais domésticos, pets', 'Cachorro', 'Latido', 'Latido agudo', 'Uivo', 'Au-au', 'Rosnado', 'Chilrear (cachorro)', 'Gato', 'Ronronar', 'Miar', 'Chiado', 'Grito estridente', 'Animais de fazenda, animais de trabalho', 'Cavalo', 'Trote', 'Relincho', 'Gado', 'Mugido', 'Sino de vaca', 'Porco', 'Grunhido', 'Bode', 'Balido', 'Ovelha', 'Aves', 'Galinha, galo', 'Cocoricó', 'Canto do galo', 'Peru', 'Gluglutar', 'Pato', 'Quack', 'Ganso', 'Grunhido', 'Animais selvagens', 'Gatos rugindo (leões, tigres)', 'Rugido', 'Pássaro', 'Vocalização de pássaro, chamada de pássaro, canto de pássaro', 'Piar', 'Grito agudo', 'Pombo, rola', 'Coo', 'Corvo', 'Graxnado', 'Coruja', 'Gurro', 'Voo de pássaro, batidas das asas', 'Canídeos, cães, lobos', 'Roedores, ratos, camundongos', 'Camundongo', 'Patar', 'Inseto', 'Grilo', 'Mosquito', 'Mosca', 'Zumbido', 'Abelha, vespa, etc.', 'Sapo', 'Coaxar', 'Cobra', 'Chocalho', 'Vocalização de baleia', 'Música', 'Instrumento musical', 'Instrumento de cordas dedilhadas', 'Guitarra', 'Guitarra elétrica', 'Baixo', 'Guitarra acústica', 'Guitarra steel, guitarra slide', 'Técnica de dedilhado (guitarra)', 'Dedilhar', 'Banjo', 'Sitar', 'Bandolim', 'Zítero', 'Ukulele', 'Teclado (musical)', 'Piano', 'Piano elétrico', 'Órgão', 'Órgão eletrônico', 'Órgão Hammond', 'Sintetizador', 'Sampler', 'Cravo', 'Percussão', 'Bateria', 'Bateria eletrônica', 'Tambor', 'Caixa', 'Rimshot', 'Rolar tambor', 'Bombo', 'Tímpanos', 'Tabla', 'Prato', 'Hi-hat', 'Bloco de madeira', 'Tamborim', 'Chocalho (instrumento)', 'Maraca', 'Gongo', 'Sinos tubulares', 'Percussão com malhete', 'Marimba, xilofone', 'Glockenspiel', 'Vibrafone', 'Steelpan', 'Orquestra', 'Instrumento de metal', 'Trompa', 'Trompete', 'Trombone', 'Instrumento de cordas com arco', 'Seção de cordas', 'Violino, fiddle', 'Pizzicato', 'Violoncelo', 'Contrabaixo', 'Instrumento de sopro, madeiras', 'Flauta', 'Saxofone', 'Clarinete', 'Harpa', 'Sino', 'Sino de igreja', 'Sino de guizo', 'Sino de bicicleta', 'Diapasão', 'Sino musical', 'Carrilhão e sinos', 'Harmônica', 'Acordeão', 'Gaita de foles', 'Didgeridoo', 'Shofar', 'Theremin', 'Tigela cantadora', 'Raspar (técnica de performance)', 'Música pop', 'Hip hop', 'Beatboxing', 'Rock', 'Heavy metal', 'Punk rock', 'Grunge', 'Rock progressivo', 'Rock and roll', 'Rock psicodélico', 'Ritmo e blues', 'Soul', 'Reggae', 'Country', 'Swing', 'Bluegrass', 'Funk', 'Folk', 'Música do Oriente Médio', 'Jazz', 'Disco', 'Música clássica', 'Ópera', 'Música eletrônica', 'House', 'Techno', 'Dubstep', 'Drum and bass', 'Eletrônica', 'Música eletrônica de dança', 'Música ambiente', 'Trance', 'Música da América Latina', 'Salsa', 'Flamenco', 'Blues', 'Música infantil', 'New-age', 'Música vocal', 'A capella', 'Música da África', 'Afrobeat', 'Música cristã', 'Gospel', 'Música da Ásia', 'Carnatic', 'Música de Bollywood', 'Ska', 'Música tradicional', 'Música independente', 'Canção', 'Música de fundo', 'Tema musical', 'Jingle', 'Trilha sonora', 'Canção de ninar', 'Música de videogame', 'Música de Natal', 'Música de dança', 'Música de casamento', 'Música alegre', 'Música triste', 'Música suave', 'Música animada', 'Música zangada', 'Música assustadora', 'Vento', 'Folhas farfalhando', 'Ruído de vento (microfone)', 'Tempestade', 'Trovão', 'Água', 'Chuva', 'Gota de chuva', 'Chuva na superfície', 'Córrego', 'Cachoeira', 'Oceano', 'Ondas, surf', 'Vapor', 'Borbulhar', 'Fogo', 'Crepitar', 'Veículo', 'Barco, veículo aquático', 'Veleiro', 'Canoa, caiaque, bote', 'Lancha, barco a motor', 'Navio', 'Veículo motorizado (estrada)', 'Carro', 'Buzina de carro', 'Buzinar', 'Alarme de carro', 'Vidros elétricos', 'Derrapagem', 'Chilrear de pneus', 'Carro passando', 'Carro de corrida', 'Caminhão', 'Freio a ar', 'Buzina de caminhão', 'Bips de ré', 'Carrinho de sorvete', 'Ônibus', 'Veículo de emergência', 'Carro de polícia (sirene)', 'Ambulância (sirene)', 'Caminhão de bombeiros (sirene)', 'Moto', 'Ruído de trânsito', 'Transporte ferroviário', 'Trem', 'Apito de trem', 'Buzina de trem', 'Vagão de trem', 'Rangido de rodas de trem', 'Metrô', 'Aeronave', 'Motor de avião', 'Motor a jato', 'Hélice', 'Helicóptero', 'Avião', 'Bicicleta', 'Skate', 'Motor', 'Motor leve (alta frequência)', 'Broca dentária', 'Cortador de grama', 'Motosserra', 'Motor médio (frequência média)', 'Motor pesado (baixa frequência)', 'Batida do motor', 'Motor ligando', 'Motor em marcha lenta', 'Acelerando, rugindo', 'Porta', 'Campainha', 'Ding-dong', 'Porta deslizante', 'Bater porta', 'Batida', 'Toque', 'Rangeido', 'Armário abrindo ou fechando', 'Gaveta abrindo ou fechando', 'Louça, panelas', 'Talheres', 'Cortar (comida)', 'Fritar (comida)', 'Micro-ondas', 'Liquidificador', 'Torneira', 'Pia (enchendo ou lavando)', 'Banheira (enchendo ou lavando)', 'Secador de cabelo', 'Descarga', 'Escova de dentes', 'Escova de dentes elétrica', 'Aspirador de pó', 'Zíper (roupas)', 'Chaves tilintando', 'Moeda caindo', 'Tesoura', 'Barbeador elétrico', 'Embaralhar cartas', 'Digitar', 'Máquina de escrever', 'Teclado de computador', 'Escrever', 'Alarme', 'Telefone', 'Telefone tocando', 'Toque do telefone', 'Discagem do telefone', 'Tom de discagem', 'Sinal de ocupado', 'Despertador', 'Sirene', 'Sirene civil', 'Buzina', 'Detector de fumaça', 'Alarme de incêndio', 'Buzina de nevoeiro', 'Apito', 'Apito a vapor', 'Mecanismos', 'Catraca', 'Relógio', 'Tique-taque', 'Engrenagens', 'Polias', 'Máquina de costura', 'Ventilador mecânico', 'Ar-condicionado', 'Caixa registradora', 'Impressora', 'Câmera', 'Câmera reflex', 'Ferramentas', 'Martelo', 'Marreta', 'Serrar', 'Lixar', 'Lixar com lima', 'Ferramenta elétrica', 'Furadeira', 'Explosão', 'Tiro', 'Metralhadora', 'Rajada', 'Fogo de artilharia', 'Arma de brinquedo', 'Fogos de artifício', 'Estouro', 'Erupção', 'Bum', 'Madeira', 'Cortar', 'Lasca', 'Estalo', 'Vidro', 'Barulho de vidro', 'Estilhaçar', 'Líquido', 'Borrifar', 'Agitar', 'Chacoalhar', 'Pingando', 'Derramando', 'Gotejando', 'Jorrar', 'Encher (com líquido)', 'Borrifar', 'Bombear (líquido)', 'Mexer', 'Fervendo', 'Sonar', 'Flecha', 'Whoosh', 'Batida', 'Tombar', 'Afinador eletrônico', 'Unidade de efeitos', 'Efeito coro', 'Quicar bola de basquete', 'Bang', 'Tapa', 'Bater forte', 'Esmagar', 'Quebrar', 'Quicar', 'Chicote', 'Bater asas', 'Arranhar', 'Raspar', 'Esfregar', 'Rolagem', 'Esmagar', 'Amassar', 'Rasgar', 'Bip', 'Ping', 'Ding', 'Estrondo', 'Grito', 'Gritar', 'Rangeido', 'Farfalhar', 'Zumbido', 'Barulho de metal', 'Sibilo', 'Clicar', 'Barulho de carrinho', 'Rugido', 'Plop', 'Sino', 'Zumbido', 'Zing', 'Boing', 'Mastigar', 'Silêncio', 'Onda senoidal', 'Harmônico', 'Tom de pio', 'Efeito sonoro', 'Pulso', 'Interior, sala pequena', 'Interior, salão grande', 'Interior, espaço público', 'Exterior, urbano ou artificial', 'Exterior, rural ou natural', 'Reverberação', 'Eco', 'Ruído', 'Ruído ambiental', 'Zumbido da rede elétrica', 'Distorção', 'Sidetone', 'Cacofonia', 'Ruído branco', 'Ruído rosa', 'Pulsação', 'Vibração', 'Televisão', 'Rádio', 'Gravação de campo']

# 🎧 Upload de áudio
# uploaded = files.upload()
# filename = list(uploaded.keys())[0]

filename = "audios/audio.ogg"


# 🎙️ Processar o áudio
waveform, sr = librosa.load(filename, sr=16000)
scores, embeddings, spectrogram = yamnet_model(waveform)

# 📊 Resultado
mean_scores = tf.reduce_mean(scores, axis=0)
top_class = tf.argmax(mean_scores)
top_score = mean_scores[top_class].numpy()
top_label = traducao[top_class]



print(f"\n🔍 Detecção: {top_label}")
print(f"📈 Confiança: {top_score:.2f}")
# 📉 Exibir gráfico
# plt.figure(figsize=(10, 4))
# plt.plot(waveform)
# plt.title(f"Áudio: {filename}")
# plt.xlabel("Amostras")
# plt.ylabel("Amplitude")
# plt.show()

# 🧠 Verificar se é voz/fala/canto e transcrever
speech_related = ['Fala', 'Fala infantil, criança falando', 'Conversa', 'Narrrativa, monólogo', 'Balbucio', 'Sintetizador de fala', 'Tagare lice', 'Barulho confuso, fala indistinta', 'Canto', 'Criança cantando', 'Canto sintético', 'Rap', 'Sussurro']

if top_label in speech_related:
    print("\n🗣️ Detecção é de fala/canto, iniciando transcrição com Whisper...")

    # model = whisper.load_model(
    #     "base"
    # )  # você pode trocar por 'small' ou 'tiny' para ser mais rápido
    # result = model.transcribe(filename, language="pt")

    # print("\n📝 Texto detectado:")
    # print(result["text"])
else:
    print("\n🔇 Áudio não classificado como fala, não será transcrito.")
