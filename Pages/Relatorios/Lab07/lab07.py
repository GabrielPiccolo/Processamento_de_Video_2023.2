import cv2
import numpy as np

# Carrega o classificador Haarcascade para detecção de objetos
cascade_path = "haarcascade_frontalface_default.xml"  # Caminho relativo para o arquivo XML

cascade = cv2.CascadeClassifier(cascade_path)

# Função para detecção de objetos azuis na imagem
def detectar_objetos_azuis(imagem):
    # Converte a imagem para o espaço de cores HSV
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

    # Definição das faixas de cor azul
    faixa_azul_clara = np.array([90, 100, 100])
    faixa_azul_escura = np.array([130, 255, 255])

    # Segmenta a imagem utilizando as faixas de cor
    mask = cv2.inRange(hsv, faixa_azul_clara, faixa_azul_escura)

    # Realiza a detecção dos objetos na imagem segmentada
    objetos_detectados = cascade.detectMultiScale(mask)

    return objetos_detectados

# Carrega a imagem
imagem_path = "caneta_azul.JPG"  # Insira o caminho para a imagem que deseja analisar
imagem = cv2.imread(imagem_path)

# Executa a detecção de objetos azuis na imagem
objetos_azuis = detectar_objetos_azuis(imagem)

# Desenha retângulos ao redor dos objetos detectados na imagem original
for (x, y, w, h) in objetos_azuis:
    cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Exibe a imagem com os objetos detectados
cv2.imshow("Detecção de Objetos Azuis", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()