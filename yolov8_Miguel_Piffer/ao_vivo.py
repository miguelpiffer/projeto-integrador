import cv2
from pytorchyolo import Detector

# Definir o caminho para o arquivo best.pt treinado por você
weights_file = '/Users/miguelpiffer/Desktop/estudos/projeto integrador/runs/detect/train7/weights/best.pt'
class_names_file = '/Users/miguelpiffer/Desktop/estudos/projeto integrador/classe.txt'  # Arquivo com os nomes das classes

# Inicializar o detector
detector = Detector(weights_file, class_names_file)

# Inicializar a webcam
camera = cv2.VideoCapture(0)

while True:
    # Capturar o frame da webcam
    ret, frame = camera.read()

    # Executar a detecção
    detections = detector.detect(frame)

    # Desenhar as caixas delimitadoras e rótulos nas detecções
    for detection in detections:
        class_name, confidence, bbox = detection
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'{class_name}: {confidence:.2f}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Mostrar o frame com as detecções
    cv2.imshow('Object Detection', frame)

    # Sair se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a webcam e fechar as janelas
camera.release()
cv2.destroyAllWindows()
