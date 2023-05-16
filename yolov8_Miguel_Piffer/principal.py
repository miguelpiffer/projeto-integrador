from ultralytics import YOLO

# carregar o modelo do yolo
model = YOLO("yolov8n.yaml") # construii um novo modelo do zero

#use o modelo
results = model.train(data="configuração.yaml",epochs=15) # treine o modelo

# coco128.yaml significa : COCO 128.yaml: O arquivo "COCO 128.yaml" refere-se a um arquivo de configuração que contém informações sobre o modelo YOLOv8 que você está usando. Especificamente, ele contém informações sobre o número de classes que a rede deve detectar e os caminhos para os arquivos de treinamento, validação e teste. O arquivo de configuração geralmente segue o formato YAML e pode ser personalizado de acordo com suas necessidades.
# epochs :Epochs: Em aprendizado de máquina, as épocas (epochs, em inglês) referem-se ao número de vezes que um algoritmo de treinamento passa por todo o conjunto de dados de treinamento. Cada época consiste em um ciclo completo de alimentar o conjunto de dados à rede neural, calcular a função de perda e ajustar os pesos da rede por meio de um algoritmo de otimização, como gradiente descendente. O número de épocas é um hiperparâmetro que você define antes de iniciar o treinamento. Geralmente, treinar por mais épocas permite que a rede neural aprenda mais dos dados, mas treinar por muitas épocas pode levar ao overfitting (quando a rede se ajusta demais aos dados de treinamento e não generaliza bem para novos dados).

# data="configuração.yaml"é o caminho que a inteligencia vai seguir para treinar com o dataset pessoal.
