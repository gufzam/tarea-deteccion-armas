from ultralytics import YOLO
model = YOLO('armas.pt')
print('Modelo cargado OK. Clases:', model.names)
