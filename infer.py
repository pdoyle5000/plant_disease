import torch
from dataset import inference_composer, STDDEV, MEAN, CLASS_MAP
from cnn import PlantDiseaseNet
from typing import Tuple, List
from PIL.JpegImagePlugin import JpegImageFile


class PlantDiseaseClassifier:
    def __init__(self, model_path: str):
        self.model = self._load_model(model_path)
        self.model.eval()

    def _load_model(self, model_path: str):
        model = PlantDiseaseNet()
        model.load_state_dict(torch.load(model_path))
        return model

    def predict(self, image: JpegImageFile) -> Tuple[str, int, List[float]]:
        composer = inference_composer(STDDEV, MEAN)
        image = composer(image)
        with torch.no_grad():
            output = self.model(image.float().unsqueeze(0)).squeeze(0)
            softmax = torch.nn.Softmax(dim=0)
            predictions = softmax(output)
            argmax = predictions.max(0)[1]
            return CLASS_MAP[torch.max(argmax)], torch.max(argmax), output
