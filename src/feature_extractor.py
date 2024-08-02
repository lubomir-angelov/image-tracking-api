# use a pre-trained ResNet, to extract features from the segmented objects.
import torch
from torchvision import models, transforms

# Load a pre-trained model
feature_extractor = models.resnet50(pretrained=True)
feature_extractor.eval()

# Define transformation
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    #transforms.PILToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image):
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = feature_extractor(image)
    return features.numpy().flatten()
