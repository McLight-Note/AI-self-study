from PIL import Image
from torchvision import transforms

img = Image.open("cat.png").convert("RGB")

resize = transforms.Resize((224,224))
img = resize(img)

tensor = transforms.ToTensor()(img)

print("Tensor shape:", tensor.shape)
print("First pixel before normalization:")
print(tensor[:,0,0])

normalized = transforms.Normalize(
    mean=[0.485,0.456,0.406],
    std=[0.229,0.224,0.225]
)(tensor)

print("\nFirst pixel after normalization:")
print(normalized[:,0,0])