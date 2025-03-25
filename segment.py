from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from models.birefnet import BiRefNet
from utils import check_state_dict

birefnet = BiRefNet(bb_pretrained=False)
state_dict = torch.load("weights/BiRefNet-general-epoch_244.pth", map_location='cpu')
state_dict = check_state_dict(state_dict)
birefnet.load_state_dict(state_dict)

torch.set_float32_matmul_precision(['high', 'highest'][0])
birefnet.to('cuda')
birefnet.eval()
birefnet.half()

def extract_object(birefnet, imagepath):
    # Data settings
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(imagepath)
    input_images = transform_image(image).unsqueeze(0).to('cuda').half()

    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    image.putalpha(mask)
    return image, mask

# Visualization
plt.axis("off")
plt.imshow(extract_object(birefnet, imagepath='C:/Users/ZODNGUY1/datasets/zeiss/brain-bg/input/143542_0000.png')[0])
plt.show()