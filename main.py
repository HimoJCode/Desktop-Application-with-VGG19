import eel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
import base64
import io

# Initialize Eel for the front-end interface
eel.init('UI')

# Set device to GPU if available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image size based on device capability
imsize = 512 if torch.cuda.is_available() else 128

# Define image transformation pipeline: resize and convert to tensor
loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()
])

def image_loader(image_data):
    """
    Load an image from base64 encoded data, apply transformations, and return as tensor.
    Args:
        image_data (str): Base64 encoded image data
    Returns:
        torch.Tensor: Transformed image tensor
    """
    image = Image.open(io.BytesIO(base64.b64decode(image_data.split(',')[1])))
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

class ContentLoss(nn.Module):
    """
    Compute content loss as mean squared error between target and input.
    """
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    """
    Compute the Gram matrix of the input tensor.
    Args:
        input (torch.Tensor): Input feature map
    Returns:
        torch.Tensor: Gram matrix
    """
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    """
    Compute style loss as mean squared error between target and input Gram matrices.
    """
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

# Load VGG19 model pre-trained on ImageNet and set to evaluation mode
cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    """
    Normalize an image with mean and standard deviation.
    """
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)

    def forward(self, img):
        return (img - self.mean) / self.std

# Default layers to compute content and style losses
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    """
    Create a model with content and style loss layers.
    Args:
        cnn (nn.Module): The pre-trained VGG19 model
        normalization_mean (torch.Tensor): Mean for normalization
        normalization_std (torch.Tensor): Std for normalization
        style_img (torch.Tensor): Style image tensor
        content_img (torch.Tensor): Content image tensor
        content_layers (list): List of layer names for content loss
        style_layers (list): List of layer names for style loss
    Returns:
        model (nn.Sequential): The style transfer model
        style_losses (list): List of style loss layers
        content_losses (list): List of content loss layers
    """
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def get_input_optimizer(input_img):
    """
    Get optimizer for the input image.
    Args:
        input_img (torch.Tensor): The input image tensor
    Returns:
        optimizer (optim.Optimizer): The optimizer for the input image
    """
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=1000,
                       style_weight=1000000, content_weight=1):
    """
    Perform the style transfer.
    Args:
        cnn (nn.Module): The pre-trained VGG19 model
        normalization_mean (torch.Tensor): Mean for normalization
        normalization_std (torch.Tensor): Std for normalization
        content_img (torch.Tensor): Content image tensor
        style_img (torch.Tensor): Style image tensor
        input_img (torch.Tensor): Input image tensor
        num_steps (int): Number of optimization steps
        style_weight (int): Weight for style loss
        content_weight (int): Weight for content loss
    Returns:
        input_img (torch.Tensor): The transformed input image tensor
    """
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)

    input_img.requires_grad_(True)
    model.eval()

    optimizer = get_input_optimizer(input_img)
    run = [0]
    while run[0] <= num_steps:

        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            print(f"Run {run[0]} / {num_steps} - Style Loss: {style_score.item()}, Content Loss: {content_score.item()}")
            return style_score + content_score

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img

@eel.expose
def process_image(content_image, style_image, num_steps):
    """
    Process images for style transfer and return the transformed image in base64.
    Args:
        content_image (str): Base64 encoded content image
        style_image (str): Base64 encoded style image
        num_steps (int): Number of optimization steps
    Returns:
        str: Base64 encoded transformed image
    """
    try:
        content_img = image_loader(content_image)
        style_img = image_loader(style_image)

        if content_img.size() != style_img.size():
            return "Error: Content and Style images must have the same dimensions."

        input_img = content_img.clone()
        num_steps = int(num_steps)

        output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                    content_img, style_img, input_img, num_steps=num_steps)

        out_t = (output.data.squeeze())
        output_img = transforms.ToPILImage()(out_t)

        img_io = io.BytesIO()
        output_img.save(img_io, 'PNG')
        img_io.seek(0)
        base64_image = base64.b64encode(img_io.getvalue()).decode('utf-8')
        return base64_image
    except Exception as e:
        print(f"Error during processing: {e}")
        return f"Error: {str(e)}"

eel.start('index.html', size=(800, 600))
