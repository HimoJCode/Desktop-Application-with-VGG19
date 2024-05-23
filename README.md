# Desktop Application with VGG19 for Style Transfer

##Overview
This desktop application leverages the power of the VGG19 neural network to perform style transfer on images. Style transfer is a process that blends the content of one image with the style of another, creating a new, visually appealing image. This application allows users to upload a content image and a style image, adjust the transformation quality, and process the images to generate a transformed output. The final image can be viewed, downloaded, or shared directly to social media platforms.

## Features
- Upload content and style images.
- Adjust image quality using the `num_steps` parameter.
- View the transformed image.
- Download the transformed image.
- Share the transformed image on social media platforms.

## Result Display
-Transformed Image Display: Once processed, the transformed image is displayed within the application interface.
-Error Handling: Provides feedback if the images are not of the same dimensions or if there are any issues during processing.

## Social Media Sharing
-Share to Facebook: Allows users to share the transformed image directly to Facebook.
-Share to Twitter: Allows users to tweet the transformed image.
-Share to Instagram: Provides instructions for sharing the image on Instagram, which does not support direct uploads via URL.

## Download
Image Download: Users can download the transformed image as a PNG file

## Requirements
- Python 3.7+
- Torch
- Torchvision
- Pillow
- Eel

## Technologies Used
- Python
- Eel (for creating desktop applications with HTML, CSS, and JavaScript)
- PyTorch (for VGG19 model and style transfer)
- HTML, CSS, JavaScript (for the user interface)

## Run the application 
-python main.py

    
