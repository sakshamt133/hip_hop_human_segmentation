from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])

img_path = "D:\\Datasets\\Computer Vision\\Segmentation Dataset\\images"
mask_path = "D:\\Datasets\\Computer Vision\\Segmentation Dataset\\masks"
img_size = 256
batch_size = 32