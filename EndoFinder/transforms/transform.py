import torchvision.transforms as transforms
from augly.image import encoding_quality
import random
from classy_vision.dataset.transforms import (
    ClassyTransform,
    build_transform,
    register_transform,
)

class JpegCompressTransform(ClassyTransform):
    """
    Compresses an image with lower bitrate JPEG to make compression
    artifacts appear on the resulting image
    """

    def __init__(self, quality):
        """
        Args:
          quality_sampler: sampler of JPEG quality values (integers in [0, 100])
        """
        self.quality = quality

    def __call__(self, image):
        quality = random.randint(self.quality[0], self.quality[1])
        image_transformed = encoding_quality(image, quality=quality)
        return image_transformed

class MAETransform(object):
    def __init__(self, input_size):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def __call__(self, img):
        return self.transform(img)
    
class ValTransform(object):
    def __init__(self, input_size):
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            # transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
            # transforms.RandomApply([JpegCompressTransform(quality=[0, 100])], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def __call__(self, img):
        return self.transform(img)

class AdvancedTransform(object):
    def __init__(self, input_size):
        self.transform = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.Resize(int(256/224*input_size)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.8),#0.8*3 0.2*1
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
            transforms.RandomApply([JpegCompressTransform(quality=[0, 100])], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    def __call__(self, img):
        return self.transform(img)