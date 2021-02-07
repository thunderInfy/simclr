import torchvision

def get_color_distortion(s=1.0):
    color_jitter = torchvision.transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter =  torchvision.transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray =  torchvision.transforms.RandomGrayscale(p=0.2)
    color_distort =  torchvision.transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort

def deprocess_and_show(img_tensor):
    return  torchvision.transforms.Compose([
             torchvision.transforms.Normalize((0, 0, 0), (2, 2, 2)),
             torchvision.transforms.Normalize((-0.5, -0.5, -0.5), (1, 1, 1)),
             torchvision.transforms.ToPILImage()
          ])(img_tensor)

