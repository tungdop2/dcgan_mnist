from torchvision import datasets, transforms

def get_dataset():
    return datasets.MNIST(root='./data', download=True,
                          transform=transforms.Compose([
                              transforms.Resize(28),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                          ]))
    
