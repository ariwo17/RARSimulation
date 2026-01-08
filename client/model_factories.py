import client.models as models

def ComEffFlPaperCnnModel(num_classes): return models.ComEffFlPaperCnnModel()

def VGG11(num_classes): return models.VGG('VGG11')
def VGG19(num_classes): return models.VGG('VGG19')
def PreActResNet18(num_classes): return models.PreActResNet18()
def ResNeXt29_2x64d(num_classes): return models.ResNeXt29_2x64d()
def MobileNet(num_classes): return models.MobileNet()
def MobileNetV2(num_classes): return models.MobileNetV2(num_classes)
def DPN92(num_classes): return models.DPN92()
def ShuffleNetG2(num_classes): return models.ShuffleNetG2()
def SENet18(num_classes): return models.SENet18()
def ShuffleNetV2_net_1(num_classes): return models.ShuffleNetV2(1)
def EfficientNetB0(num_classes): return models.EfficientNetB0()
def RegNetX_200MF(num_classes): return models.RegNetX_200MF()

def ResNet9(num_classes): return models.ResNet9(num_classes)
def ResNet18(num_classes): return models.ResNet18(num_classes)
def ResNet50(num_classes): return models.ResNet50(num_classes)
def DenseNet121(num_classes): return models.DenseNet121(num_classes)
def GoogLeNet(num_classes): return models.GoogLeNet(num_classes)
def LeNet(num_classes): return models.LeNet(num_classes)
def ToyNet(num_classes): return models.ToyNet(num_classes)
def McCandlishMNIST(num_classes): return models.McCandlishMNIST()

# n_features hardcoded #feature for a4a/a9a
def LogisticRegression(num_classes): return models.LogisticRegression(n_features=123, num_classes=1)
