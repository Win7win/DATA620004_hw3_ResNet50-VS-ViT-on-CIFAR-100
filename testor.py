import torch
import torch.nn as nn
from vision_transformer import VisionTransformer
from resnet50 import ResNet50
from load_cifar100 import load_cifar100


def test_my_model(model, test_loader, device="cuda"):
    model.eval()  # 设置模型为评估模式
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 禁用梯度计算
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    average_test_loss = test_loss / len(test_loader)
    test_accuracy = 100. * correct / total

    print(f'Test loss: {average_test_loss}')
    print(f'Test accuracy: {test_accuracy}%')

    return average_test_loss, test_accuracy

if __name__ == "__main__":
    trainloader, validloader, testloader = load_cifar100(batch_size=128, val_part=0.1)
    vit_model = VisionTransformer().to("cuda")
    model_path = "../vit_models/vit_0.01_50_0.9_0.5/vit_epoch_50.pth"
    vit_model.load_state_dict(torch.load(model_path))
    test_loss, test_accuracy = test_my_model(vit_model, testloader)
    print("test_loss:", test_loss)
    print("test_accuracy:", test_accuracy)