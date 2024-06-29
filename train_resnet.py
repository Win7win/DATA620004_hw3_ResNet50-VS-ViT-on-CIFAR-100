from load_cifar100 import load_cifar100
from resnet50 import ResNet50
from trainer import train_with_cutmix
from testor import test_my_model

# 载入数据
trainloader, validloader, testloader = load_cifar100(batch_size=128, val_part=0.1)

# 开始训练
# 网格搜索
lrs = [0.001]
epochs = [50]
momentums = [0.6, 0.9]
gammas = [0.5, 0.7]
for lr in lrs:
    for epoch in epochs:
        for momentum in momentums:
            for gamma in gammas:
                # 定义模型
                vit_model = ResNet50(num_classes=100).to("cuda")
                # 训练
                model = train_with_cutmix(tensorboard_dir=f"./runs_2/resnet50/{lr}_{epoch}_{momentum}_{gamma}/", save_dir=f"../resnet_models_2/resnet_{lr}_{epoch}_{momentum}_{gamma}/",model=vit_model, train_loader=trainloader, val_loader=validloader, learning_rate=lr, momentum=momentum, decay_steps=20, gamma=gamma, cut_mix_alpha=1.0, epochs=epoch, save_steps=25)
                # 测试
                test_loss, test_accuracy = test_my_model(model, testloader)
                with open("resnet50_records.txt", "a+", encoding="utf-8") as f:
                    f.write(f"{lr}_{epoch}_{momentum}_{gamma}  " + "test_loss: " + str(test_loss) + "test_accuracy: "+ str(test_accuracy)+"\n")
                