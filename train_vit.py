from load_cifar100 import load_cifar100
from vision_transformer import VisionTransformer
from trainer import train_with_cutmix
from testor import test_my_model

# 载入数据
trainloader, validloader, testloader = load_cifar100(batch_size=128, val_part=0.1)

# 开始训练
lrs = [0.05, 0.01, 0.001]
epochs = [50]
momentums = [0.6, 0.9]
gammas = [0.5, 0.7]
for lr in lrs:
    for epoch in epochs:
        for momentum in momentums:
            for gamma in gammas:
                vit_model = VisionTransformer().to("cuda")
                model = train_with_cutmix(tensorboard_dir=f"./runs_2/vit/{lr}_{epoch}_{momentum}_{gamma}/", save_dir=f"../vit_models_2/vit_{lr}_{epoch}_{momentum}_{gamma}/",model=vit_model, train_loader=trainloader, val_loader=validloader, learning_rate=lr, momentum=momentum, decay_steps=20, gamma=gamma, cut_mix_alpha=1.0, epochs=epoch, save_steps=25)
                test_loss, test_accuracy = test_my_model(model, testloader)
                with open("vit_records.txt", "a+", encoding="utf-8") as f:
                    f.write(f"{lr}_{epoch}_{momentum}_{gamma}  " + "test_loss: " + str(test_loss) + "test_accuracy: "+ str(test_accuracy)+"\n")