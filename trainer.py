from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from cutmix import cutmix
import os


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def train_with_cutmix(tensorboard_dir, save_dir, model, train_loader, val_loader, learning_rate=0.01, momentum=0.9, decay_steps=20, gamma=0.5, cut_mix_alpha=1.0, epochs=10, save_steps=5):
    ensure_dir(save_dir)
    device = "cuda"
    model = model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    scheduler = StepLR(optimizer, step_size=decay_steps, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    # TensorBoard summary writer
    writer = SummaryWriter(tensorboard_dir)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_train_batches = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data, targets_a, targets_b, lam = cutmix(data, target, cut_mix_alpha)
            targets_a, targets_b = targets_a.to(device), targets_b.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_train_batches += 1

        average_train_loss = running_loss / total_train_batches
        writer.add_scalar('Loss/train', average_train_loss, epoch)

        # Validate the model
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        average_val_loss = val_loss / len(val_loader)
        val_accuracy = 100. * correct / total
        writer.add_scalar('Loss/val', average_val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        print(f'Epoch {epoch + 1} training loss: {average_train_loss}')
        print(f'Epoch {epoch + 1} validation loss: {average_val_loss}')
        print(f'Epoch {epoch + 1} validation accuracy: {val_accuracy}%')

        scheduler.step()
        if (epoch + 1) % save_steps == 0:
            torch.save(model.state_dict(), save_dir + f'vit_epoch_{epoch+1}.pth')
        
    print('Training complete')
    writer.close()
    return model

