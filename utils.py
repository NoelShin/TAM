import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns; sns.set()


def save_attention_map(tensor, file_name, heatmap=True):
    assert tensor.shape[0] == 1, "Batch size should be 1. Current value is {}".format(tensor.shape[0])
    np_image = tensor.squeeze().cpu().numpy().astype(np.float64)

    if len(np_image.shape) == 3:
        np_image = np.mean(np_image, axis=0)
        np_image -= np_image.min()
        np_image /= np_image.max()
    np_image *= 255.0

    np_image = np_image.astype(np.uint8)

    if heatmap:
        plt.figure()
        sns.heatmap(np_image, cmap='jet', cbar=False, xticklabels=False, yticklabels=False,
                    square=True)
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
        pil_image = Image.open(file_name).resize((np_image.shape[0], np_image.shape[1]))
        pil_image.save(file_name)
        plt.close()

    else:
        pil_image = Image.fromarray(np_image)
        pil_image.save(file_name)


def adjust_lr(optimizer, epoch, init_lr, milestones, gamma=0.1):
    count = 0
    for i in milestones:
        if epoch // i >= 1:
            count += 1

    lr = init_lr * (gamma ** count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("Epoch: {:d}, Learning_rate: {:.6f}".format(epoch, lr))


def cal_top1_and_top5(output, label):
    batch_size = float(output.shape[0])
    _, index = output.topk(5, dim=1, largest=True, sorted=True)
    correct = index.eq(label.view(-1, 1).expand_as(index))
    top1 = correct[:, :1].float().sum().mul_(100. / batch_size)
    top5 = correct[:, :5].float().sum().mul_(100. / batch_size)
    return 100.0 - top1, 100.0 - top5
