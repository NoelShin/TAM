if __name__ == '__main__':
    import os
    import random
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import numpy as np
    from options import BaseOptions
    from utils import adjust_lr, cal_top1_and_top5
    from tqdm import tqdm
    from datetime import datetime

    opt = BaseOptions().parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids

    random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed_all(opt.manual_seed)

    dataset_name = opt.dataset
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.backends.cudnn.benchmark = True

    if dataset_name == 'CIFAR10':
        from pipeline import CustomCIFAR10

        dataset = CustomCIFAR10(opt, val=False)
        test_dataset = CustomCIFAR10(opt, val=True)

    elif dataset_name == 'CIFAR100':
        from pipeline import CustomCIFAR100

        dataset = CustomCIFAR100(opt, val=False)
        test_dataset = CustomCIFAR100(opt, val=True)

    elif dataset_name == 'ImageNet':
        from pipeline import CustomImageNet1K

        dataset = CustomImageNet1K(opt, val=False)
        test_dataset = CustomImageNet1K(opt, val=True)

    elif dataset_name == 'SVHN':
        from pipeline import CustomSVHN

        dataset = CustomSVHN(opt, val=False)
        test_dataset = CustomSVHN(opt, val=True)

    else:
        raise NotImplementedError(
            "Invalid dataset {}. Choose among ['CIFAR10', 'CIFAR100', 'ImageNet', 'SVHN']".format(dataset_name))

    data_loader = DataLoader(dataset,
                             batch_size=opt.batch_size,
                             num_workers=opt.n_workers,
                             shuffle=True)

    test_data_loader = DataLoader(test_dataset,
                                  batch_size=opt.batch_size,
                                  num_workers=opt.n_workers,
                                  shuffle=False)

    backbone_network = opt.backbone_network
    n_layers = opt.n_layers

    if backbone_network == 'MobileNet':
        from models import MobileNet

        model = MobileNet(width_multiplier=opt.width_multiplier,
                          attention=opt.attention_module,
                          group_size=opt.group_size)

    if backbone_network == 'ResNet':
        from models import ResidualNetwork

        model = ResidualNetwork(n_layers=n_layers,
                                dataset=opt.dataset,
                                attention=opt.attention_module,
                                group_size=opt.group_size)

    elif backbone_network == 'ResNext':
        from models import ResNext

        model = ResNext(n_layers=n_layers,
                        n_groups=opt.n_groups,
                        dataset=opt.dataset,
                        attention=opt.attention_module,
                        group_size=opt.group_size)

    elif backbone_network == 'WideResNet':
        from models import WideResNet

        model = WideResNet(n_layers=n_layers,
                           widening_factor=opt.widening_factor,
                           dataset=opt.dataset,
                           attention=opt.attention_module,
                           group_size=opt.group_size)

    model = nn.DataParallel(model).to(device)

    criterion = nn.CrossEntropyLoss()

    if dataset_name in ['CIFAR10', 'CIFAR100']:
        optim = torch.optim.SGD(model.parameters(),
                                lr=opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)

        milestones = [150, 225]

    elif dataset_name == 'ImageNet':
        optim = torch.optim.SGD(model.parameters(),
                                lr=opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)
        milestones = [30, 60]

    elif dataset_name == 'SVHN':
        optim = torch.optim.SGD(model.parameters(),
                                lr=opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)

        milestones = [80, 120]

    else:
        """
                For other datasets
        """
        raise NotImplementedError

    dict_best_top1 = {'Epoch': 0, 'Top1': 100.}
    dict_best_top5 = {'Epoch': 0, 'Top5': 100.}

    if opt.resume:
        state_dict = torch.load(opt.path_model)
        model.load_state_dict(state_dict['state_dict'])
        optim.load_state_dict(state_dict['optimizer'])

        dict_best_top1.update({'Epoch': opt.epoch_top1, 'Top1': opt.top1})
        dict_best_top5.update({'Epoch': opt.epoch_top5, 'Top5': opt.top5})

    st = datetime.now()
    iter_total = 0
    top1_hist = list(100 for i in range(100))
    top5_hist = list(100 for i in range(100))  # to see 100 latest top5 error
    for epoch in range(opt.epoch_recent, opt.epochs):
        adjust_lr(optim, epoch, opt.lr, milestones=milestones, gamma=0.1)
        list_loss = list()
        model.train()
        for input, label in tqdm(data_loader):
            iter_total += 1
            input, label = input.to(device), label.to(device)

            output = model(input)

            loss = criterion(output, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

            top1, top5 = cal_top1_and_top5(output, label)
            top1_hist[iter_total % 100] = np.float(top1)
            top5_hist[iter_total % 100] = np.float(top5)

            list_loss.append(loss.detach().item())

            if iter_total % opt.iter_report == 0:
                print("Iteration: {}, Top1: {:.3f}, Top5: {:.3f} Loss: {:.4f} Top1_hist: {:.3f} Top5_hist: {:.3f}"
                      .format(iter_total, top1, top5, loss.detach().item(), np.mean(top1_hist), np.mean(top5_hist)))

            if opt.debug:
                break

        with open(os.path.join(opt.dir_analysis, 'train.txt'), 'a') as log:
            log.write(str(epoch + 1) + ', ' +
                      str(loss.detach().item()) + ', ' +
                      str(top1.item()) + ', ' +
                      str(top5.item()) + '\n')
            log.close()

        with torch.no_grad():
            model.eval()
            list_top1, list_top5 = list(), list()

            for input, label in tqdm(test_data_loader):
                input, label = input.to(device), label.to(device)

                output = model(input)

                top1, top5 = cal_top1_and_top5(output, label)
                list_top1.append(top1.cpu().numpy())
                list_top5.append(top5.cpu().numpy())

            state = {'epoch': epoch + 1,
                     'state_dict': model.state_dict(),
                     'optimizer': optim.state_dict()}
            torch.save(state, os.path.join(opt.dir_model, 'latest.pt'.format(epoch + 1)))

            avg_top1, avg_top5 = np.mean(list_top1), np.mean(list_top5)

            if avg_top1 < dict_best_top1['Top1']:
                dict_best_top1.update({'Epoch': epoch + 1, 'Top1': avg_top1})
                torch.save(state, os.path.join(opt.dir_model, 'top1_best.pt'.format(epoch + 1)))

            if avg_top5 < dict_best_top5['Top5']:
                dict_best_top5.update({'Epoch': epoch + 1, 'Top5': avg_top5})
                torch.save(state, os.path.join(opt.dir_model, 'top5_best.pt'.format(epoch + 1)))

            with open(os.path.join(opt.dir_analysis, 'log.txt'), 'a') as log:
                log.write(str(epoch + 1) + ', ' +
                          str(dict_best_top1['Epoch']) + ', ' +
                          str(dict_best_top5['Epoch']) + ', ' +
                          str(np.mean(list_loss)) + ', ' +
                          str(np.mean(list_top1)) + ', ' +
                          str(np.mean(list_top5)) + '\n')
                log.close()

        if opt.debug:
            break

    print("Total time taken: ", datetime.now() - st)
