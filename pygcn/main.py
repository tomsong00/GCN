
'''代码主函数开始'''
# Training settings
import argparse
import train as tr
import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data
from pygcn.models import GCN
import numpy as np
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--no_visual', action='store_true', default=True,
                        help='visualization of ground truth and test result')

    args = parser.parse_args()


    # 显示args
    tr.show_Hyperparameter(args)

    # 是否使用CUDA
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # 设置随机数种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data()           # 返回可视化要用的labels

    # Model
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,     # 对Cora数据集，为7，即类别总数。
                dropout=args.dropout)
    # optimizer
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    # to CUDA
    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    # Train model
    t_total = time.time()
    for epoch in range(args.epochs):
        tr.train(epoch,model,optimizer,features,adj,idx_train,labels,args,idx_val)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    output=tr.test(model,features,adj,idx_test,labels)           # 返回output

    if not args.no_visual:
        # 计算预测值
        preds = output.max(1)[1].type_as(labels)

        # output的格式转换
        output = output.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        preds = preds.cpu().detach().numpy()

        # Visualization with visdom
        vis = tr.Visdom(env='pyGCN Visualization')

        # ground truth 可视化
        result_all_2d = tr.t_SNE(output, 2)
        tr.Visualization(vis, result_all_2d, labels,
                      title='[ground truth of all samples]\n Dimension reduction to %dD' % (result_all_2d.shape[1]))
        result_all_3d = tr.t_SNE(output, 3)
        tr.Visualization(vis, result_all_3d, labels,
                      title='[ground truth of all samples]\n Dimension reduction to %dD' % (result_all_3d.shape[1]))

        # 预测结果可视化
        result_test_2d = tr.t_SNE(output[idx_test.cpu().detach().numpy()], 2)
        tr.Visualization(vis, result_test_2d, preds[idx_test.cpu().detach().numpy()],
                      title='[prediction of test set]\n Dimension reduction to %dD' % (result_test_2d.shape[1]))
        result_test_3d = tr.t_SNE(output[idx_test.cpu().detach().numpy()], 3)
        tr.Visualization(vis, result_test_3d, preds[idx_test.cpu().detach().numpy()],
                      title='[prediction of test set]\n Dimension reduction to %dD' % (result_test_3d.shape[1]))

