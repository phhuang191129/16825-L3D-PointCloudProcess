import numpy as np
import argparse

import torch
from models import cls_model
from utils import create_dir, viz_cls, rotate_x


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_240')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')
    parser.add_argument('--rotate',action='store_true', default=False, help='Rotation')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Classification Task ------
    model = cls_model().to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy(np.load(args.test_label))

    # ------ TO DO: Make Prediction ------
    print("Test data shape: ", test_data.shape)
    if args.rotate:
        degree = 60
        test_data = rotate_x(test_data, torch.tensor(degree*np.pi/180))
    pred_label = torch.zeros(test_label.size(), dtype=torch.long)
    for i in range(len(test_data)):
        pred_label[i] = torch.argmax(model(test_data[i].unsqueeze(0).to(args.device)), dim=1)
    # pred_label = torch.argmax(model(test_data.to(args.device)), dim=1)

    print(pred_label.shape, test_label.shape)
    # Compute Accuracy
    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
    print ("test accuracy: {}".format(test_accuracy))

    # Visualize Classification Result (Pred VS Ground Truth)
    incorrect_cases = (pred_label != test_label).nonzero(as_tuple=True)[0]
    print(len(incorrect_cases))
    if args.rotate:
        save_path = "{}/data_{}_rot_{}_gt_{}_pred_{}.gif".format(args.output_dir, args.i, degree, int(test_label[args.i]), \
                                                                    pred_label[args.i])
    else:
        save_path = "{}/data_{}_pts_{}_gt_{}_pred_{}.gif".format(args.output_dir, args.i, args.num_points, int(test_label[args.i]), \
                                                                    pred_label[args.i])
    viz_cls(test_data[args.i], save_path, args.device, num_points=args.num_points)
        # viz_cls(test_data[i], "{}/data_{}_gt_{}_pred_{}.gif".format(args.output_dir, i, int(test_label[i]), \
                                                                    #  pred_label[i]), args.device)