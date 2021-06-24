import  torch, os
import  numpy as np
from    pathVQA_maml import PathVQA_maml
import  scipy.stats
from    torch.utils.data import DataLoader
import  argparse
import time
from meta import Meta


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main(args):
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    # add other class (n_way + 1)
    args.n_way += 1
    print(args)

    # hidden layer dimension config
    if args.imgsz == 84:
        dim_hidden_linear = 32 * 5 * 5
    elif args.imgsz == 128:
        dim_hidden_linear = 32 * 11 * 11

    # model config
    config = [
        ('conv2d', [32, 3, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 1, 0]),
        ('flatten', []),
        ('linear', [args.n_way, dim_hidden_linear])
    ]

    # initial MAML model
    device = torch.device('cuda')
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    # Load dataset
    # batchsz here means total episode number
    mini = PathVQA_maml(args.data, mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=10000, resize=args.imgsz, t = args.t_dst)
    mini_test = PathVQA_maml(args.data, mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=600, resize=args.imgsz)

    # Train model
    for epoch in range(args.epoch//10000):
        # fetch meta_batchsz num of episode each time
        db = DataLoader(mini, args.task_num, shuffle=True, num_workers=1, pin_memory=True)
        start = time.time()
        for step, (x_spt, y_spt, x_qry, y_qry, _, _) in enumerate(db):
            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

            accs = maml(x_spt, y_spt, x_qry, y_qry)

            if step % 50 == 0:
                end = time.time()
                print('step:', step, '\ttraining acc:', accs, '\ttime:', end - start)
                start = time.time()

            if (step % 1000 == 0 and step != 0) or (step == len(db) - 1):  # evaluation
                db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
                start = time.time()
                accs_all_test = []

                for x_spt, y_spt, x_qry, y_qry, _, _ in db_test:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                    accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                    accs_all_test.append(accs)

                # [b, update_step+1]
                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                end = time.time()
                print('***************** Test acc:', accs, '\ttime:', end - start)
                start = time.time()

        # Save model
        model_dir = args.output + '/maml%d_miccai2021_optimization_newmethod_%dway_%dshot_t%d'%(args.imgsz, args.n_way, args.k_spt, args.t_dst)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, 'model_epoch%d.pth' % epoch)
        print('saving model to:', model_dir)
        torch.save(maml.state_dict(), model_path)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=10000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--output', type=str, help='output directory for saving models', default='saved_models')
    argparser.add_argument('--data', type=str, help='data directory', default='data/pathvqa_maml/')
    argparser.add_argument('--t_dst', type=int, help='t-th step', default=0)
    args = argparser.parse_args()

    main(args)
