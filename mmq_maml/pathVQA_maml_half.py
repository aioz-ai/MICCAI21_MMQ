import  torch, os
import  numpy as np
from    pathVQA_maml import PathVQA_maml
import  scipy.stats
from    torch.utils.data import DataLoader
import  argparse
import time
from meta import Meta
import shutil

def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h

def unlabel_processing(dataset, unlabel_pool, threshold_fre = 27, threshold_max = 9):
    # TODO
    count = 0
    new_path = dataset.path + '_unlabel_add'
    for i in unlabel_pool:
        if len(unlabel_pool[i]) > threshold_fre:
            unique = set(unlabel_pool[i])
            ele_count = {}
            for j in unique:
                ele_count[j] = unlabel_pool[i].count(j)
            max_key = max(ele_count, key=ele_count.get)
            max_value = ele_count[max_key]
            all_values = list(ele_count.values())
            if all_values.count(max_value) == 1 and max_value > threshold_max:
                label = int(max_key)
                if label != 20:
                    count += 1
                    class_name = dataset.label2class[label]
                    dst_dir = os.path.join(new_path, class_name)
                    if not os.path.exists(dst_dir):
                        os.makedirs(dst_dir)
                    dst = os.path.join(dst_dir, i.split('/')[-1])
                    os.rename(i, dst)
    print('The number of additional unlabeled images:', count)
    return count

def label_processing(dataset, label_pool, threshold=0.3):
    count = 0
    new_path = dataset.path + '_label_remove'
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    for i in label_pool:
        if max(label_pool[i]) < threshold:
            count+=1
            dst = os.path.join(new_path, i.split('/')[-1])
            os.rename(i, dst)
    print('The number of removed labeled images:', count)
    return count

def final_processing(dataset):
    root = dataset.path

    src = root + '_unlabel_add'
    list_dirs = os.listdir(src)
    for dir in list_dirs:
        files = os.listdir(os.path.join(src, dir))
        for file in files:
            shutil.move(os.path.join(src, dir, file), os.path.join(root, dir))
    shutil.rmtree(src)

    src = root + '_label_remove'
    list_files = os.listdir(src)
    for i in list_files:
        shutil.move(os.path.join(src, i), root + '_unlabel')
    shutil.rmtree(src)

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

    model_dir = args.input + '/maml%d_miccai2021_optimization_newmethod_%dway_%dshot_t%d' % (args.imgsz, args.n_way, args.k_spt, args.t_dst - 1)
    model_path = os.path.join(model_dir, 'model_epoch%d.pth' % args.epoch)
    print('-------load model weight from:', model_path)
    maml.load_state_dict(torch.load(model_path))
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    # source and destination directory
    t_dst = args.t_dst
    src = os.path.join(args.data, 't%d'%(t_dst - 1))
    dst = os.path.join(args.data, 't%d'%(t_dst))

    # Copy original dataset to the destination folder
    shutil.copytree(src, dst)

    ####################################################################
    ## PROCESS UNLABEL DATA
    ####################################################################
    # batchsz here means total episode number
    mini_test_unlabel = PathVQA_maml(args.data, mode='train', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=600, resize=args.imgsz, unlabel=True, t=t_dst)

    db_test_unlabel = DataLoader(mini_test_unlabel, 1, shuffle=True, num_workers=1, pin_memory=True)
    start = time.time()
    unlabel_pool = {}
    for x_spt, y_spt, x_qry, y_spt_real, flatten_query_x in db_test_unlabel:
        x_spt, y_spt, x_qry, y_spt_real = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                     x_qry.squeeze(0).to(device), y_spt_real.squeeze(0).to(device)

        results = maml.unlabel_pooling(x_spt, y_spt, x_qry, y_spt_real, flatten_query_x)
        for i in results:
            if i not in unlabel_pool.keys():
                unlabel_pool[i] = []
            unlabel_pool[i] += results[i]

    end = time.time()
    print('***************** time:', end - start)
    unlabel_processing(mini_test_unlabel, unlabel_pool)

    ####################################################################
    ## PROCESS LABEL DATA
    ####################################################################
    # batchsz here means total episode number
    mini_test_label = PathVQA_maml(args.data, mode='train', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=600, resize=args.imgsz, unlabel=False, t = t_dst)

    db_test_label = DataLoader(mini_test_label, 1, shuffle=True, num_workers=1, pin_memory=True)
    start = time.time()
    label_pool = {}
    for x_spt, y_spt, x_qry, y_qry, y_qry_real, flatten_query_x in db_test_label:
        x_spt, y_spt, x_qry, y_qry, y_qry_real = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                     x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device), y_qry_real.squeeze(0).to(device)

        results = maml.label_pooling(x_spt, y_spt, x_qry, y_qry, y_qry_real, flatten_query_x)
        for i in results:
            if i not in label_pool.keys():
                label_pool[i] = []
            label_pool[i] += results[i]

    end = time.time()
    print('***************** time:', end - start)
    label_processing(mini_test_label, label_pool)

    ####################################################################
    ## CREATE NEW LABEL AND UNLABEL DATASET and REMOVE TEMP FOLDERS
    ####################################################################
    final_processing(mini_test_label)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=0)
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
    argparser.add_argument('--input', type=str, help='input directory for saving models', default='saved_models')
    argparser.add_argument('--data', type=str, help='data directory', default='data/pathvqa_maml/')
    argparser.add_argument('--t_dst', type=int, help='t-th step', default=1)
    args = argparser.parse_args()

    main(args)
