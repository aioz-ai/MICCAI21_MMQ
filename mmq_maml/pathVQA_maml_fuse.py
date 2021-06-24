import  torch, os
import  numpy as np
from    pathVQA_maml import PathVQA_maml
import  scipy.stats
from    torch.utils.data import DataLoader
import  argparse
import time
from meta import Meta
import pickle as p

def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h

def num_tensors(model):
    tmp = filter(lambda x: x.requires_grad, model.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    return num

def feature_extraction(maml, db_test, device):
    for i in range(args.nums_t):
        model_dir = args.input + '/maml%d_miccai2021_optimization_newmethod_%dway_%dshot_t%d' % (args.imgsz, args.n_way, args.k_spt, i)
        model_path = os.path.join(model_dir, 'model_epoch%d.pth' % args.epoch)
        print('-------load model weight from:', model_path)
        maml.load_state_dict(torch.load(model_path))

        start = time.time()
        logit_softmax_list = []
        feature_list = []
        path_list = []
        for x_spt, y_spt, x_qry, y_qry, _, flatten_query_x in db_test:
            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                         x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

            logit_softmax, feature = maml.feature_validation(x_spt, y_spt, x_qry, y_qry)
            logit_softmax_list += logit_softmax
            feature_list += list(feature.cpu())
            path_list += flatten_query_x
        if not os.path.exists(args.feature_output):
            os.makedirs(args.feature_output)
        p.dump(feature_list, open(args.feature_output + '/feature_t%d.pkl'%(i), 'wb'))
        p.dump(logit_softmax_list, open(args.feature_output + '/logit_softmax_t%d.pkl'%(i), 'wb'))
        p.dump(path_list, open(args.feature_output + '/path_t%d.pkl'%(i), 'wb'))
        # [b, update_step+1]
        end = time.time()
        print('***************** Time:', end - start)

def load_feature_and_logit(args):
    features = []
    logits = []
    for t in range(args.nums_t):
        features.append(torch.stack(p.load(open(args.feature_output + '/feature_t%d.pkl' % (t), 'rb'))))
        logits.append(p.load(open(args.feature_output + '/logit_softmax_t%d.pkl' % (t), 'rb')))
    features = torch.stack(features).cpu().transpose(0, 1)
    logits = torch.tensor(logits).transpose(0, 1)
    return features, logits

def fuse_score(features, logits, alpha=0.8):
    # TODO
    cos = torch.nn.CosineSimilarity(dim=0)
    results = []
    for idx in range(len(features)):
        if idx%100 == 0:
            print('%d / %d'%(idx, len(features)))
        feature_sample = features[idx]
        logit_sample = logits[idx]
        sample_results = []
        for i in range(len(feature_sample)):
            row_results = []
            for j in range(len(feature_sample)):
                if i != j:
                    sim = cos(feature_sample[i], feature_sample[j])
                    div = 1 - sim
                    row_results.append(div)
            row_results = sum(row_results)
            fuse_score = (alpha * logit_sample[i]) + ((1 - alpha) * row_results)
            sample_results.append(fuse_score)
        results.append(sample_results)
    return results

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
    print(maml)
    print('Total trainable tensors:', num_tensors(maml))

    # Load validation dataset
    # batchsz here means total episode number
    mini_test = PathVQA_maml(args.data, mode='val', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=600, resize=args.imgsz)

    db_test = DataLoader(mini_test, 1, shuffle=False, num_workers=1, pin_memory=True)

    # Extract validation features, logits for each model and save them
    feature_extraction(maml, db_test, device)

    # Load all features and logits
    features, logits = load_feature_and_logit(args)

    # Compute fuse score
    results = fuse_score(features, logits)

    # Show results to select suitable models
    results = torch.tensor(results).mean(dim=0)
    print(results)
    print('------- sort', torch.sort(results, descending=True))

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
    argparser.add_argument('--nums_t', type=int, help='num models', default=6) # m refinement models + first model (5 + 1 = 6)
    argparser.add_argument('--feature_output', type=str, help='input directory for saving feature', default='features')
    args = argparser.parse_args()

    main(args)
