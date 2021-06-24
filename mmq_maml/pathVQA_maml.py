import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
from PIL import Image
import random


class PathVQA_maml(Dataset):
    """
    put mini-imagenet files as :
    root :
        |- images/*.jpg includes all imgeas
        |- train.csv
        |- test.csv
        |- val.csv
    NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets
    sets: conains n_way * k_shot for meta-train set, n_way * n_query for meta-test set.
    """

    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, resize, startidx=0, unlabel = False, t = 0):
        """

        :param root: root path of mini-imagenet
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param resize: resize to
        :param startidx: start to index label from startidx
        """

        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.setsz = self.n_way * self.k_shot  # num of samples per set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.resize = resize  # resize to
        self.startidx = startidx  # index label not from 0, but from startidx
        self.unlabel = unlabel
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, resize:%d' % (
        mode, batchsz, n_way, k_shot, k_query, resize))

        if mode == 'train':
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 # transforms.RandomHorizontalFlip(),
                                                 # transforms.RandomRotation(5),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])
        else:
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])

        # self.path = os.path.join(root, 'maml', mode)
        # self.path = os.path.join(root, 'maml%d'%(self.resize), mode)
        if mode == 'train':
            self.path = os.path.join(root, 't%d'%(t), mode)
        else:
            self.path = os.path.join(root, mode)
        print('--loading data from:', self.path)
        data = self.loadData(self.path)# image path
        self.data = []
        self.img2label = {}
        self.label2class = []
        for i, (k, v) in enumerate(data.items()):
            self.data.append(v)  # [[img1, img2, ...], [img111, ...]]
            self.img2label[k] = i + self.startidx  # {"img_name[:9]":label}
            self.label2class.append(k)
        self.img2label['x'] = i + 1 + self.startidx
        self.cls_num = len(self.data)
        if self.unlabel:
            self.path_unlabel = os.path.join(root, 't%d'%(t), mode + '_unlabel')
            self.data_unlabel = [os.path.join(self.path_unlabel, file) for file in os.listdir(self.path_unlabel)]
            self.create_batch_unlabel(self.batchsz)
        else:
            self.create_batch(self.batchsz)

    def loadData(self, path):
        """
        return a dict saving the information of csv
        :param splitFile: csv file name
        :return: {label:[file1, file2 ...]}
        """
        dictLabels = {}
        for (dirpath, dirnames, _) in os.walk(path):
            for dir in dirnames:
                dictLabels[dir] = [os.path.join(dir, file) for file in os.listdir(os.path.join(dirpath, dir))]
        return dictLabels
    def create_other(self, selected_cls):
        other_pool = []
        for idx, i in enumerate(self.data):
            if idx not in selected_cls:
                other_pool += i
        other_pool = [i + 'x' for i in other_pool]
        return other_pool

    def create_batch(self, batchsz):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param batchsz: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly
            selected_cls = np.random.choice(self.cls_num, self.n_way - 1, False)  # no duplicate
            np.random.shuffle(selected_cls)
            support_x = []
            query_x = []
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class
                try:
                    selected_imgs_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, False)
                except:
                    selected_imgs_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, True)
                np.random.shuffle(selected_imgs_idx)
                indexDtrain = np.array(selected_imgs_idx[:self.k_shot])  # idx for Dtrain
                indexDtest = np.array(selected_imgs_idx[self.k_shot:])  # idx for Dtest
                support_x.append(
                    np.array(self.data[cls])[indexDtrain].tolist())  # get all images filename for current Dtrain
                query_x.append(np.array(self.data[cls])[indexDtest].tolist())

            other_images = self.create_other(selected_cls)
            try:
                selected_imgs_idx_other = np.random.choice(len(other_images), self.k_shot + self.k_query, False)
            except:
                selected_imgs_idx_other = np.random.choice(len(other_images), self.k_shot + self.k_query, True)
            np.random.shuffle(selected_imgs_idx_other)
            indexDtrain_other = np.array(selected_imgs_idx_other[:self.k_shot])  # idx for Dtrain
            indexDtest_other = np.array(selected_imgs_idx_other[self.k_shot:])
            support_x.append(
                np.array(other_images)[indexDtrain_other].tolist())  # get all images filename for current Dtrain
            query_x.append(np.array(other_images)[indexDtest_other].tolist())

            # shuffle the correponding relation between support set and query set
            random.shuffle(support_x)
            random.shuffle(query_x)

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets

    def create_batch_unlabel(self, batchsz):
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch

        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly
            selected_cls = np.random.choice(self.cls_num, self.n_way - 1, False)  # no duplicate
            np.random.shuffle(selected_cls)
            support_x = []
            query_x = []
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class
                try:
                    selected_imgs_idx = np.random.choice(len(self.data[cls]), self.k_shot, False)
                except:
                    selected_imgs_idx = np.random.choice(len(self.data[cls]), self.k_shot, True)

                np.random.shuffle(selected_imgs_idx)
                indexDtrain = np.array(selected_imgs_idx)  # idx for Dtrain
                support_x.append(
                    np.array(self.data[cls])[indexDtrain].tolist())  # get all images filename for current Dtrain

                selected_imgs_idx_test = np.random.choice(len(self.data_unlabel), self.k_query, False)
                indexDtest = np.array(selected_imgs_idx_test)  # idx for Dtest
                query_x.append(np.array(self.data_unlabel)[indexDtest].tolist())

            other_images = self.create_other(selected_cls)
            try:
                selected_imgs_idx_other = np.random.choice(len(other_images), self.k_shot, False)
            except:
                selected_imgs_idx_other = np.random.choice(len(other_images), self.k_shot, True)
            np.random.shuffle(selected_imgs_idx_other)
            indexDtrain_other = np.array(selected_imgs_idx_other)
            support_x.append(
                np.array(other_images)[indexDtrain_other].tolist())

            selected_imgs_idx_test_other = np.random.choice(len(self.data_unlabel), self.k_query, False)
            indexDtest_other = np.array(selected_imgs_idx_test_other)  # idx for Dtest
            query_x.append(np.array(self.data_unlabel)[indexDtest_other].tolist())

            # shuffle the correponding relation between support set and query set
            random.shuffle(support_x)
            random.shuffle(query_x)

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        # [setsz, 3, resize, resize]
        support_x = torch.FloatTensor(self.setsz, 3, self.resize, self.resize)
        # [setsz]
        support_y = np.zeros((self.setsz), dtype=np.int)
        # [querysz, 3, resize, resize]
        query_x = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)
        # [querysz]


        flatten_support_x = [os.path.join(self.path, item if item[-1] != "x" else item[:-1])
                             for sublist in self.support_x_batch[index] for item in sublist]
        support_y = np.array(
            [self.img2label[item.split('/')[0] if item[-1] != "x" else "x"]  # filename:n0153282900000005.jpg, the first 9 characters treated as label
             for sublist in self.support_x_batch[index] for item in sublist]).astype(np.int32)
        if not self.unlabel:
            flatten_query_x = [os.path.join(self.path, item if item[-1] != "x" else item[:-1])
                           for sublist in self.query_x_batch[index] for item in sublist]
        else:
            flatten_query_x = [item for sublist in self.query_x_batch[index] for item in sublist]
        # print('global:', support_y, query_y)
        # support_y: [setsz]
        # query_y: [querysz]
        # unique: [n-way], sorted
        unique = np.unique(support_y)
        random.shuffle(unique)
        # relative means the label ranges from 0 to n-way
        support_y_relative = np.zeros(self.setsz)

        for idx, l in enumerate(unique):
            support_y_relative[support_y == l] = idx

        # print('relative:', support_y_relative, query_y_relative)

        for i, path in enumerate(flatten_support_x):
            support_x[i] = self.transform(path)

        for i, path in enumerate(flatten_query_x):
            query_x[i] = self.transform(path)
        # print(support_set_y)
        # return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)
        if  not self.unlabel:
            query_y_relative = np.zeros(self.querysz)
            query_y = np.zeros((self.querysz), dtype=np.int)
            query_y = np.array([self.img2label[item.split('/')[0] if item[-1] != "x" else "x"]
                                for sublist in self.query_x_batch[index] for item in sublist]).astype(np.int32)
            for idx, l in enumerate(unique):
                query_y_relative[query_y == l] = idx
            return support_x, torch.LongTensor(support_y_relative), query_x, torch.LongTensor(query_y_relative), torch.LongTensor(query_y), flatten_query_x
        return support_x, torch.LongTensor(support_y_relative), query_x, torch.LongTensor(unique), flatten_query_x

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz

