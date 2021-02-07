import argparse
import torch
import os
import torchvision
import utils
import simclr
from PIL import Image
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# making a command line interface
parser = argparse.ArgumentParser(description="This is the command line interface for the linear evaluation model")

parser.add_argument('datapath', type=str ,help="Path to the data root folder which contains train and test folders")

parser.add_argument('model_path', type=str, help="Path to the trained self-supervised model")

parser.add_argument('respath', type=str, help="Path to the results where the evaluation metrics would be stored. ")

parser.add_argument('-bs','--batch_size',default=250, type=int, help="The batch size for evaluation")

parser.add_argument('-nw','--num_workers',default=2,type=int,help="The number of workers for loading data")

parser.add_argument('-c','--cuda',action='store_true')

parser.add_argument('--multiple_gpus', action='store_true')

parser.add_argument('--remove_top_layers', default=1, type=int)


class TrainDataset(torch.utils.data.Dataset):

    def __init__(self, args):
        self.args = args
        
        with open(os.path.join(args.datapath,'train','train.json')) as f:
            self.filedict = json.load(f)

        with open(os.path.join(args.datapath,'mapper.json')) as f:
            self.mapper = json.load(f)

        self.filenames = list(self.filedict)
    
    def __len__(self):
        return len(self.filenames)

    def tensorify(self, img):
        return torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(
            torchvision.transforms.ToTensor()(img)
            )

    def __getitem__(self, idx):
        return {
        'image':self.tensorify(
                    torchvision.transforms.Resize((224, 224))(
                                Image.open(os.path.join(args.datapath, 'train', self.filenames[idx])).convert('RGB')
                            )
                    ), 
        'label':self.mapper[self.filedict[self.filenames[idx]]]
        }


class TestDataset(torch.utils.data.Dataset):

    def __init__(self, args):
        self.args = args
        
        with open(os.path.join(args.datapath,'test','test.json')) as f:
            self.filedict = json.load(f)

        with open(os.path.join(args.datapath,'mapper.json')) as f:
            self.mapper = json.load(f)

        self.filenames = list(self.filedict)
    
    def __len__(self):
        return len(self.filenames)

    def tensorify(self, img):
        return torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(
            torchvision.transforms.ToTensor()(img)
            )

    def __getitem__(self, idx):
        return {
        'image':self.tensorify(
                    torchvision.transforms.Resize((224, 224))(
                                Image.open(os.path.join(args.datapath, 'test', self.filenames[idx])).convert('RGB')
                            )
                    ), 
        'label':self.mapper[self.filedict[self.filenames[idx]]]
        }


if __name__ == '__main__':
    args = parser.parse_args()
    args.device = torch.device('cuda' if args.cuda else 'cpu')
    model = utils.model.get_model(args)

    dataloaders = {}
    
    dataloaders['train'] = torch.utils.data.DataLoader(
        TrainDataset(args), 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
        )
    
    dataloaders['test'] = torch.utils.data.DataLoader(
        TestDataset(args), 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
        )

    simclrobj = simclr.SimCLR(
        model, 
        None, 
        dataloaders, 
        None
        )
    simclrobj.load_model(args)


    reprs = {}

    for mode in ['train', 'test']:
        reprs[mode] = simclrobj.get_representations(args, mode=mode)    

    scaler = StandardScaler().fit(reprs['train']['X'])

    Xtrain = scaler.transform(reprs['train']['X'])
    Xtest = scaler.transform(reprs['test']['X'])

    clf = LogisticRegression(
        multi_class='multinomial', 
        max_iter=1000, 
        n_jobs=16,
        ).fit(
        Xtrain, reprs['train']['Y']
        )
    
    ypred = clf.predict(Xtest)
    print(
        classification_report(
        reprs['test']['Y'], 
        ypred, 
        digits=4, 
        target_names=['car', 'airplane', 'elephant', 'dog', 'cat'])
    )
