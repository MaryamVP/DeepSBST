import os
import sys
import logging
import data_loader
from utils import *
import random
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
import math
import numpy as np
from tqdm import tqdm

class SearchEngine:
    def __init__(self, args, conf=None):
        self.data_path = args.data_path + args.dataset+'/' 
        self.train_params = conf.get('training_params', dict())
        self.data_params = conf.get('data_params',dict())
        self.model_params = conf.get('model_params',dict())
        
        self._eval_sets = None
        
        self._code_reprs = None
        self._codebase = None
        self._codebase_chunksize = 2000000

    ##### Model Loading / saving #####
    def save_model(self, model, epoch):
        model_path = f"./output/{model.__class__.__name__}/models/"
        os.makedirs(model_path, exist_ok=True)
        model.save(model_path + f"epo{epoch}_code.h5", model_path + f"epo{epoch}_desc.h5", overwrite=True)
        
    def load_model(self, model, epoch):
        model_path = f"./output/{model.__class__.__name__}/models/"
        assert os.path.exists(model_path + f"epo{epoch}_code.h5"),f"Weights at epoch {epoch} not found"
        assert os.path.exists(model_path + f"epo{epoch}_desc.h5"),f"Weights at epoch {epoch} not found"
        model.load(model_path + f"epo{epoch}_code.h5", model_path + f"epo{epoch}_desc.h5")


    ##### Training #####
    def train(self, model):
        if self.train_params['reload']>0:
            self.load_model(model, self.train_params['reload'])
        valid_every = self.train_params.get('valid_every', None)
        save_every = self.train_params.get('save_every', None)
        batch_size = self.train_params.get('batch_size', 128)
        nb_epoch = self.train_params.get('nb_epoch', 10)
        split = self.train_params.get('validation_split', 0)
        
        val_loss = {'loss': 1., 'epoch': 0}
        chunk_size = self.train_params.get('chunk_size', 100000)
        
        for i in range(self.train_params['reload']+1, nb_epoch):
            print('Epoch %d :: \n' % i, end='')  
            
            logger.debug('loading data chunk..')
            offset = (i-1)*self.train_params.get('chunk_size', 100000)
            
            
            names = data_loader.load_hdf5(self.data_path+self.data_params['train_methname'], offset, chunk_size)
            descs = data_loader.load_hdf5(self.data_path+self.data_params['train_desc'], offset, chunk_size)
            
            #print('names','names', names)
            File_object = open(r"names.txt","a")
            File_object.write(str(names))
            
            File_object = open(r"desc.txt","a")
            File_object.write(str(descs))
    
            logger.debug('padding data..')
            methnames = pad(names, self.data_params['methname_len'])
            good_descs = pad(descs,self.data_params['desc_len'])
            bad_descs=[desc for desc in descs]
            random.shuffle(bad_descs)
            bad_descs = pad(bad_descs, self.data_params['desc_len'])

            hist = model.fit([methnames, good_descs, bad_descs], epochs=1, batch_size=batch_size, validation_split=split)

            if hist.history['val_loss'][0] < val_loss['loss']:
                val_loss = {'loss': hist.history['val_loss'][0], 'epoch': i}
            print('Best: Loss = {}, Epoch = {}'.format(val_loss['loss'], val_loss['epoch']))
            
            if save_every is not None and i % save_every == 0:
                self.save_model(model, i)

            if valid_every is not None and i % valid_every == 0:                
                acc, mrr, map, ndcg = self.valid(model, 1000, 1)             

    ##### Evaluation in the develop set #####
    def valid(self, model, poolsize, K):
        """
        validate in a code pool. 
        param: poolsize - size of the code pool, if -1, load the whole test set
        """
        def ACC(real,predict):
            sum=0.0
            for val in real:
                try: index=predict.index(val)
                except ValueError: index=-1
                if index!=-1: sum=sum+1  
            return sum/float(len(real))
        def MAP(real,predict):
            sum=0.0
            for id,val in enumerate(real):
                try: index=predict.index(val)
                except ValueError: index=-1
                if index!=-1: sum=sum+(id+1)/float(index+1)
            return sum/float(len(real))
        def MRR(real,predict):
            sum=0.0
            for val in real:
                try: index=predict.index(val)
                except ValueError: index=-1
                if index!=-1: sum=sum+1.0/float(index+1)
            return sum/float(len(real))
        def NDCG(real,predict):
            dcg=0.0
            idcg=IDCG(len(real))
            for i,predictItem in enumerate(predict):
                if predictItem in real:
                    itemRelevance=1
                    rank = i+1
                    dcg+=(math.pow(2,itemRelevance)-1.0)*(math.log(2)/math.log(rank+1))
            return dcg/float(idcg)
        def IDCG(n):
            idcg=0
            itemRelevance=1
            for i in range(n):
                idcg+=(math.pow(2, itemRelevance)-1.0)*(math.log(2)/math.log(i+2))
            return idcg

        #load valid dataset
        
        print('self._eval_sets,', self._eval_sets)
        if self._eval_sets is None:
            methnames = data_loader.load_hdf5(self.data_path+self.data_params['valid_methname'], 0, poolsize)
            descs = data_loader.load_hdf5(self.data_path+self.data_params['valid_desc'], 0, poolsize) 
            self._eval_sets={'methnames':methnames, 'descs':descs}
            
        accs,mrrs,maps,ndcgs = [], [], [], []
        data_len = len(self._eval_sets['descs'])
        print('data_len',data_len)
        for i in tqdm(range(data_len)):
            desc=self._eval_sets['descs'][i]#good desc
            descs = pad([desc]*data_len,self.data_params['desc_len'])
            methnames = pad(self._eval_sets['methnames'],self.data_params['methname_len'])
            n_results = K          
            sims = model.predict([methnames, descs], batch_size=data_len).flatten()
            negsims= np.negative(sims)
            predict = np.argpartition(negsims, kth=n_results-1)
            predict = predict[:n_results]   
            predict = [int(k) for k in predict]
            real=[i]
            accs.append(ACC(real,predict))
            mrrs.append(MRR(real,predict))
            maps.append(MAP(real,predict))
            ndcgs.append(NDCG(real,predict))                          
        logger.info(f'ACC={np.mean(accs)}, MRR={np.mean(mrrs)}, MAP={np.mean(maps)}, nDCG={np.mean(ndcgs)}')        
        return accs,mrrs,maps,ndcgs
    
    
    ##### Compute Representation #####
    def repr_code(self, model):
        logger.info('Loading the use data ..')
        methnames = data_loader.load_hdf5(self.data_path+self.data_params['use_methname'],0,-1)
        methnames = pad(methnames, self.data_params['methname_len'])
        
        logger.info('Representing code ..')
        vecs= model.repr_code([methnames], batch_size=10000)
        vecs= vecs.astype(np.float)
        vecs= normalize(vecs)
        return vecs
            
    