import os
import sys
import random
import time
import traceback
from tensorflow.keras.optimizers import RMSprop, Adam
from scipy.stats import rankdata
import math
import numpy as np
from tqdm import tqdm
import argparse
random.seed(42)
import threading 
import configs
import re
import DNNTestSuite
from record import record
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

from utils import cos_np, normalize, cos_np_for_normalized, pad, convert, revert
import models, configs, data_loader

class SearchEngine:
    def __init__(self, args, conf=None):
        self.data_path = args.data_path + args.dataset+'/' 
        self.train_params = conf.get('training_params', dict())
        self.data_params = conf.get('data_params', dict())
        self.model_params = conf.get('model_params', dict())
        
        self._eval_sets = None
        
        self._code_reprs=None
        self._codebase=None
        self._codebase_chunksize=2000000

    ##### Model Loading / saving #####
    def save_model(self, model, epoch):
        model_path = "./output/{}/models/".format(model.__class__.__name__)
        #model_path = f"./output/{model.__class__.__name__}/models/"
        os.makedirs(model_path, exist_ok=True)
        model.save(model_path + "epo{}_code.h5".format(epoch), model_path + "epo{}_desc.h5".format(epoch), overwrite=True)
        
    def load_model(self, model, epoch):
        model_path = "./output/{}/models/".format(model.__class__.__name__)
        assert os.path.exists(model_path + "epo{}_code.h5".format(epoch)),"Weights at epoch {} not found".format(epoch)
        assert os.path.exists(model_path + "epo{}_desc.h5".format(epoch)),"Weights at epoch {} not found".format(epoch)
        model.load(model_path + "epo{}_code.h5".format(epoch), model_path + "epo{}_desc.h5".format(epoch))


    ##### Training #####
    def train(self, model):
        if self.train_params['reload'] > 0:
            self.load_model(model, self.train_params['reload'])
        valid_every = self.train_params.get('valid_every', None)
        save_every = self.train_params.get('save_every', None)
        batch_size = self.train_params.get('batch_size', 128)
        nb_epoch = self.train_params.get('nb_epoch', 10)
        split = self.train_params.get('validation_split', 0)
        
        val_loss = {'loss': 1., 'epoch': 0}
        chunk_size = self.train_params.get('chunk_size', 100000)
        
        for i in range(self.train_params['reload']+1, nb_epoch):
            print('Epoch %d :: \n' % i)  
            
            logger.debug('loading data chunk..')
            offset = (i-1)*self.train_params.get('chunk_size', 100000)
            
            names = data_loader.load_hdf5(self.data_path+self.data_params['train_methname'], offset, chunk_size)
            apis = data_loader.load_hdf5(self.data_path+self.data_params['train_apiseq'], offset, chunk_size)
            tokens = data_loader.load_hdf5(self.data_path+self.data_params['train_tokens'], offset, chunk_size)
            descs = data_loader.load_hdf5(self.data_path+self.data_params['train_desc'], offset, chunk_size)
            
            logger.debug('padding data..')
            methnames = pad(names, self.data_params['methname_len'])
            apiseqs = pad(apis, self.data_params['apiseq_len'])
            tokens = pad(tokens, self.data_params['tokens_len'])
            good_descs = pad(descs, self.data_params['desc_len'])
            bad_descs = [desc for desc in descs]
            random.shuffle(bad_descs)
            bad_descs = pad(bad_descs, self.data_params['desc_len'])

            hist = model.fit([methnames, apiseqs, tokens, good_descs, bad_descs], epochs=1, batch_size=batch_size, validation_split=split)

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
        if self._eval_sets is None:
            methnames = data_loader.load_hdf5(self.data_path+self.data_params['valid_methname'], 0, poolsize)
            apiseqs= data_loader.load_hdf5(self.data_path+self.data_params['valid_apiseq'], 0, poolsize)
            tokens = data_loader.load_hdf5(self.data_path+self.data_params['valid_tokens'], 0, poolsize)
            descs = data_loader.load_hdf5(self.data_path+self.data_params['valid_desc'], 0, poolsize) 
            self._eval_sets={'methnames':methnames, 'apiseqs':apiseqs, 'tokens':tokens, 'descs':descs}
            
        acc,mrr,map,ndcg = 0,0,0,0
        data_len = len(self._eval_sets['descs'])
        for i in tqdm(range(data_len)):
            desc=self._eval_sets['descs'][i]#good desc
            descs = pad([desc]*data_len,self.data_params['desc_len'])
            methnames = pad(self._eval_sets['methnames'],self.data_params['methname_len'])
            apiseqs= pad(self._eval_sets['apiseqs'],self.data_params['apiseq_len'])
            tokens= pad(self._eval_sets['tokens'],self.data_params['tokens_len'])
            n_results = K          
            sims = model.predict([methnames, apiseqs,tokens, descs], batch_size=data_len).flatten()
            negsims= np.negative(sims)
            predict= np.argsort(negsims)#predict = np.argpartition(negsims, kth=n_results-1)
            predict = predict[:n_results]   
            predict = [int(k) for k in predict]
            real=[i]
            acc+=ACC(real,predict)
            mrr+=MRR(real,predict)
            map+=MAP(real,predict)
            ndcg+=NDCG(real,predict)               
        acc = acc / float(data_len)
        mrr = mrr / float(data_len)
        map = map / float(data_len)
        ndcg= ndcg/ float(data_len)
        logger.info('ACC={}, MRR={}, MAP={}, nDCG={}'.format(acc,mrr,map,ndcg))        
        return acc,mrr,map,ndcg
    
    
    ##### Compute Representation #####
    def repr_code(self, model):
        logger.info('Loading the use data ..')
        methnames = data_loader.load_hdf5(self.data_path+self.data_params['use_methname'],0,-1)
        apiseqs = data_loader.load_hdf5(self.data_path+self.data_params['use_apiseq'],0,-1)
        tokens = data_loader.load_hdf5(self.data_path+self.data_params['use_tokens'],0,-1) 
        methnames = pad(methnames, self.data_params['methname_len'])
        apiseqs = pad(apiseqs, self.data_params['apiseq_len'])
        tokens = pad(tokens, self.data_params['tokens_len'])
        
        logger.info('Representing code ..')
        vecs= model.repr_code([methnames, apiseqs, tokens], batch_size=1000)
        vecs= vecs.astype('float32')
        vecs= normalize(vecs)
        return vecs
            
    
    def search(self, model, vocab, query, n_results=10):
        desc=[convert(vocab, query)]#convert desc sentence to word indices
        padded_desc = pad(desc, self.data_params['desc_len'])
        desc_repr = model.repr_desc([padded_desc])
        desc_repr = desc_repr.astype('float32')
        
        codes, sims = [], []
        threads = []
        for i,code_reprs_chunk in enumerate(self._code_reprs):
            t = threading.Thread(target=self.search_thread, args = (codes,sims,desc_repr,code_reprs_chunk,i,n_results))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:#wait until all sub-threads finish
            t.join()
        return codes,sims
                 
    def search_thread(self, codes, sims, desc_repr, code_reprs, i, n_results):        
    #1. compute similarity
        chunk_sims=cos_np_for_normalized(normalize(desc_repr),code_reprs) 
        
    #2. choose top results
        negsims=np.negative(chunk_sims[0])
        maxinds = np.argpartition(negsims, kth=n_results-1)
        maxinds = maxinds[:n_results]        
        chunk_codes=[self._codebase[i][k] for k in maxinds]
        chunk_sims=chunk_sims[0][maxinds]
        codes.extend(chunk_codes)
        sims.extend(chunk_sims)
        
    def postproc(self,codes_sims):
        codes_, sims_ = zip(*codes_sims)
        codes= [code for code in codes_]
        sims= [sim for sim in sims_]
        final_codes=[]
        final_sims=[]
        n=len(codes_sims)        
        for i in range(n):
            is_dup=False
            for j in range(i):
                if codes[i][:80]==codes[j][:80] and abs(sims[i]-sims[j])<0.01:
                    is_dup=True
            if not is_dup:
                final_codes.append(codes[i])
                final_sims.append(sims[i])
        return zip(final_codes,final_sims)

    
def parse_args():
    parser = argparse.ArgumentParser("Train and Test Code Search(Embedding) Model")
    parser.add_argument("--data_path", type=str, default='./data/', help="working directory")
    parser.add_argument("--model", type=str, default="JointEmbeddingModel", help="model name")
    parser.add_argument("--dataset", type=str, default="github", help="dataset name")
    parser.add_argument("--mode", choices=["train","eval","repr_code","search"], default='train',
                        help="The mode to run. The `train` mode trains a model;"
                        " the `eval` mode evaluat models in a test set "
                        " The `repr_code/repr_desc` mode computes vectors"
                        " for a code snippet or a natural language description with a trained model.")
    parser.add_argument("--verbose",action="store_true", default=True, help="Be verbose")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = getattr(configs, 'config_'+args.model)()
    engine = SearchEngine(args, config)

    ##### Define model ######
    logger.info('Build Model')

    #tf.compat.v1.global_variables_initializer()
    model = getattr(models, args.model)(config)  # initialize the model
    model.build()
    model.summary(export_path = "./output/{}/".format(args.model))
    
    optimizer = config.get('training_params', dict()).get('optimizer', 'adam')
    model.compile(optimizer=optimizer)  

    data_path = args.data_path+args.dataset+'/'
    mode = 'test'
    filename = './log_folder/record.txt'
    threshold_CC = 0
    threshold_MC = 0.7
    symbols_SQ = 2
    seq = '[0 ,1]'
    seq = re.findall(r"\d+\.?\d*", seq)
    TargMetri = None
    CoverageStop = 0.9
    TestCaseNum = 2000
    minimalTest = 0
    r = record(filename, time.time())
    data_param = config.get('data_params', dict())
    
    ### Loading training set
    names = data_loader.load_hdf5(data_path + data_param['train_methname'], 0, data_param.get('chunk_size', 100000))
    methnames = pad(names, data_param['methname_len'])
    print(methnames[0])
    z = methnames[0].reshape(1, 6)
    print(z)
    z[0][1] = 0
    print(z)
    apis = data_loader.load_hdf5(data_path + data_param['train_apiseq'], 0, data_param.get('chunk_size', 100000),)
    tokens = data_loader.load_hdf5(data_path + data_param['train_tokens'], 0, data_param.get('chunk_size', 100000),)
    apiseqs = pad(apis, data_param['apiseq_len'])
    tokens = pad(tokens, data_param['tokens_len'])
    
    
    
    
    mutated_mn, mutated_api, mutated_token = DNNTestSuite.generateTestSuite(r, threshold_CC, threshold_MC, symbols_SQ, seq, TestCaseNum, minimalTest, TargMetri, CoverageStop, model, methnames, apiseqs, tokens)
    descs = data_loader.load_hdf5(data_path + data_param['train_desc'], 0, 100000)
    
    
    number_of_samples = 5
    
    #mutation for description
    good_descs = pad(descs, data_param['desc_len'])
    print(good_descs)
    bad_descs = [desc for desc in descs]
    random.shuffle(bad_descs)
    bad_descs = pad(bad_descs, data_param['desc_len'])
    for i in range(0, number_of_samples):
        loc_del = random.randint(0, 29)
        loc_ins = random.randint(0, 29)
        loc_swap = random.randint(0, 29)
        loc_swap1 = random.randint(0, 29)
        while loc_swap1 == loc_swap:
            loc_swap1 = random.randint(0, 29)
        good_descs[i][loc_del] = 0
        bad_descs[i][loc_del] = 0
        good_descs[i][loc_ins] = random.randint(0, 10000)
        bad_descs[i][loc_ins] = random.randint(0, 10000)
        tempg = good_descs[i][loc_swap]
        tempd = good_descs[i][loc_swap]
        good_descs[i][loc_swap] = good_descs[i][loc_swap1]
        good_descs[i][loc_swap1] = tempg
        bad_descs[i][loc_swap] = bad_descs[i][loc_swap1]
        bad_descs[i][loc_swap1] = tempd
    

    
    methnames = np.array(methnames).reshape(100000, 6)
    apiseqs = np.array(apiseqs).reshape(100000, 30)
    tokens = np.array(tokens).reshape(100000, 50)
    methnames = methnames[:number_of_samples, ]
    apiseqs = apiseqs[:number_of_samples, ]
    tokens = tokens[:number_of_samples, ]

    good_descs = good_descs[:2*number_of_samples, ]
    bad_descs = bad_descs[:2*number_of_samples, ]

    mutated_mn = np.array(mutated_mn).reshape(number_of_samples, 6)
    methnames = np.concatenate((methnames, mutated_mn), axis=0)
    mutated_api = np.array(mutated_api).reshape(number_of_samples, 30)
    apiseqs = np.concatenate((apiseqs, mutated_api))
    mutated_token = np.array(mutated_token).reshape(number_of_samples, 50)
    tokens = np.concatenate((tokens, mutated_token))
    
    
    #retrain the model
    hist = model.fit([methnames, apiseqs, tokens, good_descs, bad_descs], epochs=1, batch_size=128, validation_split=0.2)
    
    #validate the trained model
    engine.valid(model, 2*number_of_samples , int(number_of_samples/2))


