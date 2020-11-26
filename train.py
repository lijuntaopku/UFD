import torch
from model import Adaptor_domain, Adaptor_global, Max_Discriminator, Min_Discriminator, Classifier, Combine_features_map
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch
import sys 
import time



def load_train(filename):
    """
    load file for training
    """
    data = []
    with open (filename, 'r') as F:
        for line in F:
            data.append(line.split('\t'))
        print ("number of training pairs is", len(data))

    return data


def load_test(filename):
    """
    load file for evaluation or testing
    """
    with open(filename, 'r') as F:
        data = []
        for line in F:
            data.append(line.split('\t'))
        print ("number of testing pairs is", len(data))
        return data


def load_raw(filename):
    """
    load file for evaluation or distillation
    """
    with open(filename, 'r') as F:
        data = []
        for line in F:
            data.append([0,line.strip()])
        print ("number of raw pairs is", len(data))
        return data


def xlm_r(data):
    """
    obtain pretrained xlm_r representations
    """
    xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.large').to(device)
    xlmr.eval()
    embeded_data=[]
    with torch.no_grad():
        for pair in data:
            tokens = xlmr.encode(pair[1])
            if list(tokens.size())[0] > 512:
                tokens=tokens[:512]        
            
            last_layer_features = xlmr.extract_features(tokens) #1 * length * embedding_size
            mean_features=torch.mean(last_layer_features, dim=1).to(device) # 1 * embedding_size
            tem_label = torch.tensor([[float(pair[0])]]).to(device)
            new_pair = torch.cat((tem_label, mean_features), dim=1)
            embeded_data.append(new_pair)
        return embeded_data # number_paris * 1 * embedding_size


class DeepInfoMaxLoss(nn.Module):
    """
    the objective of unsuperivised feature decomposition module
    """
    def __init__(self, alpha=0.3, beta=1, gamma=0.2, delta=1):
        super().__init__()
        self.max_d = Max_Discriminator(dim_hidden, initrange)
        self.min_d = Min_Discriminator(dim_hidden, initrange)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        print ('alpha is', self.alpha)
        print ('beta is', self.beta)
        print ('gamma is', self.gamma)
        print ('delta is', self.delta)

    def forward(self, x, x_n, f_g, fg_n, f_d, fd_n, y_g, y_d, yd_n):
        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf
        # Global reserved for xlm_r representations and global shared representations, Max-MI
        # Local refers to, Min-MI

        Ej = -F.softplus(-self.max_d(y_g, f_g)).mean()
        Em = F.softplus(self.max_d(y_g, fg_n)).mean()
        GLOBAL_A = (Em - Ej) * self.alpha


        Ej = -F.softplus(-self.max_d(x, y_g)).mean()
        Em = F.softplus(self.max_d(x_n, y_g)).mean()
        GLOBAL_B = (Em - Ej) * self.delta

        Ej = -F.softplus(-self.min_d(y_d, y_g)).mean()
        Em = F.softplus(self.min_d(yd_n, y_g)).mean()
        Local_B = (Ej - Em) * self.gamma

        return GLOBAL_A + GLOBAL_B + Local_B #training objective, see equation (6) of https://www.ijcai.org/Proceedings/2020/0508.pdf


def train_adaptor(train_data):
    """
    function for training the adaptor
    """   
    t_loss = 0 
    data = DataLoader(train_data, batch_size=b_size, shuffle=True)
    for i, pairs in enumerate(data):
        # print (pairs.size())
        x = pairs[:, :, 1:]
        x=torch.squeeze(x)
        x = x.to(device)
               
        optim.zero_grad()
        loss_optim.zero_grad()   

        y_g, f_g = adaptor_global(x)
        y_d, f_d = adaptor_domain(x)

        x_n = torch.cat((x[1:], x[0].unsqueeze(0)), dim=0)
        fg_n = torch.cat((f_g[1:], f_g[0].unsqueeze(0)), dim=0)
        fd_n = torch.cat((f_d[1:], f_d[0].unsqueeze(0)), dim=0)
        yd_n = torch.cat((y_d[1:], y_d[0].unsqueeze(0)), dim=0)

        loss_a = loss_fn(x, x_n, f_g, fg_n, f_d, fd_n, y_g, y_d, yd_n)
        t_loss += loss_a.item()

        loss_a.backward()
        optim.step()
        loss_optim.step()

    return t_loss


def train(train_data, cross_domain = False):
    """
    function for training the classifier
    """
    # Train the model
    train_loss = 0
    train_acc = 0
    data = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    for i, pairs in enumerate(data):
        text = pairs[:, :, 1:]
        index = pairs[:, :, :1]
        text=torch.squeeze(text)
        index = torch.squeeze(index)
        index = index.long()
        
        optimizer.zero_grad()
        optim_fusion.zero_grad()
        text, index = text.to(device), index.to(device)

        if cross_domain:
            global_f,_ = adaptor_global(text)
            output = model(global_f)
        else:
            global_f,_ = adaptor_global(text)
            domain_f,_ = adaptor_domain(text)
            features = torch.cat((global_f, domain_f), dim=1)
            output=model(maper(features))

        loss = criterion(output, index)
        train_loss += loss.item()
        loss.backward(retain_graph=True)
        optimizer.step()
        optim_fusion.step()
        train_acc += (output.argmax(1) == index).sum().item()

    return train_loss / len(train_data), train_acc / len(train_data)


def test(test_data, cross_domain = False):
    """
    function for evaluating the classifier
    """
    loss = 0
    acc = 0
    data = DataLoader(test_data, batch_size=BATCH_SIZE)
    for i, pairs in enumerate(data):
        text = pairs[:, :, 1:]
        index = pairs[:, :, :1]
        text=torch.squeeze(text)
        index = torch.squeeze(index)
        index = index.long()
        text, index = text.to(device), index.to(device)
        with torch.no_grad():
            if cross_domain:
                global_f,_ = adaptor_global(text)
                output = model(global_f)
            else:
                global_f,_ = adaptor_global(text)
                domain_f,_ = adaptor_domain(text)
                features = torch.cat((global_f, domain_f), dim=1)
                output=model(maper(features))
            l = criterion(output, index)
            loss += l.item()
            acc += (output.argmax(1) == index).sum().item()           

    return loss / len(test_data), acc / len(test_data)



if __name__ == '__main__':
    #hyper-parameters for UFD
    device = torch.device('cuda:0')
    b_size = 16
    epochs = 30
    in_dim = 1024
    dim_hidden = 1024
    out_dim = 1024
    
    #setting for the whole model
    initrange = 0.1

    #settings for task-specific module
    N_EPOCHS = 60
    BATCH_SIZE = 8    
    NUN_CLASS =2
    min_valid_loss = float('inf')

    print ('raw batch size', b_size)
    print ('Num_class', NUN_CLASS)
    print ('Embedding size', out_dim)
    print ('Number of epochs', N_EPOCHS)
    print ('Batch size', BATCH_SIZE)
    print ('initrange', initrange)
    
    #adaptor for domain-specific feature extraction
    adaptor_domain = Adaptor_domain(in_dim, dim_hidden, out_dim,initrange).to(device)
    #adaptor for domain-invariant feature extraction    
    adaptor_global = Adaptor_global(in_dim, dim_hidden, out_dim,initrange).to(device)
    #combine domain-invariant feature and domain-specific feature
    maper = Combine_features_map(in_dim,initrange).to(device)

    optim_fusion = Adam(maper.parameters(), lr=1e-4)
    loss_fn = DeepInfoMaxLoss().to(device)
    optim = Adam([{"params":adaptor_domain.parameters()},{"params":adaptor_global.parameters()}], lr=1e-4)
    loss_optim = Adam(loss_fn.parameters(), lr=1e-4)
    
    #load raw text for training UFD
    raw_data = load_raw('./data/raw.15.txt')
    #load data from source language and source domain
    train_book = load_train('./data/en/books/train.txt')
    train_dvd = load_train('./data/en/dvd/train.txt')
    train_music = load_train('./data/en/music/train.txt')

    #processing data with XLM-R model
    raw_data=xlm_r(raw_data)
    train_book=xlm_r(train_book)
    train_dvd=xlm_r(train_dvd)
    train_music=xlm_r(train_music)

    #load validation sets
    valid_fr_book = load_test('./data/fr/books/sampled.txt')
    valid_fr_dvd =  load_test('./data/fr/dvd/sampled.txt')
    valid_fr_music = load_test('./data/fr/music/sampled.txt')
    valid_jp_book = load_test('./data/jp/books/sampled.txt')
    valid_jp_dvd = load_test('./data/jp/dvd/sampled.txt')
    valid_jp_music = load_test('./data/jp/music/sampled.txt')
    valid_de_book = load_test('./data/de/books/sampled.txt')
    valid_de_dvd = load_test('./data/de/dvd/sampled.txt')
    valid_de_music = load_test('./data/de/music/sampled.txt')

    #processing validation sets with XLM-R model
    valid_fr_book=xlm_r(valid_fr_book)
    valid_fr_dvd=xlm_r(valid_fr_dvd)
    valid_fr_music=xlm_r(valid_fr_music)
    valid_jp_book=xlm_r(valid_jp_book)
    valid_jp_dvd=xlm_r(valid_jp_dvd)
    valid_jp_music=xlm_r(valid_jp_music)
    valid_de_book=xlm_r(valid_de_book)
    valid_de_dvd=xlm_r(valid_de_dvd)
    valid_de_music=xlm_r(valid_de_music)

    #load test sets
    test_fr_book = load_test('./data/fr/books/test.txt')
    test_fr_dvd =  load_test('./data/fr/dvd/test.txt')
    test_fr_music = load_test('./data/fr/music/test.txt')
    test_jp_book = load_test('./data/jp/books/test.txt')
    test_jp_dvd = load_test('./data/jp/dvd/test.txt')
    test_jp_music = load_test('./data/jp/music/test.txt')
    test_de_book = load_test('./data/de/books/test.txt')
    test_de_dvd = load_test('./data/de/dvd/test.txt')
    test_de_music = load_test('./data/de/music/test.txt')

    #processing test sets with XLM-R model
    test_fr_book=xlm_r(test_fr_book)
    test_fr_dvd=xlm_r(test_fr_dvd)
    test_fr_music=xlm_r(test_fr_music)
    test_jp_book=xlm_r(test_jp_book)
    test_jp_dvd=xlm_r(test_jp_dvd)
    test_jp_music=xlm_r(test_jp_music)
    test_de_book=xlm_r(test_de_book)
    test_de_dvd=xlm_r(test_de_dvd)
    test_de_music=xlm_r(test_de_music)

    # de, fr, jp represent the target language are German, French, and Japanese, respectively; b,d,m represent the target domains are book, dvd, music, respectively.
    #l refers to loss; valid refer to the performance on validation set
    de_b_valid_l_m =10000.0
    #a corresponds to accuracy
    de_b_test_a_m =0

    de_b_valid_l_d =10000.0
    de_b_test_a_d =0        

    de_d_valid_l_b =10000.0
    de_d_test_a_b =0

    de_d_valid_l_m =10000.0
    de_d_test_a_m =0

    de_m_valid_l_b =10000.0
    de_m_test_a_b =0     

    de_m_valid_l_d =10000.0
    de_m_test_a_d =0  

    #French
    fr_b_valid_l_m =10000.0
    fr_b_test_a_m =0

    fr_b_valid_l_d =10000.0
    fr_b_test_a_d =0        

    fr_d_valid_l_b =10000.0
    fr_d_test_a_b =0

    fr_d_valid_l_m =10000.0
    fr_d_test_a_m =0

    fr_m_valid_l_b =10000.0
    fr_m_test_a_b =0     

    fr_m_valid_l_d =10000.0
    fr_m_test_a_d =0  

    #Japanese 
    jp_b_valid_l_m =10000.0
    jp_b_test_a_m =0

    jp_b_valid_l_d =10000.0
    jp_b_test_a_d =0        

    jp_d_valid_l_b =10000.0
    jp_d_test_a_b =0

    jp_d_valid_l_m =10000.0
    jp_d_test_a_m =0

    jp_m_valid_l_b =10000.0
    jp_m_test_a_b =0     

    jp_m_valid_l_d =10000.0
    jp_m_test_a_d =0  


    for epoch in range(epochs):
        start_time = time.time()
        # train_adaptor(raw_data)
        a_loss = train_adaptor(raw_data)

        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60

        # print('Epoch: %d' %(epoch), " | time in %d minutes, %d seconds" %(mins, secs))

        if epoch >0:
            model = Classifier(out_dim, NUN_CLASS, initrange).to(device)
            criterion = torch.nn.CrossEntropyLoss().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            de_b_d_valid_loss=10000.0
            de_b_d_test_accuracy=0.0

            de_b_m_valid_loss=10000.0
            de_b_m_test_accuracy=0.0

            fr_b_d_valid_loss=10000.0
            fr_b_d_test_accuracy=0.0

            fr_b_m_valid_loss=10000.0
            fr_b_m_test_accuracy=0.0

            jp_b_d_valid_loss=10000.0
            jp_b_d_test_accuracy=0.0

            jp_b_m_valid_loss=10000.0
            jp_b_m_test_accuracy=0.00            

            for ep in range(N_EPOCHS):
                if ep >5:
                    start_time = time.time()
                    train_loss, train_acc = train(train_book)
                    
                    #source language and source domain is English book, target language and domain is German dvd
                    eval_de_b_d_loss, eval_de_b_d_loss_acc = test(valid_de_dvd)
                    test_de_b_d_loss, test_de_b_d_loss_acc = test(test_de_dvd) 

                    #source language and source domain is English book, target language and domain is German music                   
                    eval_de_b_m_loss, eval_de_b_m_loss_acc = test(valid_de_music)
                    test_de_b_m_loss, test_de_b_m_loss_acc = test(test_de_music) 

                    #source language and source domain is English book, target language and domain is French dvd
                    eval_fr_b_d_loss, eval_fr_b_d_loss_acc = test(valid_fr_dvd)
                    test_fr_b_d_loss, test_fr_b_d_loss_acc = test(test_fr_dvd) 

                    #source language and source domain is English book, target language and domain is French music  
                    eval_fr_b_m_loss, eval_fr_b_m_loss_acc = test(valid_fr_music)
                    test_fr_b_m_loss, test_fr_b_m_loss_acc = test(test_fr_music) 

                    #source language and source domain is English book, target language and domain is Japanese dvd 
                    eval_jp_b_d_loss, eval_jp_b_d_loss_acc = test(valid_jp_dvd)
                    test_jp_b_d_loss, test_jp_b_d_loss_acc = test(test_jp_dvd) 

                    #source language and source domain is English book, target language and domain is Japanese music
                    eval_jp_b_m_loss, eval_jp_b_m_loss_acc = test(valid_jp_music)
                    test_jp_b_m_loss, test_jp_b_m_loss_acc = test(test_jp_music) 

                    #pick up the checkpoint with the best validation loss
                    if eval_de_b_d_loss<de_b_d_valid_loss:
                        de_b_d_valid_loss= eval_de_b_d_loss
                        de_b_d_test_accuracy=test_de_b_d_loss_acc

                    if eval_de_b_m_loss<de_b_m_valid_loss:
                        de_b_m_valid_loss= eval_de_b_m_loss
                        de_b_m_test_accuracy=test_de_b_m_loss_acc

                    if eval_fr_b_d_loss<fr_b_d_valid_loss:
                        fr_b_d_valid_loss= eval_fr_b_d_loss
                        fr_b_d_test_accuracy=test_fr_b_d_loss_acc
                    
                    if eval_fr_b_m_loss<fr_b_m_valid_loss:
                        fr_b_m_valid_loss= eval_fr_b_m_loss
                        fr_b_m_test_accuracy=test_fr_b_m_loss_acc

                    if eval_jp_b_d_loss<jp_b_d_valid_loss:
                        jp_b_d_valid_loss= eval_jp_b_d_loss
                        jp_b_d_test_accuracy=test_jp_b_d_loss_acc

                    if eval_jp_b_m_loss<jp_b_m_valid_loss:
                        jp_b_m_valid_loss= eval_jp_b_m_loss
                        jp_b_m_test_accuracy=test_jp_b_m_loss_acc

            model = Classifier(out_dim, NUN_CLASS, initrange).to(device)
            criterion = torch.nn.CrossEntropyLoss().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            de_d_b_valid_loss=10000.0
            de_d_b_test_accuracy=0.0

            de_d_m_valid_loss=10000.0
            de_d_m_test_accuracy=0.0

            fr_d_b_valid_loss=10000.0
            fr_d_b_test_accuracy=0.0

            fr_d_m_valid_loss=10000.0
            fr_d_m_test_accuracy=0.0

            jp_d_b_valid_loss=10000.0
            jp_d_b_test_accuracy=0.0

            jp_d_m_valid_loss=10000.0
            jp_d_m_test_accuracy=0.00            

            for ep in range(N_EPOCHS):
                if ep >5:
                    start_time = time.time()
                    train_loss, train_acc = train(train_dvd)

                    eval_de_d_b_loss, eval_de_d_b_loss_acc = test(valid_de_book)
                    test_de_d_b_loss, test_de_d_b_loss_acc = test(test_de_book) 

                    eval_de_d_m_loss, eval_de_d_m_loss_acc = test(valid_de_music)
                    test_de_d_m_loss, test_de_d_m_loss_acc = test(test_de_music) 

                    eval_fr_d_b_loss, eval_fr_d_b_loss_acc = test(valid_fr_book)
                    test_fr_d_b_loss, test_fr_d_b_loss_acc = test(test_fr_book) 

                    eval_fr_d_m_loss, eval_fr_d_m_loss_acc = test(valid_fr_music)
                    test_fr_d_m_loss, test_fr_d_m_loss_acc = test(test_fr_music) 

                    eval_jp_d_b_loss, eval_jp_d_b_loss_acc = test(valid_jp_book)
                    test_jp_d_b_loss, test_jp_d_b_loss_acc = test(test_jp_book) 

                    eval_jp_d_m_loss, eval_jp_d_m_loss_acc = test(valid_jp_music)
                    test_jp_d_m_loss, test_jp_d_m_loss_acc = test(test_jp_music) 

                    #pick up the checkpoint with the best validation loss
                    if eval_de_d_b_loss<de_d_b_valid_loss:
                        de_d_b_valid_loss= eval_de_d_b_loss
                        de_d_b_test_accuracy=test_de_d_b_loss_acc

                    if eval_de_d_m_loss<de_d_m_valid_loss:
                        de_d_m_valid_loss= eval_de_d_m_loss
                        de_d_m_test_accuracy=test_de_d_m_loss_acc

                    if eval_fr_d_b_loss<fr_d_b_valid_loss:
                        fr_d_b_valid_loss= eval_fr_d_b_loss
                        fr_d_b_test_accuracy=test_fr_d_b_loss_acc
                    
                    if eval_fr_d_m_loss<fr_d_m_valid_loss:
                        fr_d_m_valid_loss= eval_fr_d_m_loss
                        fr_d_m_test_accuracy=test_fr_d_m_loss_acc

                    if eval_jp_d_b_loss<jp_d_b_valid_loss:
                        jp_d_b_valid_loss= eval_jp_d_b_loss
                        jp_d_b_test_accuracy=test_jp_d_b_loss_acc

                    if eval_jp_d_m_loss<jp_d_m_valid_loss:
                        jp_d_m_valid_loss= eval_jp_d_m_loss
                        jp_d_m_test_accuracy=test_jp_d_m_loss_acc
       
            model = Classifier(out_dim, NUN_CLASS, initrange).to(device)
            criterion = torch.nn.CrossEntropyLoss().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            de_m_d_valid_loss=10000.0
            de_m_d_test_accuracy=0.0

            de_m_b_valid_loss=10000.0
            de_m_b_test_accuracy=0.0

            fr_m_d_valid_loss=10000.0
            fr_m_d_test_accuracy=0.0

            fr_m_b_valid_loss=10000.0
            fr_m_b_test_accuracy=0.0

            jp_m_d_valid_loss=10000.0
            jp_m_d_test_accuracy=0.0

            jp_m_b_valid_loss=10000.0
            jp_m_b_test_accuracy=0.00            

            for ep in range(N_EPOCHS):
                if ep >5:
                    # print ('**************************')
                    start_time = time.time()
                    train_loss, train_acc = train(train_music)

                    eval_de_m_d_loss, eval_de_m_d_loss_acc = test(valid_de_dvd)
                    test_de_m_d_loss, test_de_m_d_loss_acc = test(test_de_dvd) 

                    eval_de_m_b_loss, eval_de_m_b_loss_acc = test(valid_de_book)
                    test_de_m_b_loss, test_de_m_b_loss_acc = test(test_de_book) 

                    eval_fr_m_d_loss, eval_fr_m_d_loss_acc = test(valid_fr_dvd)
                    test_fr_m_d_loss, test_fr_m_d_loss_acc = test(test_fr_dvd) 

                    eval_fr_m_b_loss, eval_fr_m_b_loss_acc = test(valid_fr_book)
                    test_fr_m_b_loss, test_fr_m_b_loss_acc = test(test_fr_book) 

                    eval_jp_m_d_loss, eval_jp_m_d_loss_acc = test(valid_jp_dvd)
                    test_jp_m_d_loss, test_jp_m_d_loss_acc = test(test_jp_dvd) 

                    eval_jp_m_b_loss, eval_jp_m_b_loss_acc = test(valid_jp_book)
                    test_jp_m_b_loss, test_jp_m_b_loss_acc = test(test_jp_book) 

                    #pick up the checkpoint with the best validation loss
                    if eval_de_m_d_loss<de_m_d_valid_loss:
                        de_m_d_valid_loss= eval_de_m_d_loss
                        de_m_d_test_accuracy=test_de_m_d_loss_acc

                    if eval_de_m_b_loss<de_m_b_valid_loss:
                        de_m_b_valid_loss= eval_de_m_b_loss
                        de_m_b_test_accuracy=test_de_m_b_loss_acc

                    if eval_fr_m_d_loss<fr_m_d_valid_loss:
                        fr_m_d_valid_loss= eval_fr_m_d_loss
                        fr_m_d_test_accuracy=test_fr_m_d_loss_acc
                    
                    if eval_fr_m_b_loss<fr_m_b_valid_loss:
                        fr_m_b_valid_loss= eval_fr_m_b_loss
                        fr_m_b_test_accuracy=test_fr_m_b_loss_acc

                    if eval_jp_m_d_loss<jp_m_d_valid_loss:
                        jp_m_d_valid_loss= eval_jp_m_d_loss
                        jp_m_d_test_accuracy=test_jp_m_d_loss_acc

                    if eval_jp_m_b_loss<jp_m_b_valid_loss:
                        jp_m_b_valid_loss= eval_jp_m_b_loss
                        jp_m_b_test_accuracy=test_jp_m_b_loss_acc

            #pick up the checkpoint with the best validation loss
            if de_m_d_valid_loss<de_m_valid_l_d:
                de_m_valid_l_d= de_m_d_valid_loss
                de_m_test_a_d=de_m_d_test_accuracy

            if de_m_b_valid_loss<de_m_valid_l_b:
                de_m_valid_l_b= de_m_b_valid_loss
                de_m_test_a_b=de_m_b_test_accuracy

            if fr_m_b_valid_loss<fr_m_valid_l_b:
                fr_m_valid_l_b= fr_m_b_valid_loss
                fr_m_test_a_b=fr_m_b_test_accuracy
                    
            if fr_m_d_valid_loss<fr_m_valid_l_d:
                fr_m_valid_l_d= fr_m_d_valid_loss
                fr_m_test_a_d=fr_m_d_test_accuracy

            if jp_m_d_valid_loss<jp_m_valid_l_d:
                jp_m_valid_l_d= jp_m_d_valid_loss
                jp_m_test_a_d=jp_m_d_test_accuracy

            if jp_m_b_valid_loss<jp_m_valid_l_b:
                jp_m_valid_l_b= jp_m_b_valid_loss
                jp_m_test_a_b=jp_m_b_test_accuracy            
          
            if de_b_d_valid_loss<de_b_valid_l_d:
                de_b_valid_l_d= de_b_d_valid_loss
                de_b_test_a_d=de_b_d_test_accuracy

            if de_b_m_valid_loss<de_b_valid_l_m:
                de_b_valid_l_m= de_b_m_valid_loss
                de_b_test_a_m=de_b_m_test_accuracy

            if fr_b_m_valid_loss<fr_b_valid_l_m:
                fr_b_valid_l_m= fr_b_m_valid_loss
                fr_b_test_a_m=fr_b_m_test_accuracy
                    
            if fr_b_d_valid_loss<fr_b_valid_l_d:
                fr_b_valid_l_d= fr_b_d_valid_loss
                fr_b_test_a_d=fr_b_d_test_accuracy

            if jp_b_d_valid_loss<jp_b_valid_l_d:
                jp_b_valid_l_d= jp_b_d_valid_loss
                jp_b_test_a_d=jp_b_d_test_accuracy

            if jp_b_m_valid_loss<jp_b_valid_l_m:
                jp_b_valid_l_m= jp_b_m_valid_loss
                jp_b_test_a_m=jp_b_m_test_accuracy 

            if de_d_m_valid_loss<de_d_valid_l_m:
                de_d_valid_l_m= de_d_m_valid_loss
                de_d_test_a_m=de_d_m_test_accuracy

            if de_d_b_valid_loss<de_d_valid_l_b:
                de_d_valid_l_b= de_d_b_valid_loss
                de_d_test_a_b=de_d_b_test_accuracy

            if fr_d_b_valid_loss<fr_d_valid_l_b:
                fr_d_valid_l_b= fr_d_b_valid_loss
                fr_d_test_a_b=fr_d_b_test_accuracy
                    
            if fr_d_m_valid_loss<fr_d_valid_l_m:
                fr_d_valid_l_m= fr_d_m_valid_loss
                fr_d_test_a_m=fr_d_m_test_accuracy

            if jp_d_m_valid_loss<jp_d_valid_l_m:
                jp_d_valid_l_m= jp_d_m_valid_loss
                jp_d_test_a_m=jp_d_m_test_accuracy

            if jp_d_b_valid_loss<jp_d_valid_l_b:
                jp_d_valid_l_b= jp_d_b_valid_loss
                jp_d_test_a_b=jp_d_b_test_accuracy 

    print("de_m_valid_l_d",de_m_valid_l_d)
    print("de_m_test_a_d",de_m_test_a_d)

    print("de_m_valid_l_b",de_m_valid_l_b)
    print("de_m_test_a_b",de_m_test_a_b)
 
    print("fr_m_valid_l_b",fr_m_valid_l_b)
    print("fr_m_test_a_b",fr_m_test_a_b)

    print("fr_m_valid_l_d",fr_m_valid_l_d)
    print("fr_m_test_a_d",fr_m_test_a_d)

    print("jp_m_valid_l_d",jp_m_valid_l_d)
    print("jp_m_test_a_d",jp_m_test_a_d)

    print("jp_m_valid_l_b",jp_m_valid_l_b)
    print("jp_m_test_a_b",jp_m_test_a_b)
 
    print("de_b_valid_l_d",de_b_valid_l_d)
    print("de_b_test_a_d",de_b_test_a_d)

    print("de_b_valid_l_m",de_b_valid_l_m)
    print("de_b_test_a_m",de_b_test_a_m)

    print("fr_b_valid_l_m",fr_b_valid_l_m)
    print("fr_b_test_a_m",fr_b_test_a_m)

    print("fr_b_valid_l_d",fr_b_valid_l_d)
    print("fr_b_test_a_d",fr_b_test_a_d)
 
    print("jp_b_valid_l_d",jp_b_valid_l_d)
    print("jp_b_test_a_d",jp_b_test_a_d)

    print("jp_b_valid_l_m",jp_b_valid_l_m)
    print("jp_b_test_a_m",jp_b_test_a_m)

    print("de_d_valid_l_m",de_d_valid_l_m)
    print("de_d_test_a_m",de_d_test_a_m)

    print("de_d_valid_l_b",de_d_valid_l_b)
    print("de_d_test_a_b",de_d_test_a_b)
 
    print("fr_d_valid_l_b",fr_d_valid_l_b)
    print("fr_d_test_a_b",fr_d_test_a_b)

    print("fr_d_valid_l_m",fr_d_valid_l_m)
    print("fr_d_test_a_m",fr_d_test_a_m)

    print("jp_d_valid_l_m",jp_d_valid_l_m)
    print("jp_d_test_a_m",jp_d_test_a_m)

    print("jp_d_valid_l_b",jp_d_valid_l_b)
    print("jp_d_test_a_b",jp_d_test_a_b)
 

