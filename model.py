import torch
import torch.nn as nn
import torch.nn.functional as F




class Adaptor_global(nn.Module):
    """
    domain-invariant feature extractor
    """
    def __init__(self, in_dim, dim_hidden, out_dim,initrange):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, out_dim)
        self.act = F.relu
        self.init_weights(initrange)

    def init_weights(self, initrange):
        self.lin1.weight.data.uniform_(-initrange, initrange)
        self.lin1.bias.data.zero_()
        self.lin2.weight.data.uniform_(-initrange, initrange)
        self.lin2.bias.data.zero_()

    def forward(self, input):
        x = self.lin1(input)
        x = self.act(x)
        local_features = x + input
        x = self.lin2(local_features)
        x = self.act(x)+ local_features
        x = x
        # x = F.dropout(x, p=self.dropout)
        return x, local_features


class Adaptor_domain(nn.Module):
    """
    domain-specific feature extractor
    """
    def __init__(self, in_dim, dim_hidden, out_dim, initrange):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, out_dim)
        self.act = F.relu
        self.init_weights(initrange)

    def init_weights(self, initrange):
        self.lin1.weight.data.uniform_(-initrange, initrange)
        self.lin1.bias.data.zero_()
        self.lin2.weight.data.uniform_(-initrange, initrange)
        self.lin2.bias.data.zero_()

    def forward(self, input):
        x = self.act(input)
        x = self.lin1(x)
        local_features = x
        x = self.lin2(local_features)
        x = self.act(x)
        x = F.normalize(x, p=2, dim=1)      

        # x = F.dropout(x, p=self.dropout)
        return x, local_features



class Max_Discriminator(nn.Module):
    """
    Discriminator for calculating mutual information maximization
    """
    def __init__(self, hidden_g, initrange):
        super().__init__()
        self.l0 = nn.Linear(2*hidden_g, 1) 
        self.init_weights(initrange)


    def init_weights(self, initrange):
        self.l0.weight.data.uniform_(-initrange, initrange)
        self.l0.bias.data.zero_()

    def forward(self, f_g, f_d):
        h = torch.cat((f_g, f_d), dim=1)
        h = self.l0(F.relu(h))

        return h



class Min_Discriminator(nn.Module):
    """
    Discriminator for calculating mutual information minimization
    """
    def __init__(self,hidden_l, initrange):
        super().__init__()
        self.l0 = nn.Linear(2*hidden_l, 1)
        # self.l1 = nn.Linear(hidden_l, hidden_l)
        # self.l1 = nn.Linear(hidden_l, 1)
        self.init_weights(initrange)

    def init_weights(self, initrange):
        self.l0.weight.data.uniform_(-initrange, initrange)
        self.l0.bias.data.zero_()

    def forward(self, f_g, f_d):
        h = torch.cat((f_g, f_d), dim=1)
        h = self.l0(F.relu(h))
        h = F.normalize(h, p=2, dim=1)
        return h



class Combine_features_map(nn.Module):
    """
    Combing domain-invariant and domain-specific features with a linear mapping layer
    """
    def __init__(self, embed_dim, initrange):
        super().__init__()
        self.fc = nn.Linear(2*embed_dim, embed_dim)
        self.init_weights(initrange)
        self.act=F.relu

    def init_weights(self, initrange):
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, input):
        return self.fc(self.act(input))



class Combine_features_gate(nn.Module):
    """
    Combing domain-invariant and domain-specific features with a gate 
    """    
    def __init__(self, embed_dim, initrange):
        super().__init__()
        self.fc = nn.Linear(2*embed_dim, 1)
        self.init_weights(initrange)
        self.act=F.sigmoid

    def init_weights(self, initrange):
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, input):
        return self.fc(self.act(input))



class Classifier(nn.Module):
    """
    Task-specific classifier
    """
    def __init__(self, embed_dim, num_class, initrange):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights(initrange)
        self.act=F.relu

    def init_weights(self, initrange):
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, input):
        return self.fc(self.act(input))
