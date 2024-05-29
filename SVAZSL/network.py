import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, drop_out_rate, sr_dim, unsr_dim, resSize):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(sr_dim + unsr_dim, resSize),
            # nn.Linear(embedding_size, embedding_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(drop_out_rate),
            nn.Linear(resSize, resSize),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, z):
        x = self.decoder(z)
        return x


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.opt = opt
        self.encoder = nn.Sequential(
            nn.Linear(opt.embedding_size, opt.sr_dim + opt.unsr_dim),
            nn.Linear(opt.sr_dim + opt.unsr_dim, opt.embedding_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(opt.drop_rate),
        )

    def forward(self, z):
        x = self.encoder(z)
        return x


class RelationNet(nn.Module):
    def __init__(self, embedding_size, att_size):
        super(RelationNet, self).__init__()
        self.fc1 = nn.Linear(embedding_size + att_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, 1)

    def forward(self, s, c):

        c_ext = c.unsqueeze(0).repeat(s.shape[0], 1, 1)
        cls_num = c_ext.shape[1]
        # print('cls_num is:', cls_num) 128

        s_ext = torch.transpose(s.unsqueeze(0).repeat(cls_num, 1, 1), 0, 1)
        relation_pairs = torch.cat((s_ext, c_ext), 2).view(-1, c.shape[1] + s.shape[1])
        relation = nn.ReLU()(self.fc1(relation_pairs))
        relation = nn.Sigmoid()(self.fc2(relation))
        return relation

class AE(nn.Module):
    def __init__(self, opt):
        super(AE, self).__init__()
        self.opt = opt
        self.encoder = nn.Sequential(
            nn.Linear(opt.resSize, opt.sr_dim + opt.unsr_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(opt.drop_rate)
        )
        self.decoder = nn.Sequential(
            nn.Linear(opt.sr_dim + opt.unsr_dim, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(opt.drop_rate),
            nn.Linear(2048, opt.resSize),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(opt.drop_rate),
        )

    def forward(self, x):
        z = self.encoder(x)
        s = z[:, :self.opt.sr_dim]
        ns = z[:, self.opt.unsr_dim:]
        # x1 = self.decoder(z)
        return s, ns




