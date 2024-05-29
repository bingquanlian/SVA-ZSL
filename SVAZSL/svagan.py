from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import math

import classifierNew
import network
import util
import classifier
import classifier2
import sys
import svaClassifier
import model
import matplotlib.pyplot as plt
from network import Decoder, Encoder, RelationNet
from torch.nn import functional as F
import itertools
import numpy as np

#  指定参数，   Import Version, the more reasonable version
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CUB', help='AWA2, CUB, SUN, FLO')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='sent')
parser.add_argument('--syn_num', type=int, default=300, help='number features to generate per class')
parser.add_argument('--gzsl', action='store_true', default=True, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=True,
                    help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=1024, help='size of semantic features')
parser.add_argument('--nz', type=int, default=1024, help='size of the latent z vector')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=4096, help='size of the hidden units in discriminator')
parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
# parser.add_argument('--cls_weight', type=float, default=1, help='weight of the classification loss') # CUB设置的为1
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=0, help='number of GPUs to use')
parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netG_name', default='')
parser.add_argument('--netD_name', default='')
parser.add_argument('--outf', default='./checkpoint/', help='folder to output data and model checkpoints')
parser.add_argument('--outname', help='folder to output data and model checkpoints')
parser.add_argument('--save_every', type=int, default=100)
parser.add_argument('--print_every', type=int, default=1)
parser.add_argument('--val_every', type=int, default=10)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
#  SVA
parser.add_argument('--embedding_size', type=int, default=2048, help='size of embedding_size')
parser.add_argument('--drop_rate', type=float, default=0.2, help='size of drop_rate')
parser.add_argument('--weight_1', type=float, default=1.0, help='size of weight_1')
parser.add_argument('--weight_2', type=float, default=3.0, help='size of weight_2')
parser.add_argument('--weight_3', type=float, default=0.5, help='size of weight_3')
parser.add_argument('--structure_weight', type=float, default=1.0, help='size of structure_weight')  # 0.0001
parser.add_argument('--sr_dim', type=float, default=2048, help='size of structure_weight')
parser.add_argument('--unsr_dim', type=float, default=2048, help='size of structure_weight')
parser.add_argument('--weight_decay', type=float, default=1e-8, help='weight_decay')
parser.add_argument('--sva_dis', type=float, default=3, help='Discriminator weight')
parser.add_argument('--sva_dis_step', type=float, default=2, help='Discriminator update interval')
parser.add_argument('--sva_beta', type=float, default=0.003, help='tc weight')

parser.add_argument("--encoder_layer_sizes", type=list, default=[2048, 4096])
parser.add_argument("--decoder_layer_sizes", type=list, default=[4096, 2048])
parser.add_argument('--gammaD', type=int, default=10, help='weight on the W-GAN loss')
parser.add_argument('--gammaG', type=int, default=10, help='weight on the W-GAN loss')
parser.add_argument('--recons_weight', type=float, default=0.01, help='recons_weight for decoder')
parser.add_argument("--latent_size", type=int, default=312)
parser.add_argument('--encoded_noise', action='store_true', default=True, help='enables validation mode')
parser.add_argument('--freeze_dec', action='store_true', default=False, help='Freeze Decoder for fake samples')
parser.add_argument('--dec_lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--kl_warmup', type=float, default=0.001, help='kl warm-up for VAE')
parser.add_argument('--tc_warmup', type=float, default=0.0001, help='tc warm-up')

#  解析命令行参数，将解析结果存放在opt中
opt = parser.parse_args()
opt.encoder_layer_sizes[0] = opt.resSize
opt.decoder_layer_sizes[-1] = opt.resSize
opt.latent_size = opt.attSize


print(opt)
#  创建一个文件夹
try:
    os.makedirs(opt.outf)
except OSError:
    pass

#  生成随机种子，使用相同的随机种子可以保证程序在每次运行时生成相同的随机数序列，从而方便调试和比较不同模型的性能
if opt.manualSeed is None:
    opt.manualSeed = 3483
    # opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load data
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)
print("# of test samples: ", data.ntest)
print("# of unseen samples:", data.nunseen)
print("# of seen categories:", data.seenclasses)
print("# of unseen categories:", len(data.unseenclasses))
print("# of test_seen numbers:", len(data.ntest_seen_label))  #
print("# of test_seen categories:", data.ntest_class)
print("# of test_seen_instance:", data.ntest_seen_label_instance)
print("# of test_unseen categories:", data.ntest_unseen_label)
# print('of data attribute: ', data.attribute) 每一类有唯一的一组属性


#  initialize generator and discriminator
netE = model.Encoder(opt)
netG = model.Generator(opt)
netDec = model.AttDec(opt, opt.attSize)
#  判断是否指定了预训练生成模型的检查点文件
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = model.Discriminator_D1(opt)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# netP = model.Embedding_Net(opt)

# netRelationnet = network.RelationNet(opt.embedding_size, opt.embedding_size)
# print(netRelationnet)

cls_criterion = nn.NLLLoss()

# 存储输入的图像特征、属性信息、噪声向量和标签
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
# print('input res:', input_res)
# print('input res row:', input_res.size(0))
# print('input res column:', input_res.size(1))
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
# print('input att', input_att)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
# 将one = torch.FloatTensor([1])改为了one = torch.tensor(1)
# one = 1,mone = -1,用于计算损失函数的梯度和更新参数
one = torch.tensor(1, dtype=torch.float)
mone = one * -1
input_label = torch.LongTensor(opt.batch_size)

# fc_projection = nn.Linear(opt.noise_dim, opt.embedding_size)

input_att_unseen = torch.FloatTensor(opt.batch_size, opt.attSize)
input_label_unseen = torch.LongTensor(opt.batch_size)
# input_res_unseen = torch.FloatTensor(opt.batch_size, opt.resSize)
relation_seen_1 = torch.FloatTensor(opt.batch_size, opt.batch_size)
relation_unseen_1 = torch.FloatTensor(opt.batch_size, opt.batch_size)

# fc_projection = nn.Linear(opt.resSize, opt.embedding_size)
# fc_projection1 = nn.Linear(128, 2048)
projection_related = network.Encoder(opt)
projection_unrelated = network.Encoder(opt)
ae = network.AE(opt)
# netRelation = network.Encoder(opt)
decoder = network.Decoder(opt.drop_rate, opt.sr_dim, opt.unsr_dim, opt.resSize)
relation = network.RelationNet(opt.embedding_size, opt.attSize)

if opt.cuda:
    netE.cuda()
    netD.cuda()
    netG.cuda()
    # netP.cuda()
    netDec.cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    one = one.cuda()
    mone = mone.cuda()
    cls_criterion.cuda()
    mse = nn.MSELoss().cuda()
    input_label = input_label.cuda()
    # fc_projection = fc_projection.cuda()
    # fc_projection1 = fc_projection1.cuda()
    projection_related = projection_related.cuda()
    projection_unrelated = projection_unrelated.cuda()
    # netRelation.cuda()
    decoder = decoder.cuda()
    relation = relation.cuda()
    ae = ae.cuda()
    # input_res_unseen = input_res_unseen.cuda()
    input_att_unseen = input_att_unseen.cuda()
    input_label_unseen = input_label_unseen.cuda()
    relation_seen_1 = relation_seen_1.cuda()
    relation_unseen_1 = relation_unseen_1.cuda()


def loss_fn(recon_x, x, mean, log_var):
    # BCE = torch.nn.functional.binary_cross_entropy(recon_x + 1e-12, x.detach(), size_average=False)
    BCE = torch.nn.functional.binary_cross_entropy(recon_x + 1e-12, x.detach(), reduction='sum')
    BCE = BCE.sum() / x.size(0)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.size(0)
    return (BCE + KLD)


def WeightedL1(pred, gt):
    wt = (pred - gt).pow(2)
    wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0), wt.size(1))
    loss = wt * (pred - gt).abs()
    return loss.sum() / loss.size(0)


#  采样了一个batch_size的特征、标签和属性并把值赋值给了定义的变量
def sample():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))


def sample_unseen():
    batch_label_unseen, batch_att_unseen = data.next_batch_unseen(opt.batch_size)
    input_att_unseen.copy_(batch_att_unseen)
    input_label_unseen.copy_(util.map_label(batch_label_unseen, data.unseenclasses))
    # input_res_unseen.copy_(batch_feature_unseen)


#  为（不可见类）的所有样本生成特征并给标签，eg：50*100，总共5000个特征和标签
def generate_syn_feature(netG, classes, attribute, num):
    #  计算类别数量
    nclass = classes.size(0)
    # print(nclass)
    # 总共生成nclass * num个样本，每个样本的特征向量大小为opt.resSize,用来保存合成的特征向量
    # 定义了一个变量
    syn_feature = torch.FloatTensor(nclass * num, opt.sr_dim)
    #  每个样本都有一个标签
    syn_label = torch.LongTensor(nclass * num)
    #  每个合成特征样本对应的属性信息大小
    syn_att = torch.FloatTensor(num, opt.attSize)
    #  每个样本对应的噪声大小
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
    # number = 0
    for i in range(nclass):
        # 类别的序号
        iclass = classes[i]
        # print('iclass:', iclass)
        # print('i:', i)
        # 相应类别的属性
        iclass_att = attribute[iclass]
        # 一次生成num个特征，因此会将同一类别的属性复制num(100)次
        syn_att.copy_(iclass_att.repeat(num, 1))
        # print("syn_att", syn_att)
        # 生成了num个符合高斯分布的312维的噪声向量,上边声明时已经定义了数目和维度，现在只需要使其满足高斯分布
        syn_noise.normal_(0, 1)
        #  生成特征向量，用num个噪声和num个属性为每个类别生成num个特征
        with torch.no_grad():
            output = netG(syn_noise, c=syn_att)
        syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)
    return syn_feature, syn_label


# setup optimizer
optimizer = optim.Adam(netE.parameters(), lr=opt.lr)
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

optimizerSVA = optim.Adam(itertools.chain(relation.parameters(), decoder.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay= opt.weight_decay)
optimizerDec = optim.Adam(netDec.parameters(), lr=opt.dec_lr, betas=(opt.beta1, 0.999))
ae_optimizer = optim.Adam(ae.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)


#  提高训练的稳定性和生成图像的质量
def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    # print real_data.size()
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, Variable(input_att))

    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty




Wasserstein_D_Array = []
Loss_D_Array = []
Loss_G_Array = []
c_errG_Array = []
H_accuracy_Array = []
Top_accuracy_Array = []
epoch_array = np.arange(1, 1 + opt.nepoch)



def kl_loss(prediction, targets):
    T = 0.1
    return F.kl_div(F.log_softmax(prediction / T, dim=1),
                    F.log_softmax(targets / T, dim=1),  # 1.2 0.1 0.2 0.3
                    reduction='sum', log_target=True) / prediction.numel()


def cal_similarity(key_embeds,
                   ref_embeds,
                   method='cosine',
                   temperature=-1):
    assert method in ['dot_product', 'cosine', 'euclidean']

    if key_embeds.size(0) == 0 or ref_embeds.size(0) == 0:
        return torch.zeros((key_embeds.size(0), ref_embeds.size(0)),
                           device=key_embeds.device)

    if method == 'cosine':
        key_embeds = F.normalize(key_embeds, p=2, dim=1)
        ref_embeds = F.normalize(ref_embeds, p=2, dim=1)
        return torch.mm(key_embeds, ref_embeds.t())
    elif method == 'euclidean':
        return euclidean_dist(key_embeds, ref_embeds)
    elif method == 'dot_product':
        if temperature > 0:
            dists = cal_similarity(key_embeds, ref_embeds, method='cosine')
            dists /= temperature
            return dists
        else:
            return torch.mm(key_embeds, ref_embeds.t())


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def sva_loss_G(fake_res_seen, att, label, fake_res_unseen, att_unseen, label_unseen):
    fake_seen_embeddings = fake_res_seen
    # fake_seen_related = projection_related(fake_seen_embeddings)
    # fake_seen_unrelated = projection_unrelated(fake_seen_embeddings)
    fake_seen_related, fake_seen_unrelated = ae(fake_seen_embeddings)
    fake_seen_unrelated_noise = torch.FloatTensor(fake_seen_unrelated.shape[0],
                                                  fake_seen_unrelated.shape[1]).cuda().normal_(0, 1)
    rec_fake_seen = decoder(torch.cat([fake_seen_related, fake_seen_unrelated], dim=-1))
    # note follow
    label_numpy = np.array(label.cpu())
    label_count_seen = label.unique().shape[0]
    unique_label_seen = label.unique().cpu()
    att_unique = torch.from_numpy(np.array([data.train_att[i, :] for i in label.unique()])).cuda()
    # print("att_unique", att_unique.shape)
    unique_label_seen_array = np.array(unique_label_seen)
    re_batch_labels = []
    for label in label_numpy:
        if np.argwhere(unique_label_seen_array == label).size > 0:
            index = np.argwhere(unique_label_seen_array == label)
            re_batch_labels.append(index[0][0])
    re_batch_labels = torch.LongTensor(re_batch_labels)
    one_hot_labels = torch.zeros(opt.batch_size, label_count_seen).scatter_(1, re_batch_labels.view(-1, 1), 1).cuda()
    relations_1 = relation(fake_seen_related, att_unique).view(-1, label_count_seen)
    relation_seen_1 = F.cross_entropy(relations_1, one_hot_labels, reduction="mean")
    # relation_seen_1 = mse(relations_1, one_hot_labels)

    kl_unrelated_1 = kl_loss(fake_seen_unrelated, fake_seen_unrelated_noise)
    kl_related_3 = kl_loss(cal_similarity(fake_seen_related, fake_seen_related),
                           cal_similarity(att, att))
    rec_loss_1 = F.l1_loss(rec_fake_seen, fake_seen_embeddings)
    # unseen
    fake_unseen_embeddings = fake_res_unseen
    # fake_unseen_related = projection_related(fake_unseen_embeddings)
    # fake_unseen_unrelated  = projection_unrelated(fake_unseen_embeddings)
    fake_unseen_related, fake_unseen_unrelated = ae(fake_unseen_embeddings)
    fake_unseen_unrelated_noise = torch.FloatTensor(fake_unseen_unrelated.shape[0],
                                                    fake_unseen_unrelated.shape[1]).cuda().normal_(0, 1)
    rec_fake_unseen = decoder(torch.cat([fake_unseen_related, fake_unseen_unrelated], dim=-1))

    label_unseen_numpy = np.array(label_unseen.cpu())
    label_count_unseen = label_unseen.unique().shape[0]
    unique_label_unseen = label_unseen.unique().cpu()
    att_unique_unseen = torch.from_numpy(np.array([data.test_att[i, :] for i in label_unseen.unique()])).cuda()
    # print("att_unique", att_unique.shape)
    unique_label_unseen_array = np.array(unique_label_unseen)
    re_batch_labels = []
    for label in label_unseen_numpy:
        if np.argwhere(unique_label_unseen_array == label).size > 0:
            index = np.argwhere(unique_label_unseen_array == label)
            re_batch_labels.append(index[0][0])
    re_batch_labels = torch.LongTensor(re_batch_labels)
    one_hot_labels = torch.zeros(opt.batch_size, label_count_unseen).scatter_(1, re_batch_labels.view(-1, 1), 1).cuda()
    relations = relation(fake_unseen_related, att_unique_unseen).view(-1, label_count_unseen)
    relation_unseen_1 = F.cross_entropy(relations, one_hot_labels, reduction="mean")

    kl_unseen_1 = kl_loss(cal_similarity(fake_unseen_related, fake_unseen_related),
                          cal_similarity(att_unseen, att_unseen))
    kl_unseen_2 = kl_loss(cal_similarity(fake_seen_related, fake_unseen_related),
                          cal_similarity(att, att_unseen))
    rec_loss_4 = F.l1_loss(rec_fake_unseen, fake_unseen_embeddings)
    kl_unrelated_4 = kl_loss(fake_unseen_unrelated, fake_unseen_unrelated_noise)
    kl_loss_all = (kl_related_3 + kl_unseen_1 + kl_unseen_2) / 3.0
    relation_loss = 0.2 * (relation_seen_1 + relation_unseen_1) / 2.0
    rec_loss = (rec_loss_1 + rec_loss_4) / 2.0
    unrelated = 0.5 * (kl_unrelated_1 + kl_unrelated_4) / 2.0
    sva_G_loss = opt.weight_1 * (
            kl_loss_all + unrelated) + opt.weight_2 * relation_loss + opt.weight_3 * rec_loss
    return sva_G_loss


def sva_loss_D(fake_res_seen, att, label, real_res_seen):
    fake_seen_embeddings = fake_res_seen
    # fake_seen_related = projection_related(fake_seen_embeddings)
    # fake_seen_unrelated = projection_unrelated(fake_seen_embeddings)
    fake_seen_related, fake_seen_unrelated = ae(fake_seen_embeddings)
    fake_seen_unrelated_noise = torch.FloatTensor(fake_seen_unrelated.shape[0],
                                                  fake_seen_unrelated.shape[1]).cuda().normal_(0, 1)
    rec_fake_seen = decoder(torch.cat([fake_seen_related, fake_seen_unrelated], dim=-1))
    # note follow
    label_numpy = np.array(label.cpu())
    label_count_seen = label.unique().shape[0]
    unique_label_seen = label.unique().cpu()
    att_unique = torch.from_numpy(np.array([data.train_att[i, :] for i in label.unique()])).cuda()
    # print("att_unique", att_unique.shape)
    unique_label_seen_array = np.array(unique_label_seen)
    re_batch_labels = []
    for label in label_numpy:
        if np.argwhere(unique_label_seen_array == label).size > 0:
            index = np.argwhere(unique_label_seen_array == label)
            re_batch_labels.append(index[0][0])
    re_batch_labels = torch.LongTensor(re_batch_labels)
    one_hot_labels = torch.zeros(opt.batch_size, label_count_seen).scatter_(1, re_batch_labels.view(-1, 1), 1).cuda()
    relations_1 = relation(fake_seen_related, att_unique).view(-1, label_count_seen)
    relation_seen_1 = F.cross_entropy(relations_1, one_hot_labels)

    kl_unrelated_1 = kl_loss(fake_seen_unrelated, fake_seen_unrelated_noise)
    kl_related_3 = kl_loss(cal_similarity(fake_seen_related, fake_seen_related),
                           cal_similarity(att, att))
    rec_loss_1 = F.l1_loss(rec_fake_seen, fake_seen_embeddings)
    # real
    real_seen_embeddings = real_res_seen
    real_seen_related, real_seen_unrelated = ae(real_seen_embeddings)
    # real_seen_related = projection_related(real_seen_embeddings)
    # real_seen_unrelated = projection_unrelated(real_seen_embeddings)
    real_seen_unrelated_noise = torch.FloatTensor(real_seen_unrelated.shape[0],
                                                    real_seen_unrelated.shape[1]).cuda().normal_(0, 1)
    rec_real_seen = decoder(torch.cat([real_seen_related, real_seen_unrelated], dim=-1))

    label_real_numpy = label_numpy
    label_count_real = label_count_seen
    unique_label_real = unique_label_seen
    att_unique_real = att_unique
    # print("att_unique", att_unique.shape)
    unique_label_real_array = np.array(unique_label_real)
    re_batch_labels = []
    for label in label_real_numpy:
        if np.argwhere(unique_label_real_array == label).size > 0:
            index = np.argwhere(unique_label_real_array == label)
            re_batch_labels.append(index[0][0])
    re_batch_labels = torch.LongTensor(re_batch_labels)
    one_hot_labels = torch.zeros(opt.batch_size, label_count_real).scatter_(1, re_batch_labels.view(-1, 1), 1).cuda()
    relations = relation(real_seen_related, att_unique_real).view(-1, label_count_real)
    relation_unseen_1 = F.cross_entropy(relations, one_hot_labels)

    kl_unseen_1 = kl_loss(cal_similarity(real_seen_related, real_seen_related),
                          cal_similarity(att, att))
    kl_unseen_2 = kl_loss(cal_similarity(fake_seen_related, real_seen_related),
                          cal_similarity(att, att))
    rec_loss_4 = F.l1_loss(rec_real_seen, real_seen_embeddings)
    kl_unrelated_4 = kl_loss(real_seen_unrelated, real_seen_unrelated_noise)
    kl_loss_all = (kl_related_3 + kl_unseen_1 + kl_unseen_2) / 3.0
    relation_loss = 0.2 * (relation_seen_1 + relation_unseen_1) / 2.0
    rec_loss = (rec_loss_1 + rec_loss_4) / 2.0
    unrelated = 0.5 * (kl_unrelated_1 + kl_unrelated_4) / 2.0
    sva_D_loss = opt.weight_1 * (
            kl_loss_all + unrelated) + opt.weight_2 * relation_loss + opt.weight_3 * rec_loss
    return sva_D_loss



# for循环从外层循环开始，先执行结束内层循环 在第一阶段使用sva损失先固定下Encoder、RelationNet的参数
best_gzsl_acc = 0
best_zsl_acc = 0
best_epoch = 0
for epoch in range(opt.nepoch):

    mean_lossD = 0
    mean_lossG = 0
    # 从0开始到特征样本的总数量，stride = batch_size，控制每次从训练数据集中取多少数据作为一个批次（batch）用于训练。
    # 内第一层循环总共循环14次
    for i in range(0, data.ntrain, opt.batch_size):
        ############################
        # (1) Update D network: optimize WGAN-GP objective
        ###########################
        #  将神经网络模型 netD 中所有的参数的 requires_grad 属性设置为 True，
        #  以便在后续的反向传播过程中计算梯度并更新模型参数。
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

        for p in netDec.parameters():
            p.requires_grad = True
        for p in relation.parameters():
            p.requires_grad = True
        for p in decoder.parameters():
            p.requires_grad = True

        for p in ae.parameters():
            p.requires_grad = True


        gp_sum = 0
        for iter_d in range(opt.critic_iter):
            sample()
            #  清空模型梯度的参数信息
            netD.zero_grad()
            # train with realG
            # sample a mini-batch
            # 计算张量input_res[1]中非零元素的数量，相减得到一个稀疏矩阵的实际大小，以进行模型参数的初始化和优化
            sparse_real = opt.resSize - input_res[1].gt(0).sum()
            input_resv = input_res
            input_attv = input_att
            input_labelv = input_label

            netDec.zero_grad()
            recons = netDec(input_resv)
            R_cost = opt.recons_weight * WeightedL1(recons, input_attv)
            R_cost.backward()
            optimizerDec.step()


            means, log_var = netE(input_resv, input_attv)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn([opt.batch_size, opt.latent_size]).cpu()
            eps = eps.cuda()
            noisev = eps * std + means  # torch.Size([64, 312])
            fake = netG(noisev, c=input_attv)
            optimizerSVA.zero_grad()
            ae_optimizer.zero_grad()
            sva_d = sva_loss_D(fake, input_attv, input_labelv, input_resv)
            sva_d.backward()
            optimizerSVA.step()
            ae_optimizer.step()


            # 判别输入的batch_size个样本
            criticD_real = netD(input_resv, input_attv)
            criticD_real = opt.gammaD * criticD_real.mean()
            criticD_real.backward(mone)

            if opt.encoded_noise:
                means, log_var = netE(input_resv, input_attv)
                std = torch.exp(0.5 * log_var)
                eps = torch.randn([opt.batch_size, opt.latent_size]).cpu()
                eps = eps.cuda()
                noisev = eps * std + means  # torch.Size([64, 312])
            else:
                noise.normal_(0, 1)
                noisev = noise
                # 一次循环会判别生成的batch_size个样本
            fake = netG(noisev, c=input_attv)
            fake_norm = fake.data[0].norm()
            sparse_fake = fake.data[0].eq(0).sum()

            criticD_fake = netD(fake.detach(), input_attv)
            criticD_fake = opt.gammaD * criticD_fake.mean()
            criticD_fake.backward(one)

            # gradient penalty
            gradient_penalty = opt.gammaD * calc_gradient_penalty(netD, input_resv, fake.data, input_att)
            gp_sum += gradient_penalty
            gradient_penalty.backward()

            Wasserstein_D = criticD_real - criticD_fake
            # Wasserstein_D_Array.append(Wasserstein_D.item())
            # print(Wasserstein_D.item())
            D_cost = criticD_fake - criticD_real + gradient_penalty
            # Loss_D_Array.append(D_cost.item())
            # 更新判别器的参数rt
            optimizerD.step()

        gp_sum /= (opt.gammaD * opt.lambda1 * opt.critic_iter)
        if (gp_sum > 1.05).sum() > 0:
            opt.lambda1 *= 1.1
        elif (gp_sum < 1.001).sum() > 0:
            opt.lambda1 /= 1.1

        ############################
        # (2) Update G network: optimize WGAN-GP objective
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = False  # avoid computation
        if opt.recons_weight > 0 and opt.freeze_dec:
            for p in netDec.parameters():  # freeze decoder
                p.requires_grad = False
        for p in ae.parameters():
            p.requires_grad = True
        for p in relation.parameters():
            p.requires_grad = True
        for p in decoder.parameters():
            p.requires_grad = True

        netE.zero_grad()
        netG.zero_grad()
        input_resv = input_res # torch.Size([128, 2048])
        input_attv = input_att  # torch.Size([128, 1024])
        means, log_var = netE(input_resv, input_attv)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn([opt.batch_size, opt.latent_size]).cpu()
        eps = Variable(eps.cuda())
        z = eps * std + means  #  torch.Size([128, 1024])

        recon_x = netG(z, c=input_attv) # torch.Size([128, 2048])
        vae_loss_seen = loss_fn(recon_x, input_resv, means, log_var)
        if opt.encoded_noise:
            criticG_fake = netD(recon_x, input_attv).mean()
            fake_seen = recon_x
        else:
            noise.normal_(0, 1)
            noisev = Variable(noise)

            fake_seen = netG(noisev, c=input_attv)
            criticG_fake = netD(fake_seen, input_attv).mean()

        G_cost = -criticG_fake
        netDec.zero_grad()
        recons_fake = netDec(fake_seen)
        R_cost = WeightedL1(recons_fake, input_attv)

        # process unseen class
        sample_unseen()
        input_attv_unseen = input_att_unseen
        input_label_unseen = input_label_unseen
        fake_unseen = netG(noisev, c=input_attv_unseen)


        optimizerSVA.zero_grad()
        ae_optimizer.zero_grad()
        sva_G_cost = sva_loss_G(fake_seen, input_attv, input_labelv, fake_unseen, input_attv_unseen, input_label_unseen)
        errG = opt.gammaG * G_cost + vae_loss_seen + opt.recons_weight * R_cost + opt.structure_weight * sva_G_cost
        errG.backward()
        optimizer.step()
        # 根据计算出的梯度来更新生成器网络的权重
        optimizerG.step()
        optimizerSVA.step()
        ae_optimizer.step()
        if opt.recons_weight > 0 and not opt.freeze_dec:  # not train decoder at feedback time
            optimizerDec.step()

    mean_lossG /= data.ntrain / opt.batch_size
    mean_lossD /= data.ntrain / opt.batch_size

    #  将 D_cost.data[0]改成了D_cost.item()
    Wasserstein_D_Array.append(Wasserstein_D.item())
    print('[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f, vae_cost:%.4f, R_cost:%.4f, sva_cost:%.4f'
          % (epoch, opt.nepoch, D_cost.item(), G_cost.item(), Wasserstein_D.item(), vae_loss_seen.item(), R_cost.item(), sva_G_cost.item()))

    # evaluate the model, set G to evaluation mode
    netG.eval()
    netDec.eval()
    ae.eval()

    # Generalized zero-shot learning
    if opt.gzsl:
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)
        nclass = opt.nclass_all

        gzsl_cls = classifierNew.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, 25,
                                            opt.syn_num, generalized=True, netDec=netDec, dec_size=opt.attSize,
                                            dec_hidden_size=4096)
        if best_gzsl_acc < gzsl_cls.H:
            best_acc_seen, best_acc_unseen, best_gzsl_acc = gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H
            best_epoch = epoch
        print('GZSL: seen=%.4f, unseen=%.4f, h=%.4f' % (gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H))
        H_accuracy_Array.append(gzsl_cls.H)


    # Zero-shot learning
    else:
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
        zsl_cls = classifierNew.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses),
                                        data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25,
                                        opt.syn_num, generalized=False, netDec=netDec, dec_size=opt.attSize,
                                        dec_hidden_size=4096)
        acc = zsl_cls.acc
        Top_accuracy_Array.append(acc)
        if best_zsl_acc < acc:
            best_zsl_acc = acc
            best_epoch = epoch
        print('ZSL: unseen accuracy=%.4f' % (acc))

    # reset G to training mode
    netG.train()
    netDec.train()

print('Dataset', opt.dataset)
if opt.gzsl:
    print('the best epoch is', best_epoch)
    print('the best GZSL seen accuracy is', best_acc_seen)
    print('the best GZSL unseen accuracy is', best_acc_unseen)
    print('the best GZSL H is', best_gzsl_acc)

    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('H')
    plt.plot(epoch_array, H_accuracy_Array)
    plt.show()
else:
    print('the best epoch is', best_epoch)
    print('the best ZSL unseen accuracy is', best_zsl_acc)
    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('Top1 accuracy')
    plt.plot(epoch_array, Top_accuracy_Array)
    plt.show()

plt.figure()
plt.xlabel('epoch')
plt.ylabel('wasserstein_dist')
plt.plot(epoch_array, Wasserstein_D_Array)


