import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import util
import torchvision.models as models
from .stylegan2_layers import Downsample
from networks import init_net, init_weights

def gan_loss(pred, should_be_classified_as_real):
    bs = pred.size(0)
    if should_be_classified_as_real:
        return F.softplus(-pred).view(bs, -1).mean(dim=1)
    else:
        return F.softplus(pred).view(bs, -1).mean(dim=1)


def feature_matching_loss(xs, ys, equal_weights=False, num_layers=6):
    loss = 0.0
    for i, (x, y) in enumerate(zip(xs[:num_layers], ys[:num_layers])):
        if equal_weights:
            weight = 1.0 / min(num_layers, len(xs))
        else:
            weight = 1 / (2 ** (min(num_layers, len(xs)) - i))
        loss = loss + (x - y).abs().flatten(1).mean(1) * weight
    return loss


class IntraImageNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, query, target):
        num_locations = min(query.size(2) * query.size(3), self.opt.intraimage_num_locations)
        bs = query.size(0)
        patch_ids = torch.randperm(num_locations, device=query.device)

        query = query.flatten(2, 3)
        target = target.flatten(2, 3)

        # both query and target are of size B x C x N
        query = query[:, :, patch_ids]
        target = target[:, :, patch_ids]

        cosine_similarity = torch.bmm(query.transpose(1, 2), target)
        cosine_similarity = cosine_similarity.flatten(0, 1)
        target_label = torch.arange(num_locations, dtype=torch.long, device=query.device).repeat(bs)
        loss = self.cross_entropy_loss(cosine_similarity / 0.07, target_label)
        return loss


class VGG16Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_convs = torchvision.models.vgg16(pretrained=True).features
        self.register_buffer('mean',
                             torch.tensor([0.485, 0.456, 0.406])[None, :, None, None] - 0.5)
        self.register_buffer('stdev',
                             torch.tensor([0.229, 0.224, 0.225])[None, :, None, None] * 2)
        self.downsample = Downsample([1, 2, 1], factor=2)

    def copy_section(self, source, start, end):
        slice = torch.nn.Sequential()
        for i in range(start, end):
            slice.add_module(str(i), source[i])
        return slice

    def vgg_forward(self, x):
        x = (x - self.mean) / self.stdev
        features = []
        for name, layer in self.vgg_convs.named_children():
            if "MaxPool2d" == type(layer).__name__:
                features.append(x)
                if len(features) == 3:
                    break
                x = self.downsample(x)
            else:
                x = layer(x)
        return features

    def forward(self, x, y):
        y = y.detach()
        loss = 0
        weights = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1.0]
        #weights = [1] * 5
        total_weights = 0.0
        for i, (xf, yf) in enumerate(zip(self.vgg_forward(x), self.vgg_forward(y))):
            loss += F.l1_loss(xf, yf) * weights[i]
            total_weights += weights[i]
        return loss / total_weights


class NCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, query, target, negatives):
        query = util.normalize(query.flatten(1))
        target = util.normalize(target.flatten(1))
        negatives = util.normalize(negatives.flatten(1))
        bs = query.size(0)
        sim_pos = (query * target).sum(dim=1, keepdim=True)
        sim_neg = torch.mm(query, negatives.transpose(0, 1))
        all_similarity = torch.cat([sim_pos, sim_neg], axis=1) / 0.07
        #sim_target = util.compute_similarity_logit(query, target)
        #sim_target = torch.mm(query, target.transpose(0, 1)) / 0.07
        #sim_query = util.compute_similarity_logit(query, query)
        #util.set_diag_(sim_query, -20.0)

        #all_similarity = torch.cat([sim_target, sim_query], axis=1)

        #target_label = torch.arange(bs, dtype=torch.long,
        #                            device=query.device)
        target_label = torch.zeros(bs, dtype=torch.long, device=query.device)
        loss = self.cross_entropy_loss(all_similarity,
                                       target_label)
        return loss


class ScaleInvariantReconstructionLoss(nn.Module):
    def forward(self, query, target):
        query_flat = query.transpose(1, 3)
        target_flat = target.transpose(1, 3)
        dist = 1.0 - torch.bmm(
            query_flat[:, :, :, None, :].flatten(0, 2),
            target_flat[:, :, :, :, None].flatten(0, 2),
        )

        target_spatially_flat = target.flatten(1, 2)
        num_samples = min(target_spatially_flat.size(1), 64)
        random_indices = torch.randperm(num_samples, dtype=torch.long, device=target.device)
        randomly_sampled = target_spatially_flat[:, random_indices]
        random_indices = torch.randperm(num_samples, dtype=torch.long, device=target.device)
        another_random_sample = target_spatially_flat[:, random_indices]

        random_similarity = torch.bmm(
            randomly_sampled[:, :, None, :].flatten(0, 1),
            torch.flip(another_random_sample, [0])[:, :, :, None].flatten(0, 1)
        )

        return dist.mean() + random_similarity.clamp(min=0.0).mean()

class PatchSim(nn.Module):
    """Calculate the similarity in selected patches"""
    def __init__(self, patch_nums=256, patch_size=None, norm=True):
        super(PatchSim, self).__init__()
        self.patch_nums = patch_nums
        self.patch_size = patch_size
        self.use_norm = norm

    def forward(self, feat, patch_ids=None):
        """
        Calculate the similarity for selected patches
        """
        B, C, W, H = feat.size()
        feat = feat - feat.mean(dim=[-2, -1], keepdim=True)
        feat = F.normalize(feat, dim=1) if self.use_norm else feat / np.sqrt(C)
        query, key, patch_ids = self.select_patch(feat, patch_ids=patch_ids)
        patch_sim = query.bmm(key) if self.use_norm else torch.tanh(query.bmm(key)/10)
        if patch_ids is not None:
            patch_sim = patch_sim.view(B, len(patch_ids), -1)

        return patch_sim, patch_ids

    def select_patch(self, feat, patch_ids=None):
        """
        Select the patches
        """
        B, C, W, H = feat.size()
        pw, ph = self.patch_size, self.patch_size
        feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2) # B*N*C
        if self.patch_nums > 0:
            if patch_ids is None:
                patch_ids = torch.randperm(feat_reshape.size(1), device=feat.device)
                patch_ids = patch_ids[:int(min(self.patch_nums, patch_ids.size(0)))]
            feat_query = feat_reshape[:, patch_ids, :]       # B*Num*C
            feat_key = []
            Num = feat_query.size(1)
            if pw < W and ph < H:
                pos_x, pos_y = patch_ids // W, patch_ids % W
                # patch should in the feature
                left, top = pos_x - int(pw / 2), pos_y - int(ph / 2)
                left, top = torch.where(left > 0, left, torch.zeros_like(left)), torch.where(top > 0, top, torch.zeros_like(top))
                start_x = torch.where(left > (W - pw), (W - pw) * torch.ones_like(left), left)
                start_y = torch.where(top > (H - ph), (H - ph) * torch.ones_like(top), top)
                for i in range(Num):
                    feat_key.append(feat[:, :, start_x[i]:start_x[i]+pw, start_y[i]:start_y[i]+ph]) # B*C*patch_w*patch_h
                feat_key = torch.stack(feat_key, dim=0).permute(1, 0, 2, 3, 4) # B*Num*C*patch_w*patch_h
                feat_key = feat_key.reshape(B * Num, C, pw * ph)  # Num * C * N
                feat_query = feat_query.reshape(B * Num, 1, C)  # Num * 1 * C
            else: # if patch larger than features size, use B * C * N (H * W)
                feat_key = feat.reshape(B, C, W*H)
        else:
            feat_query = feat.reshape(B, C, H*W).permute(0, 2, 1) # B * N (H * W) * C
            feat_key = feat.reshape(B, C, H*W)  # B * C * N (H * W)

        return feat_query, feat_key, patch_ids



class SpatialCorrelativeLoss(nn.Module):
    """
    learnable patch-based spatially-correlative loss with contrastive learning
    """
    def __init__(self, loss_mode='cos', patch_nums=256, patch_size=32, norm=True, use_conv=True,
                 init_type='normal', init_gain=0.02, gpu_ids=[], T=0.1):
        super(SpatialCorrelativeLoss, self).__init__()
        self.patch_sim = PatchSim(patch_nums=patch_nums, patch_size=patch_size, norm=norm)
        self.patch_size = patch_size
        self.patch_nums = patch_nums
        self.norm = norm
        self.use_conv = use_conv
        self.conv_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids
        self.loss_mode = loss_mode
        self.T = T
        self.criterion = nn.L1Loss() if norm else nn.SmoothL1Loss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()


    def create_conv(self, feat, layer):
        """
        create the 1*1 conv filter to select the features for a specific task
        :param feat: extracted features from a pretrained VGG or encoder for the similarity and dissimilarity map
        :param layer: different layers use different filter
        :return:
        """
        input_nc = feat.size(1)
        output_nc = max(32, input_nc // 4)
        conv = nn.Sequential(*[nn.Conv2d(input_nc, output_nc, kernel_size=1),
                               nn.ReLU(),
                               nn.Conv2d(output_nc, output_nc, kernel_size=1)])
        conv.to(feat.device)
        setattr(self, 'conv_%d' % layer, conv)
        init_net(conv, self.init_type, self.init_gain, self.gpu_ids)
        
    def update_init_(self):
        self.conv_init = True

    def cal_sim(self, f_src, f_tgt, f_other=None, layer=0, patch_ids=None):
        """
        calculate the similarity map using the fixed/learned query and key
        :param f_src: feature map from source domain
        :param f_tgt: feature map from target domain
        :param f_other: feature map from other image (only used for contrastive learning for spatial network)
        :return:
        """
        if self.use_conv:
            if not self.conv_init:
                self.create_conv(f_src, layer)
            conv = getattr(self, 'conv_%d' % layer)
            f_src, f_tgt = conv(f_src), conv(f_tgt)
            f_other = conv(f_other) if f_other is not None else None
        sim_src, patch_ids = self.patch_sim(f_src, patch_ids)
        sim_tgt, patch_ids = self.patch_sim(f_tgt, patch_ids)
        if f_other is not None:
            sim_other, _ = self.patch_sim(f_other, patch_ids)
        else:
            sim_other = None

        return sim_src, sim_tgt, sim_other

    def compare_sim(self, sim_src, sim_tgt, sim_other):
        """
        measure the shape distance between the same shape and different inputs
        :param sim_src: the shape similarity map from source input image
        :param sim_tgt: the shape similarity map from target output image
        :param sim_other: the shape similarity map from other input image
        :return:
        """
        B, Num, N = sim_src.size()
        if self.loss_mode == 'info' or sim_other is not None:
            sim_src = F.normalize(sim_src, dim=-1)
            sim_tgt = F.normalize(sim_tgt, dim=-1)
            sim_other = F.normalize(sim_other, dim=-1)
            sam_neg1 = (sim_src.bmm(sim_other.permute(0, 2, 1))).view(-1, Num) / self.T
            sam_neg2 = (sim_tgt.bmm(sim_other.permute(0, 2, 1))).view(-1, Num) / self.T
            sam_self = (sim_src.bmm(sim_tgt.permute(0, 2, 1))).view(-1, Num) / self.T
            sam_self = torch.cat([sam_self, sam_neg1, sam_neg2], dim=-1)
            loss = self.cross_entropy_loss(sam_self, torch.arange(0, sam_self.size(0), dtype=torch.long, device=sim_src.device) % (Num))
        else:
            tgt_sorted, _ = sim_tgt.sort(dim=-1, descending=True)
            num = int(N / 4)
            src = torch.where(sim_tgt < tgt_sorted[:, :, num:num + 1], 0 * sim_src, sim_src)
            tgt = torch.where(sim_tgt < tgt_sorted[:, :, num:num + 1], 0 * sim_tgt, sim_tgt)
            if self.loss_mode == 'l1':
                loss = self.criterion((N / num) * src, (N / num) * tgt)
            elif self.loss_mode == 'cos':
                sim_pos = F.cosine_similarity(src, tgt, dim=-1)
                loss = self.criterion(torch.ones_like(sim_pos), sim_pos)
            else:
                raise NotImplementedError('padding [%s] is not implemented' % self.loss_mode)

        return loss

    def loss(self, f_src, f_tgt, f_other=None, layer=0):
        """
        calculate the spatial similarity and dissimilarity loss for given features from source and target domain
        :param f_src: source domain features
        :param f_tgt: target domain features
        :param f_other: other random sampled features
        :param layer:
        :return:
        """
        sim_src, sim_tgt, sim_other = self.cal_sim(f_src, f_tgt, f_other, layer)
        # calculate the spatial similarity for source and target domain
        loss = self.compare_sim(sim_src, sim_tgt, sim_other)
        return loss


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(23, 26):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(26, 28):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(28, 30):
            self.relu5_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        #for param in self.parameters():
        #    param.requires_grad = False

    def forward(self, x, layers=None, encode_only=False, resize=False):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)

        relu4_1 = self.relu4_1(relu3_3)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)

        relu5_1 = self.relu5_1(relu4_3)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
        }
        if encode_only:
            if len(layers) > 0:
                feats = []
                for layer, key in enumerate(out):
                    if layer in layers:
                        feats.append(out[key])
                return feats
            else:
                return out['relu3_1']
        return out