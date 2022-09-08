from base64 import encode
import torch
import util
from models import BaseModel
import models.networks as networks
import models.networks.loss as loss
from .networks.patchnce import PatchNCELoss


class SwappingAutoencoderModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        BaseModel.modify_commandline_options(parser, is_train)
        parser.add_argument("--spatial_code_ch", default=8, type=int)
        parser.add_argument("--global_code_ch", default=2048, type=int)
        parser.add_argument("--lambda_R1", default=10.0, type=float)
        parser.add_argument("--lambda_patch_R1", default=1.0, type=float)
        parser.add_argument("--lambda_L1", default=1.0, type=float)
        parser.add_argument("--lambda_GAN", default=1.0, type=float)
        parser.add_argument("--lambda_PatchGAN", default=1.0, type=float)
        parser.add_argument("--patch_min_scale", default=1 / 8, type=float)
        parser.add_argument("--patch_max_scale", default=1 / 4, type=float)
        parser.add_argument("--patch_num_crops", default=8, type=int)
        parser.add_argument("--patch_use_aggregation",
                            type=util.str2bool, default=True)

        # CUT arguments
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        
        # parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        # parser.add_argument('--netF_nc', type=int, default=256)

        parser.add_argument('--init_type', type=str, default='xavier', choices=['normal', 'xavier', 'kaiming', 'orthogonal'], help='network initialization')
        # parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        
        parser.add_argument('--nce_layers', type=str, default='4,7,9', help='compute NCE loss on which layers')
        parser.add_argument('--patch_nums', type=float, default=256, help='select how many patches for shape consistency, -1 use all')
        parser.add_argument('--nce_patch_size', type=int, default=32, help='patch size to calculate the attention')
        parser.add_argument('--loss_mode', type=str, default='cos', help='which loss type is used, cos | l1 | info')

        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for similarity')

        parser.add_argument('--use_vgg', type=util.str2bool, default=True, help="feature extractor")
        parser.add_argument('--use_norm', type=util.str2bool, default=True, help="normalize the feature map for FLSeSim")
        parser.add_argument('--learned_attn', type=util.str2bool, default=True, help="use the learnable attention map")
        parser.add_argument('--augment', type=util.str2bool, default=True, help="use data augmentation for contrastive learning")


        parser.add_argument('--lambda_NCE', type=float, default=-2.0, help='weight for NCE loss: NCE(G(X), X)')
        # parser.add_argument('--lambda_spatial', type=float, default=10.0, help='weight for spatially-correlative loss')
        # parser.add_argument('--lambda_spatial_idt', type=float, default=0.0, help='weight for idt spatial loss')
        # parser.add_argument('--lambda_perceptual', type=float, default=0.0, help='weight for feature consistency loss')
        # parser.add_argument('--lambda_style', type=float, default=0.0, help='weight for style loss')
        # parser.add_argument('--lambda_identity', type=float, default=0.0, help='use identity mapping')
        # parser.add_argument('--lambda_gradient', type=float, default=0.0, help='weight for the gradient penalty')

        return parser

    def initialize(self, input):
        """
        Initialize the Model
        """
        self.isTrain = self.opt.isTrain
        self.gpu_ids = []
        for i in range(self.opt.num_gpus):
            self.gpu_ids.append(i)

        # return network instances
        self.E = networks.create_network(self.opt, self.opt.netE, "encoder")    # opt.netE="StyleGAN2Resnet" + "encoder"
        self.G = networks.create_network(self.opt, self.opt.netG, "generator")  # opt.netG="StyleGAN2Resnet" + "generator"

        if self.isTrain:
            if self.opt.lambda_GAN > 0.0:                       # already added in gather_options()
                self.D = networks.create_network(
                    self.opt, self.opt.netD, "discriminator")   # opt.netD="StyleGAN2" + "discriminator"
            if self.opt.lambda_PatchGAN > 0.0:                  # already added in gather_options()
                self.Dpatch = networks.create_network(          # opt.netPatchD="StyleGAN2" + "patch_discriminator" 
                    self.opt, self.opt.netPatchD, "patch_discriminator")
        
            # Count the iteration count of the discriminator
            # Used for lazy R1 regularization (c.f. Appendix B of StyleGAN2)
            # all non-required gradient tensor created by register_buffer
            self.register_buffer(
                "num_discriminator_iters", torch.zeros(1, dtype=torch.long)
            )

            self.l1_loss = torch.nn.L1Loss()    # reconstruction loss

            # initialize netF
            if self.opt.lambda_NCE > 0.0:
                # net F to select better features
                self.netF = loss.SpatialCorrelativeLoss(self.opt.loss_mode, self.opt.patch_nums, self.opt.nce_patch_size, self.opt.use_norm, 
                                                                    self.opt.learned_attn, gpu_ids=self.gpu_ids, T=self.opt.nce_T)
                if self.opt.use_vgg:
                    self.netPre = loss.VGG16().to(self.device)
                else:
                    self.netPre = self.E
                
                if not self.opt.learned_attn and self.opt.use_vgg:
                    self.set_requires_grad([self.netPre], False)

                self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

                self.set_input(input)
                # bs_per_gpu = self.real.size(0) // max(len(self.gpu_ids), 1)
                # real_per_gpu = self.real[:bs_per_gpu]

                _ = self.Spatial_Loss(self.netPre, self.real_A, self.real_B, None)


                """
                #self.netF = networks.define_F(self.opt.netF, self.opt.init_type, self.opt.init_gain, self.gpu_ids, self.opt.netF_nc)
                feat_k = self.E(prepare_data_per_gpu, self.nce_layers)
                _, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
                _, _ = self.netF(feat_k, self.opt.num_patches, sample_ids)
                
                self.criterionNCE = []
                for nce_layer in self.nce_layers:
                    self.criterionNCE.append(PatchNCELoss(self.opt).to(self.device))
                """
        # load check_point file
        # if not train or true continue train
        if (not self.opt.isTrain) or self.opt.continue_train:
            self.load()

        # model to gpu
        if self.opt.num_gpus > 0:
            self.to("cuda:0")


    def set_input(self, input):   # return batch tensor
        if self.opt.use_unaligned:
            self.real_A = input["real_A"].to(self.device)# night
            self.real_B = input["real_B"].to(self.device) # day
            # A = A[torch.randperm(A.size(0))] # shuffle A
            # B = B[torch.randperm(B.size(0))] # shuffle B
            # c = list(A.shape)
            # c[0] = 2*c[0]
            # A = torch.cat([A, B], dim=1).view(tuple(c))
            self.real = torch.cat([self.real_A, self.real_B], dim=0).to(self.device)

            if self.opt.isTrain and self.opt.augment:
                self.aug_A = input['aug_A'].to(self.device)
                self.aug_B = input['aug_B'].to(self.device)
        else:
            self.real = input["real_A"].to(self.device)

    def compute_forward(self):
        self.sp, self.gl = self.E(self.real)   # encoder forward with real images, return strucure code and global code for each sample
        self.B = self.real.size(0)        # check if batch size is even
        assert self.B % 2 == 0, "Batch size must be even on each GPU."

        # To save memory, compute the GAN loss on only
        # half of the reconstructed images
        # self.rec = self.G(self.sp[:B // 2], self.gl[:B // 2])
        self.rec = self.G(self.sp, self.gl)
        #self.rec_A, self.rec_B = rec[:B//2], rec[B//2:]
        #self.mix_B, self.mix_A = self.mix[:B//2], self.mix[B//2:]
        self.mix = self.G(self.sp, self.swap(self.gl))
        
    def compute_netF_losses(self, input):
        """
        Calculate the contrastive loss for learned spatially-correlative loss
        """
        # TODO normalize and augment
        self.set_input(input)
        self.compute_forward()
        norm_real_A, norm_real_B, norm_fake_B = (self.real_A + 1) * 0.5, (self.real_B + 1) * 0.5, (self.mix[:self.B//2].detach() + 1) * 0.5
    
        if self.opt.augment:
            norm_aug_A, norm_aug_B = (self.aug_A + 1) * 0.5, (self.aug_B + 1) * 0.5
            norm_real_A = torch.cat([norm_real_A, norm_real_A], dim=0)
            norm_fake_B = torch.cat([norm_fake_B, norm_aug_A], dim=0)
            norm_real_B = torch.cat([norm_real_B, norm_aug_B], dim=0)

        # loss_spatial_A = self.Spatial_Loss(self.netPre, self.real_A, self.mix[:self.B//2], self.real_B)
        # loss_spatial_B = self.Spatial_Loss(self.netPre, self.real_B, self.mix[self.B//2:], self.real_A)
        # loss_spatial = 0.5*loss_spatial_A + 0.5*loss_spatial_B
        loss_spatial_A = self.Spatial_Loss(self.netPre, norm_real_A, norm_fake_B, norm_real_B)

        norm_real_A, norm_real_B, norm_fake_A = (self.real_A + 1) * 0.5, (self.real_B + 1) * 0.5, (self.mix[self.B//2:].detach() + 1) * 0.5

        if self.opt.augment:
            norm_real_B = torch.cat([norm_real_B, norm_real_B], dim=0)
            norm_fake_A = torch.cat([norm_fake_A, norm_aug_B], dim=0)
            norm_real_A = torch.cat([norm_real_A, norm_aug_A], dim=0)
            
        loss_spatial_B = self.Spatial_Loss(self.netPre, norm_real_B, norm_fake_A, norm_real_A)

        loss_spatial = 0.5 * (loss_spatial_A + loss_spatial_B)

        return loss_spatial

    def swap(self, x):
        """ Swaps (or mixes) the ordering of the minibatch """
        shape = x.shape
        assert shape[0] % 2 == 0, "Minibatch size must be a multiple of 2"
        B = shape[0]//2
        x = torch.cat([x[B:], x[:B]], dim=0)
        # new_shape = [shape[0] // 2, 2] + list(shape[1:])
        # x = x.view(*new_shape)
        # x = torch.flip(x, [1])

        return x

    def get_random_crops(self, x, crop_window=None):
        """ Make random crops.
            Corresponds to the yellow and blue random crops of Figure 2.

            Output tensor = (batch size, num_crops, channel, height, width)
        """
        crops = util.apply_random_crop(
            x, self.opt.patch_size,
            (self.opt.patch_min_scale, self.opt.patch_max_scale),
            num_crops=self.opt.patch_num_crops
        )
        return crops

    def compute_image_discriminator_losses(self):
        if self.opt.lambda_GAN == 0.0:
            return {}

        pred_real = self.D(self.real)
        pred_rec = self.D(self.rec)
        pred_mix = self.D(self.mix)

        losses = {}
        losses["D_real"] = loss.gan_loss(
            pred_real, should_be_classified_as_real=True
        ) * self.opt.lambda_GAN

        losses["D_rec"] = loss.gan_loss(
            pred_rec, should_be_classified_as_real=False
        ) * (1.0 * self.opt.lambda_GAN)
        losses["D_mix"] = loss.gan_loss(
            pred_mix, should_be_classified_as_real=False
        ) * (1.0 * self.opt.lambda_GAN)

        return losses

    def compute_patch_discriminator_losses(self):
        losses = {}
        real_feat = self.Dpatch.extract_features(
            self.get_random_crops(self.real),
            aggregate=self.opt.patch_use_aggregation
        )
        target_feat = self.Dpatch.extract_features(self.get_random_crops(self.real))
        mix_feat = self.Dpatch.extract_features(self.get_random_crops(self.mix))
 
        losses["PatchD_real"] = loss.gan_loss(
            self.Dpatch.discriminate_features(real_feat, target_feat),
            should_be_classified_as_real=True,
        ) * self.opt.lambda_PatchGAN

        losses["PatchD_mix"] = loss.gan_loss(
            self.Dpatch.discriminate_features(real_feat, mix_feat),
            should_be_classified_as_real=False,
        ) * self.opt.lambda_PatchGAN

        return losses

    def compute_discriminator_losses(self, input):
        self.set_input(input)
        self.compute_forward()
        self.num_discriminator_iters.add_(1)
        """
        sp, gl = self.E(real)   # encoder forward with real images, return strucure code and global code for each sample
        B = real.size(0)        # check if batch size is even
        assert B % 2 == 0, "Batch size must be even on each GPU."

        # To save memory, compute the GAN loss on only
        # half of the reconstructed images
        rec = self.G(sp[:B // 2], gl[:B // 2])  # here sp and gl are pairs, half of them are input to generator to get reconstructed images
        mix = self.G(self.swap(sp), gl)         # swap the orderof two continious sps, get the mix images, all resultant sp/gl are mixed
        """
        losses = self.compute_image_discriminator_losses()    # self.D loss = "D_real", "D_rec" and "D_mix"

        if self.opt.lambda_PatchGAN > 0.0:
            patch_losses = self.compute_patch_discriminator_losses() # self.Dpatch loss = "PatchD_real" and "PatchD_mix"
            losses.update(patch_losses) # add path_losses into losses dict

        metrics = {}  # no metrics to report for the Discriminator iteration

        return losses, metrics

    def compute_R1_loss(self):
        losses = {}
        if self.opt.lambda_R1 > 0.0:
            self.real.requires_grad_()
            pred_real = self.D(self.real).sum()
            grad_real, = torch.autograd.grad(
                outputs=pred_real,
                inputs=[self.real],
                create_graph=True,
                retain_graph=True,
            )
            grad_real2 = grad_real.pow(2)
            dims = list(range(1, grad_real2.ndim))
            grad_penalty = grad_real2.sum(dims) * (self.opt.lambda_R1 * 0.5)
        else:
            grad_penalty = 0.0

        if self.opt.lambda_patch_R1 > 0.0:
            real_crop = self.get_random_crops(self.real).detach()
            real_crop.requires_grad_()
            target_crop = self.get_random_crops(self.real).detach()
            target_crop.requires_grad_()

            real_feat = self.Dpatch.extract_features(
                real_crop,
                aggregate=self.opt.patch_use_aggregation)
            target_feat = self.Dpatch.extract_features(target_crop)
            pred_real_patch = self.Dpatch.discriminate_features(
                real_feat, target_feat
            ).sum()

            grad_real, grad_target = torch.autograd.grad(
                outputs=pred_real_patch,
                inputs=[real_crop, target_crop],
                create_graph=True,
                retain_graph=True,
            )

            dims = list(range(1, grad_real.ndim))
            grad_crop_penalty = grad_real.pow(2).sum(dims) + \
                grad_target.pow(2).sum(dims)
            grad_crop_penalty *= (0.5 * self.opt.lambda_patch_R1 * 0.5)
        else:
            grad_crop_penalty = 0.0

        losses["D_R1"] = grad_penalty + grad_crop_penalty

        return losses

    def compute_generator_losses(self, input):
        losses, metrics = {}, {}
        """
        
        sp, gl = self.E(real)
        rec = self.G(sp[:B // 2], gl[:B // 2])  # only on B//2 to save memory
        sp_mix = self.swap(sp)

        if self.opt.crop_size >= 1024:
        # another momery-saving trick: reduce #outputs to save memory
        real = real[B // 2:]
        gl = gl[B // 2:]
        sp_mix = sp_mix[B // 2:]
        
        mix = self.G(sp_mix, gl)
        """
        #B = self.real.size(0)
        # record the error of the reconstructed images for monitoring purposes
        self.set_input(input)
        self.compute_forward()

        metrics["L1_dist"] = self.l1_loss(self.rec, self.real)

        if self.opt.lambda_L1 > 0.0:
            losses["G_L1"] = metrics["L1_dist"] * self.opt.lambda_L1

        if self.opt.lambda_GAN > 0.0:
            losses["G_GAN_rec"] = loss.gan_loss(
                self.D(self.rec),
                should_be_classified_as_real=True
            ) * (self.opt.lambda_GAN * 1.0)

            losses["G_GAN_mix"] = loss.gan_loss(
                self.D(self.mix),
                should_be_classified_as_real=True
            ) * (self.opt.lambda_GAN * 1.0)

        if self.opt.lambda_PatchGAN > 0.0:
            real_feat = self.Dpatch.extract_features(
                self.get_random_crops(self.real),
                aggregate=self.opt.patch_use_aggregation).detach()
            mix_feat = self.Dpatch.extract_features(self.get_random_crops(self.mix))
            # patchGAN loss
            losses["G_mix"] = loss.gan_loss(
                self.Dpatch.discriminate_features(real_feat, mix_feat),
                should_be_classified_as_real=True,
            ) * self.opt.lambda_PatchGAN

        # TODO Normalize
        if self.opt.lambda_NCE > 0.0:
            norm_real_A, norm_real_B = (self.real_A + 1) * 0.5, (self.real_B + 1) * 0.5
            norm_fake_B, norm_fake_A = (self.mix[:self.B//2].detach() + 1) * 0.5, (self.mix[self.B//2:].detach() + 1) * 0.5
            losses["G_spatial_loss"] = self.Spatial_Loss(self.netPre, norm_real_A, norm_fake_B, None) * self.opt.lambda_NCE * 0.5 + \
                                        self.Spatial_Loss(self.netPre, norm_real_B, norm_fake_A, None) * self.opt.lambda_NCE * 0.5

        return losses, metrics

    def Spatial_Loss(self, net, src, tgt, other=None):
        """given the source (real) and target images (mix) to calculate the spatial similarity and dissimilarity loass"""

        n_layers = len(self.nce_layers)
        feats_src = net(src, self.nce_layers, encode_only=True)   # list of features
        feats_tgt = net(tgt, self.nce_layers, encode_only=True)
        if other is not None:
            feats_oth = net(torch.flip(other, [2, 3]), self.nce_layers, encode_only=True)
        else:
            feats_oth = [None for _ in range(n_layers)]

        total_loss = 0.0
        for i, (feat_src, feat_tgt, feat_oth) in enumerate(zip(feats_src, feats_tgt, feats_oth)):
            loss = self.netF.loss(feat_src, feat_tgt, feat_oth, i)
            total_loss += loss.mean()

        if not self.netF.conv_init:
            self.netF.update_init_()

        return total_loss / n_layers

    """
    def calculate_NCE_loss(self, net, src, tgt):
        # given the source (real) and target images (mix) to calculate the NCE loss
    
        n_layers = len(self.nce_layers)

        feats_src = net(src, self.nce_layers)   # list of features
        feats_tgt = net(tgt, self.nce_layers)
        

        feat_k_pool, sample_ids = self.netF(feats_src, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feats_tgt, self.opt.num_patches, sample_ids)

        total_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_loss += loss.mean()

        return total_loss / n_layers
    """


    def get_visuals_for_snapshot(self):
        # if self.opt.isTrain:
        #     # avoid the overhead of generating too many visuals during training
        #     real = real[:2] if self.opt.num_gpus > 1 else real[:4]
        #real = torch.cat([self.real_A[:2], self.real_B[:2]], dim=0)
        #sp, gl = self.E(real)
        # layout = util.resize2d_tensor(util.visualize_spatial_code(sp), real)
        # rec = self.G(sp, gl)
        # mix = self.G(sp, self.swap(gl))

        visual_ids_A = torch.randperm(self.real_A.size(0), device=self.real_A.device)[:2]
        visual_ids_B = visual_ids_A + self.real_A.size(0)

        real = torch.cat([self.real[visual_ids_A], self.real[visual_ids_B]], dim=1).view(*([4]+list(self.real.shape[1:])))
        sp = torch.cat([self.sp[visual_ids_A], self.sp[visual_ids_B]], dim=1).view(*([4]+list(self.sp.shape[1:])))
        layout = util.resize2d_tensor(util.visualize_spatial_code(sp), real)
        rec = torch.cat([self.rec[visual_ids_A], self.rec[visual_ids_B]], dim=1).view(*([4]+list(self.rec.shape[1:])))
        mix = torch.cat([self.mix[visual_ids_A], self.mix[visual_ids_B]], dim=1).view(*([4]+list(self.mix.shape[1:])))

        visuals = {"real": real, "layout": layout, "rec": rec, "mix": mix}

        return visuals

    def fix_noise(self, sample_image=None):
        """ The generator architecture is stochastic because of the noise
        input at each layer (StyleGAN2 architecture). It could lead to
        flickering of the outputs even when identical inputs are given.
        Prevent flickering by fixing the noise injection of the generator.
        """
        if sample_image is not None:
            # The generator should be run at least once,
            # so that the noise dimensions could be computed
            sp, gl = self.E(sample_image)
            self.G(sp, gl)
        noise_var = self.G.fix_and_gather_noise_parameters()
        return noise_var

    def encode(self, image, extract_features=False):
        return self.E(image, extract_features=extract_features)

    def decode(self, spatial_code, global_code):
        return self.G(spatial_code, global_code)

    def get_parameters_for_mode(self, mode):    # return network parameters
        if mode == "generator":
            return list(self.G.parameters()) + list(self.E.parameters())
        elif mode == "discriminator":
            Dparams = []
            if self.opt.lambda_GAN > 0.0:
                Dparams += list(self.D.parameters())
            if self.opt.lambda_PatchGAN > 0.0:
                Dparams += list(self.Dpatch.parameters())
            return Dparams
        elif mode == 'netF' and self.opt.lambda_NCE > 0.0:
            return list(self.netF.parameters())

    def per_gpu_initialize(self):
        pass