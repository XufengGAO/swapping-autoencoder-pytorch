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
        parser.add_argument('--CUT_mode', type=str, default="None", choices='(CUT, cut, FastCUT, fastcut)')
        parser.add_argument('--lambda_NCE', type=float, default=0.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        parser.add_argument('--init_type', type=str, default='xavier', choices=['normal', 'xavier', 'kaiming', 'orthogonal'], help='network initialization')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True
            )

        return parser

    def initialize(self, prepare_data):
        # return network instances
        self.E = networks.create_network(self.opt, self.opt.netE, "encoder")    # opt.netE="StyleGAN2Resnet" + "encoder"
        self.G = networks.create_network(self.opt, self.opt.netG, "generator")  # opt.netG="StyleGAN2Resnet" + "generator"
        if self.opt.lambda_GAN > 0.0:       # already added in gather_options()
            self.D = networks.create_network(
                self.opt, self.opt.netD, "discriminator")   # opt.netD="StyleGAN2" + "discriminator"
        if self.opt.lambda_PatchGAN > 0.0:  # already added in gather_options()
            self.Dpatch = networks.create_network(          # opt.netPatchD="StyleGAN2" + "patch_discriminator" 
                self.opt, self.opt.netPatchD, "patch_discriminator" 
            )
        
        self.gpu_ids = []
        for i in range(self.opt.num_gpus):
            self.gpu_ids.append(i)

        

        # Count the iteration count of the discriminator
        # Used for lazy R1 regularization (c.f. Appendix B of StyleGAN2)
        # all non-required gradient tensor created by register_buffer
        self.register_buffer(
            "num_discriminator_iters", torch.zeros(1, dtype=torch.long)
        )
        self.l1_loss = torch.nn.L1Loss()    # reconstruction loss

        # set netF
        if self.opt.lambda_NCE > 0.0:
            self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
            self.netF = networks.define_F(self.opt.netF, self.opt.init_type, self.opt.init_gain, self.gpu_ids, self.opt.netF_nc)
            preimages = self.prepare_images(prepare_data)

            bs_per_gpu = preimages.size(0) // max(len(self.opt.gpu_ids), 1)
            pre_images_per_gpu = preimages[:bs_per_gpu]

            feat_k = self.E(pre_images_per_gpu, self.nce_layers)
            _, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
            _, _ = self.netF(feat_k, self.opt.num_patches, sample_ids)
            
            self.criterionNCE = []
            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(self.opt).to(self.device))


        # load check_point file
        # if not train or true continue train
        if (not self.opt.isTrain) or self.opt.continue_train:
            self.load()

        # model to gpu
        if self.opt.num_gpus > 0:
            self.to("cuda:0")

    def prepare_images(self, data_i):   # return batch tensor
        A = data_i["real_A"] # night
        if "real_B" in data_i:
            B = data_i["real_B"] # day
            A = A[torch.randperm(A.size(0))] # shuffle A
            B = B[torch.randperm(B.size(0))] # shuffle B
            c = list(A.shape)
            c[0] = 2*c[0]
            A = torch.cat([A, B], dim=1).view(tuple(c))

        return A

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        

    def per_gpu_initialize(self):
        pass

    def swap(self, x):
        """ Swaps (or mixes) the ordering of the minibatch """
        shape = x.shape
        assert shape[0] % 2 == 0, "Minibatch size must be a multiple of 2"
        new_shape = [shape[0] // 2, 2] + list(shape[1:])
        x = x.view(*new_shape)
        x = torch.flip(x, [1])
        return x.view(*shape)

    def compute_image_discriminator_losses(self, real, rec, mix):
        if self.opt.lambda_GAN == 0.0:
            return {}

        pred_real = self.D(real)
        pred_rec = self.D(rec)
        pred_mix = self.D(mix)

        losses = {}
        losses["D_real"] = loss.gan_loss(
            pred_real, should_be_classified_as_real=True
        ) * self.opt.lambda_GAN

        losses["D_rec"] = loss.gan_loss(
            pred_rec, should_be_classified_as_real=False
        ) * (0.5 * self.opt.lambda_GAN)
        losses["D_mix"] = loss.gan_loss(
            pred_mix, should_be_classified_as_real=False
        ) * (0.5 * self.opt.lambda_GAN)

        return losses

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

    def compute_patch_discriminator_losses(self, real, mix):
        losses = {}
        real_feat = self.Dpatch.extract_features(
            self.get_random_crops(real),
            aggregate=self.opt.patch_use_aggregation
        )
        target_feat = self.Dpatch.extract_features(self.get_random_crops(real))
        mix_feat = self.Dpatch.extract_features(self.get_random_crops(mix))

        losses["PatchD_real"] = loss.gan_loss(
            self.Dpatch.discriminate_features(real_feat, target_feat),
            should_be_classified_as_real=True,
        ) * self.opt.lambda_PatchGAN

        losses["PatchD_mix"] = loss.gan_loss(
            self.Dpatch.discriminate_features(real_feat, mix_feat),
            should_be_classified_as_real=False,
        ) * self.opt.lambda_PatchGAN

        return losses

    def compute_discriminator_losses(self, real):
        self.num_discriminator_iters.add_(1)

        sp, gl = self.E(real)   # encoder forward with real images, return strucure code and global code for each sample
        B = real.size(0)        # check if batch size is even
        assert B % 2 == 0, "Batch size must be even on each GPU."

        # To save memory, compute the GAN loss on only
        # half of the reconstructed images
        rec = self.G(sp[:B // 2], gl[:B // 2])  # here sp and gl are pairs, half of them are input to generator to get reconstructed images
        mix = self.G(self.swap(sp), gl)         # swap the orderof two continious sps, get the mix images, all resultant sp/gl are mixed

        losses = self.compute_image_discriminator_losses(real, rec, mix)    # self.D loss = "D_real", "D_rec" and "D_mix"

        if self.opt.lambda_PatchGAN > 0.0:
            patch_losses = self.compute_patch_discriminator_losses(real, mix) # self.Dpatch loss = "PatchD_real" and "PatchD_mix"
            losses.update(patch_losses) # add path_losses into losses dict

        metrics = {}  # no metrics to report for the Discriminator iteration

        return losses, metrics, sp.detach(), gl.detach()

    def compute_R1_loss(self, real):
        losses = {}
        if self.opt.lambda_R1 > 0.0:
            real.requires_grad_()
            pred_real = self.D(real).sum()
            grad_real, = torch.autograd.grad(
                outputs=pred_real,
                inputs=[real],
                create_graph=True,
                retain_graph=True,
            )
            grad_real2 = grad_real.pow(2)
            dims = list(range(1, grad_real2.ndim))
            grad_penalty = grad_real2.sum(dims) * (self.opt.lambda_R1 * 0.5)
        else:
            grad_penalty = 0.0

        if self.opt.lambda_patch_R1 > 0.0:
            real_crop = self.get_random_crops(real).detach()
            real_crop.requires_grad_()
            target_crop = self.get_random_crops(real).detach()
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

    def compute_generator_losses(self, real, sp_ma, gl_ma):
        losses, metrics = {}, {}
        B = real.size(0)

        sp, gl = self.E(real)

        rec = self.G(sp[:B // 2], gl[:B // 2])  # only on B//2 to save memory
        sp_mix = self.swap(sp)
        
        # record the error of the reconstructed images for monitoring purposes
        metrics["L1_dist"] = self.l1_loss(rec, real[:B // 2])

        if self.opt.lambda_L1 > 0.0:
            losses["G_L1"] = metrics["L1_dist"] * self.opt.lambda_L1

        if self.opt.crop_size >= 1024:
            # another momery-saving trick: reduce #outputs to save memory
            real = real[B // 2:]
            gl = gl[B // 2:]
            sp_mix = sp_mix[B // 2:]

        mix = self.G(sp_mix, gl)

        if self.opt.lambda_GAN > 0.0:
            losses["G_GAN_rec"] = loss.gan_loss(
                self.D(rec),
                should_be_classified_as_real=True
            ) * (self.opt.lambda_GAN * 0.5)

            losses["G_GAN_mix"] = loss.gan_loss(
                self.D(mix),
                should_be_classified_as_real=True
            ) * (self.opt.lambda_GAN * 1.0)

        if self.opt.lambda_PatchGAN > 0.0:
            real_feat = self.Dpatch.extract_features(
                self.get_random_crops(real),
                aggregate=self.opt.patch_use_aggregation).detach()
            mix_feat = self.Dpatch.extract_features(self.get_random_crops(mix))
            # patchGAN loss
            losses["G_mix"] = loss.gan_loss(
                self.Dpatch.discriminate_features(real_feat, mix_feat),
                should_be_classified_as_real=True,
            ) * self.opt.lambda_PatchGAN

        if self.opt.lambda_NCE > 0.0:
           losses["G_NCE"] = self.calculate_NCE_loss(real, mix)  # check the mutual information, src=real, tgt=mix

        return losses, metrics

    def calculate_NCE_loss(self, src, tgt):
        # src=real, tgt=mix
        n_layers = len(self.nce_layers)
        feat_q = self.E(tgt, self.nce_layers)
        feat_k = self.E(src, self.nce_layers)

        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def get_visuals_for_snapshot(self, real):
        if self.opt.isTrain:
            # avoid the overhead of generating too many visuals during training
            real = real[:2] if self.opt.num_gpus > 1 else real[:4]
        sp, gl = self.E(real)
        layout = util.resize2d_tensor(util.visualize_spatial_code(sp), real)
        rec = self.G(sp, gl)
        mix = self.G(sp, self.swap(gl))

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
