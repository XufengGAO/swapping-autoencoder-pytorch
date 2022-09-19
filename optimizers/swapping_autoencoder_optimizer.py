from distutils.cmd import Command
import torch
import util
from models import MultiGPUModelWrapper
from optimizers.base_optimizer import BaseOptimizer


class SwappingAutoencoderOptimizer(BaseOptimizer):
    """ Class for running the optimization of the model parameters.
    Implements Generator / Discriminator training, R1 gradient penalty,
    decaying learning rates, and reporting training progress.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--lr", default=0.002, type=float)
        parser.add_argument("--beta1", default=0.0, type=float)
        parser.add_argument("--beta2", default=0.99, type=float)
        parser.add_argument(
            "--R1_once_every", default=16, type=int,
            help="lazy R1 regularization. R1 loss is computed "
                 "once in 1/R1_freq times",
        )
        return parser

    def __init__(self, model: MultiGPUModelWrapper):
        self.opt = model.opt
        opt = self.opt
        self.model = model
        self.train_mode_counter = 0
        self.discriminator_iter_counter = 0

        self.Gparams = self.model.get_parameters_for_mode("generator")      # list parameters: self.G, self.E
        self.Dparams = self.model.get_parameters_for_mode("discriminator")  # self.D, self.patchD


        self.optimizer_G = torch.optim.Adam(
            self.Gparams, lr=opt.lr, betas=(opt.beta1, opt.beta2)   # see above
        )

        # c.f. StyleGAN2 (https://arxiv.org/abs/1912.04958) Appendix B
        c = opt.R1_once_every / (1 + opt.R1_once_every)
        self.optimizer_D = torch.optim.Adam(
            self.Dparams, lr=opt.lr * c, betas=(opt.beta1 ** c, opt.beta2 ** c)
        )

        # TODO lr and beta values for netF
        if self.opt.use_NCE:
            self.Fparams = self.model.get_parameters_for_mode("netF")
            self.optimizer_F = torch.optim.Adam(
                self.Fparams, lr=opt.lr, betas=(opt.beta1, opt.beta2)
            )


    def set_requires_grad(self, params, requires_grad):
        """ For more efficient optimization, turn on and off
            recording of gradients for |params|.
        """
        for p in params:
            p.requires_grad_(requires_grad)

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

    def toggle_training_mode(self):
        modes = ["discriminator", "generator"]
        self.train_mode_counter = (self.train_mode_counter + 1) % len(modes)
        return modes[self.train_mode_counter]

    def train_one_step(self, data_i, total_steps_so_far=0):
        #images_minibatch = self.prepare_images(data_i)
        # self.model(data_i, command="set_input" )
        # self.model(command="compute_forward")

        if self.opt.use_NCE:
            spatial_loss = self.train_netF_one_step(data_i)

        if self.toggle_training_mode() == "generator":
            losses = self.train_discriminator_one_step(data_i)
        else:
            losses = self.train_generator_one_step(data_i)

        if self.opt.use_NCE:
            losses['netF_spatial_loss'] = spatial_loss

        return util.to_numpy(losses)

    def train_netF_one_step(self, data_i):
        self.set_requires_grad(self.Fparams, True)  # only record F's gradient
        self.set_requires_grad(self.Dparams, False)
        self.set_requires_grad(self.Gparams, False)
        self.optimizer_F.zero_grad()

        spatial_loss = self.model(data_i, command="compute_netF_losses")
        spatial_loss.backward()
        self.optimizer_F.step()

        return spatial_loss
            
    def train_generator_one_step(self, data_i):
        self.set_requires_grad(self.Dparams, False)
        self.set_requires_grad(self.Gparams, True)  # only record G's gradient

        if self.opt.use_NCE:
            self.set_requires_grad(self.Fparams, False)
        
        
        self.optimizer_G.zero_grad()

        g_losses, g_metrics = self.model(data_i, command="compute_generator_losses")
        g_loss = sum([v.mean() for v in g_losses.values()])
        g_loss.backward()
        
        self.optimizer_G.step()

        g_losses["G_total"] = g_loss
        g_losses.update(g_metrics)
        return g_losses

    def train_discriminator_one_step(self, data_i):
        if self.opt.lambda_GAN == 0.0 and self.opt.lambda_PatchGAN == 0.0:
            return {}
        self.set_requires_grad(self.Dparams, True)  # only record D's gradients
        self.set_requires_grad(self.Gparams, False)
        if self.opt.use_NCE:
            self.set_requires_grad(self.Fparams, False)

        self.discriminator_iter_counter += 1
        self.optimizer_D.zero_grad()
        d_losses, d_metrics = self.model(data_i, command="compute_discriminator_losses")
        #self.previous_sp = sp.detach()  # record the calculated sp/gl, and detach it (just store values)
        #self.previous_gl = gl.detach()

        d_loss = sum([v.mean() for v in d_losses.values()])
        d_loss.backward()   # backward D, G, E, but below just step D, so no worries to G and E
        self.optimizer_D.step()

        needs_R1 = self.opt.lambda_R1 > 0.0 or self.opt.lambda_patch_R1 > 0.0
        needs_R1_at_current_iter = needs_R1 and \
            self.discriminator_iter_counter % self.opt.R1_once_every == 0
        if needs_R1_at_current_iter:        # R1 regularization
            self.optimizer_D.zero_grad()
            r1_losses = self.model(data_i, command="compute_R1_loss")
            d_losses.update(r1_losses)
            r1_loss = sum([v.mean() for v in r1_losses.values()])
            r1_loss = r1_loss * self.opt.R1_once_every
            r1_loss.backward()
            self.optimizer_D.step()

        d_losses["D_total"] = sum([v.mean() for v in d_losses.values()])
        d_losses.update(d_metrics)
        return d_losses

    def get_visuals_for_snapshot(self, data_i):
        with torch.no_grad():
            return self.model(data_i, command="get_visuals_for_snapshot")

    def save(self, epoch, total_steps_so_far):
        self.model.save(epoch, total_steps_so_far)
