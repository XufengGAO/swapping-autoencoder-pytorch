from email.policy import default
import numpy as np
import torch
import os
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE
from func_timeout import func_timeout, FunctionTimedOut


if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s/%s.png' % (label, name)
        os.makedirs(os.path.join(image_dir, label), exist_ok=True)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--server', type=str, default='izar')
        parser.add_argument("--display_port", default=8097)
        parser.add_argument("--display_ncols", default=2)
        parser.add_argument("--display_env", default="izar_default")
        parser.add_argument("--no_html", type=util.str2bool, nargs='?', const=True, default=True)

        return parser

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.display_id = np.random.randint(1000000) * 10  # just a random display id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.crop_size
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False 
        if self.display_id > 0:
            # connect to a visdom server
            import visdom
            self.plot_data = {}
            self.ncols = opt.display_ncols
            if "tensorboard_base_url" in os.environ:
                self.vis = visdom.Visdom(
                    port=2004,
                    base_url=os.environ['tensorboard_base_url'] + '/visdom',
                    env=opt.display_env,
                    #raise_exceptions=False,
                )
                print("setting up visdom server for sensei")
            else:
                print('')
                self.vis = visdom.Visdom(
                    server="http://"+opt.server,
                    port=opt.display_port,
                    env=opt.display_env,
                    raise_exceptions=False)

                
                # test code
                self.vis.text('Hello Visdom!')
                image = np.random.randint(0, 255, (3, 256, 256), dtype=np.uint8)
                self.vis.image(image)
                print("http://", opt.server, opt.display_port, opt.display_env)                
                
            if not self.vis.check_connection():
                self.create_visdom_connections()

        if self.use_html:
            # Create an HTML object at <checkpoints_dir>/web/;
            # Images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])

        # create a logging file to store training losses
        self.log_name = os.path.join(
            opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a+") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server,
        this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. '
              '\n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch,
                                save_result=None, max_num_images=4):
        """Display current results on visdom;
        save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        if save_result is None:
            save_result = not self.opt.no_html
        if self.display_id > 0:  # show images in the browser using visdom
            ncols = self.ncols
            if ncols > 0:        # show all the images in one visdom panel
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)  # create a table css
                # create a table of images.
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    if image.size(3) < 64:
                        image = torch.nn.functional.interpolate(
                            image, size=(64, 64),
                            mode='bilinear', align_corners=False)
                    image_numpy = util.tensor2im(image[:max_num_images])
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(
                    image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                try:
                    func_timeout(15, self.vis.images,
                                 args=(images, ncols, 2, self.display_id + 1,
                                       None, dict(title=title + ' images')))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html,
                                  win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except FunctionTimedOut:
                    print("visdom call to display image timed out")
                    pass
                except VisdomExceptionBase:
                    self.create_visdom_connections()

            else:     # show each image in a separate visdom panel;
                idx = 1
                try:
                    for label, image in visuals.items():
                        image_numpy = util.tensor2im(image[:4])
                        try:
                            func_timeout(5, self.vis.image, args=(
                                image_numpy.transpose([2, 0, 1]),
                                self.display_id + idx,
                                None,
                                dict(title=label)
                            ))
                        except FunctionTimedOut:
                            print("visdom call to display image timed out")
                            pass
                        idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()

        needs_save = save_result or not self.saved
        if self.use_html and needs_save:
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image[:4])
                img_path = os.path.join(
                    self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(
                self.web_dir, 'Experiment name = %s' % self.name, refresh=0)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if len(losses) == 0:
            return

        plot_name = '_'.join(list(losses.keys()))

        if plot_name not in self.plot_data:
            self.plot_data[plot_name] = {'X': [], 'Y': [], 'legend': list(losses.keys())}

        plot_data = self.plot_data[plot_name]
        plot_id = list(self.plot_data.keys()).index(plot_name)

        plot_data['X'].append(epoch + counter_ratio)
        plot_data['Y'].append([losses[k] for k in plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
                Y=np.array(plot_data['Y']),
                opts={
                    'title': self.name,
                    'legend': plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id - plot_id)
        except VisdomExceptionBase:
            self.create_visdom_connections()

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, iters, times, losses):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(iters: %d' % (iters)
        for k, v in times.items():
            message += ", %s: %.3f" % (k, v)
        message += ") "
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v.mean())

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
