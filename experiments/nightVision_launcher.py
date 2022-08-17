from random import shuffle
from .tmux_launcher import Options, TmuxLauncher

class Launcher(TmuxLauncher):
    def options(self):
        opt = Options()
        opt.set(
            dataroot="./datasets/nightVisionDatesets/",
            dataset_mode="nightVision",
            checkpoints_dir="./checkpoints/",

            # num_workers=int(opt.num_gpus)
            num_gpus=8, batch_size=16,

            # scale the image such that the short side is |load_size|, and
            # crop a square window of |crop_size|.
            preprocess="scale_shortside_and_crop",
            load_size=512, crop_size=512,

            display_freq=1600, print_freq=480,
        )

        return [
            opt.specify(
                name="nightVision_default",
                # patch_use_aggregation=False,
            ),
        ]

    def train_options(self):
        common_options = self.options()
        return [opt.specify(
            continue_train=False,
            evaluation_metrics="swap_visualization",
            evaluation_freq=50000,
        ) for opt in common_options]

    def test_options(self):
        return super().test_options()