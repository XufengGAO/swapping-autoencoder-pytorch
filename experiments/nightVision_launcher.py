from random import shuffle
from .tmux_launcher import Options, TmuxLauncher

class Launcher(TmuxLauncher):
    def options(self):
        opt = Options()
        opt.set(
            dataroot="/home/xugao/gitRepo/swapping-autoencoder-pytorch/datasets/nightVisionDatasets",
            dataset_mode="nightVision",
            checkpoints_dir="./checkpoints/",

            # num_workers=int(opt.num_gpus)
            num_gpus=2, batch_size=16,

            # scale the image such that the short side is |load_size|, and
            # crop a square window of |crop_size|.
            preprocess="scale_shortside_and_crop",
            load_size=512, crop_size=512,
            server='izar',
        )

        return [
            opt.specify(
                name="nightVision_smallPatch_smallDsp",
                netE_num_downsampling_sp=2,
                patch_min_scale=1/16,
                patch_max_scale=1/8,
                patch_num_crops = 8,
                patch_size=64,
                evaluation_metrics="none",
                display_env="izar_smallPatch_smallDsp",
                tb_folder="./runs/smallPatch_smallDsp/"
            ), 
            opt.specify(
                name="nightVision_unaligned", 
                use_unaligned=True,
                dataset_mode="unaligned",
                netE_num_downsampling_sp=2,
                patch_min_scale=1/16,
                patch_max_scale=1/8,
                patch_num_crops = 8,
                patch_size=64,
                evaluation_metrics="none",
                display_env="izar_unaligned",
                tb_folder="./runs/unaligned/"
            ),
            opt.specify(
                name="nightVision_unaligned_default",
                use_unaligned=True,
                dataset_mode="unaligned",
                evaluation_metrics="none",
                display_env="izar_unaligned_default",
                tb_folder="./runs/unaligned_default/"
            ),
            opt.specify(
                name="nightVision_aligned_extreme",
                use_unaligned=False,

                batch_size=64,
                load_size=256, crop_size=256,
                netE_num_downsampling_sp=2,
                evaluation_metrics="none",
                patch_min_scale=1/16,
                patch_max_scale=1/8,
                patch_num_crops = 16,
                patch_size=32,
                netPatchD_scale_capacity=2.0,
                netD_scale_capacity=0.5,
                display_env="izar_aligned_extreme",
                tb_folder="./runs/aligned_extreme/"
            ),
            opt.specify(
                name="nightVision_unaligned_extreme_smallSize",
                use_unaligned=True,
                dataset_mode="unaligned",

                batch_size=32,
                load_size=256, crop_size=256,
                netE_num_downsampling_sp=3,
                evaluation_metrics="none",
                patch_min_scale=1/16,
                patch_max_scale=1/8,
                patch_num_crops = 16,
                patch_size=32,
                netPatchD_scale_capacity=2.5,
                netD_scale_capacity=0.8,
                display_env="izar_unaligned_extreme_smallSize",
                tb_folder="./runs/unaligned_extreme_smallSize/"
            ),
            opt.specify(
                name="nightVision_unaligned_sc",
                use_unaligned=True,
                dataset_mode="unaligned",
                num_gpus=1,
                server='izar',
                batch_size=8,
                load_size=286, crop_size=256,
                netE_num_downsampling_sp=2,
                evaluation_metrics="none",
                patch_min_scale=1/16,
                patch_max_scale=1/8,
                patch_num_crops = 16,
                patch_size=32,
                display_env="izar_unaligned_sc",
                tb_folder="./runs/unaligned_sc/"
            ),
        ]

    def train_options(self):
        common_options = self.options()
        return [opt.specify(
            continue_train=False,
        ) for opt in common_options
        ]

    def test_options(self):
        opt = self.options()[-2]
        return [
            opt.tag("simple_swapping").specify(
                num_gpus=1,
                batch_size=1,
                dataroot="./datasets/nightVisionDatasets/",  # dataroot is ignored.
                result_dir="./results/",
                preprocess="scale_shortside",
                load_size=512,
                evaluation_metrics="simple_swapping",
                # Specify the two images here.
                #input_texture_image ="/home/xugao/gitRepo/swapping-autoencoder-pytorch/testphotos/nightVision/val_night/1645596786082.jpg",
                input_texture_image ="/home/xugao/gitRepo/swapping-autoencoder-pytorch/datasets/nightVisionDatasets/day_images/3891.jpg",
                input_structure_image ="/home/xugao/gitRepo/swapping-autoencoder-pytorch/testphotos/nightVision/val_night/1645596068345.jpg",
                # alpha == 1.0 corresponds to full swapping.
                # 0 < alpha < 1 means interpolation
                texture_mix_alpha=1.0,
            ),
            # Simple interpolation images for quick testing
            opt.tag("simple_interpolation").specify(
                num_gpus=1,
                batch_size=1,
                dataroot="./datasets/nightVisionDatasets/",  # dataroot is ignored.
                result_dir="./results/",
                preprocess="resize", load_size=1024, crop_size=1024,
                evaluation_metrics="simple_swapping",
                # Specify the two images here.
                input_texture_image ="/home/xugao/gitRepo/swapping-autoencoder-pytorch/datasets/nightVisionDatasets/day_images/3891.jpg",
                input_structure_image ="/home/xugao/gitRepo/swapping-autoencoder-pytorch/testphotos/nightVision/val_night/1645596068345.jpg",
                texture_mix_alpha='0.0 0.25 0.5 0.75 1.0',
            )
        ]