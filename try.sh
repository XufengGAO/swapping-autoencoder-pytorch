set -ex

cd /home/bozorgta/xugao/gitLocal/swapping-autoencoder-pytorch
# python -m experiments --name nightVision --cmd train --id nightVision_unaligned_sc
python -m experiments --name nightVision --cmd test --id day_swapping