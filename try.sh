set -ex

conda active yolo
cd /home/bozorgta/xugao/gitLocal/swapping-autoencoder-pytorch
python -m experiments --name nightVision --cmd train --id nightVision_cut