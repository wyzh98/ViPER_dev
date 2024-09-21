# ViPER: Visibility-based Pursuit-Evasion via Reinforcement Learning

This repository hosts the code for [ViPER](https://openreview.net/pdf?id=EPujQZWemk), accepted for [CoRL 2024](https://www.corl.org/).

## Setup instructions

Use conda and pip to setup environments:

```bash
conda create -n viper python=3.11 scikit-image imageio tensorboard matplotlib pytorch pytor
ch-cuda=11.8 -c pytorch -c nvidia -y
conda activate viper
pip install ray opencv-python wandb 
```

## Training

Download our map dataset (4000 maps) for training. It will be extracted to `ViPER/maps_train/`:

```bash
wget -O maps_train.zip "https://www.dropbox.com/scl/fi/b0hosxxfnzv7j0tcoiwq1/maps_train.zip?rlkey=au844wb4dvvo2bbq32x9p2p8x&dl=1"
unzip maps_train.zip && rm maps_train.zip
```
Set appropriate parameters in `parameter.py` and run `python driver.py` to train the model.

## Evaluation

Download pretrained model checkpoint.
```bash
wget -O ""
```

Download our map test dataset (100 maps) for testing. It will be extracted to `ViPER/maps_test/`.
```bash
wget -O maps_test.zip "https://www.dropbox.com/scl/fi/8ktuwxxzw4iz4x4hht3o7/maps_test.zip?rlkey=ahcoqn6932z1sepq01kjfqtya&dl=1"
unzip maps_test.zip && rm maps_test.zip
```
### Draw Your Map
You can also create your own map by running `python map_creator.py`, which will open a canvas.
- Use _Obstacle_ and _Free Space_ brushes to draw your map. Adjust the brush size with the thickness slider.
- Click _Start_ to place the starting position of agents.
- Click _Reset_ to clear the canvas, setting it entirely to obstacles or free space.
- Click _Save Map_ before closing the canvas; your map will be saved as `maps_spec/map.png`.

Test your map by running `python test_worker.py`. You can visualize the solution in `results/gifs/`.

## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@inproceedings{wang2024viper,
  title={ViPER: Visibility-based Pursuit-Evasion via Reinforcement Learning},
  author={Wang, Yizhuo and Cao, Yuhong and Chiun, Jimmy and Koley, Subhadeep and Pham, Mandy and Sartoretti, Guillaume},
  booktitle = {8th Annual Conference on Robot Learning},
  year = {2024}
}
```
Authors:
[Yizhuo Wang](https://www.yizhuo-wang.com/),
[Yuhong Cao](https://www.yuhongcao.online/),
[Jimmy Chiun](https://www.linkedin.com/in/jimmychiun/),
[Subhadeep Koley](https://www.linkedin.com/in/subhadeep-koley-70251b1bb/),
[Mandy Pham](https://www.linkedin.com/in/phamandy24/),
[Guillaume Sartoretti](https://cde.nus.edu.sg/me/staff/sartoretti-guillaume-a/)
