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

## Evaluation

### Download pretrained model and dataset

```bash
bash ./utils/download.sh
```

Set appropriate parameters in `test_parameter.py` and run `python test_driver.py` to evaluate.

### Interactive demo

You can also create your own map by running `python viper_demo.py`, which opens a canvas for you to draw on.

- Use _Obstacle_ and _Free Space_ brushes to draw your map. Adjust the brush size with the thickness slider.
- Click _Reset_ to clear the canvas, setting it entirely to obstacles or free space.
- Click _Place Agents_ to place multiple agents in the **free space**.
- Click _Play_ to observe how ViPER agents plan their path.

Alternatively, you can save the map you created.

- Click _Start Position_ to place the starting position of agents.
- Click _Save Map_ before closing the canvas. Your map will be saved as `maps_spec/map.png`. 

## Training

Make sure you have downloaded the map dataset.
Set appropriate parameters in `parameter.py` and run `python driver.py` to train the model.


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
