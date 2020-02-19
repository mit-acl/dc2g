# DC2G: Deep Cost-to-Go Planning Algorithm (IROS '19)

*Planning Beyond the Sensing Horizon Using a Learned Context*

Michael Everett, Justin Miller, Jonathan P. How

IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2019

Paper: [https://arxiv.org/abs/1908.09171](https://arxiv.org/abs/1908.09171)

Video: [https://youtu.be/yVlnbqEFct0](https://youtu.be/yVlnbqEFct0)

![network architecture](./misc/dc2g_architecture.png)

**Note:** This repo contains the code that was approved for release:
- Pre-trained cost-to-go estimation network
- Bing Maps Driveway Dataset (~80 houses)
- Gridworld evaluation environment (built on gym-minigrid)
- Jupyter Notebook to explain code

```sh
cd <code directory>
git clone --recursive <this repo>
cd dc2g
python -m pip install -e gym-minigrid
python -m pip install -e .
```

Test a network in the gridworld:
```sh
python -m dc2g.run_episode -p dc2g
```

### If you find this code useful, please consider citing our paper:
```
@inproceedings{Everett19_IROS,
	Address = {Macau, China},
	Author = {Michael Everett and Justin Miller and Jonathan P. How},
	Booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
	Date-Modified = {2019-06-22 06:18:08 -0400},
	Month = {November},
	Title = {Planning Beyond The Sensing Horizon Using a Learned Context},
	Year = {2019}
}
```
