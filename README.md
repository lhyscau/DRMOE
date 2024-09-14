# DRMOE: Towards Better Mixture of Experts via Dual Routing Strategy

------

This is  the implementation of the paper "DRMOE: Towards Better Mixture of Experts via Dual Routing Strategy".



## Running

------

You can implement out model according to the following steps:

1. Install the necessary packages. Run the command:

   ```shell
   pip install -r requirements.txt
   ```

2. To train the DRMOE and generate the answer to test, please run the command:

   ```bash
   bash ./experiments/train.sh
   ```

3. Finally, you can get the answer at results/eval_metric.csv.



## Requirements

------

To ease the configuration of the environment, I list versions of my hardware and software equipments:

- Hardware:
  - GPU: RTX 3090
  - Cuda: 11.3.1
  - Driver version: 535.146.02
- Software:
  - Python: 3.9.5
  - Pytorch: 1.12.0+cu113
  - transformers: 4.28.1
  - deepspeed: 0.9.4

You can also visit this [link](https://drive.google.com/file/d/1pqRZUa8HB02qxcwGU1TzZqUipAYzpFLm/view?usp=sharing) to get the tar.gz file containing the complete virtual environment.