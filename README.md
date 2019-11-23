# REPROLANG 2020 @ LREC - Task A.1: VecMap, Cross-Lingual Word Embedding Mappings

This is a fork from the original open source [implementation](https://github.com/artetxem/vecmap) of the framework to learn cross-lingual word embedding mappings.

This repository is strictly limited to the reproduction of the original paper of [Artetxe et al. 2018 (ACL)](https://aclweb.org/anthology/P18-1073).


## Dataset
- The dataset is available here: https://vecmap-submission.s3.amazonaws.com/dataset.tar.gz
- The checksum of the dataset is: `69409adb48f668ce8872d924caec4519`
- Untar the dataset using the following command: `tar -xvf dataset.tar.gz`

## Usage
The reproduction of the Table 1 and 2 of our paper can easily be done using the docker image that we provide.

First, create the output directories on the host:

```
mkdir -p output/tables_and_plots
```

It is mandatory to have a GPU on the host with [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker).
Otherwise the docker image will crash.
One can generate the tables with the following command, but its takes about a **day** to generate all the results;

```
docker run --gpus all -ti --rm -v ${PWD}/data:/input -v ${PWD}/output/tables_and_plots:/output/tables_and_plots registry.gitlab.com/nicolasgarneau/vecmap
```

It is also possible to launch the script in test mode, which is a subset of only `1999` words, using the following command:
```
docker run --gpus all -ti --rm -v ${PWD}/data:/input -v ${PWD}/output/tables_and_plots:/output/tables_and_plots registry.gitlab.com/nicolasgarneau/vecmap /run_all.sh /run.sh "--test=True"
```

## Reproducing all the results

In our analysis, we presented more results in order to assess the robustness of the algorithm.

To recreate these results, one should parallelize the computation otherwise it will take **many** days to gather the results.

Using the same docker image that we provide, it is possible to create *all* the tables and plots with the following command (assuming the directory ...);

```
docker run --gpus all -ti --rm -v ${PWD}/data:/input -v ${PWD}/output/tables_and_plots:/output/tables_and_plots registry.gitlab.com/nicolasgarneau/vecmap /run_all.sh 
```

It is also possible to launch the script in test mode, as describe previously, using the following command:

```
docker run --gpus all -ti --rm -v ${PWD}/data:/input -v ${PWD}/output/tables_and_plots:/output/tables_and_plots registry.gitlab.com/nicolasgarneau/vecmap /run_all.sh "--test=True"
```


## Publications

If you use this software for academic research, please cite the relevant paper(s) as follows for the original model:
```
@inproceedings{artetxe2018acl,
  author    = {Artetxe, Mikel  and  Labaka, Gorka  and  Agirre, Eneko},
  title     = {A robust self-learning method for fully unsupervised cross-lingual mappings of word embeddings},
  booktitle = {Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year      = {2018},
  pages     = {789--798}
}
```

License
-------

Copyright (C) 2016-2018, Mikel Artetxe

Licensed under the terms of the GNU General Public License, either version 3 or (at your option) any later version. A full copy of the license can be found in LICENSE.txt.
