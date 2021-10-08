# CAME

CAME is a tool for Cell-type Assignment and Module Extraction, based on a heterogeneous graph neural network.


### Installation

It's recommended to create a conda environment for running CAME:

```shell
conda create -n came python=3.8
conda activate came
```

Install required packages:

```shell
pip install "scanpy[leiden]"
pip install torch  # >=1.8 
pip install dgl  
```

See [scanpy](https://scanpy.readthedocs.io/en/stable/), 
[PyTorch](https://pytorch.org/) and [DGL](https://www.dgl.ai/) 
for detailed installation guide (especially for GPU version).

Install CAME from source code:

```shell
git clone https://github.com/XingyanLiu/CAME.git
cd CAME
python setup.py install
```

### Testing the package

Before running the testing code, make sure that the sample data 
(`./came/sample_data/`, currently stored in `./came/sample_data.zip`) exist, which contains:

- gene_matches_1v1_human2mouse.csv (optional)
- gene_matches_1v1_mouse2human.csv (optional)
- gene_matches_human2mouse.csv
- gene_matches_mouse2human.csv
- raw-Baron_mouse.h5ad
- raw-Baron_human.h5ad 

> NOTE: the file `raw-Baron_mouse.h5ad` is subsample from the original data 
> for code testing. The resulting annotation accuracy may not be as good as 
> using full dataset as the reference.

To test the package, run the python file `test_pipeline.py`:

```shell
python test_pipeline.py 
```


### Example code for analysis

See code in files in `./notebooks`


### Citation

If CAME is useful for your research, consider citing our preprint:

> Cross-species cell-type assignment of single-cell RNA-seq by a heterogeneous graph neural network.
   Xingyan Liu, Qunlun Shen, Shihua Zhang.
   bioRxiv 2021.09.25.461790; doi: https://doi.org/10.1101/2021.09.25.461790

