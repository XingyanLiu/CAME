# CAME
Cross-species cell-type assignment and gene module extraction of single-cell RNA-seq by graph neural network

### Installation

It's recommended to create an environment for running CAME:

```shell
conda create -n came python=3.8
conda activate came
```

Install required packages:

```shell
pip install "scanpy[leiden]"
pip install torch>=1.8 
pip install dgl  
```

See [scanpy](https://scanpy.readthedocs.io/en/stable/), 
[PyTorch](https://pytorch.org/) and [DGL](https://www.dgl.ai/) 
for detailed installation guide (for GPU version).

Install CAME from source code:

```shell
git clone https://github.com/XingyanLiu/CAME
python setup.py install
```

### Testing the package

Before running the testing code, make sure that the sample data 
(`./came/sample_data/`, currently stored in `./came/sample_data.rar`) exist, which contains:

- gene_matches_1v1_human2mouse.csv (optional)
- gene_matches_1v1_mouse2human.csv (optional)
- gene_matches_human2mouse.csv
- gene_matches_mouse2human.csv
- raw-Baron_mouse.h5ad
- raw-Baron_human.h5ad 


To test the package, run the python file `test_pipeline.py`:

```shell
python test_pipeline.py 
```


### Example code for analysis

See code in `case-unaligned.py`

TODO:
* Jupyter notebooks of tutorials in different cases, that is dataset pairs with
  * Aligned features.
  * Unaligned features.
* Documentation of API.
