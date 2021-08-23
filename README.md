# CAME
Cross-species cell-type assignment and gene module extraction of single-cell RNA-seq by graph neural network

### Installation
 Install from source code:
```shell
python setup.py install
```

### Testing the package
To test the package, run the python file `test_pipeline.py`:

```shell
python test_pipeline.py 
```

When running the testing code, make sure that the sample data (`/came/sample_data/`) exist, which contains:
- gene_matches_1v1_human2mouse.csv (optional)
- gene_matches_1v1_mouse2human.csv (optional)
- gene_matches_human2mouse.csv
- gene_matches_mouse2human.csv
- raw-Baron_mouse.h5ad
- raw-Baron_human.h5ad 

NOTE:
Currently, files in folder `/came/sample_data/` are ignored, i.e., NOT included in the project folder.

### Example code for analysis

See code in `case-unaligned.py`

TODO:
* Jupyter notebooks of tutorials in different cases, that is dataset pairs with
  * Aligned features.
  * Unaligned features.
* Documentation of API.
