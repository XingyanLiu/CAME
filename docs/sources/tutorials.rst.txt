Tutorials
=========

The example data
----------------

The tutorials are based on the example data attached to the CAME package.
It is initially saved in compressed form (`CAME/came/sample_data.zip`),
and will be automatically decompressed to the default directory
(`CAME/came/sample_data/`) when necessary, which contains the following files:

- gene_matches_1v1_human2mouse.csv (optional)
- gene_matches_1v1_mouse2human.csv (optional)
- gene_matches_human2mouse.csv
- gene_matches_mouse2human.csv
- raw-Baron_mouse.h5ad
- raw-Baron_human.h5ad

You can access these data by :doc:`generated/came.load_example_data`.

If you tend to apply CAME to analyze your own datasets, you need to
prepare at least the last two files for the same species (e.g., cross-dataset
integration);

For cross-species analysis, you need to provide another `.csv`
file where the first column contains the genes in the reference species and the
second contains the corresponding query homologous genes.

**NOTE**

   The file `raw-Baron_human.h5ad` is a subsample from the original data
   for code testing. The resulting annotation accuracy may not be as good as
   using the full dataset as the reference.


Getting started
---------------

.. toctree::
   :maxdepth: 1

   tut_notebooks/getting_started_pipeline_un
   tut_notebooks/getting_started_pipeline_aligned
   tut_notebooks/load_results
