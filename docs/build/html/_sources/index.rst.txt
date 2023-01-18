.. CAME documentation master file, created by
   sphinx-quickstart on Sat Jul 24 10:14:53 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CAME
====

CAME is a tool for Cell-type Assignment and Module Extraction,
based on a heterogeneous graph neural network.

To get started with CAME's pipeline, browse the tutorials below:
  * :doc:`tut_notebooks/getting_started_pipeline_un`
  * :doc:`tut_notebooks/getting_started_pipeline_aligned`


.. image:: _static/Fig1ABC.png
   :width: 600px

CAME outputs the quantitative cell-type assignment for each query cell, that is,
the probabilities of cell types that exist in the reference species, which
enables the identification of the unresolved cell states in the query data.

Besides, CAME gives the aligned cell and gene embeddings across species, which
facilitates low-dimensional visualization and joint gene module extraction.


.. image:: _static/Fig1D.png
   :width: 600px

.. toctree::
   :caption: Contents
   :maxdepth: 1

   installation
   tutorials
   api
   faqs
   citation


Citation
--------

If CAME is useful for your research, consider citing our paper:

   Liu X, Shen Q, Zhang S. Cross-species cell-type assignment of single-cell RNA-seq by a heterogeneous graph neural network[J]. Genome Research, 2022: gr. 276868.122.




Contribute
----------

- Issue Tracker: https://github.com/XingyanLiu/CAME/issues
- Source Code: https://github.com/XingyanLiu/CAME

Support
-------

If you are having issues, please let us know.
We have a mailing list located at:

* xingyan@amss.ac.cn
* 544568643@qq.com


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
