Installation
============

It's recommended to create a conda environment for running CAME:

.. code-block:: shell

   conda create -n env_came python=3.8
   conda activate env_came


Install required packages:

.. code-block:: shell

   pip install "scanpy[leiden]"
   pip install torch  # >=1.8
   pip install dgl

See scanpy (https://scanpy.readthedocs.io/en/stable/),
PyTorch (https://pytorch.org/) and DGL(https://www.dgl.ai/)
for detailed installation guide (especially for GPU version).

Install CAME
~~~~~~~~~~~~

To install CAME with PyPI, run:

.. code-block:: shell

   pip install came


Or fetch from GitHub and manually install:

.. code-block:: shell

   git clone https://github.com/zhanglabtools/CAME.git
   cd CAME
   python setup.py install

