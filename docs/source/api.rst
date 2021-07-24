.. module:: came
.. automodule:: came
   :noindex:

API Reference
=============


Import CAME::

    import came


Pipeline
--------

.. module:: came.pipeline
.. :currentmodule:: came

.. autosummary::
   :toctree: generated/

   main_for_aligned
   main_for_unaligned
   preprocess_aligned
   preprocess_unaligned
   gather_came_results

Preprocessing ``came.pp.*``
------------------------

.. module:: came.utils.preprocess
.. :currentmodule:: came

Preprocessing functions
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   align_adata_vars
   normalize_default
   change_names
   group_mean_adata


DataPair and AlignedDataPair
----------------------------
.. module:: came
.. py:currentmodule:: came

.. autosummary::
   :toctree: generated/

   aligned_datapair_from_adatas
   datapair_from_adatas
   AlignedDataPair
   DataPair

Graph Neural Network Model
--------------------------
.. autosummary::
   :toctree: generated/

   CGGCNet
   CGCNet


