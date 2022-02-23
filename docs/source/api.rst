.. module:: came
.. automodule:: came
   :noindex:

API References
==============


Import CAME:

.. code:: ipython3

    import came

Example data
------------

.. module:: came

.. autosummary::
   :toctree: generated/

   load_example_data


Pipeline ``came.pipeline.*``
----------------------------

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
---------------------------

.. module:: came.utils.preprocess
.. :currentmodule:: came

.. autosummary::
   :toctree: generated/

   align_adata_vars
   normalize_default
   quick_preprocess
   quick_pre_vis
   group_mean
   group_mean_adata
   wrapper_scale
   make_bipartite_adj
   take_1v1_matches
   subset_matches
   get_homologies
   take_adata_groups
   remove_adata_groups
   merge_adata_groups
   split_adata


DataPair and AlignedDataPair
----------------------------
.. module:: came
.. py:currentmodule:: came

.. autosummary::
   :toctree: generated/

   make_features
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


I/O Functions
-------------
.. autosummary::
   :toctree: generated/

   load_dpair_and_model
   load_hidden_states
   save_hidden_states
   save_pickle
   load_pickle

Analysis
--------
.. autosummary::
   :toctree: generated/

   weight_linked_vars
   make_abstracted_graph


Plotting Functions ``came.pl.*``
--------------------------------

.. module:: came.utils.plot
.. :currentmodule:: came

.. autosummary::
   :toctree: generated/

   plot_stacked_bar
   heatmap_probas
   wrapper_heatmap_scores
   plot_confus_mat
   plot_contingency_mat
   embed_with_values
   adata_embed_with_values
   umap_with_annotates
   plot_multipartite_graph
