FAQs
====

About the input format
----------------------

**Q**: I processed my data using Seurat, and transformed them into .h5ad files.
But an error occurred when I passed them into CAME's default pipeline.

**A**: The problem is caused by "the h5ad files converted from seurat-object by
``SeuratDisk``".
CAME process the data from the raw-count matrices.
So please use scanpy to construct the AnnData object from the raw-count matrices
(e.g., read from the ``*.mtx`` and ``*.txt`` files by ``scanpy.read()``)

You can also use the following R code to export the filtered scRNA-seq data
into an ``.h5`` file, which takes less time and space.

.. code-block:: R

    # R code
    library(rhdf5)
    library(Matrix)

    save_h5mat = function(mat, fp_h5, feature_type, genome=""){
      # save sparse.mat ('dgCMatrix' format) into a h5 file
      # ======= Test code ======
      # tmp = Seurat::Read10X_h5(fp_h5)
      # all(tmp@x == mat@x)
      # all(tmp@i == mat@i)
      # all(tmp@p == mat@p)

      message(fp_h5)

      h5createFile(fp_h5)
      root = "matrix"
      h5createGroup(fp_h5, root)

      h5write(dim(mat), fp_h5, paste(root, "shape", sep='/'))
      h5write(mat@x, fp_h5, paste(root, "data", sep='/'))
      h5write(mat@i, fp_h5, paste(root, "indices", sep='/'))  # mat@i - 1 ?
      h5write(mat@p, fp_h5, paste(root, "indptr", sep='/'))
      h5write(colnames(mat), fp_h5, paste(root, "barcodes", sep='/'))


      feat_root = paste(root, "features", sep='/')
      h5createGroup(fp_h5, feat_root)

      h5write(rownames(mat), fp_h5, paste(feat_root, "id", sep='/'))
      h5write(rownames(mat), fp_h5, paste(feat_root, "name", sep='/'))

      h5write(rep(feature_type, dim(mat)[1]),
              fp_h5, paste(feat_root, "feature_type", sep='/'))

      h5write(rep("", dim(mat)[1]),
              fp_h5, paste(feat_root, "derivation", sep='/'))
      h5write(rep(genome, dim(mat)[1]),  # "mm10"
              fp_h5, paste(feat_root, "genome", sep='/'))
      h5write(c("genome", "derivation"),
              fp_h5, paste(feat_root, "_all_tag_keys", sep='/'))

      h5closeAll()
      message("Done!")
    }

    # save_h5mat_peak = function(mat, fp_h5, genome=""){
    #   save_h5mat(mat, fp_h5, feature_type = "Peaks", genome = genome)
    # }

    save_h5mat_gex = function(mat, fp_h5, genome=""){
      save_h5mat(mat, fp_h5, feature_type = "Gene Expression", genome = genome)
    }
    # save the raw-counts in a Seurat-object "seurat_obj"
    mat = seurat_obj[["RNA"]]@counts
    save_h5mat_gex(mat, "matrix.raw.h5", genome="")

    # save the meta-data into a csv file:
    meta_data = seurat_obj@meta.data
    write.csv(meta_data, "metadata.csv")


And read the h5 file using Scanpy's build-in function:

.. code-block:: Python

    # python-code
    import pandas as pd
    import scanpy as sc

    fp_mat = 'matrix.raw.h5'
    fp_meta = 'metadata.csv'
    adata_raw = sc.read_10x_h5(fp_mat)
    metadata = pd.read_csv(fp_meta, index_col=0)
    # add meta-data
    for c in metadata.columns:
        adata_raw.obs[c] = metadata[c]






