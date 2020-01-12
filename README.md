# cdm-icml
Code that Accompanies "Discovering Context Effects from Raw Choice Data"

The following code uses the open source machine learning framework pytorch 
(https://pytorch.org/) to implement the CDM as described in "Discovering Context
Effects from Raw Choice Data" by Arjun Seshadri, Alexander Peysakhovich and 
Johan Ugander in ICML 2019 (http://proceedings.mlr.press/v97/seshadri19a.html).

Below is code for a sample run to train the CDM on SFShop, a dataset 
explored in the paper (as run from the code directory): 
```
import cdm_pytorch as cp
val_loss, tr_loss, gv, train_ds, val_ds, model, opt = cp.default_run(dataset='SFshop',
                                                                     batch_size=None,
                                                                     epochs=500, 
                                                                     embedding_dim=3, 
                                                                     lr=5e-2, 
                                                                     seed=0,
                                                                     wd=0)
print(f'Val Loss: {val_loss.item()}, Tr Loss: {tr_loss.item()}, final_grad: {gv.item()}')
```

The above function is a wrapper function to use the CDM. The inputs used are 
described below, but see in-line documentation for more details. 

*dataset* - name of dataset to run CDM on. Default options are 
'SFwork', 'SFshop', or 'syn_nature_triplet' to run the datasets in the paper.

*batch_size* - hyperparameter for training, number of points to be used for 
each gradient step (None means full batch)

*epochs* - number of epochs to perform optimization. If batch_size is set to
None, this is just the number of gradient steps taken

*embedding_dim* - dimension of CDM (r in the paper)

*lr* - learning rate for Adam optimizer (the default optimizer)

*seed* - random seed that sets dataset splits + initialization for 
reproducibility (None means no seed)

*wd* - weight decay parameter for the model (similar to l2 regularization,
but faster - see pytorch docs for more info)
