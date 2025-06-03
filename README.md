# TransformLib
A simple and easy to understand transformer implementation using PyTorch

## Library Features 
The library currently contains the following classes 

1) Input Embeddings
2) Positional Encodings
3) Layer Normalization
4) Feed-Forward Blocks
5) Multi-Headed Attention Blocks
6) Residual Connections
7) Encoder Class
8) Decoder Class

## Sample Decoder outputs
Decoder.ipynb contains a small decoder model with the following architecure
1) embedding dimebsion (d_model) =32
2) Decoder Layers (n_layers) = 8
3) Head Size (h) = 8
4) Feed-Forward size (d_ff) = 128

We then proceed to train this on a small corpus with a vocabulary size of 308

### Outputs
```
i want a pan card internet related regarding security due the to mall settled machine with account activation any and required done valid password transfer
```

```
i applied for booking transaction my reset refunded name problem with application days to mall settled machine my able some making delays updated
```

```
i applied for account new the to shopping but joint previously last although loan new emi machine all old new is credit security
```

While the sentences have no semantic meaning due to the very small size of both the model and the dataset, we do notice that similar words such as "transaction" and "refund" are often grouped together


