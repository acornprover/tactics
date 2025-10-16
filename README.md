# tactics

Code for training and evaluating a "tactics" model, which suggests proof steps generatively.

To generate training data from acornlib:

```
acorn --lib <path/to/acornlib> --generate-training data/proofs
```

This dumps out all proof certificates in a structured format.
