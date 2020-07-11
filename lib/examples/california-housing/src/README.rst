Differences against CAIR example
================================
CAIR example from which this code originated: https://github.com/cair/pyTsetlinMachine/tree/fecaa3845615548f767a18afd2e1e5fcde1e3793

Tsetlini example:

* uses Tsetlin Machine Regressor model with non-weighted clauses,
* uses smaller Threshold of 500 against CAIR's 5000,
* rounds transformed target values to integers,

Data binarization schemes
=========================

CAIR binarization
-----------------
With CAIR binarization scheme applied to input data gives an output like the one below:

.. code-block::

  ENSEMBLE 1
  Average RMSD on test data: 0.611 +/- 0.000 (12.10s)
  ENSEMBLE 2
  Average RMSD on test data: 0.616 +/- 0.006 (14.29s)
  ENSEMBLE 3
  Average RMSD on test data: 0.618 +/- 0.005 (14.43s)
  ENSEMBLE 4
  Average RMSD on test data: 0.616 +/- 0.006 (13.98s)
  ENSEMBLE 5
  Average RMSD on test data: 0.617 +/- 0.005 (13.98s)
  ENSEMBLE 6
  Average RMSD on test data: 0.620 +/- 0.007 (13.92s)
  ENSEMBLE 7
  Average RMSD on test data: 0.619 +/- 0.006 (13.93s)
  ENSEMBLE 8
  Average RMSD on test data: 0.620 +/- 0.005 (14.17s)
  ENSEMBLE 9
  Average RMSD on test data: 0.620 +/- 0.005 (14.25s)
  ENSEMBLE 10
  Average RMSD on test data: 0.620 +/- 0.004 (14.25s)
  ENSEMBLE 11
  Average RMSD on test data: 0.620 +/- 0.004 (14.13s)
  ENSEMBLE 12
  Average RMSD on test data: 0.619 +/- 0.005 (14.28s)
  ENSEMBLE 13
  Average RMSD on test data: 0.619 +/- 0.005 (14.20s)
  ENSEMBLE 14
  Average RMSD on test data: 0.619 +/- 0.004 (14.25s)
  ENSEMBLE 15
  Average RMSD on test data: 0.619 +/- 0.004 (14.25s)
  ENSEMBLE 16
  Average RMSD on test data: 0.619 +/- 0.004 (14.21s)
  ENSEMBLE 17
  Average RMSD on test data: 0.619 +/- 0.004 (14.21s)
  ENSEMBLE 18
  Average RMSD on test data: 0.619 +/- 0.003 (14.30s)
  ENSEMBLE 19
  Average RMSD on test data: 0.619 +/- 0.003 (14.15s)
  ENSEMBLE 20
  Average RMSD on test data: 0.619 +/- 0.004 (14.23s)
  ENSEMBLE 21
  Average RMSD on test data: 0.618 +/- 0.004 (14.30s)
  ENSEMBLE 22
  Average RMSD on test data: 0.618 +/- 0.004 (14.23s)
  ENSEMBLE 23
  Average RMSD on test data: 0.618 +/- 0.003 (14.25s)
  ENSEMBLE 24
  Average RMSD on test data: 0.618 +/- 0.003 (14.38s)
  ENSEMBLE 25
  Average RMSD on test data: 0.619 +/- 0.003 (14.15s)

To recreate the dataset which uses CAIR's scheme run `produce_dataset.py`.

Tsetlini binarization
---------------------
We have tested another binarization scheme which uses ``KBinsDiscretizer`` from ``scikit-learn``.
To encode each feature into 10 bits we use ``quantile`` strategy to transform continuous features
into ordinals representing 11 buckets. These ordinal values are then encoded into sequences
of 0s and 1s using the same scheme as CAIR example.
Ordinal value of 0 will yield a sequence of ten 0s, and ordinal value of 10 will yield a sequence of ten 1s.

To recreate the dataset transformed using this approach run ``produce_dataset_alt.py``.

Output from one of the runs is as follows:

.. code-block::

  ENSEMBLE 1
  Average RMSD on test data: 0.589 +/- 0.000 (11.57s)
  ENSEMBLE 2
  Average RMSD on test data: 0.592 +/- 0.003 (13.15s)
  ENSEMBLE 3
  Average RMSD on test data: 0.589 +/- 0.005 (13.97s)
  ENSEMBLE 4
  Average RMSD on test data: 0.588 +/- 0.004 (13.88s)
  ENSEMBLE 5
  Average RMSD on test data: 0.588 +/- 0.003 (14.07s)
  ENSEMBLE 6
  Average RMSD on test data: 0.587 +/- 0.003 (13.97s)
  ENSEMBLE 7
  Average RMSD on test data: 0.586 +/- 0.003 (14.02s)
  ENSEMBLE 8
  Average RMSD on test data: 0.585 +/- 0.003 (14.28s)
  ENSEMBLE 9
  Average RMSD on test data: 0.585 +/- 0.003 (14.30s)
  ENSEMBLE 10
  Average RMSD on test data: 0.584 +/- 0.004 (14.33s)
  ENSEMBLE 11
  Average RMSD on test data: 0.584 +/- 0.004 (14.19s)
  ENSEMBLE 12
  Average RMSD on test data: 0.585 +/- 0.004 (14.20s)
  ENSEMBLE 13
  Average RMSD on test data: 0.586 +/- 0.004 (14.53s)
  ENSEMBLE 14
  Average RMSD on test data: 0.586 +/- 0.004 (14.27s)
  ENSEMBLE 15
  Average RMSD on test data: 0.586 +/- 0.004 (14.41s)
  ENSEMBLE 16
  Average RMSD on test data: 0.586 +/- 0.003 (14.30s)
  ENSEMBLE 17
  Average RMSD on test data: 0.586 +/- 0.003 (14.45s)
  ENSEMBLE 18
  Average RMSD on test data: 0.586 +/- 0.003 (14.21s)
  ENSEMBLE 19
  Average RMSD on test data: 0.585 +/- 0.003 (14.62s)
  ENSEMBLE 20
  Average RMSD on test data: 0.586 +/- 0.003 (14.33s)
  ENSEMBLE 21
  Average RMSD on test data: 0.586 +/- 0.003 (14.62s)
  ENSEMBLE 22
  Average RMSD on test data: 0.586 +/- 0.003 (14.24s)
  ENSEMBLE 23
  Average RMSD on test data: 0.586 +/- 0.003 (14.47s)
  ENSEMBLE 24
  Average RMSD on test data: 0.586 +/- 0.003 (14.36s)
  ENSEMBLE 25
  Average RMSD on test data: 0.587 +/- 0.003 (14.32s)

There seems to be a noticeable improvement over results achieved for input data binarized using CAIR's scheme.
