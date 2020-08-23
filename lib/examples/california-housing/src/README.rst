Differences against CAIR example
================================
CAIR example from which this code originated: https://github.com/cair/pyTsetlinMachine/tree/fecaa3845615548f767a18afd2e1e5fcde1e3793

Tsetlini example:

* rounds transformed target values to integers instead of truncating,
* test data is not used for target scaling. Instead, scaling parameters are
  determined using train targets only and then test targets are transformed,

Data binarization schemes
=========================

CAIR binarization
-----------------
With CAIR binarization scheme applied to input data gives an output like the one below:

.. code-block::

  ENSEMBLE 1
  Average RMSD on test data: 0.612 +/- 0.000 (5.71s)
  ENSEMBLE 2
  Average RMSD on test data: 0.615 +/- 0.003 (5.79s)
  ENSEMBLE 3
  Average RMSD on test data: 0.613 +/- 0.003 (6.14s)
  ENSEMBLE 4
  Average RMSD on test data: 0.613 +/- 0.002 (6.91s)
  ENSEMBLE 5
  Average RMSD on test data: 0.612 +/- 0.002 (6.86s)
  ENSEMBLE 6
  Average RMSD on test data: 0.611 +/- 0.003 (6.95s)
  ENSEMBLE 7
  Average RMSD on test data: 0.610 +/- 0.003 (6.89s)
  ENSEMBLE 8
  Average RMSD on test data: 0.609 +/- 0.003 (6.95s)
  ENSEMBLE 9
  Average RMSD on test data: 0.609 +/- 0.003 (6.87s)
  ENSEMBLE 10
  Average RMSD on test data: 0.609 +/- 0.003 (6.89s)
  ENSEMBLE 11
  Average RMSD on test data: 0.609 +/- 0.003 (6.89s)
  ENSEMBLE 12
  Average RMSD on test data: 0.610 +/- 0.002 (6.88s)
  ENSEMBLE 13
  Average RMSD on test data: 0.609 +/- 0.003 (6.99s)
  ENSEMBLE 14
  Average RMSD on test data: 0.609 +/- 0.002 (6.91s)
  ENSEMBLE 15
  Average RMSD on test data: 0.609 +/- 0.002 (7.23s)
  ENSEMBLE 16
  Average RMSD on test data: 0.609 +/- 0.002 (7.01s)
  ENSEMBLE 17
  Average RMSD on test data: 0.610 +/- 0.003 (6.87s)
  ENSEMBLE 18
  Average RMSD on test data: 0.610 +/- 0.002 (7.03s)
  ENSEMBLE 19
  Average RMSD on test data: 0.610 +/- 0.002 (6.98s)
  ENSEMBLE 20
  Average RMSD on test data: 0.610 +/- 0.002 (6.98s)
  ENSEMBLE 21
  Average RMSD on test data: 0.610 +/- 0.002 (6.97s)
  ENSEMBLE 22
  Average RMSD on test data: 0.610 +/- 0.002 (6.99s)
  ENSEMBLE 23
  Average RMSD on test data: 0.611 +/- 0.003 (7.00s)
  ENSEMBLE 24
  Average RMSD on test data: 0.612 +/- 0.003 (6.95s)
  ENSEMBLE 25
  Average RMSD on test data: 0.611 +/- 0.003 (7.10s)

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
  Average RMSD on test data: 0.593 +/- 0.000 (5.58s)
  ENSEMBLE 2
  Average RMSD on test data: 0.591 +/- 0.003 (5.63s)
  ENSEMBLE 3
  Average RMSD on test data: 0.589 +/- 0.005 (5.65s)
  ENSEMBLE 4
  Average RMSD on test data: 0.584 +/- 0.008 (6.17s)
  ENSEMBLE 5
  Average RMSD on test data: 0.581 +/- 0.009 (6.87s)
  ENSEMBLE 6
  Average RMSD on test data: 0.582 +/- 0.008 (6.82s)
  ENSEMBLE 7
  Average RMSD on test data: 0.584 +/- 0.007 (6.76s)
  ENSEMBLE 8
  Average RMSD on test data: 0.585 +/- 0.007 (6.80s)
  ENSEMBLE 9
  Average RMSD on test data: 0.585 +/- 0.006 (6.83s)
  ENSEMBLE 10
  Average RMSD on test data: 0.584 +/- 0.006 (6.99s)
  ENSEMBLE 11
  Average RMSD on test data: 0.584 +/- 0.005 (6.88s)
  ENSEMBLE 12
  Average RMSD on test data: 0.585 +/- 0.005 (6.79s)
  ENSEMBLE 13
  Average RMSD on test data: 0.585 +/- 0.005 (6.78s)
  ENSEMBLE 14
  Average RMSD on test data: 0.586 +/- 0.005 (6.76s)
  ENSEMBLE 15
  Average RMSD on test data: 0.587 +/- 0.005 (6.75s)
  ENSEMBLE 16
  Average RMSD on test data: 0.587 +/- 0.004 (6.85s)
  ENSEMBLE 17
  Average RMSD on test data: 0.587 +/- 0.004 (6.84s)
  ENSEMBLE 18
  Average RMSD on test data: 0.588 +/- 0.004 (6.92s)
  ENSEMBLE 19
  Average RMSD on test data: 0.588 +/- 0.004 (7.00s)
  ENSEMBLE 20
  Average RMSD on test data: 0.589 +/- 0.004 (6.86s)
  ENSEMBLE 21
  Average RMSD on test data: 0.589 +/- 0.004 (6.92s)
  ENSEMBLE 22
  Average RMSD on test data: 0.588 +/- 0.004 (6.92s)
  ENSEMBLE 23
  Average RMSD on test data: 0.588 +/- 0.004 (6.90s)
  ENSEMBLE 24
  Average RMSD on test data: 0.588 +/- 0.004 (6.85s)
  ENSEMBLE 25
  Average RMSD on test data: 0.588 +/- 0.004 (6.87s)

There seems to be a noticeable improvement over results achieved for input data binarized using CAIR's scheme.
