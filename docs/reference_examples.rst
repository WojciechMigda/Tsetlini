Reference examples
==================

Noisy XOR
---------

Appeared in the first Tsetlin Machine paper [TM2018]_, Section 5.4, pp 28.

   The dataset consists of :math:`10~000` examples with twelve binary inputs,
   :math:`X = [x_1, x_2, \ldots, x_{12}]`, and a binary output, :math:`y`.
   Ten of the inputs are completely random. The two remaining inputs, however,
   are related to the output :math:`y` through an XOR-relation,
   :math:`y = \mathrm{XOR}(x_{k_1}, x_{k_2})`. Finally, :math:`40\%` of the outputs
   are inverted. (...) We partition the dataset into training and test data,
   using :math:`50\%` of the data for training.
   
   The Tsetlin Machine used here contains :math:`20` clauses, and uses
   an :math:`s`-value of :math:`3.9` and a summation target
   :math:`T` of :math:`15`. Furthermore, the individual Tsetlin Automata
   each has :math:`100` states. The Tsetlin Machine is run for
   :math:`200` epochs, and it is the accuracy after the final epoch, that we report.

Noisy XOR example reproduced here uses the same parameters. The only distinction is
in specification of the number of clauses. ``Tsetlini``'s configuration that corresponds
to the referenced values uses ``number_of_clauses_per_label`` equal ``20``. In addition to that,
[TM2018]_ does not explicitly state configuration of ``boost_true_positive_feedback`` parameter,
but the source code referenced in it uses value of ``0`` [#xor_code1]_ [#xor_code2]_.

``Tsetlini``'s Noisy XOR example source code can be found in the
`lib/examples/noisy-xor <https://github.com/WojciechMigda/Tsetlini/tree/main/lib/examples/noisy-xor>`_ folder.

When run it will produce output below::

   $ ./noisy-xor
   Accuracy on test data (no noise): 1
   Accuracy on training data (40% noise): 0.603
   
   Prediction: x1 = 1, x2 = 0, ... -> y = 1
   Prediction: x1 = 0, x2 = 1, ... -> y = 1
   Prediction: x1 = 0, x2 = 0, ... -> y = 0
   Prediction: x1 = 1, x2 = 1, ... -> y = 0

Output produced by code from Granmo's paper [TM2018code]_::

   $ python NoisyXORDemo.py 
   Accuracy on test data (no noise): 1.0
   Accuracy on training data (40% noise): 0.603
   
   Prediction: x1 = 1, x2 = 0, ... -> y =  1
   Prediction: x1 = 0, x2 = 1, ... -> y =  1
   Prediction: x1 = 0, x2 = 0, ... -> y =  0
   Prediction: x1 = 1, x2 = 1, ... -> y =  0

The Binary Iris Dataset
-----------------------

Appeared in the first Tsetlin Machine paper [TM2018]_, section 5.2, pp 25.

   We first evaluate the Tsetlin Machine on the classical Iris dataset.
   This dataset consists of 150 examples with four inputs (Sepal Length, Sepal Width,
   Petal Length and Petal Width), and three possible outputs (Setosa, Versicolour,
   and Virginica).
   
   We increase the challenge by transforming the four input values into one consecutive
   sequence of :math:`16` bits, four bits per float. It is thus necessary to also learn
   how to segment the :math:`16` bits into four partitions, and extract the numeric
   information. We refer to the new dataset as the The Binary Iris Dataset.
   
   We partition this dataset into a training set and a test set, with 80 percent
   of the data being used for training. We here randomly produce :math:`1000`
   training and test data partitions. For each ensemble, we also randomly
   reinitialize the competing algorithms, to gain information on stability
   and robustness.
   
   (...)
   
   The Tsetlin Machine (In this experiment, we use a Multi-Class Tsetlin Machine,
   described in Section 6.1. We also apply Boosting of True Positive Feedback
   to Include Literal actions as described in Section 3.3.3.)
   used here employs :math:`300` clauses, and uses an :math:`s`-value
   of :math:`3.0` and a summation target :math:`T` of :math:`10`. Furthermore,
   the individual Tsetlin Automata each has :math:`100` states.
   This Tsetlin Machine is run for :math:`500` epochs, and it is the accuracy
   after the final epoch that is reported.

Binary Iris Dataset example reproduced here uses the same parameters.
The only distinction is in specification of the number of clauses.
``Tsetlini``'s configuration that corresponds to the referenced values uses
``number_of_clauses_per_label`` equal ``200``.

``Tsetlini``'s Binary Iris Dataset example source code can be found in the
`lib/examples/binary-iris <https://github.com/WojciechMigda/Tsetlini/tree/main/lib/examples/binary-iris>`_ folder.

When run it will produce output below::

   $ ./binary-iris
   ENSEMBLE 1
   Average accuracy on test data: 93.3 +/- 0.0
   Average accuracy on train data: 97.5 +/- 0.0
   ENSEMBLE 2
   Average accuracy on test data: 93.3 +/- 0.0
   Average accuracy on train data: 97.1 +/- 0.6
   ENSEMBLE 3
   Average accuracy on test data: 90.0 +/- 5.3
   Average accuracy on train data: 97.5 +/- 0.8
   ENSEMBLE 4
   Average accuracy on test data: 91.7 +/- 4.9
   Average accuracy on train data: 97.3 +/- 0.7
   ENSEMBLE 5
   Average accuracy on test data: 92.0 +/- 4.0
   Average accuracy on train data: 97.2 +/- 0.6
   ENSEMBLE 6
   Average accuracy on test data: 92.2 +/- 3.3
   Average accuracy on train data: 97.2 +/- 0.5
   ENSEMBLE 7
   Average accuracy on test data: 92.4 +/- 2.9
   Average accuracy on train data: 97.0 +/- 0.6
   ENSEMBLE 8
   Average accuracy on test data: 93.3 +/- 3.1
   Average accuracy on train data: 96.9 +/- 0.6
   ENSEMBLE 9
   Average accuracy on test data: 93.7 +/- 2.8
   Average accuracy on train data: 96.9 +/- 0.5
   ENSEMBLE 10
   Average accuracy on test data: 94.0 +/- 2.6
   Average accuracy on train data: 96.8 +/- 0.5
   (...)
   ENSEMBLE 991
   Average accuracy on test data: 95.1 +/- 0.3
   Average accuracy on train data: 96.5 +/- 0.0
   ENSEMBLE 992
   Average accuracy on test data: 95.1 +/- 0.3
   Average accuracy on train data: 96.5 +/- 0.0
   ENSEMBLE 993
   Average accuracy on test data: 95.1 +/- 0.3
   Average accuracy on train data: 96.5 +/- 0.0
   ENSEMBLE 994
   Average accuracy on test data: 95.1 +/- 0.3
   Average accuracy on train data: 96.5 +/- 0.0
   ENSEMBLE 995
   Average accuracy on test data: 95.1 +/- 0.3
   Average accuracy on train data: 96.5 +/- 0.0
   ENSEMBLE 996
   Average accuracy on test data: 95.1 +/- 0.3
   Average accuracy on train data: 96.5 +/- 0.0
   ENSEMBLE 997
   Average accuracy on test data: 95.2 +/- 0.3
   Average accuracy on train data: 96.5 +/- 0.0
   ENSEMBLE 998
   Average accuracy on test data: 95.2 +/- 0.3
   Average accuracy on train data: 96.5 +/- 0.0
   ENSEMBLE 999
   Average accuracy on test data: 95.2 +/- 0.3
   Average accuracy on train data: 96.5 +/- 0.0
   ENSEMBLE 1000
   Average accuracy on test data: 95.2 +/- 0.3
   Average accuracy on train data: 96.5 +/- 0.0

`Full execution log <https://github.com/WojciechMigda/Tsetlini/blob/main/lib/examples/binary-iris/src/output.txt>`_.

Run [TM2018code]_ reference example in `Colab Notebook <https://colab.research.google.com/github/WojciechMigda/Tsetlini/blob/main/lib/examples/binary-iris/src/CAIR_The_Binary_Iris_Dataset.ipynb>`_.


.. [TM2018] Granmo, O.C., 2018. The Tsetlin Machine--A Game Theoretic Bandit Driven Approach to Optimal Pattern Recognition with Propositional Logic. `arXiv preprint arXiv:1804.01508 <https://arxiv.org/abs/1804.01508>`_.

.. [TM2018code] `Tsetlin Machine <https://github.com/cair/TsetlinMachine>`_.

.. rubric:: Footnotes

.. [#xor_code1] `<https://github.com/cair/TsetlinMachine/blob/master/NoisyXORDemo.py#L34>`_.
.. [#xor_code2] `<https://github.com/cair/TsetlinMachine/blob/master/MultiClassTsetlinMachine.pyx#L58>`_
