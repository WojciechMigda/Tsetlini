Reference examples
==================

Noisy XOR
---------

Appeared in the first Tsetlin Machine paper [TM2018]_.

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
but the source code referenced in it uses value of ``0`` [#f1]_ [#f2]_.

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



.. [TM2018] Granmo, O.C., 2018. The Tsetlin Machine--A Game Theoretic Bandit Driven Approach to Optimal Pattern Recognition with Propositional Logic. `arXiv preprint arXiv:1804.01508 <https://arxiv.org/abs/1804.01508>`_.

.. rubric:: Footnotes

.. [#f1] `<https://github.com/cair/TsetlinMachine/blob/master/NoisyXORDemo.py#L34>`_.
.. [#f2] `<https://github.com/cair/TsetlinMachine/blob/master/MultiClassTsetlinMachine.pyx#L58>`_
