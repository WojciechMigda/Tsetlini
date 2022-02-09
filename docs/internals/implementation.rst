Implementation
==============

.. note:: Implementation details described here may change in the future,
          may already be outdated, incorrect, or incomplete.
          When in doubt, please refer to the actual source code.

          If you feel something should be corrected, please
          `raise an issue <https://github.com/WojciechMigda/Tsetlini/issues/new>`_.

Training code flow
******************

Interfaces
----------

On the very top there are two kinds of public interfaces. One is for online training
(``partial_fit()``) and the other is for training from scratch (``fit()``).
These public interfaces exist as class methods for each of supported estimators.
They are implemented as thin wrappers, which delegate further work to
freestanding functions, called with estimator's state as one of their arguments.
Doing so enables us to take advantage of hiding implementation and departing from
Object-oriented paradigm.
These functions are called ``partial_fit_impl``, and ``fit_impl``, and are overloads for respective estimator state type.

What happens underneath ``partial_fit_impl``, and ``fit_impl`` is driven by a need to converge into single invocation of either ``fit_classifier_online_impl`` or ``fit_regressor_online_impl``.

.. uml::
   :caption: Portion of Classifier's call trace

   hide empty description
   state "public interface" as public_interface {
      state "<font:courier><classifier>.fit()" as fit #palegreen
      state "<font:courier><classifier>.partial_fit()" as partial_fit #palegreen
   }
   state "<font:courier>fit_impl()" as fit_impl #yellow
   state "<font:courier>fit_classifier_impl<>()" as fit_classifier_impl
   state "<font:courier>partial_fit_impl()" as partial_fit_impl #yellow
   state "<font:courier>fit_classifier_online_impl<>()" as fit_classifier_online_impl
   state "..." as ellipsis #skyblue

   state is_fitted <<choice>>

   fit --> fit_impl
   partial_fit --> partial_fit_impl
   partial_fit_impl --> is_fitted : is fitted?
   is_fitted -left-> fit_impl : [no]
   is_fitted --> fit_classifier_online_impl : [yes]
   note on link
     ""check_X_y""
   end note
   fit_impl --> fit_classifier_impl
   fit_classifier_impl --> fit_classifier_online_impl
   note on link
     ""check_X_y""
     ""check_labels""
   end note
   fit_classifier_online_impl --> ellipsis
   note on link
     ""check_labels""
   end note


