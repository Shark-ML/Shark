Label Formats
=============

This tutorial covers label conventions of :doxy:`LabeledData`,
and converters between different label formats.
For other aspects of Shark data containers, please see the
complete list of :ref:`data tutorials <label_for_data_tutorials>`.
Also, please read the following sentences carefully: this tutorial
does **not** cover in what format you can/should/may bring your
input data files. It **only** covers how Shark treats labels
internally. So, in most cases, you will want to read the
:doc:`tutorial on importing data <import_data>` first to figure
out how to get your data into Shark. Then, to learn about how
Shark treats labels and to work with them by writing Shark code,
read this tutorial here. To prevent a classical pitfall, don't try
to bring your data files into a one-hot encoding just because it
is mentioned on this page.


Label Conventions
-----------------

Many algorithms in Shark are fully templatized. The kernel methods
for example are designed to work with any input format, and this
allows for easy employment of, e.g., sparse vector formats.
Templatization also applies to the labels in a :doxy:`LabeledData`
data container: they can either be simple integers for classification,
or real values for regression, or arbitrarily complex structured types
for more specialized applications.
However, while the :doxy:`Data` class and its subclasses do not impose
any restrictions on label formats, several *algorithms* within Shark
(i.e., some which *work on or with* datasets) in fact do: they may apply
in a standard classification or regression setting, and might expect
integer-valued or real-valued labels in accordance with fixed conventions.
In other words, using custom labels is supported, however, it corresponds
to non-standard learning tasks and as such might require custom error
functions, losses, and eventually even the adaptation of trainers.

In detail, there exists a convention for algorithms in typical
classification settings, and a convention for regression settings:

* The format for a classification label is a single unsigned integer
  (C++ type ``unsigned int``) in the range ``0,...,d-1``. For binary
  (two-class) labels, the wide-spread binary labels ``-1``/``+1`` are
  no longer supported; instead, ``0``/``1`` is used for the sake of
  consistency with the multi-class case. When required, binary labels
  -1/+1  are converted to 0/1 by setting all -1 labels to 0.

* Labels for regression are  of type ``RealVector``. This is also the
  case for single-dimensional regression problems. In this case the
  label vectors are one-dimensional. The C++ type ``double`` is not
  used.

Based on the method (model, algorithm), classification labels are
interpreted differently. The most common interpretation is that of a
unique atom. By convention, ``d`` different atoms (in a classification
task with ``d`` classes) are chosen as ``0,...,d-1``. Such a value can
also serve as an index (e.g., indexing output neurons) in certain
circumstances.
The label ``c`` can also be interpreted as a ``d``-dimensional unit-vector
for which the ``c``-th component is one. This enables the application of,
e.g., the mean-squared error measure on the output of neural networks
for classification.


Conversions of class labels
---------------------------

Often Models in Shark do not produce the correct output format for
classification. For example, a neural network for a ``d``-class
classification problem usually encodes its prediction into an output
of size ``d``, with the prediction being understood as the
index of the output neuron with highest activity. The network output
is thus a ``RealVector`` of dimension ``d``, not an unsigned integer.
This is often not a problem, the loss function can interpret
outputs accordingly and thus a neural network can easily be trained
in a classification setting, even though the network only returns
vectors instead of (integer) labels.
However, if integer labels are indeed needed for further
post-processing then the output of a Network needs to be transformed.
This can be done with a helper model

* :doxy:`Classifier`:   As input it assumes a d-dimensional
  ``RealVector`` for classification. It converts the vector to a
  discrete label in the range 0,...,d-1 by finding the index of the
  largest component (the arg max). This is useful for turning the output
  of a support vector machine or neural network for multi-category
  classification into a discrete class label. In the case of d=2 it will transform
  the output to an output based on the sign of the label. positive sign is transformed
  to label, a negative sign to label 0. Additionally the decision thresholds can be adapted
  using bias weights for each model output.
  
The classifier holds the trained model as its decision function. Therefore creating 
a Classifier from a model can be done via::

  Classifier<ModelType> classifier;//used after training for classification
  ModelType& model = classifier.decisionFunction();//used -during training

Some models are predefined for convenience:
The :doxy:`KernelClassifier` is a classifier using a :doxy:`KernelExpansion`
and the :doxy:`LinearClassifier` is a classifier using a :doxy:`LinearModel`.
