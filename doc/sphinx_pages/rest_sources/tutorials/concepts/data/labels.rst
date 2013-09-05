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

Most algorithms in Shark are fully templated. The kernel methods
for example are designed to work with any input format, and this
allows for easy employment of sparse vector formats. Templatization
also applies to the labels in a :doxy:`LabeledData` dataset: they can either
be simple integers for classification, or real values for regression, or
arbitrarily complex structured types for more specialized applications.
However, while the :doxy:`Data` class and its subclasses do not impose
any restrictions on label formats, several *algorithms* within Shark
(i.e., some which *work on or with* datasets) in fact do: they may be tailored
towards a standard classification or regression setting, and might expect
integer or real-valued encoded labels in accordance with fixed conventions.
In other words, using custom templates for the labels might make necessary
writing one's own custom trainers, error functions, losses, etc.

In detail, there exist a convention used by algorithms
in typical classification settings, and a convention for regression
settings::

* The format for a classification label is a single unsigned integer (C++ type
   ``unsigned int``) in the range ``0,...,d-1``. For binary data, the wide-spread
   binary labels ``-1``/``+1`` are no longer supported; instead, ``0``/``1``
   is used for the sake of consistency with the multi-class case. When required, 
   Binary labels -1/+1  are converted to 0/1 by setting all -1 labels to 0.

* Labels for regression are  of type ``RealVector``, also for
  single-dimensional regression problems. In this case the label
  vectors are one-dimensional.

Based on the Method, classification labels are interpreted differently. The most common interpretation
is that of a simple index. But the label *c* can also be
interpreted as a unit-vector for which the *c*-th component is one. Through this it is possible to use
mean-squared error mesaures on the output of neural networks in classification.


Conversions
-----------

Often Models in Shark do not produce the correct output for classification. This is most often not a problem,
as the loss function can interpret outputs accordingly and thus a neural network can be easily trained in a
classification setting, even though the ntwork only returns vectors instead of labels. However, if the labels
are indeed needed, the output of a Ntowkr needs to transformed for which the following converter exist:

* :doxy:`ThresholdConverter`: The class converts single dimensional
  ``RealVector`` inputs to binary 0/1 class-labels by assigning the value 1 if the 
  value of the input is higher than a certain threshold.
  This is useful, for example, for converting the output of a support 
  vector machine of neural network for binary classification into a 
  discrete class label.

* :doxy:`ArgMaxConverter`:  Is a generalization of the ThresholdConverter.
  It class assumes d-dimensional ``RealVector`` inputs for classification. 
  It converts the vector to a discrete label in the range 0,...,d-1 by finding the index of
  the largest component (the arg max). This is useful for turning
  the output of a support vector machine or neural network for
  multi-category classification into a discrete class label.