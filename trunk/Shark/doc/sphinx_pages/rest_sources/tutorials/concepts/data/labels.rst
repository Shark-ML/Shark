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

In detail, there exist **two** main conventions which are used by algorithms
in typical classification settings, **plus** one additional convention for regression
settings. We first list these three conventions, and after that provide a 
list of classes or algorithms which rely on data having the labels formatted
accordingly:

* Labels for **classification** are stored in one of two formats, and it depends
  on the algorithm whether it accepts both equally or just a specific one of the two,
  although most often, the single integer coding will be supported.
  Assume a d-class problem:

  + The default format for one label is a single unsigned integer (C++ type
    ``unsigned int``) in the range ``0,...,d-1``. For binary data, the wide-spread
    binary labels ``-1``/``+1`` are no longer supported; instead, ``0``/``1``
    is used for the sake of consistency with the multi-class case. When required, 
    Binary labels -1/+1  are converted to 0/1 by setting all -1 labels to 0.
    
  + Alternatively, a class label can be encoded by a d-dimensional
    RealVector. For example, the class :doxy:`CrossEntropy`
    can then interpret the real value at position ``i`` as the
    probability of the corresponding sample having class ``i`` as its
    label. However, in general the values may not all be positive, and
    they may not sum to one (for example, neural network outputs).
    Note that the deterministic case then corresponds to a so-called
    one-hot encoding of the labels, in which all entries are set to zero
    and the denoted label's entry is set to one. All algorithms relying or
    accepting a probabilistic encoding via a RealVector also accept
    a RealVector of length one -- which is then understood to denote the
    probability of the label being of the positive class ``0`` in a binary
    classification dataset.

* Labels for **regression** are always of type ``RealVector``, also for
  single-dimensional regression problems. In this case the label
  vectors are one-dimensional. The use of type double is
  discouraged, since many classes in Shark (such as models,
  loss functions, and trainers) already assume the ``RealVector``
  encoding.

Conversions
-----------

Shark offers three different converters between label formats.
These are found in the file :doxy:`Converter.h`.
The available conversions are

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

* :doxy:`OneHotConverter`: This class takes a discrete label in
  the range 0,...,d-1 and converts it into a one-hot-encoded
  ``RealVector`` of size d. This is useful for converting a discrete
  class label into a target value for neural network training.

All three classes are implemented as models. This allows for 
concatenating the converters with actual
predictive models, such as a :doxy:`KernelExpansion` and an
:doxy:`ArgMaxConverter`, in order to obtain a multi-category
support vector machine model with class labels as outputs. This
concatenation is realized by the :doxy:`ConcatenatedModel` class.
