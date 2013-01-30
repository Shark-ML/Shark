Data Containers
===============

.. warning::

    The Data class interface is subject to change in the near future and the recent
    past. Thus this tutorial might be in parts outdated or will turn out to be
    outdated in the near future. However on a conceptionel level the contents are
    correct.

Data handling is an important aspect of a machine learning
library. Shark ships with three container classes tailored
to holding data for machine learning applications:
:doxy:`Data`, :doxy:`UnlabeledData`, and :doxy:`LabeledData`.
After familiarizing yourself with the basic concepts, also see the
complete list of :ref:`data tutorials <label_for_data_tutorials>`

The containers presented in this tutorial can all be used by including::

  #include<shark/Data/Dataset.h>


The Data<Input> container
-------------------------


Key properties
&&&&&&&&&&&&&&

The central container for data in Shark is simply called :doxy:`Data`.
This template class can store any data that could also be held in one
of the standard template library containers. In contrast to a ``std::vector``, 
the Data class has three abilities that are important in the context 
of machine learning:

* Data is stored in groups or "batches". In detail, a number or group of
  data points are regarded and treated as a grouped entity, on and with
  which computations can be carried out block-wise -- in "batches". These
  batches serve the purpose (and are sized such) that memory access patterns
  allow for more efficient processing and thus faster implementations.
  For example, a batch of vectors is stored as a matrix with consecutive
  memory, instead of several vectors with memory locations all over
  the heap. This is achieved through Shark's :doc:`batch mechanism <batches>`.

* A :doxy:`Data` object can be used to create subsets. This is useful,
  for example, for splitting data into training, validation, and test sets. 
  Conceptually, each batch of data is here regarded as atomic and required to 
  reside in one subset. Thus it is not possible to assign one half of a batch 
  to one subset and the other half to another. If this is needed, the batch 
  must first be split physically.

* Data can be shared among different :doxy:`Data` instances. Thus creating
  subsets on the level of batches is quite cheap as it does not need a physical
  copy of the contents of the set. 


Creating datasets
&&&&&&&&&&&&&&&&&

Creating a dataset is quite easy and can be achived in several ways. The first and
by far easiest way is by directly loading the dataset form a file or generate them
using an artificial distribution of data. Examples for this are given in the
tutorial on :doc:` importing data <general_optimization_tasks>`. In some cases
data is already in memory, in this case a dataset can be created using::

  std::vector<RealVector> points;//vector of points
  Data<RealVector> data(points);


More on views and splits
&&&&&&&&&&&&&&&&&&&&&&&&


Data as a collection of batches
*******************************

As seen above, the Data class provides for two different conceptual views of
a data set. The first one is a collection of batches, which partition the data
into chunks and thereby allowing for faster processing. These chunks can be
reordered and shared between several Data instances, thus allowing several views 
on the same data without the need for a copy of the points inside the batches. 
Instead only references to the batches are shared between vies and a batch is 
removed from memory when no dataset is referencing it anymore.
As a consequence, copying and restrictions to subsets are operations linear in 
space and time in the number of batches, but not in the number of training points.

A simple example of creating such a subset is by issuing

  Data<RealVector> dataset=...;
  Data<RealVector> subset=rangedSubset(dataset,0,5);
  
this will create a subset of the dataset containing the first five batches of 
the original dataset.

One should not confuse this behaviour with *lazy copying*, as in that case a
copy would be performed as soon as one of the views were to be changed. The Shark
Data class instead assumes that the user would actually always want to change
a data point for all views, and thus that there is no need to track changes.

However, a subset can be made independent from its parents and other siblings
using the :doxy:`Data::makeIndependent` method.

There are some operations in a subset which depend on the contens of the data
set not being shared between several views. This holds for example for
:doxy:`Data::splitBatch` which splits a single batch into two parts, thereby
potentially invalidating other subsets.

.. warning:

    This part of the tutorial is outdated or needs to be rewritten.

Data as a list of points
************************

The other view of the dataset is the view as a list of points. This is mostly
a compatibility feature for algorithms which are hard to transform into a form
which uses the batch view.

.. todo::

    ok, the entire tutorial needs a complete and very thorough re-reading with
    respect to clear and unambiguous terminology: now here, view is used as
    meaning the two aspects presented in the tutorial, and not as a data view
    in later code. view, copy, instance, etc. should all be very clearly used.

.. todo::

    I don't share this criticism. View is never used beforehand aside from
    "conceptional view". So the terminology _is_ clear.


For example, decision trees cannot exploit or even work under the batch
structure of data sets, because their nodes are defined using single points.

More on the Data interface
**************************

The class is a mostly standard compliant container with respect to the batches:
it provides :doxy:`Data::size` and :doxy:`Data::empty` methods, returning the
number of batches and whether the container is empty. It also provides
iterators and standard compliant typdefs. It can therefore be used with
standard algorithms. To access the i-th single batch, the method
:doxy:`Data::batch`(i) should be called.

To operate on single elements/points of the data set, the :doxy:`Data::numberOfElements`,
:doxy:`Data::elemBegin` and :doxy:`Data::elemEnd` methods return the number of samples,
as well as iterators over the range of elements. The iterators over the elements have
typedefed names of ``element_iterator`` and "const_element_iterator". Accessing a single
i-th element can be achieved using ``Data::operator()(i)``. The usual interface can be

.. todo::

    what is "usual" here? in the docs i see that it returns element_range, but
    what does "usual" mean?

accessed using the :doxy:`Data::elements` function which returns a range over the elements.
See the :doxy:`Data` reference documentation for details.

.. caution::

  The range over the elements is not a standard compliant container, as the iterators
  do not return references to the objects, but proxy objects instead, much like
  ``std::vector<bool>``. This means that standard algorithms are not required
  to work. Further note that while the iterators allow for random access,
  this is not O(1), but has the run time behavior of a skip list, as it traverses the
  list of batches during random access. This also holds for ``Data::operator()``,
  which needs to traverse the container to find the correct batch. Thus it is better
  to view the range over the elements as *list* instead of an array, and to
  assume the same run time performance. Thus, ``Data::operator()`` should only be used
  for datasets with a small number of batches or in code not critical w.r.t. performance.



UnlabeledData<Input>
---------------------

The :doxy:`UnlabeledData` class can be used as a data container class for
unsupervised learning. This is mostly a *semantic* difference, as these data
points are interpreted as input data without labels, compared to the above
mentioned Data class whose contents might store anything (for example model
outputs, labels or points).
:doxy:`UnlabeledData` is a sub-class of :doxy:`Data` with a few additional
methods for accessing the elements of the container as *inputs*.
For example, it allows shuffling the inputs using :doxy:`UnlabeledData::shuffle`.
See the full class documentation for details.


LabeledData<Input,Label>
-------------------------

:doxy:`LabeledData` stores a data set as a collection of pairs input points and
labels. It is internally implemented as a pair of Data containers: one holding
the points and one the labels. It features the same interface as the UnlabeledData
class, but always returns an object representing the pair of a batch of inputs
and labels (or a pair of single input and single label respectively). Access to
either the input or label container can be achieved using
:doxy:`LabeledData::inputs()` and :doxy:`LabeledData::labels()`.

.. caution::

  LabeledData is not a valid, standard-compliant container, as the input-label
  pairs are virtual. Thus, the same warning applies as to the element view of Data.




Querying information about a dataset
------------------------------------


Sometimes we want to query basic informations about a data set like input
dimension or the number of classes of a labeled data set. The data classes
provide several convenience functions for such queries.

For Data and UnlabeledData there are three functions::

  Data<unsigned int> data;
  std::size_t numberOfClasses(data); //returns the maximum class label minus one
  std::vector<std::size_t> sizes = classSizes(data); //returns the number of occurrences for every class label

  Data<RealVector> dataVectorial;
  std::size_t dim = dataDimensions(dataVectorial); //returns the dimensionality of the data points

For LabeledData we have a similar set of methods::

  LabeledData<RealVector,unsigned int> data;

  std::size_t classes = numberOfClasses(data); //returns the maximum class label minus one
  std::vector<std::size_t> sizes = classSizes(data); //returns the number of occurrences for every class label
  std::size_t dim = inputDimensions(data);

  LabeledData<RealVector, RealVector> dataVectorial;
  std::size_t dimLabel = labelDimension(data); //returns the dimensionality of the labels
  // number of classes assuming one-hot-encoding
  // same as labelDimension
  std::size_t classesOneHot = numberOfClasses(data);


.. todo::

    is there a line of code missing between the two comment lines or do these
    belong together? i'm not sure from the context...




Element views: DataView<Dataset>
---------------------------------


Sometimes one needs to perform intensive single-element, random access to data
points, for example in decision tree training. In this case, the performance
guarantees of Data are not sufficient, as every random access to an element needs
to be translated into a list traversal. For such scenarios, Shark provides the
class :doxy:`DataView`. It provides another type of view on a data set under the
assumption that the data will not change during the lifetime of the DataView
object. A dataview object consumes linear space, as it stores the exact position
of every element in the container (i.e., the index of the batch and position
inside the batch). Thus creating a DataView object might lead to a big inital
overhead which only pays off if the object is then used a lot. The DataView class
is made available via ``#include<shark/Data/DataView.h>``.

Using a DataView object is easy::

  Data<unsigned int> dataset;
  DataView<Data<unsigned int> > view(dataset);
  for(std::size_t i = 0; i!=view.size(); ++i){
    std::cout << view[i];
  }

Using a DataView object it is also possible to create element-wise subsets which
can then be transformed back into datasets::

   std::vector<std::size_t> indices;
   //somehow choose a set of indices
   Data<unsigned int> subset = toDataset(subset(view,indices));


.. todo::

    i'd prefer a little more information here: what happens to the batches,
    which batches does the new object have, is the data shared (i assume not)
    or copied?

And the usual methods for querying dataset informations also works for the view::

  LabeledData<RealVector,unsigned int> dataset;
  DataView<LabeledData<RealVector,unsigned int> > view(dataset);
  std::cout << numberOfClasses(view) << " " << inputDimension(view);

See the doxygen documentation for more details!

Typical Use Cases
-----------------

The :doxy:`UnlabeledData` and :doxy:`LabeledData` classes are intended
to hold (e.g., training or test) data for learning. These containers are
typically constructed early in a program, for example by loading data from
files. See the :doc:`import_data` tutorial on how this is done. Then,
depending on the learning task at hand, they are passed on to a
:doxy:`SupervisedObjectiveFunction` or an :doxy:`UnsupervisedObjectiveFunction`
(e.g., an :doxy:`ErrorFunction` computing the empirical risk of
a model on data), or to a trainer derived from :doxy:`AbstractTrainer`.

Within these classes, the data is propagated through one or more models,
yielding (intermediate) results. These results will typically be
stored in another :doxy:`Data` object. This container is then passed on
to a loss function, encoded by a sub-class of :doxy:`AbstractLoss`, to
compute the training or test error.

Models may also be used for pre- or post-processing of results, which
can lead to potentially long chains of models. The processing of such
chains can be explicit in a program, with :doxy:`Data` objects holding
intermediate results, or implicit by means of the
:doxy:`ConcatenatedModel` class.

We close with two summarizing remarks:

* A typical main program loads data into :doxy:`UnlabeledData`
  or :doxy:`LabeledData` containers. It may use a further :doxy:`Data`
  object to store model outputs.

* When writing new machine learning models, algorithms, and objective
  or loss functions the :doxy:`Data` container should be used wherever
  possible for data exchange, since it results in the most
  versatile interfaces.
