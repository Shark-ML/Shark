Data Containers
===============
Data handling is an important aspect of a machine learning
library. Shark ships with three container classes tailored
to holding data for machine learning applications:
:doxy:`Data`, :doxy:`UnlabeledData`, and :doxy:`LabeledData`.
After familiarizing yourself with the basic concepts, have a look at the
complete list of :ref:`data tutorials <label_for_data_tutorials>`

A decisive difference between Shark 3.x and previous Shark version and
other machine learning libraries is that the data is not stored in a
generic container, but in objects tailored to efficient large-scale
machine learning.

The containers presented in this tutorial can all be used by including::

  #include<shark/Data/Dataset.h>

Key properties
---------------

The data containers provided by shark can store all types of data that
could also be  held in one of the standard template library containers.
In contrast to  a ``std::vector``,  the Data class has three abilities
that are important in the context of machine learning:

* Elements of a dataset are stored in blocks called batches, such that
  computations can be carried out block by block, instead of element
  by element. These batches are optimized to allows for continuous memory access,
  which allow for more efficient processing and thus faster implementations.
  For example, a batch of vectors is stored as a matrix with consecutive
  memory with every point occupying a matrix row, instead of using several vectors
  with memory locations scattered all over the heap. This is achieved through Shark's
  :doc:`batch mechanism <../library_design/batches>`.

* A :doxy:`Data` object can be used to create subsets. This is useful,
  for example, for splitting data into training, validation, and test sets.
  Conceptually, each batch of data is here regarded as atomic and required to
  reside in one subset in full. Thus it is not possible to assign one half of
  a batch  to one subset and the other half to another. If this is needed,
  the batch  must first be split physically.

* Data can be shared among different :doxy:`Data` instances. Thus creating
  subsets on the level of batches is quite cheap as it does not need a physical
  copy of the contents of the set. On should not confuse this with the different
  concept of lazy-copying, which just delays the copy until an actual change is
  done. Instad sets are shard by default and only copied, when actually required by
  the algorithm.


Different types of Datasets
--------------------------------

The three dataset classes in shark differ not much in their implementation, as
they all use the same underlying structure. However they provide important semantic
differentiation as well as special functions tailored to this differentiation. Before
we introduce the interface of the data object we want to clarify this distinction:

* :doxy:`Data` can store arbitrary data. The data class takes the
  general role of an ``std::vector`` only adapted to the special needs
  for fast computation in a machine learning environment.

* :doxy:`UnlabeledData` represents input data which is not labeled.
  This is the input format used for unsupervised learning methods. The unlabeled
  data class is a subclass of Data and does not offer much new functionality
  compared to ``Data``.
  But it provides an important *semantic* difference, as these data
  points are interpreted as input data without labels, compared to the above
  mentioned Data class whose contents might store anything (for example model
  outputs, labels or points) . Datasets as used in machine learning are
  inherently unordered constructs, thus it is okay for an algorithm to shuffle or
  otherwise
  reorder the contents of a dataset. This is reflected in the set, that shuffling
  is actively supported using the :doxy:`UnlabeledData::shuffle` method.

* :doxy:`LabeledData` finally represents datapoints which are a pair of inputs
  and labels. An dataset of type ``LabeledData<I,L>`` can be roughly described
  as the known data object using a pair-type of inputs I and labels L, for example
  ``Data<std::pair<I,L> >``. There is however an important difference in how labels
  and inputs are treated in machine learning. We often like, especially for unsupervised
  methods, to only use the inputs, thus viewing the object as an ``UnlabeledData<I>``.
  For evaluation of the model, we also want to first get the set inputs, acquire the
  set of predictions of the model and compare this set of predictions with the set of labels
  using a loss function. Instead of seeing input-label pairs as a fixed grouping, we would
  like to view them as two separate datasets which are conveniently bound together. And this is
  how the LabeledData object is implemented.


The class Data<T>
------------------
This part of the tutorial introduces the interface of :doxy:`Data`. The following description
also applies to the two other types of datasets.

Creation and copying of datasets
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

Creating a dataset is quite easy and can be achieved in several ways. The first and
by far easiest way is by directly loading the dataset from a file or generate them
using an artificial distribution of data. Examples for this are given in the
tutorial on :doc:`importing data <../../first_steps/general_optimization_tasks>`. In some cases
data is already in memory and only needs to be imported into a dataset.
In this case a dataset can be created using::

  std::vector<RealVector> points; //vector of points
  Data<RealVector> data = createDataFromRange(points);

To create an dataset with space for *n* points, we need to define an example point which
describes the objects to be saved in the set::

  Data<RealVector> data(1000, RealVector(5));

In the above example, we create a dataset which can hold 1000
5-dimensional vectors.  The provided Vector is not copied to all 1000
elements, but it serves merely as a hint on the structure of the
objects to be stored. To understand this, remember that objects are
not stored as single entities, but grouped in batches. In the case of
the vector, the type of the batch is a matrix. But we can't store
vectors with different sizes in the dataset, and thus we must provide
the dataset with the information about how long a matrix-row needs to
be. In essence this call does not create 1000 entities of vectors
together with the same amount of memory allocations, but only a few
bigger matrices. By default a safe size is used for the number of
elements in a batch, but it can also be actively controlled by adding
the maximum size of batches as a third parameter::

  Data<RealVector> data(1000, RealVector(5),100);

Datasets can be copied and assigned using the typical operations::

  Data<RealVector> data2(data);
  data = data2;

However, note that these operations do not perform a deep-copy, but as mentioned in the
key properties, data is shared between the different instances. To check whether the content
of a set is shared, we can use::

  data.isIndependent();

and to perform a deep copy of the elements, we can use::

  data.makeIndependent();

Data sharing is thread-safe, thus it is perfectly fine to create
shares of (parts of) the data object in several threads. However, it
has to be stressed that the dataset class does not guard one from
changes to the individual batches or single elements. Changing an
element in one instance of the data object will change the respective
elements in all other containers as well.

Data as a collection of batches
*******************************

As outlined above, the Data class stores the points internally as batches and
is therefore optimized for using these batches directly instad of accessing the
single points. Therefore this part of the tutorial will explain how the dataset
provides access to the batches as well as common usage patterns.

The first thing to note is that the dataset itself does not provide direct access
using iterators or other stl-compatible means. This is done to prevent confusion
with the element methods (e.g. a size() method could be either interpreted as
returning the number of batches or the number of elements). However an
stl compatible interface can be acquired using the :doxy:`Data::batches`
method::

    typedef Data<RealVector>::batch_range Batches;
    Batches batches = data.batches();

    std::cout<<batches.size()<<std::endl;
    for(Batches::iterator pos = batches.begin(); pos != batches.end(); ++pos){
        std::cout<<*pos<<std::endl;
    }

or similarly when data is constant or a constant range is desired::

    Data<RealVector>::const_batch_range batches = data.batches();

However, the above loop still looks a bit inconvenient, we might as well use
``BOOST_FOREACH`` for traversal::

    typedef Data<RealVector>::const_batch_reference BatchRef;
    BOOST_FOREACH(BatchRef batch,data.batches()){
        std::cout<<batch<<std::endl;
    }

Or we can also just iterate using an indexed access::

   for(std::size_t i = 0; i != data.numberOfBatches(); ++i){
      std::cout<<data.batches(i)<<std::endl;
   }

We can also use this direct batch access to get direct access to the single elements,
using the methods for batch-handling and another loop::

   BOOST_FOREACH(BatchRef batch,data.batches()){
        for(std::size_t i = 0; i != boost::size(batch); ++i){
	    std::cout<<shark::get(batch,i);//prints element i of the batch
	}
   }


Data as a collection of elements
*********************************

While the data object is optimized for batch access, sometimes direct
access to elements is desired.  Thus we also provide an convenience
interface for elements, however, we can't give as good performance
guarantees as for the batch access. While the interfaces look very
similar, you must be aware of the important differences.

First of all, all elements stored in the dataset are only virtual for most input types. This means
that querying the i-th element of the set does not return a reference to it, but instead returns
a proxy object which behaves as the reference. So for example when storing vectors, instead of a vector
a row of the matrix it is stored in is returned. This is no problem most of the time, however when
using the returned value as an argument to a function like for example::

   void function(Vector&);

the compiler will complain, that a matrix row is not a vector. In the case of::

  void function(Vector const&);

the compiler is very helpful, creating a temporary vector for you and copying the
matrix row into it. However, this is slow. Be aware of this performance pitfall and use
template arguments or the correct reference type of the dataset if possible::

   void function (Data<RealVector>::element_reference);

The second pitfall is  that we can't give as strong performance guarantees for the methods called.
As we allow batch resizing and all batches having a different size, it is not easy to keep track of the
actual number of elements stored in the set, thus calling
:doxy:`Data::numberOfElements` takes time linear in the number of batches.
For the same reason, accessing the i-th element using :doxy:`Data::element` is linear in the number of batches,
as we first need to find the batch the element is located in, before we can actually access it.
Thus aside from only very small datasets or performance  uncritical code, you should never use
random-access to the dataset and use the following, more appropriate  ways to iterate over the elements::

    typedef Data<RealVector>::element_range Elements;
    typedef Data<RealVector>::const_element_reference ElementRef;

    //1: explicit iterator loop using the range over the elements
    Elements elements = data.elements();
    for(Elements::iterator pos = elements.begin(); pos != elements.end(); ++pos){
        std::cout<<*pos<<std::endl;
    }
    //2: BOOST_FOREACH
    BOOST_FOREACH(ElementRef element,data.elements()){
        std::cout<<element<<std::endl;
    }


Summary of element access
**************************
We will now summarize the above description in a more formal tabular layout. For the shortness of description,
we  only present the non-const version of every method and typedef. The rest can be looked up in the doxygen reference.

Typedefs of Data. For every reference and range there exists also an immutable version adding a ``const_`` to the
beginning:

========================   ======================================================================
Type                       Description
========================   ======================================================================
element_type               The type of elements stores in the object
element_reference          Reference to a single element. This is a proxy reference, meaning
                           that it can be something more complex than element_type&, for example
			   an object describing the row of a matrix.
element_range              Range over the elements..
batch_type                 The batch type of the Dataset. Same as Batch<element_type>::type
batch_reference            Reference to a batch of points. This is batch_type&.
batch_range                Range over the batches.
========================   ======================================================================

Methods regarding batch access. All these methods have constant time complexity:

==========================================   ======================================================================
Method                                       Description
==========================================   ======================================================================
size_t numberOfBatches () const              Returns the number of batches in the set.
batch_reference batch (size_t i)             Returns the i-th batch of the set
batch_range batches ()                       Returns an stl-compliant random-access-container over the batches.
==========================================   ======================================================================

Methods regarding batch access. All these methods have time complexity
linear in the number of batches:

==========================================   ======================================================================
Method                                       Description
==========================================   ======================================================================
size_t numberOfElements () const             Returns the number of elements in the set.
element_reference element (size_t i)         Returns the i-th element of the set
element_range elements ()                    Returns an bidirectional container over the elements. Random access
                                             is also supported, but does not meet the time complexity. Also be aware
					     that instead of references, proxy-objects are returned as elements are
					     only virtual.
==========================================   ======================================================================

further, ``LabeledData`` supports direct access to the Containers representing either elements or labels.

==========================================   ======================================================================
Method                                       Description
==========================================   ======================================================================
UnlabeledData<I>& inputs()                   Returns only the inputs of the LabeledData<I,L> object.
Data<L>& labels()                            Returns only the labels of the LabeledData<I,L> object.
==========================================   ======================================================================

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
  std::size_t classesOneHot = numberOfClasses(data);

Transformation of datasets
---------------------------------------------

In a lot of use cases, one needs to preprocess the data, before it can be used for the problem.
For example, the mean of a dataset is to be removed, or labels need to be changed in order to fit
into the shark scheme which assumes the existence of all class labels. For this, shark provides
a smart transformation mechanism. Lt's assume we have a function object f and g such that f(input)
returns the transformed input vector and g(label) the transformed label. Than we can transform
data sets by::

   Data<RealVector> data;//initial dataset;
   data = transform(data,f);//applies f to all elements of data

   LabeledData<RealVector,unsigned int> labeledData;//initial labeled dataset;
   labeledData = transformInputs(labeledData,f);//applies f to the inputs only
   labeledData = transformLabels(labeledData,g);//applies f to the labels only

The transformation mechanism itself is smart! If f does not only provide a function
f(input) but also f(Batch_of_input>) returning the same transformation for a whole batch,
this is applied instead. As batch transformations are often more efficient than applying
the same transformation to all elements one after another, this can be a real time saver.
An example for an object satisfying this requirement are the Models provided by shark::

    //a linear model, for example for whitening or making a dataset mean free
    LinearModel<> model;
    //applying the model
    labeledData = transformInputs(labeledData,model);
    //or an alternate shortcut for data:
    data = model(data);

It is easy to write your own transformation.
A simple example just adding a scalar to all elements in a dataset
could look like this: ::

  class Add {
  public:
	Add(double offset) : m_offset(offset), m_scalar(true) {}

	typedef RealVector result_type; // do not forget to specify  result type

	RealVector operator()(RealVector input) const { // const is important
		for(std::size_t i = 0; i != input.size(); ++i)
				input(i) += m_offset;
		return input;
	}
  private:
	double m_offset;
  };


It is then applied to dataset by calling something such as: ::

  data = transform(data, Add(2.0));



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
inside the batch). Thus creating a DataView object might lead to a big initial
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

After the operation, ``subset`` holds a copy of the points indexed by the subset operation.
As in all other dataset operations, the subset is organized in several batches. To control the
maximum size of the batches, ``toDataset`` also takes an optional second parameter, which controls this:

Data<unsigned int> subset = toDataset(subset(view,indices),maximumBatchSize);

And the usual methods for querying dataset informations also works for the view::

  LabeledData<RealVector,unsigned int> dataset;
  DataView<LabeledData<RealVector,unsigned int> > view(dataset);
  std::cout << numberOfClasses(view) << " " << inputDimension(view);

See the doxygen documentation for more details!
