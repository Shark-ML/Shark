
Importing Data
==============

Shark provides a number of containers for data storage.
Read the basic :ref:`data tutorials <label_for_data_tutorials>`
first if you are not familiar with these containers. This tutorial
deals with how to fill the containers with actual data.

File Formats
------------

Shark supports a number of standard file formats for data sets,
including the HDF5-based format used by http://www.mldata.org,
comma(character)-separated-values (CSV), and the LIBSVM format
(http://www.csie.ntu.edu.tw/~cjlin/libsvm/). Shark does not come
with its own data set format any more in order to avoid further
growth of the jungle of data set formats. However, data can be
serialized, which practially amounts to a data file format.

Most data formats in common use are restricted to (sparse)
vectorial input data. Thus, when dealing with non-vectorial data
the user needs to write specialized methods for loading/storing
these data. It is understood that shark can not implement any
possible data format you can dream of. However, if the input
type is serializable with boost::serialization, then the
:doxy:`Data` container can be serialized.

Generate from Artificial Distributions
++++++++++++++++++++++++++++++++++++++

Data sets can be generated using artificial distributions.
Currently, shark comes with a few distributions for testing
purposes, but if you need automatically generated (pseudo
random) data then you probably want to create your own
distribution class. To create your own data distribution,
you have to derive a class from the :doxy:`DataDistribution`
interface and overload the :doxy:`DataDistribution::draw`
method, which allows shark to draw a labeled example from
your probability distribution. Also you can choose which
types your inputs and labels should have.

As an example, let us generate inputs from the real line with
labels 0 and 1 with equal probability, with uniform and
overlapping class-conditional distributions: ::

  class YourDistribution: DataDistribution<RealVector, unsigned int>{
      public:
          void draw(RealVector& input, unsigned int& label) {
	      input.resize(2);
	      label = Rng::coinToss();
	      input(0) = Rng::uni(-1,1);
	      input(1) = Rng::uni(-1,1) + label;
	  }
  };

Once the distribution is defined it is easy to generate a data set: ::

  YourDistribution distribution;
  unsigned int numberOfSamples = 1000;
  ClassificationDataset dataset = distribution.generateDataset(numberOfSamples);


CSV
++++++++++++++++++++++++++++++++++++++++

Shark supports the simplistic but widespread CSV (comma/character
separated value) data format; however, support of this format is
currently quite limited. Not all class label types are supported
and the data must be dense.

To load csv files you have to include ::

  #include <shark/Data/Csv.h>

Since the separator in the CSV format is left open it needs to be
specified. A comma (",") is a standard choice, but spaces or tabulators
are also common. A comma is used as a default.

Now you can call one of the import routines like this: ::

  Data<RealVector> data;
  std::string separator = ","
  std::string filename = "data.csv";
  import_csv(data, filename, separator);     // import file
  export_csv(data, filename, separator);     // export file

If you want to import regression data then you have to load data and
labels from different csv files and create a LabeledData object from
both the two containers::

  Data<RealVector> inputs;
  Data<RealVector> labels;
  import_csv(inputs,"inputs.csv",separator);
  import_csv(labels,"labels.csv",separator);
  RegressionDataset dataset(inputs, labels);

Classification data can be read in from a single file. Only `unsigned int`
labels are currently supported (see also :doc:`labels`). Below are three
different use case examples::

  ClassificationDataset dataset;
  // load a csv with labels in the first column of every line
  import_csv(dataset, "data.csv", FIRST_COLUMN, separator);
  // load a csv with labels in the last column of every line
  import_csv(dataset, "data.csv", LAST_COLUMN, separator);
  
  // save the classification dataset
  export_csv(dataset, "data.csv", FIRST_COLUMN, separator);


LibSVM
++++++++++++++++++++++++++++++++++++++++

Shark can import LibSVM files.

.. todo::
   are there restrictions?

LibSVM support comes with the include directive ::

  #include <shark/Data/LibSVM.h>

Similar to the CSV import functions we can call ::

  ClassificationDataset dataset;
  import_libsvm(dataset, "data.libsvm");

For sparse libsvm data you may consider setting the third parameter
``verbose`` in :doxy:`Libsvm.h` to `true`. This tells shark to print the sparseness
ratio of the data to standard output. You can also import to sparse
data vectors: ::

  Dataset<CompressedRealVector, unsigned int> dataset;
  import_libsvm(dataset, "data.libsvm", true);


HDF5 and MLData
++++++++++++++++++++++++++++++++++++++++

.. todo:: The tutorial section on HDF5 and MLData imports will be part of the official Shark release.
