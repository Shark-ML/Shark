
Shark Data Containers Quick Reference
=====================================


Related tutorials
-----------------

* :doc:`../tutorials/concepts/data/datasets`
* :doc:`../tutorials/concepts/data/labels`
* :doc:`../tutorials/concepts/data/import_data`
* :doc:`../tutorials/concepts/data/dataset_subsets`
* :doc:`../tutorials/concepts/data/normalization`


Relevant Types
--------------

* :doxy:`Data`
* :doxy:`UnlabeledData`
* :doxy:`LabeledData` (also the typedefs ClassificationDataset, CompressedClassificationDataset, RegressionDataset)
* :doxy:`DataView`
* :doxy:`Data`
* :doxy:`DataDistribution`
* :doxy:`LabeledDataDistribution`
* :doxy:`CVFolds`


Container / View Creation
-------------------------

=================================== ===============================================================
Data<T>()                           create empty data container
Data<T>(data)                       create shallow copy with content sharing
Data<T>(N)                          create new data container with N batches
Data<T>(N, elem)                    create new data container with N elements, with blueprint elem
UnlabeledData<T>()                  create empty data container
UnlabeledData<T>(data)              create shallow copy with content sharing
UnlabeledData<T>(N)                 create new data container with N batches
UnlabeledData<T>(N, elem)           create new data container with N elements, with blueprint elem
LabeledData<I,L>()                  create empty data container
LabeledData<I,L>(input, labels)     create shallow copy with content sharing
LabeledData<I,L>(N)                 create new data container with N batches
LabeledData<I,L>(N, elem)           create new data container with N elements, with blueprint elem
DataView<DatasetType> view(data)    create view of data for fast random access to elements
:doxy:`createDataFromRange`         create from begin+end iterators, e.g., from std::vector
:doxy:`createLabeledDataFromRange`  create from two ranges for inputs and labels
:doxy:`toDataset`                   create data container from view
=================================== ===============================================================


Batch Access
------------

=================================== ===============================================================
data.empty()                        true iff data.numberOfBatches() == 0
data.numberOfBatches()              number of batches in the container
data.batch(i)                       (reference to) the i-th batch
data.batches()                      stl-compliant access to batches as a range
=================================== ===============================================================


Element Access
--------------

.. warning::
	Random access to elements is a linear time operation!
	Never iterate over elements by index. Consider employing
	a ``DataView`` for random access.

=================================== ===============================================================
data.numberOfElements()             number of elements in the container
data.element(i)                     (proxy to) the i-th elements
data.elements()                     stl-compliant access to (proxies to) elements as a range
=================================== ===============================================================


Further Methods
---------------

=================================== ===============================================================
swap()                              swap container contents (constant time)
makeIndependent()                   make sure data is not shared with other containers
shuffle()                           randomly reorder elements (not only batches)
append(data)                        concatenate containers
LabeledData::inputs()               underlying container of inputs
LabeledData::labels()               underlying container of labels
=================================== ===============================================================


Sizes and Dimensions
--------------------

=================================== ===============================================================
:doxy:`numberOfClasses`             number of classes (maximal class label + 1)
:doxy:`classSizes`                  vector of class sizes
:doxy:`dataDimension`               dimension of vectors in the data set
:doxy:`inputDimension`              dimension of input vectors in the data set
:doxy:`labelDimension`              dimension of label vectors in the data set
=================================== ===============================================================


Subset Creation and Folds for Cross-validation
----------------------------------------------

=================================== ===============================================================
:doxy:`Data::splice`                split data into front and back part (often training and test)
:doxy:`indexedSubset`               obtain subset from indices
:doxy:`rangeSubset`                 obtain subset from range
:doxy:`subset`                      create indexed subset from :doxy:`DataView`
:doxy:`createCVIID`                 create folds by i.i.d. assignment of element to folds
:doxy:`createCVSameSize`            create folds of roughly equal size
:doxy:`createCVSameSizeBalanced`    create folds of roughly equal size, stratifying classes
:doxy:`createCVIndexed`             create folds explicitly by index
:doxy:`createCVFullyIndexed`        create folds explicitly by index with reordering
=================================== ===============================================================


Import / Export
---------------

=================================== ===============================================================
:doxy:`importCSV`                   import from comma separated values (CSV) file
:doxy:`importSparseData`            import from sparse vector (libSVM) format
:doxy:`importHDF5`                  import from comma separated values (CSV) file
:doxy:`exportCSV`                   export to comma separated values (CSV) file
:doxy:`exportSparseData`            export to sparse vector (libSVM) format
=================================== ===============================================================
