
Shark Data Containers Quick Reference
=====================================


Related tutorials
-----------------

:doc:`../tutorials/concepts/data/datasets`,
:doc:`../tutorials/concepts/data/labels`,
:doc:`../tutorials/concepts/data/import_data`,
:doc:`../tutorials/concepts/data/dataset_subsets`,
:doc:`../tutorials/concepts/data/normalization`.


Relevant Types
--------------

:doxy:`Data`,
:doxy:`UnlabeledData`,
:doxy:`LabeledData` (also the typedefs ClassificationDataset, CompressedClassificationDataset, RegressionDataset),
:doxy:`DataView`,
:doxy:`Data`,
:doxy:`DataDistribution`,
:doxy:`LabeledDataDistribution`,
:doxy:`CVFolds`.


Container / View Creation
-------------------------

=================================== =============================================================== =================
Data<T>()                           create empty data container                                     ``Dataset.h``
Data<T>(data)                       create shallow copy with content sharing                        ``Dataset.h``
Data<T>(N)                          create new data container with N batches                        ``Dataset.h``
Data<T>(N, elem)                    create new data container with N elements, with blueprint elem  ``Dataset.h``
UnlabeledData<T>()                  create empty data container                                     ``Dataset.h``
UnlabeledData<T>(data)              create shallow copy with content sharing                        ``Dataset.h``
UnlabeledData<T>(N)                 create new data container with N batches                        ``Dataset.h``
UnlabeledData<T>(N, elem)           create new data container with N elements, with blueprint elem  ``Dataset.h``
LabeledData<I,L>()                  create empty data container                                     ``Dataset.h``
LabeledData<I,L>(input, labels)     create shallow copy with content sharing                        ``Dataset.h``
LabeledData<I,L>(N)                 create new data container with N batches                        ``Dataset.h``
LabeledData<I,L>(N, elem)           create new data container with N elements, with blueprint elem  ``Dataset.h``
DataView<DatasetType>(data)         create view of data for fast random access to elements          ``DataView.h``
:doxy:`createDataFromRange`         create from begin+end iterators, e.g., from std::vector         ``Dataset.h``
:doxy:`createLabeledDataFromRange`  create from two ranges for inputs and labels                    ``Dataset.h``
:doxy:`toDataset`                   create data container from view                                 ``DataView.h``
=================================== =============================================================== =================


Batch Access
------------

=================================== =============================================================== =================
data.empty()                        true iff data.numberOfBatches() == 0                            ``Dataset.h``
data.numberOfBatches()              number of batches in the container                              ``Dataset.h``
data.batch(i)                       (reference to) the i-th batch                                   ``Dataset.h``
data.batches()                      stl-compliant access to batches as a range                      ``Dataset.h``
=================================== =============================================================== =================


Element Access
--------------

.. warning::
	Random access to elements is a linear time operation!
	Never iterate over elements by index. Consider employing
	a ``DataView`` for random access.

=================================== =============================================================== =================
data.numberOfElements()             number of elements in the container                             ``Dataset.h``
data.element(i)                     (proxy to) the i-th elements                                    ``Dataset.h``
data.elements()                     stl-compliant access to (proxies to) elements as a range        ``Dataset.h``
=================================== =============================================================== =================


Further Methods
---------------

=================================== =============================================================== =================
swap()                              swap container contents (constant time)                         ``Dataset.h``
makeIndependent()                   make sure data is not shared with other containers              ``Dataset.h``
shuffle()                           randomly reorder elements (not only batches)                    ``Dataset.h``
append(data)                        concatenate containers                                          ``Dataset.h``
LabeledData::inputs()               underlying container of inputs                                  ``Dataset.h``
LabeledData::labels()               underlying container of labels                                  ``Dataset.h``
=================================== =============================================================== =================


Sizes and Dimensions
--------------------

=================================== =============================================================== =================
:doxy:`numberOfClasses`             number of classes (maximal class label + 1)                     ``Dataset.h``
:doxy:`classSizes`                  vector of class sizes                                           ``Dataset.h``
:doxy:`dataDimension`               dimension of vectors in the data set                            ``Dataset.h``
:doxy:`inputDimension`              dimension of input vectors in the data set                      ``Dataset.h``
:doxy:`labelDimension`              dimension of label vectors in the data set                      ``Dataset.h``
=================================== =============================================================== =================


Subset Creation and Folds for Cross-validation
----------------------------------------------

=================================== =============================================================== =================
:doxy:`splitAtElement`              split data into front and back part (often training and test)   ``Dataset.h``
:doxy:`subset`                      create indexed subset from :doxy:`DataView`                     ``DataView.h``
:doxy:`createCVIID`                 create folds by i.i.d. assignment of element to folds           ``CVDatasetTools.h``
:doxy:`createCVSameSize`            create folds of roughly equal size                              ``CVDatasetTools.h``
:doxy:`createCVSameSizeBalanced`    create folds of roughly equal size, stratifying classes         ``CVDatasetTools.h``
:doxy:`createCVIndexed`             create folds explicitly by index                                ``CVDatasetTools.h``
:doxy:`createCVFullyIndexed`        create folds explicitly by index with reordering                ``CVDatasetTools.h``
:doxy:`Data::splice`                split data at batch boundaries (contrary of append)             ``Dataset.h``
:doxy:`indexedSubset`               obtain subset of batches from indices                           ``Dataset.h``
:doxy:`rangeSubset`                 obtain subset of batches from range                             ``Dataset.h``
:doxy:`selectFeatures`              filter out a subset of features from :doxy:`Data`               ``Dataset.h``
:doxy:`selectInputFeatures`         filter out a subset of features from :doxy:`LabeledData`        ``Dataset.h``
=================================== =============================================================== =================


Import / Export
---------------

=================================== =============================================================== =================
:doxy:`importCSV`                   import from comma separated values (CSV) file                   ``Csv.h``
:doxy:`exportCSV`                   export to comma separated values (CSV) file                     ``Csv.h``
:doxy:`importSparseData`            import from sparse vector (libSVM) format                       ``SparseData.h``
:doxy:`exportSparseData`            export to sparse vector (libSVM) format                         ``SparseData.h``
:doxy:`importHDF5`                  import from HDF5 file used by mldata.org                        ``HDF5.h``
:doxy:`importPGM`                   import single PGM image                                         ``Pgm.h``
:doxy:`importPGMDir`                import directory of PGM images                                  ``Pgm.h``
:doxy:`importPGMSet`                import set of PGM images                                        ``Pgm.h``
:doxy:`exportPGM`                   export single PGM image                                         ``Pgm.h``
=================================== =============================================================== =================
