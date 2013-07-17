
Hello World
===========

In this section you will write your first program with Shark.

LDA classification
------------------

Using a Linear Discriminant Analysis (LDA) as Hello-World (or
Hello-Shark) program, we will try to separate two classes of inputs
using a simple linear function. The code for this tutorial can be
found in :download:`quickstartTutorial.cpp
<../../../../../examples/Supervised/quickstartTutorial.cpp>` in the
``examples/Supervised`` folder.

In order to access the LDA, we include its header file and import the Shark
namespace for convenience. We will also need the header for importing CSV files::

    #include <shark/Data/Csv.h>
    #include <shark/Algorithms/Trainers/LDA.h>
    using namespace shark;
    using namespace std;   

Data preparation
%%%%%%%%%%%%%%%%

Next we would like some data to classify. We can use the :doxy:`Dataset`
class for holding the data, and load the data with ``import_csv``::

    ClassificationDataset data;
    import_csv( data, "data/quickstartData.csv", LAST_COLUMN, " " );

The first line creates a dataset, the data structure used in Shark for holding
data for supervised learning tasks. Such containers hold pairs
of input data points and labels. The ``ClassificationDataset`` used here is
simply a typedef for
``LabeledData < RealVector, unsigned int >``, as we will use real-valued feature
vectors and integer labels (see the :ref:`LinAlg tutorials <label_for_linalg_tutorials>`
for more information on ``RealVector``). The second line loads the file ``quickstartData.csv``.
If you open that file you will see that it contains the label information in the last
column while the preceding two columns represent the input data points. For this reason,
the argument ``LAST_COLUMN`` was passed to ``import_csv``. The data formats supported by
Shark are in detail described in the :ref:`data tutorials <label_for_data_tutorials>`.

We want to use only one part of all available data for training, and
set aside another part for testing. The next line splits (i.e.,
removes) a test set from our loaded data and stores it inside a new
dataset. We choose the training set to be 80% of the available data::

    ClassificationDataset test = splitAtElement(data,0.8*data.numberOfElements());

After this operation, ``data`` is only 80% of its former size, and ``test`` holds the
remaining 20%. See the :ref:`data tutorials <label_for_data_tutorials>` for similar
such operations.

Declaring a model and trainer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Shark strictly separates the concepts of models and optimizers. This means that we
need to declare a learning algorithm and the right model for it separately (see more
information about the building blocks and implementation design rationales in the
concept tutorial :doc:`../concepts/optimization/optimizationtrainer`). Since the LDA
uses a linear classifier, we declare an instance of such. It is not needed to specify
the dimensionality of the input data or the number of classes of our problem, the LDA
will infer this from the training data itself. The LDA is a separate entity playing
the role of a trainer. ::

    LinearClassifier classifier;
    LDA lda;

To optimize the model given the training data and using a specific trainer we write ::

    lda.train ( classifier, data );

After this call, our classifier can directly be used to classify data. But we do not
know how good our solution is, so let's evaluate it.


Evaluation
%%%%%%%%%%

One way to evaluate our LDA-trained linear model is to count the number of
correctly classified test samples. We simply use ``BOOST_FOREACH`` to iterate
over all key-value pairs of the dataset::

    unsigned int correct = 0;
    BOOST_FOREACH(ClassificationDataset::element_reference point, test.elements()){
        unsigned int result = classifier(point.input);
        if (result == point.label){
            correct++;
        }
    }

Easier, faster, and more flexible ways to evaluate models are facilitated by Shark
losses and error functions, which will be introduced in the next tutorials.
In order to print the results (do not forget to include ``iostream``), issue::

    cout << "RESULTS: " << endl;
    cout << "========\n" << endl;
    cout << "test data size: " << test.numberOfElements()<< endl;
    cout << "correct classification: " << correct << endl;
    cout << "error rate: " << 1.0 - double(correct)/test.numberOfElements() << endl;

The result should read:

.. code-block:: none

    RESULTS:
    ========

    test data size: 200
    correct classification: 155
    error rate: 0.225

What you learned
----------------

You should have learned the following aspects in this Tutorial:

* What the main building blocks of a general optimization task are: Data, Error Function, Model, Optimizer

* How to load data from from a csv file.

During the course of all tutorials, you will gain a more fine grained knowledge
about these different aspects.

What next?
----------

In the next tutorial we will investigate how :doc:`general_optimization_tasks` are set up, which gives
you a deeper understanding of the main building blocks of Shark.



