Serialization
=============

Most objects in Shark can be serialized, meaning that their internal
state can be transferred from and to a stream, e.g., for loading and
saving. This short tutorial demonstrates how to use this feature.

Let us start with a basic machine learning example, similar to the one
developed in the :doc:`../../algorithms/svm` tutorial::

	#include <shark/Algorithms/Trainers/SvmTrainer.h>
	#include <shark/Models/Kernels/GaussianRbfKernel.h>
	#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
	#include <shark/Data/DataDistribution.h>
	#include <fstream>

	using namespace shark;
	using namespace std;

	int main(int argc, char** argv)
	{
		// generate synthetic data
		Chessboard prob;
		ClassificationDataset training(prob, 500);

		// define a model
		GaussianRbfKernel<> kernel(0.5, true);
		KernelExpansion<RealVector> ke(&kernel, true);

		// train the model
		CSvmTrainer<RealVector> trainer(&kernel, 1000.0);
		trainer.train(&ke, training);

		// evaluate the trained model on the training set
		Data<RealVector> output;
		ke.eval(training.inputs(), output);
		ZeroOneLoss<unsigned int, RealVector> loss;
		double trainError = loss.eval(training.labels(), output);
		cout << "training error of the original model:\t" << trainError << endl;
	}

This program trains a support vector machine and outputs its training
error. Now let's assume we want to store the trained model for later
use, e.g., as a recovery point in a long running process. We extend the
above program::

		// save the model to the file "svm.model"
		ofstream ofs("svm.model");
		boost::archive::polymorphic_text_oarchive oa(ofs);
		ke.write(oa);
		ofs.close();

Shark makes heavy use of templates. This has many great advantages,
but in this case makes life a bit harder. The kernel expansion model
internally holds a list of all support vectors, and they are objects of
an arbitrary type that comes as a template argument. In other words, the
``KernelExpansion`` code does not know anything about this type and how
to serialize it. Now, this unknown and possibly user defined type needs
to be serialized to a file, since it is an important part of the model's
state. This is where the serialization capability of boost comes into
play, since the boost serialization library offers a principled solution
to this problem.

Use of this feature is easy. We construct a boost archive object and
call the ``write`` method of the kernel expansion. The model stores its
internal state in the archive. Another interesting aspect of this
construction is the handling of the kernel parameters, in this case the
bandwidth parameter of the Gaussian RBF kernel. This parameter has been
set to 0.5 in the above example, and since the kernel is an integral
part of the kernel expansion, the kernel state it stored alongside the
other parameters.

Now let's assume disaster has happened: our long running process was
killed, maybe by a power outage. We are lucky, because we have stored
the kernel expansion model to disk. So let's continue the process with
the stored model, instead of going through the possibly lengthy training
process again::

		// load the file "svm.model" into a new model
		GaussianRbfKernel<> kernelLoad(true);
		KernelExpansion<RealVector> keLoad(&kernelLoad, true);
		ifstream ifs("svm.model");
		boost::archive::polymorphic_text_iarchive ia(ifs);
		keLoad.read(ia);
		ifs.close();

That's all. We construct a boost archive for input and invoke the
``read`` method of a fresh kernel expansion model. Note that we have
provided the kernel expansion object already with the right type of
kernel object, but we have not set its parameters. All parameters
(support vectors, weights and bias of the kernel expansion and bandwidth
of the kernel) are restored from disk, and the model is straight away
ready for evaluation::

		// evaluate the loaded model on the training set
		keLoad.eval(training.inputs(), output);
		trainError = loss.eval(training.labels(), output);
		cout << "training error of the loaded model:\t" << trainError << endl;
