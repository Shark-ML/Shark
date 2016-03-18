/*!
 * 
 *
 * \brief       Illustration of the OpenML component.
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2016
 *
 *
 * \par Copyright 1995-2016 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
 * 
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <shark/OpenML/OpenML.h>
#include <shark/Data/Arff.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Algorithms/Trainers/McSvmOVATrainer.h>
#include <iostream>

using namespace shark;
using namespace std;


void printIDs(openML::IDList const& ids)
{
	cout << "IDs [" << ids.size() << "]";
	for (size_t i=0; i<min((size_t)10, ids.size()); i++) cout << " " << ids[i];
	if (ids.size() > 10) cout << " ...";
	cout << endl;
}


int main(int argc, char** argv)
{
	// The following line sets the OpenML api key to the Shark library's
	// demo account. This account is for tutorial demonstration only.
	// It is a read-only key, which does not allow to make changed to
	// the OpenML system.
	// NOTE: Always use your own api key (attached to your OpenML account)
	// for actual experiments. Otherwise your results will be lost and you
	// cannot receive credit for your work. Creating new flows and runs
	// with this key will silently fail.
	string api_key = "8d736266baa96f8ef99f10516911d334";

	// use the user's api key instead (if provided)
	if (argc > 1) api_key = argv[1];

	// register the api key in the global openML::connection object
	openML::connection.setKey(api_key);

	// query a list of tasks
//	openML::IDList taskIDs = openML::supervisedClassificationTasks();
	// TODO: augment IDs with properties dictionary, add filtering mechanisms
	shark::openML::IDType taskID = 11;   // this should be the result of a query at some point in the future

	// instantiate the chosen task
	shared_ptr<openML::Task> task = openML::Task::get(taskID);
	task->print();

	// obtain the data set underlying the task
	shared_ptr<openML::Dataset> dataset = task->dataset();
	dataset->print();

	// setup a learning machine to solve the task
	double C = 1.0;
	double gamma = 1.0;
	GaussianRbfKernel<RealVector> kernel(gamma);
	McSvmOVATrainer<RealVector> trainer(&kernel, C, false);

	// define a flow representing the setup
	std::string flowName = trainer.name() + "." + kernel.name();
	std::vector<openML::Hyperparameter> params;
	params.push_back(openML::Hyperparameter("C", "double", "regularization parameter, must be positive"));
	params.push_back(openML::Hyperparameter("gamma", "double", "kernel bandwidth parameter, must be positive"));
	params.push_back(openML::Hyperparameter("bias", "bool", "presence or absence of the bias 'b' in the model"));
	shared_ptr<openML::Flow> flow = openML::Flow::get(flowName, "one-versus-all C-SVM with Gaussian RBF kernel", params);
	flow->print();

	// create a run object representing the results
	openML::Run run(task, flow);
	run.setHyperparameterValue("C", trainer.C());                // ideally this would be automated
	run.setHyperparameterValue("gamma", kernel.gamma());         // ideally this would be automated
	run.setHyperparameterValue("bias", trainer.trainOffset());   // ideally this would be automated
	run.print();

	// execute the learning machine and fill the run with predictions
	cout << "training and predicting " << flush;
	ClassificationDataset data;
	task->loadData(data);
	for (std::size_t r=0; r<task->repetitions(); r++)
	{
		CVFolds< LabeledData<RealVector, unsigned int> > folds = task->split(r, data);
		for (std::size_t f=0; f<task->folds(); f++)
		{
			ClassificationDataset traindata = folds.training(f);
			ClassificationDataset testdata  = folds.validation(f);
			KernelClassifier<RealVector> model;
			trainer.train(model, traindata);
			Data<unsigned int> predictions = model(testdata.inputs());
			run.storePredictions(r, f, predictions);
			cout << "." << flush;
		}
	}
	cout << " done." << endl;

	// upload the results to OpenML
	run.commit();

	// tag the run
	run.tag("shark-tutorial-test-tag");
	run.print();

	// untag it again
	run.untag("shark-tutorial-test-tag");
	run.print();
}
