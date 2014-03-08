//===========================================================================
/*!
 *
 *
 * \brief       Shark SVM training wrapper for binary linear and non-linear SVMs
 *
 *
 *
 * \author      Aydin Demircioglu
 * \date        2014
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
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
//===========================================================================


#include "boost/filesystem.hpp"
#include "boost/program_options.hpp"
#include <iostream>
#include <string>

#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/Data/Libsvm.h>
#include <shark/Data/Dataset.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h> //the used kernel for the SVM
#include <shark/Models/LinearClassifier.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>


using namespace shark;
using namespace std;


// sorry, do not see c++11 being enabled yet
class ErrorCodes
{
public:
	enum _ErrorCodes
	{
		SUCCESS, ERROR_IN_COMMAND_LINE, ERROR_UNHANDLED_EXCEPTION,
		ERROR_WRONG_KERNEL_TYPE, ERROR_WRONG_SVM_TYPE,
		ERROR_DATA,
		ERROR_MODELFILE
	};
};


class SVMTypes
{
public:
	enum _SVMTypes {CSVC};
};


class KernelTypes
{
public:
	enum _KernelTypes {LINEAR, POLYNOMIAL, RBF};
};



typedef LabeledData<RealVector, RealVector> LibSVMModelDataset;


// FIXME: refactor

/// \brief Export data to LIBSVM format.
///
/// \param  dataset     Container storing the  data
/// \param  fn          Output file
/// \param  dense       Flag for using dense output format
/// \param  oneMinusOne Flag for applying the transformation y<-2y-1 to binary labels
/// \param  sortLabels  Flag for sorting data points according to labels
/// \param  append      Flag for appending to the output file instead of overwriting it
void export_libsvmModel(LibSVMModelDataset model, const std::string &fn, bool append = false)
{
	std::size_t elements = model.numberOfElements();

	// shall we append only or overwrite?
	std::ofstream ofs;
	if(append == true)
	{
		ofs.open(fn.c_str(), std::fstream::out | std::fstream::app);
	}
	else
	{
		ofs.open(fn.c_str());
	}

	if(!ofs)
	{
		throw(SHARKEXCEPTION("[export_libsvm] file can not be opened for reading"));
	}


	for(std::size_t i = 0; i < elements; i++)
	{

		// write current labels to file
		for(std::size_t j = 0; j < labelDimension(model); j++)
		{
			if(model.labels().element(i)[j] != 0)
				ofs << setprecision(16) << model.labels().element(i)[j] << " ";
		}

		// write current input data to file
		for(std::size_t j = 0; j < inputDimension(model); j++)
		{
			if(model.inputs().element(i)[j] != 0)
				ofs << j + 1 << ":" << setprecision(16) << model.inputs().element(i)[j] << " ";
		}
		ofs << std::endl;
	}
}



int main(int argc, char** argv)
{
	std::cout << "Shark SVM train v0.1" << std::endl;
	std::cout << "Copyright 1995-2014 Shark Development Team" << std::endl << std::endl;
	try
	{
		std::string appName = boost::filesystem::basename(argv[0]);

		// parameter
		std::string trainingDataPath;
		std::string modelDataPath;

		// hyper-parameter
		double gamma = numeric_limits<double>::infinity();
		double cost = 0.0f;

		// meta-parameter
		int svmType = SVMTypes::CSVC;
		int kernelType = KernelTypes::RBF;
		uint64_t cacheSize = 0x1000000;

		// optimization parameter
		double epsilon = 0.001;
		bool bias = false;

		namespace po = boost::program_options;
		po::options_description desc("Options");
		desc.add_options()
		(",s", po::value<int>()->default_value(SVMTypes::CSVC), "svm_type : set type of SVM (default 0)\n        0 -- C-SVC")
		(",t", po::value<int>()->default_value(KernelTypes::RBF), "kernel_type : set type of kernel function (default 2)\n        0 -- linear: u'*v\n        2 -- radial basis function: exp(-gamma*|u-v|^2)")
		(",g", po::value<double>(), "gamma : set gamma in kernel function (default 1/num_features)")
		(",c", po::value<double>()->default_value(1.0), "cost : set the parameter C of C-SVC (default 1)")
		(",e", po::value<double>()->default_value(0.001), "epsilon : set tolerance of termination criterion (default 0.001)")
		(",i", po::value<bool>()->default_value(true), "bias : if set, a bias term will be used (default true)")
		(",m", po::value<uint64_t>()->default_value(0x1000000), "cachesize : set cache memory size in MB (default 128MB)")
		("training_set_file,", po::value<std::string>(&trainingDataPath)->required(), "path to training data file")
		("model_file,", po::value<std::string>(&modelDataPath), "path to model file");

		po::positional_options_description positionalOptions;
		positionalOptions.add("training_set_file", 1);
		positionalOptions.add("model_file", 1);

		po::variables_map vm;
		try
		{
			po::store(po::command_line_parser(argc, argv).options(desc).positional(positionalOptions).run(),  vm);
			po::notify(vm);

			/** --help option
			 */
			if(vm.count("help"))
			{
				std::cout << "Basic Command Line Parameter App" << std::endl
						  << desc << std::endl;
				return ErrorCodes::SUCCESS;
			}

		}
		catch(boost::program_options::required_option& e)
		{
			std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
			return ErrorCodes::ERROR_IN_COMMAND_LINE;
		}
		catch(boost::program_options::error& e)
		{
			std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
			std::cerr << desc << std::endl;
			return ErrorCodes::ERROR_IN_COMMAND_LINE;
		}

		// svm type
		if(vm.count("-s"))
		{
			svmType = vm["-s"].as<int>();
			switch(svmType)
			{
			case SVMTypes::CSVC:
				svmType = SVMTypes::CSVC;
				std::cout << "Type of SVM: C-SVC" << std::endl;
				break;
			default:
				std::cerr << "Unknown svm type. Refer to help to see the possible options. " << std::endl;
				return ErrorCodes::ERROR_WRONG_SVM_TYPE;
			}
		}

		// kernel type
		if(vm.count("-t"))
		{
			kernelType = vm["-t"].as<int>();
			switch(kernelType)
			{
			case KernelTypes::RBF:
				kernelType = KernelTypes::RBF;
				std::cout << "Type of Kernel: RBF" << std::endl;
				break;
			case KernelTypes::LINEAR:
				kernelType = KernelTypes::LINEAR;
				std::cout << "Type of Kernel: LINEAR" << std::endl;
				break;
			default:
				std::cerr << "Unknown kernel type. Refer to help to see the possible options. " << std::endl;
				return ErrorCodes::ERROR_WRONG_KERNEL_TYPE;
			}
		}

		// bias term
		if(vm.count("-i"))
		{
			bias = vm["-i"].as<bool>();
			std::cout << "Bias-term: " << bias << std::endl;
		}

		// gamma
		if(vm.count("-g"))
		{
			gamma = vm["-g"].as<double>();
			std::cout << "Gamma: " << gamma << std::endl;
		}

		// cost
		if(vm.count("-c"))
		{
			cost = vm["-c"].as<double>();
			std::cout << "Cost: " << cost << std::endl;
		}

		// epsilon
		if(vm.count("-e"))
		{
			epsilon = vm["-e"].as<double>();
			std::cout << "Epsilon: " << epsilon << std::endl;
		}

		// cache size
		if(vm.count("-m"))
		{
			// keep computations at float, just like libSVM
			cacheSize = vm["-m"].as<uint64_t>();
			std::cout << "CacheSize: " << std::setprecision(2) << (cacheSize * sizeof(float) / 1024 / 1024) << "MB" << std::endl;
		}


		// need to read the training file
		trainingDataPath = vm["training_set_file"].as<std::string>();
		std::cout << "Reading training data from " << trainingDataPath << std::endl;
		LabeledData<RealVector, unsigned int> trainingData;
		import_libsvm(trainingData, trainingDataPath);
		std::cout << "Data has " << trainingData.numberOfElements() << " points, input dimension " << inputDimension(trainingData) << std::endl;

		// check if we have a binary problem
		// FIXME: for other SVCs, fix this.
		if(numberOfClasses(trainingData) != 2)
		{
			//
			std::cerr << "Data has either only one or multiple classes. Only binary data is supported for binary C-SVCs." << std::endl;
			return ErrorCodes::ERROR_DATA;
		}

		// if we had no gamma, we need to set it
		if(gamma == numeric_limits<double>::infinity())
		{
			gamma = 1. / trainingData.numberOfElements();
			std::cout << "No gamma specified, will compute it from training data." << std::endl;
			std::cout << "Gamma: " << gamma << std::endl;
		}

		// create c-svm trainer
		if(svmType == SVMTypes::CSVC)
		{

			// now we follow the tutorial-

			// FIXME: many more switches depending on kernel blabla

			GaussianRbfKernel<> kernel(gamma);
			KernelClassifier<RealVector> kc; // (affine) linear function in kernel-induced feature space

			CSvmTrainer<RealVector> trainer(&kernel, cost, bias);
			trainer.stoppingCondition().minAccuracy = epsilon;
			trainer.setCacheSize(cacheSize);

			// now do training
			std::cout << "Starting training (Depending on the dataset, this might take several hours, days or month)\n";
			trainer.train(kc, trainingData);

			std::cout << "Needed " << trainer.solutionProperties().seconds << " seconds to reach a dual of "
					  << std::setprecision(16) << trainer.solutionProperties().value << std::endl;

			// generate some meta data FIXME: for multiclass
			uint64_t totalSV = 0;
			uint64_t totalPosSV = 0;
			uint64_t totalNegSV = 0;

			// if bias is used, get value FIXME
			double rho = 0.0;
			if(bias == true)
			{
				rho = kc.decisionFunction().offset()[0];
				std::cout << "Bias: " << rho << std::endl;
			}

			// save model file
			if(vm.count("model_file"))
			{
				modelDataPath = vm["model_file"].as<std::string>();

				// prepare the model
				Data<RealVector> supportVectors = kc.decisionFunction().basis();
				RealMatrix tmpAlphas = kc.decisionFunction().alpha();

				// convert realmatrix to data of realvectors
				Data<RealVector> alphas(tmpAlphas.size1(), row(tmpAlphas, 0));

				// TODO: can this be done better?
				for(int j = 0; j < tmpAlphas.size1(); ++j)
				{
					alphas.element(j) = row(tmpAlphas, j);

					// FIXME: this is for the binary case only...
					if(alphas.element(j)[0] > 0) totalPosSV ++;
					else totalNegSV++;
				}

				// create model
				LibSVMModelDataset model(supportVectors, alphas);

				totalSV = alphas.numberOfElements();
				std::cout << "Total number of support vectors: " << totalSV << std::endl;


				std::cout << "Saving model to " << modelDataPath << std::endl;

				// create header
				std::ofstream modelDataStream;
				modelDataStream.open(modelDataPath.c_str());

				if(!modelDataStream)
				{
					std::cerr << "Cannot open " << modelDataPath << "for writing!" << std::endl;
					return ErrorCodes::ERROR_MODELFILE;
				}

				// write header data FIXME multiclass

				// type
				modelDataStream << "svm_type c_svc" << std::endl;

				// kernel
				modelDataStream << "kernel_type rbf" << std::endl;

				// gamma
				modelDataStream << "gamma " << gamma << std::endl;
				modelDataStream << "nr_class " << 2 << std::endl;
				modelDataStream << "total_sv " << totalSV << std::endl;
				modelDataStream << "rho " << rho << std::endl;
				modelDataStream << "label 1 -1" << std::endl;
				modelDataStream << "nr_sv " << totalPosSV << " " << totalNegSV << std::endl;
				modelDataStream << "SV " << std::endl;

				// then append SVs
				export_libsvmModel(model, modelDataPath, true);
			}

			// finally do some reporting
			ZeroOneLoss<unsigned int> loss; // 0-1 loss
			Data<unsigned int> output = kc(trainingData.inputs()); // evaluate on training set
			double train_error = loss.eval(trainingData.labels(), output);
			std::cout << "Training error:\t" <<  train_error << endl;
		}
	}
	catch(std::exception& e)
	{
		std::cerr << "Unhandled Exception occured." << std::endl
				  << e.what() << ", application will now exit" << std::endl;
		return ErrorCodes::ERROR_UNHANDLED_EXCEPTION;

	}

	return ErrorCodes::SUCCESS;
} // main

