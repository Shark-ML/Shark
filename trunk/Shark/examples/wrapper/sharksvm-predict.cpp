//===========================================================================
/*!
 *
 *
 * \brief       Shark SVM predict wrapper for binary linear and non-linear SVMs
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

#include <shark/Data/Dataset.h>
#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/Data/Libsvm.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h> //the used kernel for the SVM
#include <shark/Models/LinearClassifier.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <boost/spirit/include/qi.hpp>



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
// ..LibSVMModelDataset model 
LibSVMModelDataset read_libsvmModel(const std::string &fn, double &gamma, double &rho)
{
    std::ifstream stream(fn.c_str());
    if (! stream.good()) 
        throw SHARKEXCEPTION("[shark::read_libsvmModel] failed to open file for input");


    unsigned int nr_classes = 0;
    unsigned int total_sv = 0;
    unsigned int nr_sv = 0;
    
    while(stream)
    {
        using namespace boost::spirit::qi;

        std::string line;
        std::getline(stream, line);
        if(line.empty()) continue;

        std::string::const_iterator first = line.begin();
        std::string::const_iterator last = line.end();

        vector<std::string> contents;

        const bool result = parse(first, last, 
            +(char_-' ') % space,
            contents);                                  

        // FIXME: this is stupid, but works.
        if (!contents.empty())
        {
            if (contents[0] == "SV") {
                break;
            }
            
            if (contents[0] == "rho") {
                rho = boost::lexical_cast<double>(contents[1]);
                std::cout << "rho:" << contents[1] << endl;
            }
            
            if (contents[0] == "svm_type") {
                // TODO: more than this
                if (contents[1] != "c_svc") 
                    throw SHARKEXCEPTION("[import_libsvm_reader] unsupported svm type!");
            }

            if (contents[0] == "kernel_type") {
                // TODO: more than this
                if (contents[1] != "rbf") 
                    throw SHARKEXCEPTION("[import_libsvm_reader] unsupported kernel type!");
            }

            if (contents[0] == "gamma") {
                gamma = boost::lexical_cast<double>(contents[1]);
                std::cout << "gamma:" << gamma << endl;
            }

            if (contents[0] == "nr_classes") {
                nr_classes = boost::lexical_cast<unsigned int>(contents[1]);
            }

            if (contents[0] == "total_sv") {
                total_sv = boost::lexical_cast<unsigned int>(contents[1]);
            }

            if (contents[0] == "nr_sv") {
                nr_sv = boost::lexical_cast<unsigned int>(contents[1]);
            }
            continue;
        }
        
        if(!result || first != last)
            throw SHARKEXCEPTION("[import_libsvm_reader] problems parsing file");
    }
    
    // read the alphas and the support vectors now..
    std::cout << "Reading SVs.." << std::endl;

    typedef std::pair<std::vector<double>, std::vector<std::pair<std::size_t, double> > > LibSVMPoint;
    std::vector<LibSVMPoint> contents;
    while(stream)
    {
        std::string line;
        std::getline(stream, line);
        if(line.empty()) continue;

        using namespace boost::spirit::qi;
        std::string::const_iterator first = line.begin();
        std::string::const_iterator last = line.end();

        // FIXME: does only read one alpha..
        LibSVMPoint newPoint;
        bool r = phrase_parse(
                     first, last,
                     double_    
                     >> *(uint_ >> ':' >> double_),
                     space , newPoint
                 );
        
        if (!r || first != last)
        {
            std::cout << std::string(first, last) << std::endl;
            throw SHARKEXCEPTION("[import_libsvm_reader] problems parsing file");

        }

        contents.push_back(newPoint);
    }

    //read contents of stream
    std::size_t numPoints = contents.size();


    // find dimension of alphas
    std::size_t alphaDimension = 0;
    for(std::size_t i = 0; i != numPoints; ++i)
    {
        std::vector<double> const& inputs = contents[i].first;
        if(!inputs.empty())
            alphaDimension = std::max(alphaDimension, inputs.size());
    }
    
    // find dimension of data
    std::size_t dataDimension = 0;
    for(std::size_t i = 0; i != numPoints; ++i)
    {
        std::vector<std::pair<std::size_t, double> > const& inputs = contents[i].second;
        if(!inputs.empty())
            dataDimension = std::max(dataDimension, inputs.back().first);
    }


    // check for feature index zero (non-standard, but it happens)
    bool haszero = false;
    for(std::size_t i = 0; i < numPoints; i++)
    {
        std::vector<std::pair<std::size_t, double> > const& input = contents[i].second;
        if(input.empty()) continue;
        if(input[0].first == 0)
        {
            haszero = true;
            break;
        }
    }

    // feature 0 means more input dimensions
    if (haszero == true)
        dataDimension = dataDimension + 1;
    
    // copy contents into a new dataset
    size_t batchSize = 0;
    typename LibSVMModelDataset::element_type blueprint(RealVector(dataDimension), RealVector(alphaDimension));

    
    // FIXME: could not get it to work otherwise. 
    std::vector <RealVector> tmp_alphas; 
    for (size_t l = 0; l < contents.size(); l++)
    {
        RealVector tmp (contents[l].first.size());
        std::copy(contents[l].first.begin(), contents[l].first.end(), tmp.begin());
        tmp_alphas.push_back(tmp);
    }

    std::vector <RealVector> tmp_supportVectors; 
    size_t delta = (haszero ? 0 : 1);
    for (size_t l = 0; l < contents.size(); l++)
    {
        RealVector tmp (dataDimension);

        // copy features
        std::vector<std::pair<std::size_t, double> > const& inputs = contents[l].second;
        for(std::size_t j = 0; j != inputs.size(); ++j)
        {
            tmp[inputs[j].first - delta] = inputs[j].second;
        }
        tmp_supportVectors.push_back(tmp);
    }
    
    
    LibSVMModelDataset data (createDataFromRange(tmp_supportVectors), createDataFromRange(tmp_alphas));
    return data;
}




int main(int argc, char** argv)
{
	std::cout << "Shark SVM predict v0.1" << std::endl;
	std::cout << "Copyright 1995-2014 Shark Development Team" << std::endl << std::endl;
	try
	{
		std::string appName = boost::filesystem::basename(argv[0]);

		// parameter
        std::string testDataPath;
		std::string modelDataPath;
		std::string predictionScoreFilePath;

		bool probabilityEstimates = false;

		namespace po = boost::program_options;
		po::options_description desc("Options");
		desc.add_options()
		(",b", po::value<bool>()->default_value(false), "probability_estimates: whether to predict probability estimates (default false); for one-class SVM only 0 is supported")
        ("test_file,", po::value<std::string>(&testDataPath)->required(), "path to test file")
		("model_file,", po::value<std::string>(&modelDataPath)->required(), "path to model file")
		("prediction_score_file,", po::value<std::string>(&predictionScoreFilePath), "path to prediction score output file");

		po::positional_options_description positionalOptions;
        positionalOptions.add("test_file", 1);
        positionalOptions.add("model_file", 1);
		positionalOptions.add("prediction_score_file", 1);

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

		// bias term
		if(vm.count("-b"))
		{
			// FIXME: not supported yet.
			probabilityEstimates = vm["-b"].as<bool>();
			std::cout << "probabilityEstimates: " << probabilityEstimates << std::endl;
		}

        modelDataPath = vm["model_file"].as<std::string>();
        std::cout << "Reading model data from " << modelDataPath << std::endl;

		// read model
        double gamma = 1.0;
        double bias = 0.0;
        LibSVMModelDataset libSVMModel = read_libsvmModel(modelDataPath, gamma, bias);

        // need to convert the model to something  shark understands
        GaussianRbfKernel<> kernel(gamma);
        KernelExpansion<RealVector> sharkModel (&kernel, libSVMModel.inputs(), true);
        
        // FIXME: convert back... binary only again
        RealVector parameters (libSVMModel.labels().numberOfElements() + 1);
        for (size_t l = 0; l < parameters.size() - 1; l++) 
        {
            parameters[l] = libSVMModel.labels().element(l)[0];
        }
        parameters[parameters.size()-1] = bias;
        sharkModel.setParameterVector (parameters);
        
        // create a classifier with the model
        KernelClassifier<RealVector> kc (sharkModel);
        
        
        // read test data
        testDataPath = vm["test_file"].as<std::string>();
        std::cout << "Reading test data from " << testDataPath << std::endl;
        LabeledData<RealVector, unsigned int> testData;
        import_libsvm(testData, testDataPath);
        std::cout << "Data has " << testData.numberOfElements() << " points, input dimension " << inputDimension(testData) << std::endl;

        
		// do prediction
        ZeroOneLoss<unsigned int> loss;
        Data<unsigned int> prediction = kc(testData.inputs());
        double error_rate = loss(testData.labels(), prediction);
        std::cout << "Test error rate: " << error_rate << std::endl;

        
		// save predictionScore

	}
	catch(std::exception& e)
	{
		std::cerr << "Unhandled Exception occured." << std::endl
				  << e.what() << ", application will now exit" << std::endl;
		return ErrorCodes::ERROR_UNHANDLED_EXCEPTION;

	}

	return ErrorCodes::SUCCESS;




} // main



