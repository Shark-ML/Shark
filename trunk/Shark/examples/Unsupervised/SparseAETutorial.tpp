
//###begin<includes>
#include <shark/Data/Pgm.h>
#include <shark/Data/Csv.h>//for reading in the images as csv
#include <shark/ObjectiveFunctions/SparseFFNetError.h>//the error function performing the regularisation of the hidden neurons
#include <shark/Algorithms/GradientDescent/LBFGS.h>// the L-BFGS optimization algorithm
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h> // squard loss usd for regression
#include <shark/ObjectiveFunctions/Regularizer.h> //L2 rgulariziation
#include <shark/ObjectiveFunctions/CombinedObjectiveFunction.h> //binds together the regularizer with the Error
#include <shark/LinAlg/VectorStatistics.h> //for normalization
//###end<includes>

#include <fstream>
#include <boost/format.hpp>

using namespace std;
using namespace shark;

// Image info. The images are stored as w*h vectors, so we cannot derive
// w and h from the data.
//###begin<data_generation>
const unsigned int numsamples = 10000; //number of generated patches
const unsigned int w = 512;//width of loaded image
const unsigned int h = 512;//height of loaded image
const unsigned int psize = 8;//size of a patch
//###end<data_generation>

// FFNet parameters
//###begin<ffnet>
const unsigned int numhidden = 25;
const double rho = 0.01; // Sparsity parameter
const double beta = 6.0; // Regularization parameter
const double lambda = 0.0002; // Weight decay paramater
//###end<ffnet>

// Optimizer parameters
//###begin<train>
const unsigned int maxIter = 400;
//###end<train>

//###begin<get_samples_function>
UnlabeledData<RealVector> getSamples()
//###end<get_samples_function>
{
	//###begin<data_generation>
	// Read images
	UnlabeledData<RealVector> images;
	import_csv(images, "data/images.csv");
	unsigned int n = images.numberOfElements(); // number of images
	//###end<data_generation>
	cout << "Found " << n << " images of size " << w << "x" << h << endl;

	// Create the samples at random
	// Important notes: Since the images are in csv format, the width and
	// height is hardcoded. Because width = height we only have one integer
	// distribution below.
	
	//###begin<data_generation>
	// Sample equal amount of patches per image
	size_t patchesPerImg = numsamples / n;
	typedef UnlabeledData<RealVector>::element_range::iterator ElRef;
	
	// Create patches
	vector<RealVector> patches;
	for (ElRef it = images.elements().begin(); it != images.elements().end(); ++it) {
		for (size_t i = 0; i < patchesPerImg; ++i) {
			// Upper left corner of image
			unsigned int ulx = rand() % (w - psize);
			unsigned int uly = rand() % (h - psize);
			// Transform 2d coordinate into 1d coordinate and get the sample
			unsigned int ul = ulx * h + uly;
			RealVector sample(psize * psize);
			const RealVector& img = *it;
			for (size_t j = 0; j < psize; ++j)
				for (size_t k = 0; k < psize; ++k)
					sample(j * psize + k) = img(ul + k + j * h);
			patches.push_back(sample);
		}
	}
	
	UnlabeledData<RealVector> samples = createDataFromRange(patches);
	//###end<data_generation>
	
	//###begin<normalization>
	// zero mean
	RealVector meanvec = mean(samples);
	samples = transform(samples, Shift(-meanvec));

	// Remove outliers outside of +/- 3 standard deviations
	// and normalize to [0.1, 0.9]
	RealVector pstd = 3 * sqrt(variance(samples));
	samples = transform(samples, TruncateAndRescale(-pstd, pstd, 0.1, 0.9));
	//###end<normalization>
	
	return samples;
}

void setStartingPoint(FFNet<LogisticNeuron, LogisticNeuron>& model)
{
	// Set the starting point for the optimizer. This is 0 for all bias
	// weights and in the interval [-r, r] for non-bias weights.
	double r = sqrt(6) / sqrt(numhidden + psize * psize + 1);
	vector<RealMatrix>& layers = model.layerMatrices();
	for (int k = 0; k < 2; ++k)
		for (size_t i = 0; i < layers[k].size1(); ++i)
			for (size_t j = 0; j < layers[k].size2(); ++j)
				layers[k](i,j) = ((double)rand()/(double)RAND_MAX) * 2 * r - r;
	RealVector& bias = model.bias();
	for (size_t i = 0; i < bias.size(); ++i)
		bias(i) = 0.0;
}

//###begin<export>
void exportFeatureImages(const RealMatrix& W)
{
	// Export the visualized features.
	// Each row of W corresponds to a feature. Some normalization is done and
	// then it is transformed into a psize x psize image.
	boost::format filename("feature%d.pgm");

	// Create feature images
	for (size_t i = 0; i < W.size1(); ++i)
	{
		RealVector img(W.size2());
		for (size_t j = 0; j < W.size2(); ++j)
			img(j) = W(i,j);
		exportPGM((filename % i).str().c_str(), img, psize, psize, true);
	}
}
//###end<export>
int main()
{
	// Random needs a seed
	srand(time(NULL));
	
	// Read the data
	//###begin<create_dataset>
	UnlabeledData<RealVector> samples = getSamples();
	RegressionDataset data(samples, samples);
	//###end<create_dataset>
	cout << "Generated : " << samples.numberOfElements() << " patches." << endl;
	

	// Prepare the sparse network error function
	//###begin<ffnet>
	FFNet<LogisticNeuron, LogisticNeuron> model;
	model.setStructure(psize * psize, numhidden, psize * psize, true, false, false, true);
	//###end<ffnet>
	//###begin<sparsity_error>
	SquaredLoss<RealVector> loss;
	SparseFFNetError error(&model, &loss, rho, beta);
	error.setDataset(data);
	//###end<sparsity_error>

	// Add weight regularization
	//###begin<regularization>
	TwoNormRegularizer regularizer(error.numberOfVariables());
	CombinedObjectiveFunction<VectorSpace<double>, double> func;
	func.add(error);
	func.add(lambda, regularizer);
	//###end<regularization>

	cout << "Model has: " << model.numberOfParameters() << " params." << endl;
	cout << "Model has: " << model.numberOfNeurons() << " neurons." << endl;
	cout << "Model has: " << model.inputSize() << " inputs." << endl;
	cout << "Model has: " << model.outputSize() << " outputs." << endl;

	// Train it.
	//###begin<train>
	setStartingPoint(model);
	LBFGS optimizer;
	optimizer.lineSearch().lineSearchType() = LineSearch::WolfeCubic;
	optimizer.init(func, model.parameterVector());
	//###end<train>
	clock_t start = clock();
	//###begin<train>
	for (unsigned int i = 0; i < maxIter; ++i) {
		optimizer.step(func);
		cout << "Error: " << optimizer.solution().value << endl;
	}
	//###end<train>
	cout << "Elapsed time: " << (double)(clock() - start)/CLOCKS_PER_SEC << endl;
	cout << "Function evaluations: " << error.evaluationCounter() << endl;

	//###begin<export>
	exportFeatureImages(model.layerMatrices()[0]);
	//###end<export>
}
