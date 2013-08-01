#include <fstream>

#include <shark/Data/Pgm.h>
#include <shark/Data/Csv.h>
#include <shark/ObjectiveFunctions/SparseFFNetError.h>
#include <shark/Algorithms/GradientDescent/LBFGS.h>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/ObjectiveFunctions/Regularizer.h>
#include <shark/ObjectiveFunctions/CombinedObjectiveFunction.h>
#include <shark/LinAlg/VectorStatistics.h>

#include <boost/format.hpp>

using namespace std;
using namespace shark;

// Image info. The images are stored as w*h vectors, so we cannot derive
// w and h from the data.
const unsigned int numsamples = 10000;
const unsigned int w = 512;
const unsigned int h = 512;
const unsigned int psize = 8;
const unsigned int numhidden = 25;

// FFNet parameters
const double rho = 0.01; // Sparsity parameter
const double beta = 6.0; // Regularization parameter
const double lambda = 0.0002; // Weight decay paramater

// Optimizer parameters
const unsigned int maxIter = 400;

void getSamples(UnlabeledData<RealVector>& samples)
{
	// Read images
	UnlabeledData<RealVector> images;
	import_csv(images, "data/images.csv");

	// Create patches
	vector<RealVector> patches;

	unsigned int n = images.numberOfElements(); // number of images

	cout << "Found " << n << " images of size " << w << "x" << h << endl;

	// Create the samples at random
	// Important notes: Since the images are in csv format, the width and
	// height is hardcoded. Because width = height we only have one integer
	// distribution below.

	// Sample equal amount of patches per image
	size_t patchesPerImg = numsamples / n;
	typedef UnlabeledData<RealVector>::element_range::iterator ElRef;

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

	// Create data set and normalize.
	samples = createDataFromRange(patches);
	// zero mean
	RealVector meanvec = mean(samples);
	samples = transform(samples, Shift(-meanvec));

	// Remove outliers outside of +/- 3 standard deviations
	// and normalize to [0.1, 0.9]
	RealVector pstd = 3 * sqrt(variance(samples));
	samples = transform(samples, TruncateAndRescale(-pstd, pstd, 0.1, 0.9));
}

RealVector setStartingPoint(FFNet<LogisticNeuron, LogisticNeuron>& model)
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

	// Return the starting point for use in the optimizer
	return model.parameterVector();
}

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

int main()
{
	// Random needs a seed
	srand(time(NULL));
	// Read the data
	UnlabeledData<RealVector> samples;
	getSamples(samples);
	cout << "Found : " << samples.numberOfElements() << " patches." << endl;
	RegressionDataset data(samples, samples);

	// Prepare the sparse network error function
	FFNet<LogisticNeuron, LogisticNeuron> model;
	model.setStructure(psize * psize, numhidden, psize * psize, true, false, false, true);
	SquaredLoss<RealVector> loss;
	SparseFFNetError error(&model, &loss, rho, beta);
	error.setDataset(data);

	// Add weight regularization
	TwoNormRegularizer regularizer(error.numberOfVariables());
	CombinedObjectiveFunction<VectorSpace<double>, double> func;
	func.add(error);
	func.add(lambda, regularizer);

	cout << "Model has: " << model.numberOfParameters() << " params." << endl;
	cout << "Model has: " << model.numberOfNeurons() << " neurons." << endl;
	cout << "Model has: " << model.inputSize() << " inputs." << endl;
	cout << "Model has: " << model.outputSize() << " outputs." << endl;

	// Train it.
	RealVector point = setStartingPoint(model);
	LBFGS optimizer;
	optimizer.lineSearch().lineSearchType() = LineSearch::WolfeCubic;
	optimizer.init(func, point);
	clock_t start = clock();

	for (unsigned int i = 0; i < maxIter; ++i) {
		optimizer.step(func);
		cout << "Error: " << optimizer.solution().value << endl;
	}
	cout << "Elapsed time: " << (double)(clock() - start)/CLOCKS_PER_SEC << endl;
	cout << "Function evaluations: " << error.evaluationCounter() << endl;


	const std::vector<RealMatrix>& layers = model.layerMatrices();
	exportFeatureImages(layers[0]);
}
