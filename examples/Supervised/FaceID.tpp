//###begin<data_includes>
#include <shark/Core/ZipSupport.h> //for the ZipReader class
#include <shark/Data/DataDistribution.h> //for random sampling of paths in the zip
#include <shark/Core/Images/ReadImage.h> //for reading the images
#include <shark/Core/Images/Resize.h> //for resizing the images
//###end<data_includes>
#include <shark/Models/LinearModel.h>//single dense layer
#include <shark/Models/ConvolutionalModel.h>//single convolutional layer
#include <shark/Models/PoolingLayer.h> //pooling after convolution
#include <shark/Models/ConcatenatedModel.h>//for stacking layers
#include <shark/Algorithms/GradientDescent/Adam.h>// The Adam optimization algorithm
#include <shark/ObjectiveFunctions/Loss/CrossEntropy.h> //classification loss
#include <shark/ObjectiveFunctions/ErrorFunction.h> //Error function for optimization
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h> //evaluation for testing
using namespace shark;
//###end<includes>


//###begin<define_face_pair>
struct FacePair{
	FloatVector image1;
	FloatVector image2;
};
//###end<define_face_pair>
//generates the batch magic. A batch of FacePair is equivalent to
//struct FacePairBatch{
//     FloatMatrix image1;
//     FloatMatrix image2;
//};
//this can be accessed via Batch<FacePair>::type
//it also generates a Batch<FacePair>::shape_type
//with the same structure, just with the shape of the elements, i.e.
//struct FacePairShape{
//     Shape image1;
//     Shape image2;
//};
//Shapes can become very complicated entities if you nest many of those
//classes.
//###begin<define_face_pair_batch>
namespace shark{
template<>
SHARK_CREATE_BATCH_INTERFACE(
	FacePair,
	(FloatVector, image1)(FloatVector, image2)
)
}
//###end<define_face_pair_batch>


//###begin<define_face_pair_model_base>
class FaceIdModel: public AbstractModel<FacePair, FloatVector, FloatVector>{
public:
	FaceIdModel(
		AbstractModel<FloatVector, FloatVector, FloatVector>* map, 
		AbstractModel<FloatVector, FloatVector, FloatVector>* classifier
	): m_map(map), m_classifier(classifier){
		SHARK_RUNTIME_CHECK(
			map->outputShape().numberOfElements() == classifier->inputShape().numberOfElements(), 
			"Shapes Incompatible!"
		);
		this->m_features |= HAS_FIRST_PARAMETER_DERIVATIVE;
	}
//###end<define_face_pair_model_base>
	
	//handling of parameters
//###begin<define_face_pair_model_parameters>
	std::size_t numberOfParameters() const{
		return m_map->numberOfParameters() + m_classifier->numberOfParameters();
	}
	ParameterVectorType parameterVector() const {
		return m_map->parameterVector() | m_classifier->parameterVector();
	}

	void setParameterVector(ParameterVectorType const& params){
		std::size_t cutPos = m_map->numberOfParameters();
		m_map->setParameterVector(subrange(params, 0, cutPos));
		m_classifier->setParameterVector(subrange(params, cutPos, params.size()));
	}
//###end<define_face_pair_model_parameters>

//###begin<define_face_pair_model_shapes>	
	//shape
	///\brief Returns the expected shape of the input
	shape_type<FacePair>::type inputShape() const{
		return {m_map->inputShape(), m_map->inputShape()};
	}
	///\brief Returns the shape of the output
	Shape outputShape() const{
		return m_classifier->outputShape();
	}
//###end<define_face_pair_model_shapes>		
	//handling of backpropagation state
	
//###begin<define_face_pair_model_state>	
private:
	struct InternalState: public State{
		boost::shared_ptr<State> mapState1;
		boost::shared_ptr<State> mapState2;
		boost::shared_ptr<State> classifierState;
		FloatMatrix hidden1;
		FloatMatrix hidden2;
		FloatMatrix diffOfHidden;
	};
public:
	boost::shared_ptr<State> createState()const{
		InternalState* newState = new InternalState();
		newState->mapState1 = m_map->createState();
		newState->mapState2 = m_map->createState();
		newState->classifierState = m_classifier->createState();
		return boost::shared_ptr<State>(newState);
	}
//###end<define_face_pair_model_state>	
	//implement evaluation of network response, aka forward propagation
	//inputs are batches of FacePair, output are FloatMatrix
//###begin<define_face_pair_model_eval>	
	void eval(BatchInputType const& inputs, BatchOutputType& outputs, State& state)const{
		InternalState& s = state.toState<InternalState>();//get access to internal state
//###end<define_face_pair_model_eval>		
		//map images to their hidden representation and store internal state for backpropagation
		//map images to their hidden representation and store internal state for backpropagation
		//we do this asynchronously as map is an expensive operation
//###begin<define_face_pair_model_parallel_eval>
		auto& pool = threading::globalThreadPool();
		std::future<void> event1 = pool.execute_async([&](){
//###begin<define_face_pair_model_eval>
			m_map->eval(inputs.image1, s.hidden1, *(s.mapState1));
//###end<define_face_pair_model_eval>
		});
		std::future<void> event2 = pool.execute_async([&](){
//###begin<define_face_pair_model_eval>
			m_map->eval(inputs.image2, s.hidden2, *(s.mapState2));
//###end<define_face_pair_model_eval>
		});
		pool.wait_for_all(event1,event2);//wait for results
//###end<define_face_pair_model_parallel_eval>		
		//evaluate the response of the classifier on the difference of representations
//###begin<define_face_pair_model_eval>
		s.diffOfHidden = s.hidden1 - s.hidden2;
		m_classifier->eval(s.diffOfHidden, outputs, *(s.classifierState));
	}
//###end<define_face_pair_model_eval>	
	
	//implement backpropagation
//###begin<define_face_pair_model_derivative>
	void weightedParameterDerivative(
		BatchInputType const& inputs, //as in eval
		BatchOutputType const& outputs, //as in eval 
		BatchOutputType const& coefficients,//backpropagated coefficients
		State const& state, //state computed in eval
		ParameterVectorType& gradient //parameter gradient, same structure as parameterVector()
	)const{
		InternalState const& s = state.toState<InternalState>();//get access to internal state
		
		//compute both parameter and input derivative of the classifier
		FloatVector classifierParameterGrad;
		FloatMatrix classifierInputGrad;
		m_classifier->weightedDerivatives(
			s.diffOfHidden, outputs, 
			coefficients, *(s.classifierState),
			classifierParameterGrad, classifierInputGrad
		);
		//backpropagate to the maps
		FloatVector mapParameterGrad1;
		FloatVector mapParameterGrad2;
//###end<define_face_pair_model_derivative>
		//we do this asynchronously in parallel
//###begin<define_face_pair_model_parallel_derivative>		
		auto& pool = threading::globalThreadPool();
		std::future<void> event1 = pool.execute_async([&](){
//###begin<define_face_pair_model_derivative>
			m_map->weightedParameterDerivative(
				inputs.image1, s.hidden1, 
				classifierInputGrad, *(s.mapState1),
				mapParameterGrad1
			);
//###end<define_face_pair_model_derivative>
		});
		
		std::future<void> event2 = pool.execute_async([&](){
//###begin<define_face_pair_model_derivative>
			m_map->weightedParameterDerivative(
				inputs.image2, s.hidden2, 
				classifierInputGrad, *(s.mapState2),
				mapParameterGrad2
			);
//###end<define_face_pair_model_derivative>
		});
		pool.wait_for_all(event1,event2);//wait for results
//###end<define_face_pair_model_parallel_derivative>
//###begin<define_face_pair_model_derivative>		
		//stitch gradient together
		gradient.resize(numberOfParameters());
		noalias(gradient) = (mapParameterGrad1 - mapParameterGrad2) | classifierParameterGrad;
	}
//###end<define_face_pair_model_derivative>	
	
//###begin<define_face_pair_model_base>
private:
	AbstractModel<FloatVector, FloatVector, FloatVector>* m_map;
	AbstractModel<FloatVector, FloatVector, FloatVector>* m_classifier;
};
//###end<define_face_pair_model_base>

int main(int argc, char **argv)
{
	if(argc < 2) {
		std::cerr << "usage: " << argv[0] << " path/to/att_faces.zip" << std::endl;
		return 1;
	}
	std::string archivename = argv[1];

	//Step1: set up the pipeline
//###begin<data_load_zip>
	ZipReader zip(archivename);//load the zip file
//###end<data_load_zip>
//###begin<data_enumerate_images>
	FileList filesTrain(zip.fileNames(), "s?\?/*.pgm");//random sampling of paths. we filter s1/ ... s9/ out for testing
	auto fileGenerator = filesTrain.generator(4,8);
//###end<data_enumerate_images>
	
//###begin<data_load_images>	
	Shape imageShape = {56,46, 1}; //height x width x channels of image
	// load a single image
	auto loadPointFromZip = [&](std::string const& path){
//###begin<data_load_zip>
		std::vector<unsigned char> contents = zip.readFile(path);//load a file from the zip-archive
//###end<data_load_zip>
		//load image and resize to target size
		std::pair<FloatVector, Shape> image = image::readImage<float>(contents);
		FloatVector resizedImage = image::resize(image.first, image.second, imageShape),
		// extract person id and turn into label
		auto numberString = path.substr(1,path.find_first_of('/'));
		unsigned int label = std::stoi(numberString) - 1;
		
		//return a pair of input and label
		return InputLabelPair<FloatVector, unsigned int>(resizedImage, label);
	};
//###end<data_load_images>	
	//create a generator pipeline where we first randomly sample paths from the zip
	//and then load the image. We have to give the shape of the result as an argument
	//here we have to define the shape of the images as well the the shape of the class label=number classes
//###begin<data_transform_image_load>	
	auto pointGenerator = transform(fileGenerator, loadPointFromZip, {imageShape,31});
//###end<data_transform_image_load>
	

	//now take a batch of image-label pairs and create pairs of them
	//with label 1 if the image is the same, 0 otherwise.	
//###begin<data_create_pairs>
	typedef typename Batch<InputLabelPair<FloatVector, unsigned int> >::type PointBatch;
	auto generatePairs = [](PointBatch const& batch){
		//store the datasets temporarily in a vector
		typedef InputLabelPair<FacePair, unsigned int> Element;
		std::vector< Element > pairs;
		for(std::size_t i = 0; i != batch.size(); ++i){
			auto const& pointi = getBatchElement(batch, i);
			for(std::size_t j = i; j != batch.size(); ++j){
				auto const& pointj = getBatchElement(batch, j);
				bool same = pointi.label == pointj.label;
				pairs.emplace_back(FacePair{pointi.input,pointj.input}, same);
			}
		}
		//transform the vector into a proper batch
		return createBatch(pairs);
	};
//###end<data_create_pairs>
	//now generates batches of size 64 because we took the cartesian product
//###begin<data_generator_pairs>
	auto pairGenerator = transform(pointGenerator,  generatePairs, {{imageShape,imageShape}, 2});
//###end<data_generator_pairs>	
	
	//Step 2: define model
//###begin<model_creation>
	Conv2DModel<FloatVector, RectifierNeuron> conv1(pairGenerator.shape().input.image1, {32, 5, 5});
	PoolingLayer<FloatVector> pooling1(conv1.outputShape(), {2, 2}, Pooling::Maximum, Padding::Valid);
	Conv2DModel<FloatVector, RectifierNeuron> conv2(pooling1.outputShape(), {64, 5, 5});
	PoolingLayer<FloatVector> pooling2(conv2.outputShape(), {2, 2}, Pooling::Maximum, Padding::Valid);
	LinearModel<FloatVector, RectifierNeuron> dense(pooling2.outputShape(), 128, true);
	auto map = conv1 >> pooling1 >> conv2 >> pooling2 >> dense;
	
	//simple linear model as classifier
	LinearModel<FloatVector, RectifierNeuron> inputClassifier(map.outputShape(), 16, true);
	LinearModel<FloatVector> outputClassifier(inputClassifier.outputShape(), 1, true);
	auto classifier = inputClassifier >> outputClassifier;
	
	//create faceId model from map and classifier
	FaceIdModel model(&map, &classifier);
//###end<model_creation>
	
	//Step 4 set up optimizer and run optimization
//###begin<training>
	CrossEntropy<unsigned int, FloatVector> loss;
//###begin<generator_error_usage>
	ErrorFunction<FloatVector> error(pairGenerator, &model, &loss, 2);//use two batches in every iteration
//###end<generator_error_usage>	
	std::size_t iterations = 1001;
	initRandomNormal(faceId,0.001); //init model
	Adam<FloatVector> optimizer;
	optimizer.setEta(0.01f);//learning rate of the algorithm
	error.init();
	optimizer.init(error);
	std::cout<<"Optimizing model "<<std::endl;
	for(std::size_t i = 0; i != iterations; ++i){
		optimizer.step(error);
		std::cout<<i<<" "<<optimizer.solution().value<<std::endl;
	}
//###end<training>
}

