//###begin<includes>
#include <shark/Core/ZipSupport.h> //for the ZipReader class
#include <shark/Data/DataDistribution.h> //for random sampling of paths in the zip
#include <shark/Core/Images/ReadImage.h> //for reading the images

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
//###begin<define_face_pair>
namespace shark{
template<>
SHARK_CREATE_BATCH_INTERFACE(
	FacePair,
	(FloatVector, image1)(FloatVector, image2)
)
}
//###end<define_face_pair>


//###begin<define_face_pair_model>
class FaceIdModel: public AbstractModel<FacePair, FloatVector, FloatVector>{

public:
	FaceIdModel(AbstractModel<FloatVector, FloatVector, FloatVector>* map, AbstractModel<FloatVector, FloatVector, FloatVector>* classifier)
	: m_map(map), m_classifier(classifier){
		this->m_features |= HAS_FIRST_PARAMETER_DERIVATIVE;
	}
	
	//handling of parameters
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
	
	//shape
	///\brief Returns the expected shape of the input
	Batch<FacePair>::shape_type inputShape() const{
		return {m_map->inputShape(), m_map->inputShape()};
	}
	///\brief Returns the shape of the output
	Shape outputShape() const{
		return 2;
	}
	
	//handling of backpropagation state
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
	
	//implement evaluation of network response, aka forward propagation
	//inputs are batches of FacePair, output are FloatMatrix
	void eval(BatchInputType const& inputs, BatchOutputType& outputs, State& state)const{
		InternalState& s = state.toState<InternalState>();//get access to internal state
		
		//map images to their hidden representation and store internal state for backpropagation
		//map images to their hidden representation and store internal state for backpropagation
		//we do this asynchronously as map is an expensive operation
		auto& pool = threading::globalThreadPool();
		std::future<void> event1 = pool.execute_async([&](){
			m_map->eval(inputs.image1, s.hidden1, *(s.mapState1));
		});
		std::future<void> event2 = pool.execute_async([&](){
			m_map->eval(inputs.image2, s.hidden2, *(s.mapState2));
		});
		pool.wait_for_all(event1,event2);//wait for results
		
		//evaluate the response of the classifier on the difference of representations
		s.diffOfHidden = s.hidden1 - s.hidden2;
		m_classifier->eval(s.diffOfHidden, outputs, *(s.classifierState));
	}
	
	//implement backpropagation
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
		
		//backpropagate to the maps asynchronously
		auto& pool = threading::globalThreadPool();
		FloatVector mapParameterGrad1;
		std::future<void> event1 = pool.execute_async([&](){
			m_map->weightedParameterDerivative(
				inputs.image1, s.hidden1, 
				classifierInputGrad, *(s.mapState1),
				mapParameterGrad1
			);
		});
		FloatVector mapParameterGrad2;
		std::future<void> event2 = pool.execute_async([&](){
			m_map->weightedParameterDerivative(
				inputs.image2, s.hidden2, 
				classifierInputGrad, *(s.mapState2),
				mapParameterGrad2
			);
		});
		pool.wait_for_all(event1,event2);//wait for results
		
		//stitch gradient together
		gradient.resize(numberOfParameters());
		noalias(gradient) = (mapParameterGrad1 - mapParameterGrad2) | classifierParameterGrad;
	}
	
private:
	AbstractModel<FloatVector, FloatVector, FloatVector>* m_map;
	AbstractModel<FloatVector, FloatVector, FloatVector>* m_classifier;
};


int main(int argc, char **argv)
{
	if(argc < 2) {
		std::cerr << "usage: " << argv[0] << " path/to/att_faces.zip" << std::endl;
		return 1;
	}

	//Step1: set up the pipeline
//###begin<data_load_images>
	Shape imageShape = {112,92, 1}; //height x width x channels of image
	ZipReader zip(argv[1]);//load the zip file and read single images
	FileList filesTrain(zip.fileNames(), "s?\?/*.pgm");//random sampling of paths. we filter s1/ ... s9/ out for testing
	// load a single image
	auto loadPointFromZip = [&](std::string const& path){
		std::pair<FloatVector, Shape> image = image::readImage<float>(zip.readFile(path));
		//todo: resize image
		auto numberString = path.substr(1,path.find_first_of('/'));
		unsigned int label = std::stoi(numberString) - 1;
		return InputLabelPair<FloatVector, unsigned int>(image.first, label);
	};
	//create a generator pipeline where we first randomly sample paths from the zip
	//and than load the image. We have to give the shape of the result as an argument
	//here we have to define the shape of the images as well the the shape of the class label=number classes
	auto pointGenerator = transform(filesTrain.generator(4,8), loadPointFromZip, {imageShape,31});
//###end<data_load_images>	
	
//###begin<data_create_pairs>	
	//now take a batch of image-label pairs and create pairs of them
	//with label 1 if the image is the same, 0 otherwise.
	typedef typename Batch<InputLabelPair<FloatVector, unsigned int> >::type PointBatch;
	auto generatePairs = [](PointBatch const& batch){
		std::vector<InputLabelPair<FacePair, unsigned int> > pairs;
		for(std::size_t i = 0; i != batch.size(); ++i){
			auto const& pointi = getBatchElement(batch, i);
			for(std::size_t j = i; j != batch.size(); ++j){
				auto const& pointj = getBatchElement(batch, j);
				pairs.emplace_back(FacePair{pointi.input,pointj.input}, pointi.label == pointj.label);
			}
		}
		return createBatch(pairs);
	};
	//now generates batches of size 64 because we took the cartesian product
	auto pairGenerator = transform(pointGenerator,  generatePairs, {{imageShape,imageShape}, 2});
//###end<data_create_pairs>	
	
	//Step 2: define model
//###begin<model_creation>
	Conv2DModel<FloatVector, RectifierNeuron> conv1(pairGenerator.shape().input.image1, {32, 5, 5});
	PoolingLayer<FloatVector> pooling1(conv1.outputShape(), {2, 2}, Pooling::Maximum, Padding::Valid);
	Conv2DModel<FloatVector, RectifierNeuron> conv2(pooling1.outputShape(), {64, 5, 5});
	PoolingLayer<FloatVector> pooling2(conv2.outputShape(), {2, 2}, Pooling::Maximum, Padding::Valid);
	LinearModel<FloatVector, RectifierNeuron> dense(pooling2.outputShape(), 128, true);
	auto map = conv1 >> pooling1 >> conv2 >> pooling2 >> dense;
	
	//simple linear model as classifier
	LinearModel<FloatVector> classifier(map.outputShape(), 1, true);
	
	//create faceId model from map and classifier
	FaceIdModel faceId(&map, &classifier);
//###end<model_creation>
	
	//Step 4 set up optimizer and run optimization
//###begin<training>
	CrossEntropy<unsigned int, FloatVector> loss;
	ErrorFunction<FloatVector> error(pairGenerator, &faceId, &loss, 2);
	
	std::size_t iterations = 20001;
	initRandomNormal(faceId,0.0001); //init model
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

