Variational Autoencoders
==============================================

Variational autoencoders are an extension to normal autoencoders. 
They have an underlying generative model which is trained using an lower bound of the
maximum likelihood objective function.
The optimisation problem for input data :math:`\vec{x}_1,\dots,\vec{x}_N` is stated as:

.. math ::
	\min_{\theta} \frac 1 N \sum_{i=1}^N E_{q(\vec z \| \vec x_i)}\left\{(\vec x_i - f(\vec z))^2\right\} + \lambda \cdot KL( q(\vec z \| \vec x_i)|| \mathcal{N}(0,I)) \enspace .
	
	
The encoder is represented by a distribution ``q``, that is learning a normal distribution from which we sample hidden states ``z``.
The decoder then performs the reconstruction ``f(z)`` and the squared loss between original point and reconstruction is computed. Without
an regularization term, the encoder will learn a distribution with a very small variance as larger variances would make reconstruction harder. 
Tehrefore, a second term, the KL divergence is added which measures how different the ``q`` is from a standard normal distribution.
In this tutorial, we will train such a model. The full example program is given in :doxy:`VariationalAutoencoder.cpp`.

For this tutorial, we need the following includes::


	#include <shark/Data/SparseData.h>//for reading in the images as sparseData/Libsvm format
	#include <shark/Data/Pgm.h>//for printing out reconstructions
	#include <shark/Models/LinearModel.h>//single dense layer
	#include <shark/Models/ConcatenatedModel.h>//for stacking layers
	#include <shark/Algorithms/GradientDescent/Adam.h>// The Adam optimization algorithm
	#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h> //squared loss function (can also be cross-entropy for greyscale images)
	#include <shark/ObjectiveFunctions/VariationalAutoencoderError.h> //variational autoencoder error function
	using namespace shark;
	

Defining the model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A variational autoencoder consists of two models, the encoder and the decoder. The encoder must have linear output
neurons and two times the number of outputs as the inputs of the decoder. For each input of the decoder, the encoder must learn the
mean and variance. We will showcase only a very simple pair of models::


		//build encoder network
		//note that the output layer must be linear and must have twice the number of outputs than the decoder inputs
		//as we have to model mean and variance for each decoder-input.
		LinearModel<FloatVector, RectifierNeuron> encoder1(data.inputShape(),500, true);
		LinearModel<FloatVector, LinearNeuron> encoder2(encoder1.outputShape(),2 * 300, true);
		auto encoder = encoder1 >> encoder2;
		
		//build decoder network
		//MNIST is scaled between 0 and 1 so a sigmoid output makes predicting compeltely black and completely white pixels easier
		LinearModel<FloatVector, RectifierNeuron> decoder1(300, 500, true);
		LinearModel<FloatVector, LogisticNeuron> decoder2(decoder1.outputShape(), data.inputShape(), true);
		auto decoder = decoder1 >> decoder2;
	

Instead of the :doxy:`ErrorFunction` we will use the :doxy:`VariationalAutoencoderError`. It takes the dataset, encoder and decoder models, the loss function and the strength
of regularization, lambda::


		SquaredLoss<FloatVector> loss;
		double lambda = 1.0;
		VariationalAutoencoderError<FloatVector> error(data.inputs(), &encoder, &decoder,&loss, lambda);
	

Lastly, we optimize the objective using :doxy:`Adam`. Take into account that encoder and decoder have to be initialized separately::


		std::size_t iterations = 20000;
		Adam<FloatVector> optimizer;
		optimizer.setEta(0.001);
		initRandomNormal(encoder,0.0001);
		initRandomNormal(decoder,0.0001);
		error.init();
		optimizer.init(error);
		std::cout<<"Optimizing model "<<std::endl;
		for(std::size_t i = 0; i <= iterations; ++i){
			optimizer.step(error);
			if(i % 100 == 0){
				//create some reconstructions for evaluation
				auto const& batch = data.batch(0).input;
				RealMatrix reconstructed = decoder(error.sampleZ(optimizer.solution().point, batch));
				
				std::cout<<i<<" "<<optimizer.solution().value<<" "<<loss(batch, reconstructed)/batch.size1()<<std::endl;
				//store reconstructions
				exportFiltersToPGMGrid("reconstructed"+std::to_string(i), reconstructed,28,28);
			}
		}
	
