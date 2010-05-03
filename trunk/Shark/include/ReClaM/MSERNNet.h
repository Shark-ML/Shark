//===========================================================================
/*!
 *  \file MSERNNet.h
 *
 *  \brief Offers the functions to create and to work with a
 *         recurrent neural network.
 *         The network is combined with the mean squared error
 *         measure. This combination is created due to computational
 *         efficiency.
 *
 *  \author  M. Toussaint
 *  \date    2000
 *
 *  \par Copyright (c) 1999-2001:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-27974<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *
 *  \par Project:
 *      ReClaM
 *
 *
 *  <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of ReClaM. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 2, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, write to the Free Software
 *  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
//===========================================================================

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <SharkDefs.h>
#include <Array/ArrayTable.h>
#include <Array/ArrayOp.h>
#include <Rng/GlobalRng.h>
#include <ReClaM/Model.h>
#include <ReClaM/ErrorFunction.h>

#ifndef RNNET_H
#define RNNET_H

//===========================================================================
/*!
 *  \brief A recurrent neural network regression model that learns
 *         with Back Propagation Through Time and assumes the MSE
 *         error functional.
 *
 *  This class defines a recurrent neural network regression
 *  model. The input and output arrays should, as usual, be
 *  2-dimensional and have the dimensionality
 *  (batch-length,number-of-neurons). The only difference is that
 *  here, the batch corresponds to a time series instead of
 *  independent data points. The gradient is calculated via
 *  BackPropTroughTime (BPTT). This model implies the MSE as the error
 *  functional.
 *
 *  The class can handle arbitrary network architectures. Feed-forward
 *  connections (with time delay `zero') as well as connections of any
 *  time delay can be realized --- see setStructure() for details of how
 *  to define the structure.
 *
 *  Things to note!:
 *
 *  (1) All neurons are sigmoidal! So please transform inputs and
 *  outputs to take this into account. The reason is that internally,
 *  no differences between input, hidden, or output neurons is made.
 *
 *  (2) The initialization of the state history is an important issue
 *  for dynamic systems like an RNN. Please read the documentation of
 *  setWarmUpLentgh().
 *
 *  (3) Online learning can, in principle, be realized by having a
 *  batch-length of zero (the input/output arrays still have to be
 *  2-dimensional, i.e., of dimension (1,number-of-neurons)) and by
 *  setting the WarmUpLength to zero. Note though that this is quite
 *  inefficient compared to batch-learning (because the BPTT does not
 *  save you any calculation time) and that cross-talk will make
 *  learning _very_ difficult. Maybe, even when online learning, you
 *  should still use a reasonable batch size (say 100), feed these
 *  batches sequentially, and set the WarmUpLength to zero.
 *
 *  (Internals: the only core functions are `processTimeSeries' and
 *  `calcGradBPTT' in the cpp-file. Almost all the rest it `utility'
 *  stuff.)
 *
 *  \author  M. Toussaint
 *  \date    2000
 *
 *  \par Changes:
 *      none
 *
 *  \par Status:
 *      stable
 */
class MSERNNet: public Model, public ErrorFunction
{
public:
	//! Creates an empty MSE recurrent neural network.
	MSERNNet();

	//! Creates an empty MSE recurrent neural network with a certain
	//! topology given by the connection matrix "con".
	MSERNNet(Array<int>);

	//! Creates an empty MSE recurrent neural network with a certain
	//! topology given by the connection matrix "con" and a
	//! set of weights given by weight matrix "wei".
	MSERNNet(Array<int>, Array<double>);

	//! Creates a recurrent neural network by reading the necessary
	//! information from a file named "filename".
	MSERNNet(std::string);

	/*!
	 *  \brief Sets the length of the warmup sequence.
	 *
	 *  Usually, when processing a new data series (e.g., by calling
	 *  model(...), error(...), or anything like that) then all the
	 *  `states' of the network are reset to zero. By `states' I mean the
	 *  buffered activations to which time-delayed synapses refer
	 *  to. Effectively, this means one assumes a zero activation history.
	 *
	 *  The advantage of this is, that it makes the model behavior well
	 *  defined. The disadvantage is that you can't predict a time series
	 *  well with a zero history. Thus, one should use the first few
	 *  inputs of a data series to initialize the network, i.e., to let it
	 *  converge into a `normal' dynamic state from which prediction of
	 *  new data is possible. This phase is called the warmup
	 *  phase. (Internally, the warmup phase differes from subsequent data
	 *  only in that it does not contributed to the error functional and
	 *  does not induce an error gradient.)
	 *
	 *  With this function you can set to time span of the warmup phase
	 *  (which also means the amount of data that you `waste' for the
	 *  warmup instead of the learning.)
	 *
	 *  NOTE: Sometimes, e.g., when feeding a sequence of data sequences
	 *  to the model, it is desirable not to reset the internal states to
	 *  zero. This is the case when you set a WUP to zero.
	 *
	 *  \param  WUP Length of the warmup sequence, the default value is "0".
	 */
	void includeWarmUp(unsigned WUP = 0);

	/*!
	 *  \brief Feed a data series to the model.
	 *
	 *  The output of the network is only stored internally. \em input
	 *  must be 2-dimensional, where the first index refers to the time,
	 *  the second to the neuron. To view the outputs, please use method
	 *  #model(const Array<double>& input,Array<double>& output) instead.
	 *
	 *  \param  input Input patterns for the network.
	 *  \return None.
	 */
	void model(const Array<double> & input);

	/*!
	 *  \brief Feed a data series to the model. The output (i.e., the time
	 *  series of activations of the output neurons) is copies into the
	 *  #output# buffer.
	 *
	 *  \param  input  Input patterns for the network.
	 *  \param  output Used to store the outputs of the network.
	 */
	void model(const Array<double> & input, Array<double> &output);

	/*!
	 *  \brief Evaluates the error percentage of the network output compared
	 *         to given target data.
	 *
	 *  Given the patterns in \em input and the patterns in \em target
	 *  for the comparison to the outputs of the network, the error
	 *  percentage is calculated, i.e. the mean
	 *  squared error \f$E\f$ as given in the <a href="#_details">details</a>
	 *  is calculated and the error percentage is then given as
	 *
	 *  \f$
	 *      EP = \frac{100}{(outmax - outmin)^2} \ast E
	 *  \f$
	 *
	 *  where \f$outmax\f$ is the maximum value that can be stored
	 *  in a \em double value and \f$outmin\f$ is the corresponding
	 *  minimum value.
	 *  Keep in mind, that if you have defined a warmup length greater than zero
	 *  for the network before, the first \f$warmup-length\f$ input patterns
	 *  are not used for the calculation of the error.
	 *  Of course, the number of patterns in \em input must be greater
	 *  than the warmup-length otherwise the method will exit
	 *  with an exception.
	 *
	 */
	double errorPercentage(const Array<double> &input, const Array<double> &target);

	/*!
	 *  \brief Evaluates the mean squared error of the network output compared
	 *         to given target data.
	 *
	 *  Given the patterns in \em input and the patterns in \em target
	 *  for the comparison to the outputs of the network, the mean
	 *  squared error \f$E\f$ as given in the <a href="#_details">details</a>
	 *  is calculated.
	 *
	 *  Keep in mind, that if you have defined a warmup length greater
	 *  than zero for the network before, the first \em warmup-length
	 *  input patterns are not used for the calculation of the error. Of
	 *  course, the number of patterns in \em input must be greater than
	 *  the warmup-length otherwise the method will
	 with failure.
	 *
	 *  \param  input  Input patterns for the network.
	 *  \param  target Target patterns used for comparison to the outputs
	 *                 of the network.
	 *  \return The mean squared error \f$E\f$.
	 *
	 */
	double error(Model& model, const Array<double> &input, const Array<double> &target);

	/*!
	 *  \brief Evaluates the derivative of the mean squared error.
	 *
	 *  Given the patterns in \em input and the patterns in \em target
	 *  for the comparison to the outputs of the network, the derivatives of
	 *  the mean squared error \f$E\f$ as given in the
	 *  <a href="#_details">details</a> are calculated and the results
	 *  are stored in the parameter derivative.
	 *  Additionally, the mean squared error itself is returned.
	 *  Keep in mind, that if you have defined a warmup length greater than zero
	 *  for the network before, the first \f$warmup-length\f$ input patterns
	 *  are not used for the calculation of the error.
	 *
	 *
	 *  \param  input       Input patterns for the network.
	 *  \param  target      Target patterns used for comparison to the outputs
	 *                      of the network.
	 *
	 */
	double errorDerivative(Model& model, const Array<double> &input, const Array<double> &target, Array<double>& derivative);

	/*!
	 *  \brief same as error
	 *
	 *  \param  input  Input patterns for the network.
	 *  \param  target Target patterns used for comparison to the outputs
	 *                 of the network.
	 *  \return The mean squared error \f$E\f$.
	 */	
	double meanSquaredError(const Array<double> &input, const Array<double> &target);

	/*!
	 *  \brief Returns the connection Matrix.
	 *
	 *  The 3-dimensional connection matrix of the network is returned.
	 *  The first dimension is used for the number of delays, with
	 *  each delay having a 2-dimensional connection matrix as
	 *  known from other networks.
	 *
	 *  \return The 3-dimensional connection matrix.
	 *
	 */	
	Array<int> &getConnections();

	/*!
	 *  \brief Returns the weight matrix.
	 *
	 *  The 3-dimensional weight matrix of the network is returned.
	 *  The first dimension is used for the number of delays, with
	 *  each delay having a 2-dimensional weight matrix as
	 *  known from other networks.
	 *
	 *  \return The 3-dimensional weight matrix.
	 *
	 */
	Array<double> &getWeights();

	/*!
	 *  \brief Initializes the weights of the net.
	 *
	 *  The weights of the network for all delays are initialized
	 *  by uniformally distributed numbers between \em low and \em up.
	 *
	 *  \param  low The minimum possible initialization value.
	 *  \param  up  The maximum possible initialization value.
	 *  \return None.
	 *
	 */
	void initWeights(double, double);

	/*!
	 *  \brief Initializes the random number generator and the weights of the net.
	 *
	 *  same as #Rng::seed(seed); initWeights(min, max);#
	 *
	 *  \param  seed The initialization value for the random number generator.
	 *  \param  min  The minimum possible initialization value.
	 *  \param  max  The maximum possible initialization value.
	 *  \return None.
	 */
	void initWeights(long, double, double);

	/*!
	 *  \brief Based on a given connection matrix a network is created.
	 *
	 *  This is the core function to set the structure and initialize the
	 *  sizes of all fields. All other `setStructure' methods call this
	 *  one. Any old values of the network are lost.
	 *
	 *  One has to pass a 3-dimensional connection matrix \em mat to this
	 *  function. The first dimension indicates the time delay of the
	 *  respective connection. E.g., mat[0] is the ordinary 2-dimensional
	 *  feedforward (triangular) connection matrix (where the connections
	 *  have no time delay), mat[1] is the (potentially fully connected)
	 *  connection matrix for connections of time delay 1, etc.
	 *
	 *  It is perfectly ok, if some of these `layers' mat[t] are
	 *  completely zero: the function will detect this (and store it as a
	 *  0 in the delMask(t)) and not waste time for this layer when
	 *  processing the network.
	 *
	 *  Notice, that the weight matrix #w# of the network is only adapted
	 *  to the new size, but the matrix is not explicitly initialized with
	 *  zero weights.
	 *
	 *  \param  mat The 3-dimensional connection matrix.
	 *
	 *  \return None.
	 *
	 */
	void setStructure(Array<int> &);

	/*!
	 *  \brief Based on given connection and weight matrices a network
	 *         is created.
	 *
	 *  same as #setStructure(con); setStructure(wei);#
	 *
	 *  \param  con The 3-dimensional connection matrix, where the
	 *              first dimension is used for the different time
	 *              steps with each time step having a 2-dimensional
	 *              #connectionMatrix.
	 *  \param  wei The new 3-dimensional weight matrix, where the
	 *              first dimension is used for the different time
	 *              steps with each time step having a 2-dimensional
	 *              #weightMatrix.
	 *  \return None.
	 *
	 *
	 */
	void setStructure(Array<int> &, Array<double> &);

	/*!
	 *  \brief Replaces the existing weight matrix by a new matrix.
	 *
	 *  The current weight matrix is replaced by the new weight
	 *  matrix \em wei.
	 *  This means NOT, that the whole structure of the network
	 *  is changed. The new weight matrix must be compatible
	 *  to the existing connection matrix.
	 *
	 *  \param  wei The new 3-dimensional weight matrix, where the
	 *              first dimension is used for the different time
	 *              steps with each time step having a 2-dimensional
	 *              #weightMatrix.
	 *  \return None.
	 *
	 *
	 */
	void setStructure(Array<double> &);

	/*!
	 *  \brief The current structure of the network is replaced by
	 *         a new one that is based on the information in file
	 *         "filename".
	 *
	 *  A file is used to replace the current structure of the network.
	 *  This file must have the following content:
	 *  The first line contains the number of time steps used,
	 *  followed by \f$number\ of\ time\ steps\f$ connection matrices,
	 *  followed by \f$number\ of\ time\ steps\f$ weight matrices.
	 *
	 *  \param filename Name of the file that contains the information
	 *                  for the modification of the structure.
	 *  \return None.
	 */
	void setStructure(std::string);



	/*!
	 *  \brief The current structure of the network is replaced by
	 *         based on the given paramteres.
	 *
	 *  A file is used to replace the current structure of the network.
	 *  This file must have the following content:
	 *  The first line contains the number of time steps used,
	 *  followed by \f$number\ of\ time\ steps\f$ connection matrices,
	 *  followed by \f$number\ of\ time\ steps\f$ weight matrices.
	 *
	 *  \param in number of input neurons
	 *  \param hidden number of output neurons
	 *  \param out number of input neurons
	 *  \param memory number of memory layers
	 *  \param layered use layered strucuture
	 *  \param recurrentInputs use recurrent inputs
	 *  \param bias use bias neuron
	 *  \param elman
	 *  \param previousInputs use previous inputs
	 *  \return None.
	 */
	void setStructure(unsigned in, unsigned hidden, unsigned out,
			  unsigned memory=1, bool layered=false,
			  bool recurrentInputs=true, bool bias=true, bool elman=false, bool previousInputs=false);
	/*!
	 *  \brief Writes the structure of the network to a file named
	 *         "filename".
	 *
	 *  The structure of the network is written in the following
	 *  format:
	 *
	 *  The first line only contains a single value, that denotes
	 *  the number of time steps used.
	 *  Following are \f$number\ of\ time\ steps\f$ connection
	 *  matrices, followed by \f$number\ of\ time\ steps\f$
	 *  weight matrices.
	 *
	 *  \param  filename Name of the file, the network structure is
	 *                   written to.
	 *  \return None.
	 *
	 */

	void write(std::string);

	//===========================================================================
	/*!
	 *  \brief Creates a connection matrix for a recurrent network.
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	void setStructur(unsigned in, unsigned hidden, unsigned out,
									 unsigned memory = 1, bool layered = false,
									 bool recurrentInputs = true, bool bias = true, bool elman = false, bool previousInputs = false);

	/*!
	 *  \brief Sets the history of the neuron states to certain values.
	 *
	 *  In case you chose no warmup phase (warmup length set to zero)
	 *  then, the network will not be reinitialized when new data is
	 *  feeded in. This might make sense if previous data was processes
	 *  and the current state of the model is well-defined and can ne used
	 *  for further predictions.
	 *
	 *  Alternatively, with this function, you can explicitly set the
	 *  history of the network -- I guess this makes only sense when you
	 *  have recorded a history earlier or if you set it explicitly to
	 *  zero.
	 *
	 *  The first index of the array corresponds to the index of the
	 *  neuron (and must have dimension equal to the total number of
	 *  neurons). The second index corresponds to the time IN REVERSE
	 *  ORDER (the index denotes `the time before now') and it must have
	 *  the dimensionality #delay#.
	 *
	 *  \param  Ystate The new values for the history.
	 *  \return None.
	 *
	 */
	void setHistory(const Array<double> &);

	/*!
	 *  \brief Returns the history of the neuron states.
	 *
	 *  This returns the current history of neuron activations. Use it for
	 *  #setHistory#. See the explanations of the history format there.
	 *
	 *  \return The current content of the history.
	 *
	 *  \author  M. Toussaint
	 *  \date    2000
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	Array<double> getHistory();

protected:

	//! The total number of neurons of the network (input, output and hidden).
	unsigned numberOfNeurons;

	//! The absolute number of parameters of the network
	//! (equal to the number of weights)
	unsigned numberOfParameters;

	//! the variable that counts the time (backwards!) when processing a data series
	unsigned time;

	/*! Number of different time steps in the network architecture:
	    delay = 1 means only feed forward, delay > 1 also includes
	    the history of the states into the dynamic. */
	unsigned delay;

	//
	unsigned episode;

	/*!  Number of patterns of the input sequence, which are not
		   considered for evaluating the error and gradient of the
			 error. Nevertheless, these elements are used to modify the
			 internal states of the neurons. */
	unsigned warmUpLength;

	/*! 3-dimensional connection matrix. The first dimension is the time
	    delay, the second and third dimension are the neuron indices of
	    the endpoint and starting point of the connection as given
	    in a "normal" connection matrix.*/
	ArrayTable<int> connectionMatrix;

	/*! 3-dimensional weight matrix. The first dimension is the time
		 delay, the second and third dimension are the neuron indices of
		 the endpoint and starting point of the connection, the weight
		 belongs to, as given in a "normal" weight matrix.*/
	ArrayTable<double> weightMatrix;

	/*! 1-dimensional array with the number of elements equal to the
		number of time delays in the structure. The i-th element of this
		array is set to one, if at least one connection from the i-th memory layer
		to the feed-forward layer exists. This variable is only used to
		reduce the computational time.*/
	ArrayTable<int> delMask;

	/*!Activation of the neurons prior to the processings, usually equal
		to the input pattern. Stimulus is a 1-dimensional array with the
		number of elements equal to the number of neurons.*/
	ArrayTable<double> stimulus;

	/*! Activation of the neurons after processing the time series. "Y"
		is a 2-dimensional array, the first dimension gives the neuron
		index, the second one the time step counted backwards.
		Therefore, the second dimension's number of elements is equal to
		the sum of the length of the time series and the maximum time
		delay of the structure.*/
	ArrayTable<double> Y;

	//! This array stores the errors of the neurons for every input
	//! pattern.
	ArrayTable<double> err;

	//! Stores the local delta values which are used to evaluate the
	//! gradient.
	ArrayTable<double> delta;

	/*! Derivative of the error with respect to the weights. This object is
			a 3-dimensional array, the first dimension is the time
		 delay, the second and third dimension are the neuron indices of
		 the endpoint and starting point of the connection, the weight
	 belongs to.*/
	ArrayTable<double> dEw;

	//! Activation function of all neurons.
	virtual double g(double);

	//! Computes the derivative of g(a) for all neurons.
	virtual double dg(double);

	//! Initializes some internal variables.
	void init0();

	//! Writes the values of the weights stored in the weight matrix to
	//! the parameter vector w of ModelInterface.
	void writeParameters();

	//! Reads the values of the parameter vector w of ModelInterface
	//! and stores these values in the weight matrix.
	void readParameters();

	/*! Writes the values of the gradient of the error with respect to the
			different weights from the variable dEw to the variable
			dedw in the Model Interface.*/
	void writeGradient(Array<double>& derivative);

	//! Performs some initializations that are necessary to process the
	//! time series.
	void prepareTime(unsigned t);

	//! Processes a whole time series. After processing the output can be
	//! found in the variable Y.
	void processTimeSeries(const Array<double>& input);

	//! Processes one input pattern.
	void processTimeStep();

	/*! Performs backpropagation through time to calculate the derivative
			of the error with respect to the weights. The results are stored to
			dEw.*/
	void calcGradBPTT();
};

#endif //RNNET_H









