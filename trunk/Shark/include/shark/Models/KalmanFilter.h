#ifndef SHARK_ML_MODEL_KALMANFILTER_H
#define SHARK_ML_MODEL_KALMANFILTER_H

#include <shark/Models/AbstractModel.h>
namespace shark {


//!
//! \brief Standard linear Kalman Filter.
//!
//! The implementation aims at providing flexibility w.r.t. process and
//! measurement model; no restrictive assumptions concerning specific
//! properties of the matrices are made (although in practical applications
//! often noise covariances have diagonal form). The implementation is not
//! optimized at all for computational efficiency.
//!
//! For making the algorithm robust against numerical inaccuraries, the
//! update equation for the state error covariance matrix 'P' given in
//! "Applied Optimal Estimation; The Analytic Sciences Corporation; The M.I.T.
//! Press, pp. 305" is used. For very complex modells a square root form
//! (e.g. Cholesky) based implementation might be more appropriate due to
//! its increased numerical robustness. \par
//!
//! Process Modell: <br>
//! --------------- <br>
//! State Dynamics:   \f$  x_t = F*x_{t-1} + n_x;  n_x ~ N(0,Q)  \f$\n
//! Observation   :   \f$  z_t = H*x_t + n_z;      n_z ~ N(0,R)  \f$\n
//!
//!
//! Before filtering the model matrices F,Q,H,R must be set. Furthermore
//! the initial filter states, i.e. the initial state vector x(0) and the
//! initial covariance matrix of state estimation error P(0) must be set.
//!
//! As a Model subclass the KalmanFilter holds its current a-posteriori
//! state estimate at time t (after z(t) has been observed) in the
//! Model::parameter vector.
//!
class KalmanFilter : public AbstractModel<RealVector,RealVector>
{
public:
	//! Constructor
	KalmanFilter(size_t modelSize);

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "KalmanFilter"; }

	RealVector parameterVector()const{
		return RealVector();
	}
	void setParameterVector(RealVector const& newParameters){}


	//! The model first runs the predict phase.
	//! Then it updates its internal estimates
	//! using the input. Finally the a-posteriori
	//! state estimate, that is, the model
	//! parameter vector is returned as output.
	void eval(const RealVector& pattern,RealVector& output);
	using AbstractModel<RealVector,RealVector>::eval;


	//--------------------------------------------------------------------------
	//--------------------------------------------------------------------------
	// definition of system process model
	//

	//! returns the state transition matrix 'F'
	RealMatrix& stateDynamics(){
		return m_stateTransition;
	}
	const RealMatrix& stateDynamics()const {
		return m_stateTransition;
	}

	//! returns the state noise covariance matrix 'Q'
	RealMatrix& stateNoise(){
		return m_stateNoiseCov;
	}
	const RealMatrix& stateNoise()const{
		return m_stateNoiseCov;
	}

	//--------------------------------------------------------------------------
	//--------------------------------------------------------------------------
	// definition of observation model
	//

	//! sets the observation matrix 'H'
	RealMatrix& observationModel(){
		return m_observationModel;
	}
	const RealMatrix& observationModel()const{
		return m_observationModel;
	}


	//! sets the observation noise covriance matrix 'R'
	RealMatrix& observationNoise(){
		return m_observationNoise;
	}
	const RealMatrix& observationNoise()const{
		return m_observationNoise;
	}

	//--------------------------------------------------------------------------
	//--------------------------------------------------------------------------
	// definition of initial state
	//

	//! sets the initial state vector
	void setStateVector(const RealVector& stateVec);


	//! sets the initial covariance matrix of state estimation error
	void setStateErrorCov(const RealMatrix& stateVec);


	//--------------------------------------------------------------------------
	//--------------------------------------------------------------------------
	// prediction / filtering operations
	//

	//! next time step / forward propgation (calculation of a-priori estimates)
	void predict();

	//! update state vector; parameter z: measurent vector
	void updateEstimate(const RealVector& z);

	//--------------------------------------------------------------------------
	//--------------------------------------------------------------------------
	// access of data ...
	//

	//! returs a-priori (predicted) state estimate (updated in 'propagate()')
	RealVector& statePrediction(){
		return m_predictedState;
	}
	const RealVector& statePrediction()const{
		return m_predictedState;
	}

	//! returns a-priori (predicted) covariance of state estimation error
	RealMatrix& predictedCovariance(){
		return m_predictedCovariance;
	}
	const RealMatrix& predictedCovariance()const{
		return m_predictedCovariance;
	}

	//! returns a-posteriori covariance of state estimation error (after z(t) has been observed)
	RealMatrix& observedCovariance()
	{
		return m_observedCovariance;
	}

	//! returns error in predicted measurements
	const RealVector& predictionError()const
	{
		return m_predictionError;
	}

protected:
	//! state transition matrix
	RealMatrix m_stateTransition;

	//! state noise covariance matrix
	RealMatrix m_stateNoiseCov;

	//! observation matrix
	RealMatrix m_observationModel;

	//! observation noise covariance matrix
	RealMatrix m_observationNoise;

	//! a-priori (predicted) state estimate at time t (before z(t) has been observed)
	RealVector m_predictedState;
	//! a-posteriori state at timte t (after observation of z(t))
	RealVector m_state;

	//! a-priori (predicted) covariance of state estimation error
	RealMatrix m_predictedCovariance;

	//! a-posteriori covariance of state estimation error (after z(t) has been observed)
	RealMatrix m_observedCovariance;

	//! error in predicted measurements
	RealVector m_predictionError;

	size_t m_inputSize;
};
}

#endif

