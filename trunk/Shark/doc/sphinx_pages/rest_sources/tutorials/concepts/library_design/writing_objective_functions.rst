
Writing Objective Functions
===============================

This tutorial gives an example on how to implement the different types of objective functions used throughout Shark.
Be sure to read the general tutorial on :doc:`objective_functions` first or alongside this tutorial for an overview of the interface.

It is advisable to take a look at existing objective functions while working through this tutorial. 
Especially the benchmark functions which can be found in `shark/ObjectiveFunctions/Benchmarks/` 
offer clean implementations of the different aspects of the interface.

Example Single Objective Function 
-----------------------------------

We will introduce a simple single objective benchmark function, which we will implement in several steps 
to show different aspects of the interface. We will first start by a vanilla function and enhance 
it later with a gradient. Afterwards we will take a look at constraint
handling and show two ways to handle constraints of the input space. 
Our function will be a simple benchmark function of the form :math:`f(x)=(x^Tx)^a`. The hyper parameter :math:`a` 
describees the curvature change of the function.

Let's start writing the objective function. Single objective functions are directly derived from
the abstract interface ``SingleObjectiveFunction``. This is a typedef for ``AstractObjectiveFunction`` using the template parameters
chosen throughout Shark for single objective optimizations, namely using
``RealVector`` as the ``SearchPointType`` of the objective function and double as the ``ResultType``::
	
	#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
	#include <cmath>
	
	using namespace shark;
	
	class PowerNorm : public SingleObjectiveFunction
	{
	public:
	    PowerNorm(double a = 1):m_a(a){
	    }
	
	    std::string name() {
	        return "PowerNorm";
	    }
	
	    //more to come here...
	private:
	  double m_a; 
	};

We already announced the name of our objective function in the above code. This can be used to print usefull
output, for example when an algorithm is to be applied to a set of different functions to keep track which function is
evaluated right now. We also chose a suitable default value for :math:`a`, so that it is by default a quadratic function.


Evaluating the Objective Function
-----------------------------------

The above code already compiles, but we can't instantiate it as the compiler will complain that the function ``numberOfVariables()``
is not implemented. And this is right. As our search space is vectorial, we have to describe how many input dimensions our function 
has. We also did not define how to evaluate the function - and a function which can't be evaluated is most of the times 
not very useful.

The eval function evaluates a given search point and returns the result of the function::

	ResultType eval( SearchPointType const& input )const {
		return std::pow(normSqr(input),m_a);
	}

And for now we decide that we are searching in a 3 dimensional space::

	std::size_t numberOfVariables()const{
		return 3;
	}
	
Now instantiating the function gives meaningful results::

	int main(){
	  PowerNorm f;
	  RealVector point(3);
	  point(0) = 1; point(1)=2; point(2)=3;
	  std::cout<<"evaluating "<<f.name()<<" "<<f(point);
	}

There is a small problem with the above code. First of all is our decision that our input space should be 3 dimensional completely
arbitrary. We might also want to change the dimensionality of the function for benchmarking. And finally we should check
in eval that the supplied argument matches the chosen dimensionality - at least for debug purposes. 
Taking everything into account, we arrive at the following intermediate code fragment::

	#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
	#include <cmath>
	
	using namespace shark;
	
	class PowerNorm : public SingleObjectiveFunction
	{
	public:
	    PowerNorm(double a = 2, std::size_t dimensions = 3)
	    :m_a(a),m_dimensions(dimensions){
	    }
	    std::string name() {
	        return "PowerNorm";
	    }
	    std::size_t numberOfVariables()const{
		return m_dimensions;
	    }
	    bool hasScalableDimensionality()const{
		return true;
	    }
	    void setNumberOfVariables( std::size_t numberOfVariables ){
		m_dimensions = numberOfVariables;
	    }
	    
	    ResultType eval( const SearchPointType & input )const {
		SIZE_CHECK(input.size() == m_dimensions );
		return std::pow(normSqr(input),m_a);
	    }
	private:
	    double m_a; 
	    std::size_t m_dimensions;
	};

The function ``hasScalableDimensionality()`` just announces that it is allowed to change the number of dimensions using 
``setNumberOfVariables``.

Evaluating the Derivative
------------------------------------

While this function can now be optimized by the direct-search algorithms of Shark like the CMA-ES, most algorithms require the 
derivative of the function to be present. Therefore it is very usefull to take the time to implement it. First of all, we have
to announce that this function also provides the first derivative, for this we set the flag ``HAS_FIRST_DERIVATIVE`` in the
constructor::

	PowerNorm(double a = 2, std::size_t dimensions = 3)
	:m_a(a),m_dimensions(dimensions){
	    m_features |= HAS_FIRST_DERIVATIVE;
	}
	
Afterwards we implement the function. The formula is:

.. math::
  \frac{\partial f(x)}{\partial x} = 2a(x^Tx)^{a-1} x

We implement the derivative in ``evalDerivative`` taking the ``FirstOrderDerivative`` as argument, which for now is just the same as the ``SearchPointType``.
We just calculate the function value as before and return it as well as the computed derivative::

	ResultType evalDerivative( const SearchPointType & input, FirstOrderDerivative & derivative )const {
		SIZE_CHECK(input.size() == m_dimensions );
		double norm = normSqr(input);
		derivative = 2*m_a*std::pow(norm,m_a-1) * input;
		return std::pow(norm,m_a);
	}

That's it! Now we can use gradient based optimization algorithms to optimize this function, but for now we have to provide our 
own starting points, else the optimizer will complain.

Generating Starting Points
----------------------------------------

Optimizers need proper starting points to start the optimization loop. There are basically two ways to deal with it: first of all
we could provide one during the initialization of the optimization. This is very usefull when we need a deterministic way to start 
the optimization at one specific point. But most of the time we do not care about the starting point as long as it does somehow make
sense. For a benchmark function it  is usefull to start every time at a different place to get a feeling for how an optimizer 
behaves under different starting conditions. So let's add a function which generates a uniformly gaussian distributed point::

	void proposeStartingPoint( SearchPointType & startingPoint )const {
		startingPoint.resize(m_dimensions);
		for(std::size_t i = 0; i != m_dimensions; ++i){
			startingPoint(i) = Rng::gauss(0,1);
		}
	}

And we should not forget to announce that the function can generate a starting point::

	PowerNorm(double a = 2, std::size_t dimensions = 3)
	:m_a(a),m_dimensions(dimensions){
	    m_features |= HAS_FIRST_DERIVATIVE;
	    m_features |= CAN_PROPOSE_STARTING_POINT;
	}

And for the random number generator we have to include::

	#include <shark/Rng/GlobalRng.h>

Now you should be able to create most use cases of single objective functions!

Thread Safety
----------------

By default Shark assumes that function calls are not thread safe, that means we can't evaluate multiple points in parallel, even though the algorithm could handle that.
For our benchmark function there is no problem in evaluating in paralll, so we set the flag indicating thread-safety::

	PowerNorm(double a = 2, std::size_t dimensions = 3)
	:m_a(a),m_dimensions(dimensions){
	    m_features |= HAS_FIRST_DERIVATIVE;
	    m_features |= CAN_PROPOSE_STARTING_POINT;
	    m_features |= IS_THREAD_SAFE;
	}

Handling Constraints Using Constraint Handlers
---------------------------------------------------

Shark provides two ways to handle constraints via objective functions. The first one is using a constraint handler. 
Constraint handlers serve two purposes: first they offer an reusable interface for often used types of constraints
and second they offer specific information for some kinds of constraints. Right now Shark does only offer a handler and optimizers
for simple box constraints, i.e. for every input dimension :math:`x_i` holds :math:`l_i \leq x_i \leq u_i`. An optimizer might
now query the function whether it is constrained, has an constraint handler and whether the handler itself represents box constraints.
In this case it might use a different optimization setting, or refuse to optimize as it does not work for this type of constraints.

Let's constrain our function above to the box of :math:`[0,1]^n`::

	#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
	#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>
	#include <cmath>
	
	using namespace shark;
	
	class PowerNorm : public SingleObjectiveFunction
	{
	public:
	    PowerNorm(double a = 2, std::size_t dimensions = 3)
	    :m_a(a),m_handler(dimensions,0,1){
		announceConstraintHandler(&m_handler);
	    }
	    std::string name() {
	        return "PowerNorm";
	    }
	    std::size_t numberOfVariables()const{
		return m_handler.dimensions();
	    }
	    bool hasScalableDimensionality()const{
		return true;
	    }
	    void setNumberOfVariables( std::size_t numberOfVariables ){
		m_handler.setBounds(numberOfVariables,0,1);
	    }
	    
	    ResultType eval( const SearchPointType & input )const {
		if(!m_handler.isFeasible(input))
		    throw SHARKEXCEPTION("input point not feasible");
		return std::pow(normSqr(input),m_a);
	    }
	private:
	    double m_a; 
	    BoxConstraintHandler<SearchPointType> m_handler;
	};
	
So what did happen? First of all, we removed the variable storing the dimensionality as this is now governed by the handler. 
In the constructor, we initialize it with the right dimensionality and the uniform lower and upper bound - this could also be
vectors of the correct size, if the bounds happen to be differently for every variable.
When the functions' dimensionality is changed, we have to update the handler to have the right dimensionality again.
And in eval we finally check whether the point is feasible. In the constructor there is another magic function, called 
``announceConstraintHandler``. This tells the base class that a constraint handler is available and it will set up the proper flags depending on the
capabilities of the handler.
After this call, all the virtual functions used for constraint handling can be called and are calling the handlers function. Another
nice feature in this case is that we get the starting points for free, as the handler can generate points uniformly inside the 
feasible region. But of course this behavior can still be overwritten if a different scheme is needed.
Now, that was rather simple!

Handling Constraints Without Constraint Handlers
---------------------------------------------------
We will now try to implement the above handler directly in the objective function. Please take into account that other constraints aside from the box constraints
are not well supported and that in the general interface there is no way to get more specific information about the particular constraints.

For box constraints we can support the full range of possible functions, but the minimum requirement of a constraint objective function is that it can check whether
a point is feasible or not. In our case we only hav to check whether every value of a point is between 0 and 1::

	bool isFeasible( const SearchPointType & input) const {
		SIZE_CHECK(input.size() == m_dimensions);
		for(std::size_t i = 0; i != m_dimensions; ++i){
			if (input(i) < 0 || input(i) > 1)
				return false;
		}
		return true;
	}

Secondly, some algorithms might profit from the ability of the function to find the closest feasible point to an infeasible point, this might not be 
possible for all types of constraints, so this is an optional feature::

	void closestFeasible( SearchPointType & input ) const {
		SIZE_CHECK(input.size() == m_dimensions);
		for(std::size_t i = 0; i != m_dimensions; ++i){
			input(i) = std::max(0.0,std::min(1.0,input(i)));
		}
	}

We also need to update our way we generate random points::

	void proposeStartingPoint( SearchPointType & startingPoint )const {
		startingPoint.resize(m_dimensions);
		for(std::size_t i = 0; i != m_dimensions; ++i){
			startingPoint(i) = Rng::uni(0,1);//uniform distributed between 0 and 1
		}
	}
	
Having all this in place, we have to set the proper flags announcing the capabilities of the function::

	PowerNorm(double a = 2, std::size_t dimensions = 3)
	:m_a(a),m_dimensions(dimensions){
	    m_features |= HAS_FIRST_DERIVATIVE;
	    m_features |= CAN_PROPOSE_STARTING_POINT;
	    m_features |= IS_CONSTRAINED_FEATURE;
	    m_features |= CAN_PROVIDE_CLOSEST_FEASIBLE;
	}
	
Multi-Objective Functions
---------------------------------------------------

Multi objective functions are basically the same as single-objective functions, only that they offer a vectorial return type and are derived from MultiObjectiveFunction,
which is only another typedef for AbstractObjectiveFunction with a different template parameter for the ``ResultType``. So the above tutorial
also holds for multi-objective functions - with the only quirk, that you can't calculate derivatives right now. We also need to tell the function, how many objectives we are 
going to represent with it and whether the number of objectives might be changed::

	class PowerNorm : public MultiObjectiveFunction
	{
	public:
	    PowerNorm(double a = 2, std::size_t dimensions = 3, std::size_t objectives = 2)
	    :m_a(a),m_dimensions(dimensions), m_objectives(objectives){
	    }
	    std::string name() {
	        return "PowerNorm";
	    }
	    std::size_t numberOfVariables()const{
		return m_dimensions;
	    }
	    bool hasScalableDimensionality()const{
		return true;
	    }
	    void setNumberOfVariables( std::size_t numberOfVariables ){
		m_dimensions = numberOfVariables;
	    }
	    std::size_t numberOfObjectives()const{
		return m_objectives;
	    }
	    bool hasScalableObjectives()const{
		return true;
	    }
	    void setNumberOfObjectives( std::size_t numberOfObjectives ){
		m_objectives = numberOfObjectives;
	    }
	    
	    ResultType eval( const SearchPointType & input )const {
		SIZE_CHECK(input.size() == m_dimensions );
		ResultType result(m_objectives);
		result(0) = std::pow(normSqr(input),m_a);
		//more objectives here...
		return result;
	    }
	private:
	    double m_a; 
	    std::size_t m_dimensions;
	    std::size_t m_objectives;
	};



