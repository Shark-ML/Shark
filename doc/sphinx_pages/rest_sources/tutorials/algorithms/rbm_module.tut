The RBM Module
====================================================
The following sections will briefly describe Restricted Boltzmann Machines (RBM).
After that a short summary of the design goals and the target audience is given and
in the last part the different components of the library are presented.
This documentation should only be seen as a birds eye view of the module.
Further tutorials will describe the components and the RBM in more detail.


What is an RBM?
+++++++++++++++++++++++++++++++++++++++++++++++++++
RBMs belong to the class of undirected graphical models (Markov random fields).
The undirected graph of an RBM has an bipartite structure as shown in the figure below.
RBMs hold two sets of random variables (also called neurons): one layer of visible variables 
to represent observable data and one layer of hidden variables to capture dependencies between the visible variables. 
As indicated by the missing edges between the variables inside one layer in the graph the 
variables of one layer are independent of each other given the states of the variables of the other layer.

.. figure:: ../images/rbm_graph.svg
  :scale: 75 %
  :alt: the bipartite graphical model of an RBM

As for any Markov random field with a strictly positive probability distribution 
the joint distribution of an RBM is given by a Gibbs distribution

.. math::
	 p(\vec v,\vec h)={e^{- \frac{1}{T} E(\vec v, \vec h)}}/{Z},

where :math:`\vec v` and :math:`\vec h` are the vectors of the states of the  
visible and the hidden variables respectively,
*T* is a constant called temperature (it is usually set to 1)
and *Z* is a normalization constant called the partition function:

.. math::
	Z=\sum_{ \vec v, \vec h}e^{- \frac{1}{T} E(\vec v,\vec h)}

Based on
Welling at al. [WellingEtAl2005]_, we define a generalized form of the
energy function as

.. math::
	E(\vec v,\vec h)=  f_h(\vec h) + f_v(\vec v) + \sum_{k,l} \phi_{hk}(\vec h) W_{k,l} \phi_{vl}(\vec v).

The terms :math:`f_h(\vec h)` and  :math:`f_v(\vec v)` are
associated to either only the hidden or the visible neurons and
the third term models the interaction between visible and hidden variables.
Note that the energy function given here differs from the one 
given by Welling at al. [WellingEtAl2005]_ in an important fact:
out of practical design reasons we define one function f_h (or f_v) for all hidden (or all visible) neurons jointly
instead of having one for each singe hidden (or visible) neuron. This would 
allow to introduce dependancies between the variables of one layer - and thus to define 
a model that does not correspond to an RBM.
If f_h (or analogously f_v) however is a sum of terms each depending only on the value
of one neuron we obtain the energy function of an RBM.   


In the standard case of an binary RBM [Smolensky1986]_ [Hinton2007]_ we have :math:`f_h(\vec h) = \vec h^T  \vec c`
and :math:`f_v(\vec v) = \vec v^T \vec b`, where :math:`\vec c` and :math:`\vec b`
are the vectors of the bias parameters for the hidden and the visible neurons respectively.
Furthermore, the interaction term simplifies to :math:`\vec h^T W \vec v`, where :math:`W`
is the matrix of the connection weights between hidden and visible variables, so we have just
one single 'phi-function' for each layer that is the identity function.

The parameterization of the RBM depends on the chosen energy function and thus can vary.
Training an RBM corresponds to searching for the parameters that maximize the
log-likelihood of the training data. This does not require the data to have
labels associated with them so that the RBM can be used as an unsupervised learning technique.
Unfortunately the gradient of the log-likelihood can not be computed efficiently and
thus the optimization problem is hard. Instead of computing the gradient directly,
it is approximated using Markov Chain Monte Carlo (MCMC) techniques.

Usually the MCMC techniques are based on Gibbs Sampling because the independence of the 
variables of one layer given the state of the other makes this sampling scheme 
especially easy: sampling a new state for all hidden variables in the first step and
sampling a new state for all visible variables in the second. 

The concrete form of the conditional distribution depends on the energy function.
Choosing a suitable energy function leads to conditional distributions which are
well known and can be efficiently sampled from. Adding constraints on the form of the
energy as for example the formula of Welling et al. given above, even the inverse operation
- creating a joint probability distribution from given conditional distributions -
is possible.
Thus it is possible to identify kinds of neurons with their corresponding conditional
probability distributions. This leads to names like "binary neurons" or "Gaussian neurons".





Design goals
++++++++++++++++++++++++++++++++++++++++++++++++++++
As we have seen above, RBMs are quite complex. There are a lot of different types of RBMs
and there are a lot of aspects which can be changed. Typical software tools until now
usually only supported a small range of RBMs. They are often implemented by hand for every
type of conditional distribution of the hidden and visible neurons. Often, these implementations
are very efficient but they are not useful when new ideas need to be implemented.
Often, small changes make a reimplementation of big parts of the code necessary.
Our implementation tries to avoid this by offering very flexible components. We wanted
to be able to represent all kinds of RBMs with valid energy functions.
Furthermore, we tried to optimize the library for standard  energy types.
For more complex kinds of RBMs this library will most likely not be able to compete with
specialized implementations in terms of execution speed. But it will reduce the amount of
time needed until a program is properly debugged and stable. Thus it makes it easy to try
out new kinds of RBMs.

To achieve flexibility, the usual components of a RBM learning process are separated and
highly abstracted. Every component itself covers only a small aspect of the program and all
components can be freely combined. For example different energy functions can be composed out
of any two (different) types of neurons. The neurons carry the information needed to define
the complete joint distribution.

If a completely different type of RBM needs to be implemented, the definition of the
energy can be replaced as a whole with a more suitable structure. But this should not be
needed often since the default energy can represent most of the energy functions discussed
in the literature, for example the energy of Gaussian RBMs, simple binary RBMs and those
using neurons with truncated exponentials as conditional distributions.

This level of abstraction leads to a very efficient development process. If for example a Gaussian
RBM is trained with Contrastive Divergence [Hinton2002]_ and the data turns out not to be modeled very well, the
neurons can be changed without a need to change the implementation of contrastive divergence.
When Contrastive Divergence is the problem, choosing another learning algorithm is also only a
matter of a few lines of code.


Design
++++++++++++++++++++++++++++++++++++++++++++++++++++

To achieve the desired flexibility, the module relies heavily on
templates. Most of the components described in the following depend on
the others as template parameters.  Changing a template parameter will
change the behavior of the components and give rise for a new learning
algorithm as will be described later. The advantage of templates is,
that a lot of information can be processed during compile time.  This
allows the :doxy:`Energy` function to define types based on it's
parameters and to choose the correct algorithms at compile time. For
example when the partition function is calculated (what is only
possible if the number of neurons in one of the layers is small), the
correct implementation is chosen based on the type of State Space
(e.g., :doxy:`TwoStateSpace` :doxy:`RealSpace`) the Neurons are
defined on.

The :doxy:`Energy` is the most basic concept of the RBM
module. Mathematically, it defines the family of probability
distributions modeled by the RBM. Therefore a lot of work is done by
this class.  Aside from calculating the energy, it also defines the
types of the neurons in the hidden and visible layer (e.g.,
:doxy:`BinaryLayer` or :doxy:`GaussianLayer`) .  The layers are tied
together by an interaction term which is usually a
vector-matrix-vector product. The parameters of the neurons and the
interaction term together define the parameters of the
distribution. In the most known energy functions the sets of
parameters are made up of the bias vectors of the layers and the
weight matrix of the interaction term.  But for example Gaussian
distributions can also define variance parameters. More fancy
distributions like the Beta-Distribution also require additional
weight matrices.

RBM training is based on steepest ascent on an approximation of the 
gradient of the log-likelihood. There are a lot of different approximation 
algorithms, most of them relying on Markov Chain Monte Carlo sampling schemes. 
In this implementation these sampling schemes are based on two components: 
the transition operator and the Markov chain.
The transition operator takes a pair of visible and hidden states and 
samples a new pair from them. Additionally a lot of information needed 
for calculating the gradient can be stored, for example the
conditional probability for a binary unit to be one. The most prominent example of such a transition
operator is :doxy:`GibbsOperator` [GemanGeman1984]_. For real valued cases also Hamilton Sampling can be used.
A Markov chain holds the current state of the hidden and visible variables and 
generates the transitions to the next states by a transition operator. 
It can be applied repeatedly to run the :doxy:`MarkovChain` several steps at once. 
Applying the transition operator at different temperatures leads to a tempered Markov Chain.

Most often we need samples to approximate the log-likelihood gradient, 
but also some approximations of the partition function rely on these samples.
Since different Energies lead to different log-likelihood gradients, the energy provides the information
how to approximate the gradient of the energy function given a sample. 
Still, there are different ways to organize the sampling process.
Most often, it has to be decided whether samples should be generated by one Markov Chain only (:doxy:`SingleChainApproximator`)
or whether several independent Markov chains should be used (:doxy:`MultiChainApproximator`).

A lot of standard algorithms can be created by using the components. For example a Gibbs Operator with a standard
Markov chain and a gradient approximation using several independent chains gives raise to the
Persistent Contrastive Divergence [Tieleman2008]_ algorithm. 
Using an ensemble of tempered Markov chains will create Parallel Tempering [DesjardinsEtAl2010]_.

Implementation Status
+++++++++++++++++++++++++

Not all parts described above are available in the current release. Missing are

* Hamiltonian Sampling Operator,
* Several Neurons and Energies,
* Tempered Transitions.

However, they will be available in the near future after some further testing.


What now?
+++++++++++++++++++++++++
You can see how to train a simple RBM with binary neurons in the tutorial
:doc:`../algorithms/binary_rbm`.

References
+++++++++++++++++++++++++

.. [WellingEtAl2005] M. Welling, M. Rosen-Zvi, G.E. Hinton, L.K. Saul.
   Exponential Family Harmoniums with an Application to Information Retrieval.
   Advances in Neural Information Processing Systems (NIPS 17), MIT Press, 2005, 1481-1488

.. [GemanGeman1984] S. Geman and D. Geman. Stochastic relaxation, Gibbs distributions and the Bayesian restoration of images.
	 IEEE Transactions on Pattern Analysis and Machine Intelligence, Routledge, 1984, 6, 721-741

.. [Smolensky1986] P. Smolensky Information Processing in Dynamical Systems: Foundations of Harmony Theory Parallel distributed processing:
	explorations in the microstructure of cognition, vol. 1: Foundations, MIT Press, 1986, 194-281

.. [Hinton2002] G.E. Hinton.  Training Products of Experts by Minimizing Contrastive Divergence Neural Computation, 2002, 14, 1771-1800

.. [Tieleman2008] T. Tieleman. Training restricted Boltzmann machines using approximations to the likelihood gradient.
   International Conference on Machine learning (ICML), ACM, 2008, 1064-1071

.. [DesjardinsEtAl2010] G. Desjardins, A. Courville, Y. Bengio, P. Vincent, O. Dellaleau.
	Parallel Tempering for Training of Restricted Boltzmann Machines.
	Journal of Machine Learning Research Workshop and Conference Proceedings, 2010, 9, 145-152

.. [Hinton2007] G.E. Hinton. Learning multiple layers of representation.
	 Trends in Cognitive Sciences, 2007, 11, 428-434

.. [MacKay2002]  D.J.C.MacKay.
   Information Theory, Inference & Learning Algorithms. Cambridge
   University Press, 2002.

.. [Welling2007] M. Welling.
   `Product of experts
   <http://www.scholarpedia.org/article/Product_of_experts>`_. Scholarpedia,
   2(10):3879, 2007.
