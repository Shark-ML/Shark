#include <shark/Unsupervised/RBM/BinaryRBM.h>
#include <shark/LinAlg/Base.h>
#include <shark/Rng/GlobalRng.h>

#define BOOST_TEST_MODULE RBM_Energy
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;

//structure of values which is used for the tests
struct RBMFixture {
    RBMFixture():rbm(Rng::globalRng) {
        rbm.setStructure(5,5);
        rbm.weightMatrix().clear();
        hiddenState.resize(1,5);
        visibleState.resize(1,5);

        for(std::size_t i = 0; i != 5; ++i) {
            rbm.weightMatrix()(i,i) = i;
            rbm.hiddenNeurons().bias()(i) = i;
            rbm.visibleNeurons().bias()(i) = 5+i;
            visibleState(0,i) = 10+i;
            hiddenState(0,i) = 15+i;
        }

        visibleInput = prod(hiddenState,rbm.weightMatrix());
        hiddenInput = prod(visibleState,trans(rbm.weightMatrix()));

        visibleStateVec = row(visibleState,0);
        hiddenStateVec = row(hiddenState,0);
        visibleInputVec = row(visibleInput,0);
        hiddenInputVec = row(hiddenInput,0);

        energyResult = - inner_prod(rbm.hiddenNeurons().bias(),hiddenStateVec);
        energyResult-= inner_prod(rbm.visibleNeurons().bias(),visibleStateVec);
        energyResult-= inner_prod(hiddenStateVec,prod(rbm.weightMatrix(),visibleStateVec));
    }
    BinaryRBM rbm;
    RealMatrix visibleState;
    RealMatrix hiddenState;
    RealMatrix visibleInput;
    RealMatrix hiddenInput;

    RealVector visibleStateVec;
    RealVector hiddenStateVec;
    RealVector visibleInputVec;
    RealVector hiddenInputVec;

    double energyResult;
};

BOOST_FIXTURE_TEST_SUITE (RBM_Energy, RBMFixture )

BOOST_AUTO_TEST_CASE( Energy_Input) {
    Energy<BinaryRBM> energy(rbm.energy());

    RealMatrix visibleInputResult(1,5);
    RealMatrix hiddenInputResult(1,5);

    energy.inputHidden(hiddenInputResult,visibleState);
    energy.inputVisible(visibleInputResult,hiddenState);

    BOOST_CHECK_SMALL(norm_sqr(row(visibleInputResult-visibleInput,0)), 1.e-15);
    BOOST_CHECK_SMALL(norm_sqr(row(hiddenInputResult-hiddenInput,0)), 1.e-15);
}

BOOST_AUTO_TEST_CASE( Energy_EnergyFromInput )
{
    Energy<BinaryRBM> energy(rbm.energy());

    RealVector energyVisible = energy.energyFromVisibleInput(visibleInput,hiddenState,visibleState);
    RealVector energyHidden = energy.energyFromHiddenInput(hiddenInput,hiddenState,visibleState);

    BOOST_CHECK_SMALL(energyVisible(0) - energyResult, 1.e-15);
    BOOST_CHECK_SMALL(energyHidden(0) - energyResult, 1.e-15);
}
BOOST_AUTO_TEST_CASE( Energy_SimpleEnergy )
{
    Energy<BinaryRBM> energy(rbm.energy());

    RealVector simpleEnergy = energy.energy(hiddenState,visibleState);

    BOOST_REQUIRE_SMALL(simpleEnergy(0) - energyResult, 1.e-15);

    //now some random sampling to get the energy
    {
        BinaryRBM bigRBM(Rng::globalRng);
        bigRBM.setStructure(10,18);
        initRandomNormal(bigRBM,2);

        RealMatrix inputBatch(10,10);
        RealMatrix hiddenBatch(10,18);
        RealVector energies(10);
        for(std::size_t j = 0; j != 10; ++j) {
            for(std::size_t k = 0; k != 10; ++k) {
                inputBatch(j,k)=Rng::coinToss(0.5);
            }
            for(std::size_t k = 0; k != 18; ++k) {
                hiddenBatch(j,k)=Rng::coinToss(0.5);
            }
            energies(j) = - inner_prod(bigRBM.hiddenNeurons().bias(),row(hiddenBatch,j));
            energies(j)-= inner_prod(bigRBM.visibleNeurons().bias(),row(inputBatch,j));
            energies(j)-= inner_prod(row(hiddenBatch,j),prod(bigRBM.weightMatrix(),row(inputBatch,j)));
        }
        Energy<BinaryRBM> bigEnergy(bigRBM.energy());
        RealVector testEnergies=bigEnergy.energy(hiddenBatch,inputBatch);

        for(std::size_t i = 0; i != 10; ++i) {
            BOOST_CHECK_CLOSE(energies(i),testEnergies(i),1.e-5);
        }
    }
}

BOOST_AUTO_TEST_CASE( Energy_UnnormalizedProbabilityHidden )
{
    //all possible state combinations for 2 visible units
    RealMatrix visibleStateSpace(4,2);
    visibleStateSpace(0,0)=0;
    visibleStateSpace(0,1)=0;
    visibleStateSpace(1,0)=0;
    visibleStateSpace(1,1)=1;
    visibleStateSpace(2,0)=1;
    visibleStateSpace(2,1)=0;
    visibleStateSpace(3,0)=1;
    visibleStateSpace(3,1)=1;

    //create RBM with 2 visible and 4 hidden units and initialize it randomly
    BinaryRBM rbm(Rng::globalRng);
    rbm.setStructure(2,4);
    initRandomNormal(rbm,2);

    //the hiddenstate to test is the most complex (1,1,1,1) case
    RealMatrix hiddenState = RealScalarMatrix(4,4,1.0);

    //calculate energies for the state space, we now from the previous tests, that this result is correct
    Energy<BinaryRBM> energy(rbm.energy());
    RealVector energies = energy.energy(hiddenState,visibleStateSpace);

    //now test for several choices of beta
    for(std::size_t i = 0; i <= 10; ++i) {
        //calculate unnormalized probability of the hiddenState by integrating over the visible state space
        double pTest=sum(exp(-(i*0.1)*energies));

        //calculate now the test itself
        double p=std::exp(energy.logUnnormalizedProbabilityHidden(hiddenState,blas::repeat(i*0.1,4))(0));
        BOOST_CHECK_CLOSE(pTest,p,2.e-5);
    }
}
BOOST_AUTO_TEST_CASE( Energy_UnnormalizedProbabilityVisible )
{
    //all possible state combinations for 2 hidden units
    RealMatrix hiddenStateSpace(4,2);
    hiddenStateSpace(0,0)=0;
    hiddenStateSpace(0,1)=0;
    hiddenStateSpace(1,0)=0;
    hiddenStateSpace(1,1)=1;
    hiddenStateSpace(2,0)=1;
    hiddenStateSpace(2,1)=0;
    hiddenStateSpace(3,0)=1;
    hiddenStateSpace(3,1)=1;

    //create RBM with 4 visible and 2 hidden units and initialize it randomly
    BinaryRBM rbm(Rng::globalRng);
    rbm.setStructure(4,2);
    initRandomNormal(rbm,2);

    //the hiddenstate to test is the most complex (1,1,1,1) case
    RealMatrix visibleState = RealScalarMatrix(4,4,1.0);

    //calculate energies for the state space, we now from the previous tests, that this result is correct
    Energy<BinaryRBM> energy(rbm.energy());
    RealVector energies = energy.energy(hiddenStateSpace,visibleState);

    //now test for several choices of beta
    for(std::size_t i = 0; i <= 10; ++i) {
        //calculate unnormalized probability of the visible state by integrating over the hidden state space
        double pTest=sum(exp(-(i*0.1)*energies));

        //calculate now the test itself
        double p=std::exp(energy.logUnnormalizedProbabilityVisible(visibleState,blas::repeat(i*0.1,4))(0));
        BOOST_CHECK_CLOSE(pTest,p,2.e-5);
    }
}

BOOST_AUTO_TEST_SUITE_END()
