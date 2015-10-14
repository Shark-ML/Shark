//===========================================================================
/*!
 *
 *
 * \brief       Project budget maintenance strategy
 *
 * \par
 * This is an budget strategy that simply project one of the
 * budget vectors onto the others. To save time, the smallest
 * vector (measured in 2-norm of the alpha-coefficients) will
 * be selected for projection.
 *
 *
 *
 *
 * \author      Aydin Demircioglu
 * \date        2014
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
 *
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
 *
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================


#ifndef SHARK_MODELS_PROJECTBUDGETMAINTENANCESTRATEGY_H
#define SHARK_MODELS_PROJECTBUDGETMAINTENANCESTRATEGY_H

#include <shark/Models/Converter.h>
#include <shark/Models/Kernels/AbstractKernelFunction.h>
#include <shark/ObjectiveFunctions/KernelBasisDistance.h>
#include <shark/Data/Dataset.h>
#include <shark/Data/DataView.h>


namespace shark {

    ///
    /// \brief Budget maintenance strategy that projects a vector
    ///
    /// This is an budget strategy that simply projects one of the
    /// budget vectors onto the others. The resulting coefficients are
    /// then added to the other vectors and the projected vector is
    /// removed from the budget. 
    /// 
    template<class InputType>
    class ProjectBudgetMaintenanceStrategy: public AbstractBudgetMaintenanceStrategy<InputType> {
            typedef KernelExpansion<InputType> ModelType;
            typedef LabeledData<InputType, unsigned int> DataType;
            typedef typename DataType::element_type ElementType;

        public:

            /// constructor.
            ProjectBudgetMaintenanceStrategy() {
            }


            /// add to model.
            /// this is just a fake here, as it is unclear in general how to merge two objects,
            /// one needs to specialize this template.
            ///
            /// @param[in,out]  model   the model the strategy will work with
            /// @param[in]  alpha   alphas for the new budget vector 
            /// @param[in]  supportVector the vector to add to the model by applying the maintenance strategy
            ///            
            virtual void addToModel (ModelType& model, InputType const& alpha, ElementType const& supportVector) {
                // to project we simply utilize the kernel basis distance
            }


            /// class name
            std::string name() const
            { return "ProjectBudgetMaintenanceStrategy"; }

        protected:

    };

    ///
    /// \brief Budget maintenance strategy that projects a vector
    ///
    /// \par This is an specialization of the project budget maintenance strategy 
    /// that handles simple real-valued vectors. This is a nearly 1:1 adoption of
    /// the strategy presented in Wang, Cramer and Vucetic.
    /// 
    template<>
    class ProjectBudgetMaintenanceStrategy<RealVector>: public AbstractBudgetMaintenanceStrategy<RealVector> {
            typedef RealVector InputType;
            typedef KernelExpansion<InputType> ModelType;
            typedef LabeledData<InputType, unsigned int> DataType;
            typedef typename DataType::element_type ElementType;

        public:

            /// constructor.
            ProjectBudgetMaintenanceStrategy() {
            }


                
            /// add a vector to the model.
            /// this will add the given vector to the model and merge the budget so that afterwards
            /// the budget size is kept the same. If the budget has a free entry anyway, no merging
            /// will be performed, but instead the given vector is simply added to the budget.
            ///
            /// @param[in,out]  model   the model the strategy will work with
            /// @param[in]  alpha   alphas for the new budget vector 
            /// @param[in]  supportVector the vector to add to the model by applying the maintenance strategy
            ///
            virtual void addToModel (ModelType& model, InputType const& alpha, ElementType const& supportVector) {

                // projecting should try out every budget vector
                // but as this would yield $O(B^3)$ runtime, the vector
                // with the smallest alpha is taken instead.

                // first put the new vector into place
                size_t maxIndex = model.basis().numberOfElements();
                model.basis().element(maxIndex - 1) = supportVector.input;
                row (model.alpha(), maxIndex - 1) = alpha;
                
                
                size_t firstIndex = 0;
                double firstAlpha = 0;
                findSmallestVector (model, firstIndex, firstAlpha);
                
                // if the smallest vector has zero alpha, 
                // the budget is not yet filled so we can skip projecting.
                if (firstAlpha == 0.0f)
                {
                    // as we need the last vector to be zero, we put the new
                    // vector to that place and undo our putting-the-vector-to-back-position
                    model.basis().element(firstIndex) = supportVector.input;
                    row (model.alpha(), firstIndex) = alpha;

                    // enough to zero out the alpha
                    row (model.alpha(), maxIndex - 1).clear();
                    
                    // ready.
                    return;
                }
                
                // now solve the projection equation
                
                // we need to project the one vector we have chosen down
                // to all others. so we need a model with just thet one vector
                // and then we try to approximate alphas from the rest of thet
                // vectors, such that the difference is small as possible.
                
                // create a KernelExpansion just with the one selected vector.
                std::vector<RealVector> singlePointVector (1,model.basis().element(firstIndex));
                std::vector<unsigned int> singlePointLabel (1, 0);
                ClassificationDataset singlePointData = createLabeledDataFromRange(singlePointVector, singlePointLabel);
                KernelExpansion<RealVector> singlePointExpansion(model.kernel(), singlePointData.inputs(), false, model.alpha().size2());
                row (singlePointExpansion.alpha(), 0) = row (model.alpha(), firstIndex);
                KernelBasisDistance distance(&singlePointExpansion, maxIndex - 1);

                // now create a whole new 'point' with all the other vectors. 
                // then our approximation will give us one coefficient to approximate
                // the basis, which consits only of the one selected vector.
                // thus, we approximate the one vector with the rest of them.
                size_t inputDimension = model.basis().element(0).size();
                RealVector point(inputDimension * (maxIndex - 1));

                // copy all other budget vectors into one big vector
                size_t linearIndex = 0;
                for(std::size_t t = 0; t < maxIndex; t++){
                    // do not copy the first index.
                    if (t == firstIndex)
                        continue;
                    noalias(subrange(point, linearIndex*inputDimension, (linearIndex+1)*inputDimension)) = (model.basis().element(t));
                    linearIndex++;
                }
                
                //calculate solution found by the function and check that it is close
                RealMatrix projectedAlphas = distance.findOptimalBeta(point);

                // stupid sanity check
                SHARK_ASSERT (projectedAlphas.size2() == model.alpha().size2());
                
                // add the projected values to the budget
                linearIndex = 0;
                for (std::size_t j = 0; j < maxIndex; j++)
                {
                    if (j == firstIndex)
                        continue;
            
                    // for each class we add the alpha contribution to the true alphas.
                    for (std::size_t c = 0; c < model.alpha().size2(); c++) 
                        model.alpha(j, c) += projectedAlphas(linearIndex, c);
                    
                    linearIndex++;
                }
                
                // overwrite the projected vector with the last vector
                model.basis().element(firstIndex) = supportVector.input;
                row (model.alpha(), firstIndex) = alpha;

                // zero out buffer, enough to zero out the alpha
                row (model.alpha(), maxIndex - 1).clear();
            }


            /// class name
            std::string name() const
            { return "ProjectBudgetMaintenanceStrategy"; }

        protected:

    };


}
#endif
