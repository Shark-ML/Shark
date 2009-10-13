#include <ReClaM/CMAC.h>


size_t CMACFunction::getArrayIndexForTiling(size_t indexOfTiling,Array<double> const& point)const
{

    size_t index = indexOfTiling * m_dimOffset(inputDimension);

    for(size_t dim = 0; dim != inputDimension; ++dim)
    {
        //Adjust range from (lower bound, upper bound) to (0,numberOfTiles)
        double coordinate = point(dim);
        coordinate -= m_tileBounds(dim, 0);//subtract lower bound
        coordinate -= m_offset(indexOfTiling, dim);//tiling offset
        //divide by the width of the tile to calculate the index
        coordinate /= m_tileBounds(dim, 1);
        //add index offset
        index += static_cast<size_t>(coordinate) * m_dimOffset(dim);
    }
    return index;
}

Array<size_t> CMACFunction::getIndizes(Array<double> const& point)const
{
    Array<size_t> output(m_tilings);
    output=0.0;

    for(size_t tiling = 0; tiling != m_tilings; ++tiling)
    {
        size_t index = getArrayIndexForTiling(tiling, point);
        output(tiling) = index;
    }
    return output;
}

CMACFunction::CMACFunction():m_tilings(0){}

CMACFunction::CMACFunction(int inputs, size_t numberOfTilings, size_t numberOfTiles, double lower, double upper)
    :m_offset(numberOfTilings, inputs), m_dimOffset(inputs + 1), m_tileBounds(inputs, 2), m_tilings(numberOfTilings)
{
    inputDimension  = inputs;
    outputDimension = 1;
    m_tilings       = numberOfTilings;

    //initialize bounds
    //same for all dimensions
    double tileWidth = (upper - lower) / (numberOfTiles - 1);
    for(size_t dim = 0;dim != inputDimension; ++dim)
    {
        m_tileBounds(dim, 0) = lower;
        m_tileBounds(dim, 1) = tileWidth;
    }

    //calculate number of parameters and the offsets for every input dimension
    size_t numberOfParameters = 1;
    for(size_t inputDim = 0;inputDim != inputDimension + 1; ++inputDim)
    {
        m_dimOffset(inputDim) = numberOfParameters;
        numberOfParameters *= numberOfTiles;
    }
    numberOfParameters *= m_tilings;

    //initialize parameter array with random values
    parameter.resize(numberOfParameters);
    for(size_t i=0;i!=numberOfParameters;++i)
        parameter(i) = Rng::gauss(0,1);
}

CMACFunction::CMACFunction(int inputs, size_t numberOfTilings, size_t numberOfTiles, Array<double> const& bounds)
    :m_offset(numberOfTilings, inputs), m_dimOffset(inputs + 1), m_tileBounds(inputs, 2), m_tilings(numberOfTilings)
{
    inputDimension  = inputs;
    outputDimension = 1;
    m_tilings       = numberOfTilings;

    //initialize bounds
    for(size_t dim=0; dim != inputDimension; ++dim)
    {
        double tileWidth = (bounds(dim, 1) - bounds(dim, 0)) / (numberOfTiles - 1);
        m_tileBounds(dim, 0) = bounds(dim, 0);
        m_tileBounds(dim, 1) = tileWidth;
    }

    //calculate number of parameters and the offsets for every input dimension
    size_t numberOfParameters = 1;
    for(size_t inputDim = 0; inputDim != inputDimension + 1; ++inputDim)
    {
        m_dimOffset(inputDim) = numberOfParameters;
        numberOfParameters *= numberOfTiles;
    }
    numberOfParameters *= m_tilings;
    //initialize parameter array with random values
    parameter.resize(numberOfParameters);
    for(size_t i=0;i!=numberOfParameters;++i)
        parameter(i) = Rng::gauss(0,1);
}

void CMACFunction::init(bool randomTiles)
{
    m_offset = 0;
    for(unsigned tiling = 0; tiling < m_tilings; ++tiling)
    {
        for(unsigned dim = 0; dim < inputDimension; ++dim)
        {
            if(!randomTiles)
                m_offset(tiling, dim) -= m_tileBounds(dim,1)/(tiling+1);
            else
                m_offset(tiling, dim) -= Rng::uni(0, m_tileBounds(dim,1));
        }
    }
}

void CMACFunction::model(Array<double> const& input, Array<double>& output)
{
    if (input.ndim() == 1)
    {
        output.resize(1);
        output = 0;

        Array<size_t> indizes = getIndizes(input);
        for(size_t i = 0; i != m_tilings; ++i)
            output(0) += parameter( indizes(i) );
    }
    else if (input.ndim() == 2)
    {
        output.resize(input.dim(0), 1u);
        output = 0;

        for(size_t i = 0;i != input.dim(0); ++i)
        {
            Array<size_t> indizes = getIndizes(input);
            for(size_t j = 0;j != m_tilings; ++j)
                output(i,0) += parameter( indizes(j) );
        }
    }
    else throw SHARKEXCEPTION("[CMAC::model] invalid number of dimensions.");
}

void CMACFunction::modelDerivative(Array<double> const& input, Array<double>& derivative)
{
    if (input.ndim() != 1)
        throw SHARKEXCEPTION("[CMAC::modelDerivative] invalid number of dimensions.");

    derivative.resize(1, getParameterDimension());
    derivative = 0;

    Array<size_t> indizes = getIndizes(input);
    for(size_t j=0; j != m_tilings; ++j)
        derivative(0, indizes(j) ) = 1;
}
