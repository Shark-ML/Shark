#define BOOST_TEST_MODULE LinAlg_Metrics
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/LinAlg/Metrics.h>
#include <shark/LinAlg/Base.h>
#include <shark/Rng/GlobalRng.h>
#include <shark/Core/Timer.h>

using namespace shark;

BOOST_AUTO_TEST_SUITE (LinAlg_Metrics)

BOOST_AUTO_TEST_CASE( LinAlg_Norm_distanceSqr_Vector){
	//dense - dense
	{
		IntVector vec1(3);
		vec1(0) = 2;
		vec1(1) = 4;
		vec1(2) = 6;
		
		IntVector vec2(3);
		vec2(0) = 3;
		vec2(1) = 7;
		vec2(2) = 11;
		
		int result1 = distanceSqr(vec2,vec1);
		int result2 = distanceSqr(vec1,vec2);
		
		BOOST_CHECK_EQUAL(result1,35);
		BOOST_CHECK_EQUAL(result2,35);
	}
	
	//dense - compressed
	{
		IntVector vec1(3);
		vec1(0) = 2;
		vec1(1) = 4;
		vec1(2) = 6;
		
		CompressedIntVector vec2(3);
		vec2(1) = 7;
		
		int result1 = distanceSqr(vec2,vec1);
		int result2 = distanceSqr(vec1,vec2);
		
		BOOST_CHECK_EQUAL(result1,49);
		BOOST_CHECK_EQUAL(result2,49);
	}
	
	//compressed- compressed
	{
		CompressedIntVector vec1(100);
		vec1(0) = 2;
		vec1(10) = 4;
		vec1(20) = 6;
		
		CompressedIntVector vec2(100);
		vec2(0) = 3;
		vec2(8) = 7;
		vec2(20) = 11;
		
		int result1 = distanceSqr(vec2,vec1);
		int result2 = distanceSqr(vec1,vec2);
		
		BOOST_CHECK_EQUAL(result1,91);
		BOOST_CHECK_EQUAL(result2,91);
	}
}

BOOST_AUTO_TEST_CASE( LinAlg_Norm_distanceSqr_Matrix_Vector){
	//dense - dense
	{
		IntMatrix vec1(3,3);
		vec1(0,0) = 2;
		vec1(0,1) = 4;
		vec1(0,2) = 6;
		vec1(1,0) = 3;
		vec1(1,1) = 5;
		vec1(1,2) = 7;
		vec1(2,0) = 4;
		vec1(2,1) = 6;
		vec1(2,2) = 8;
		
		IntVector vec2(3);
		vec2(0) = 3;
		vec2(1) = 7;
		vec2(2) = 11;
		
		IntVector result1 = distanceSqr(vec2,vec1);
		IntVector result2 = distanceSqr(vec1,vec2);
		
		BOOST_CHECK_EQUAL(result1(0),35);
		BOOST_CHECK_EQUAL(result1(1),20);
		BOOST_CHECK_EQUAL(result1(2),11);
		BOOST_CHECK_EQUAL(result2(0),35);
		BOOST_CHECK_EQUAL(result2(1),20);
		BOOST_CHECK_EQUAL(result2(2),11);
	}
	
	//dense - compressed
	{
		CompressedIntMatrix vec1(3,3);
		vec1(0,2) = 6;
		vec1(2,0) = 4;
		vec1(2,1) = 6;
		vec1(2,2) = 8;
		
		IntMatrix vec3(3,3,0);
		vec3(0,2) = 6;
		vec3(2,0) = 4;
		vec3(2,1) = 6;
		vec3(2,2) = 8;
		
		IntVector vec2(3);
		vec2(0) = 3;
		vec2(1) = 7;
		vec2(2) = 11;
		
		CompressedIntVector vec4(3);
		vec4(0) = 3;
		vec4(1) = 7;
		vec4(2) = 11;
		
		IntVector result1 = distanceSqr(vec2,vec1);
		IntVector result2 = distanceSqr(vec1,vec2);
		IntVector result3 = distanceSqr(vec3,vec4);
		IntVector result4 = distanceSqr(vec4,vec3);
		
		BOOST_CHECK_EQUAL(result1(0),83);
		BOOST_CHECK_EQUAL(result1(1),179);
		BOOST_CHECK_EQUAL(result1(2),11);
		BOOST_CHECK_EQUAL(result2(0),83);
		BOOST_CHECK_EQUAL(result2(1),179);
		BOOST_CHECK_EQUAL(result2(2),11);
		BOOST_CHECK_EQUAL(result3(0),83);
		BOOST_CHECK_EQUAL(result3(1),179);
		BOOST_CHECK_EQUAL(result3(2),11);
		BOOST_CHECK_EQUAL(result4(0),83);
		BOOST_CHECK_EQUAL(result4(1),179);
		BOOST_CHECK_EQUAL(result4(2),11);
	}
	
	//compressed- compressed
	{
		CompressedIntMatrix vec1(3,100);
		vec1(0,0) = 2;
		vec1(0,10) = 4;
		vec1(0,20) = 6;
		vec1(2,1) = 1;
		vec1(2,2) = 6;
		vec1(2,3) = 2;
		
		CompressedIntVector vec2(100);
		vec2(0) = 3;
		vec2(8) = 7;
		vec2(20) = 11;
		
		IntVector result1 = distanceSqr(vec2,vec1);
		IntVector result2 = distanceSqr(vec1,vec2);
		
		BOOST_CHECK_EQUAL(result1(0),91);
		BOOST_CHECK_EQUAL(result1(1),179);
		BOOST_CHECK_EQUAL(result1(2),220);
		BOOST_CHECK_EQUAL(result2(0),91);
		BOOST_CHECK_EQUAL(result2(1),179);
		BOOST_CHECK_EQUAL(result2(2),220);	
	
	}
}

BOOST_AUTO_TEST_CASE( LinAlg_Norm_distanceSqr_Matrix_Matrix){
	//small dense block 
	{
		RealMatrix mat1(2,3);
		mat1(0,0) = 2;
		mat1(0,1) = 4;
		mat1(0,2) = 6;
		mat1(1,0) = 3;
		mat1(1,1) = 5;
		mat1(1,2) = 7;
		
		RealMatrix mat2(3,3);
		mat2(0,0) = 3;
		mat2(0,1) = 5;
		mat2(0,2) = 7;
		mat2(1,0) = 4;
		mat2(1,1) = 6;
		mat2(1,2) = 8;
		mat2(2,0) = 2;
		mat2(2,1) = 1;
		mat2(2,2) = 0;
		
		RealMatrix result1 = distanceSqr(mat1,mat2);
		RealMatrix result2 = distanceSqr(mat2,mat1);
		
		
		BOOST_CHECK_CLOSE(result1(0,0),3,1e-12);
		BOOST_CHECK_CLOSE(result1(0,1),12,1e-12);
		BOOST_CHECK_CLOSE(result1(0,2),45,1e-12);
		BOOST_CHECK_CLOSE(result1(1,0),0,1e-12);
		BOOST_CHECK_CLOSE(result1(1,1),3,1e-12);
		BOOST_CHECK_CLOSE(result1(1,2),66,1e-12);
		
		BOOST_CHECK_CLOSE(result2(0,0),3,1e-12);
		BOOST_CHECK_CLOSE(result2(0,1),0,1e-12);
		BOOST_CHECK_CLOSE(result2(1,1),3,1e-12);
		BOOST_CHECK_CLOSE(result2(1,0),12,1e-12);
		BOOST_CHECK_CLOSE(result2(2,0),45,1e-12);
		BOOST_CHECK_CLOSE(result2(2,1),66,1e-12);
	}
	//small compressed block 
	{
		CompressedIntMatrix mat1(2,3);
		mat1(0,0) = 2;
		mat1(0,1) = 4;
		mat1(0,2) = 6;
		mat1(1,0) = 3;
		mat1(1,1) = 5;
		mat1(1,2) = 7;
		
		CompressedIntMatrix mat2(3,3);
		mat2(0,0) = 3;
		mat2(0,1) = 5;
		mat2(0,2) = 7;
		mat2(1,0) = 4;
		mat2(1,1) = 6;
		mat2(1,2) = 8;
		mat2(2,0) = 2;
		mat2(2,1) = 1;
		mat2(2,2) = 0;
		
		IntMatrix result1 = distanceSqr(mat1,mat2);
		IntMatrix result2 = distanceSqr(mat2,mat1);
		
		
		BOOST_CHECK_EQUAL(result1(0,0),3);
		BOOST_CHECK_EQUAL(result1(0,1),12);
		BOOST_CHECK_EQUAL(result1(0,2),45);
		BOOST_CHECK_EQUAL(result1(1,0),0);
		BOOST_CHECK_EQUAL(result1(1,1),3);
		BOOST_CHECK_EQUAL(result1(1,2),66);
		
		BOOST_CHECK_EQUAL(result2(0,0),3);
		BOOST_CHECK_EQUAL(result2(0,1),0);
		BOOST_CHECK_EQUAL(result2(1,1),3);
		BOOST_CHECK_EQUAL(result2(1,0),12);
		BOOST_CHECK_EQUAL(result2(2,0),45);
		BOOST_CHECK_EQUAL(result2(2,1),66);
	}
	
	//big dense block 
	{
		RealMatrix mat1(64,3);
		RealMatrix mat2(64,3);
		for(std::size_t i = 0; i != 64; ++i){
			mat1(i,0) = 3*i;
			mat1(i,1) = 3*i+1;
			mat1(i,2) = 3*i+2;
			mat2(i,0) = 1.5*i;
			mat2(i,1) = 1.5*i+1;
			mat2(i,2) = 1.5*i+2;
		}
		
		RealMatrix result1 = distanceSqr(mat1,mat2);
		RealMatrix result2 = distanceSqr(mat2,mat1);
		
		for(std::size_t i = 0; i != 64; ++i){
			for(std::size_t j = 0; j != 64; ++j){
				double d = distanceSqr(row(mat1,i),row(mat2,j));
				BOOST_CHECK_CLOSE(result1(i,j),d,1.e-12);
				BOOST_CHECK_CLOSE(result2(j,i),d,1.e-12);
			}
		}
	}
}



BOOST_AUTO_TEST_SUITE_END()
