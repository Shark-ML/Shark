#define BOOST_TEST_MODULE CoreScopedHandleTestModule
#include "shark/Core/utility/ScopedHandle.h"

#include <boost/bind.hpp>
#include <boost/test/unit_test.hpp>

namespace shark {

/// Fixture for testing ScopedHandle
class ScopedHandleFixture
{
public:
	/// A deleter for testing
	void deleteMe(int handle)
	{
		m_deletedHandles.push_back(handle);
	}

	/// A test verifier for testing
	bool verifyMe(int handle) const
	{
		// Return true for valid handlers
		return handle < 100;
	}

	/// Save deleted handles for verification
	std::vector<int> m_deletedHandles;
};

BOOST_FIXTURE_TEST_SUITE (Core_ScopedHandleTests, ScopedHandleFixture)

BOOST_AUTO_TEST_CASE(BasicTest)
{
	// Test that ScopedHandle is able to hold a valid handle, access it and delete it upon destruction
	const int validHandle = 10;
	const int invalidHandle = -1;
	BOOST_REQUIRE_EQUAL(m_deletedHandles.size(), 0u);
	{
		ScopedHandle<int> handle(validHandle, boost::bind(&ScopedHandleFixture::deleteMe, this, _1));
		BOOST_CHECK_EQUAL(*handle, validHandle);
	}
	BOOST_CHECK_EQUAL(m_deletedHandles.size(), 1u);
	BOOST_CHECK_EQUAL(m_deletedHandles[0], validHandle);

	m_deletedHandles.clear();

	// Test that ScopedHandle will throw exception when passed handle is invalid
	// and destruction should do no harm
	{
		BOOST_CHECK_THROW(ScopedHandle<int> handle(invalidHandle, boost::bind(&ScopedHandleFixture::deleteMe, this, _1)), Exception);
	}
	BOOST_CHECK_EQUAL(m_deletedHandles.size(), 0u);
}

BOOST_AUTO_TEST_CASE(CustomizedVerifier)
{
	// Test that ScopedHandle is able to accept customized verifier and description

	const int validHandle = 10;
	BOOST_REQUIRE_EQUAL(m_deletedHandles.size(), 0u);
	{
		ScopedHandle<int> handle(
			validHandle,
			boost::bind(&ScopedHandleFixture::deleteMe, this, _1),
			"my description 1",
			boost::bind(&ScopedHandleFixture::verifyMe, this, _1));
		BOOST_CHECK_EQUAL(*handle, validHandle);
	}
	BOOST_CHECK_EQUAL(m_deletedHandles.size(), 1u);
	BOOST_CHECK_EQUAL(m_deletedHandles[0], validHandle);
	m_deletedHandles.clear();

	// Test that ScopedHandle will save a value not a reference
	const int handleValue1 = 20;
	const int handleValue2 = 30;
	int myHandle = handleValue1;
	{
		ScopedHandle<int> handle(
			myHandle,
			boost::bind(&ScopedHandleFixture::deleteMe, this, _1),
			"my description 1",
			boost::bind(&ScopedHandleFixture::verifyMe, this, _1));
		myHandle = handleValue2; // change original one to simulate the scenario that the handle was recycled after passed to ScopedHandle
		BOOST_CHECK_EQUAL(*handle, handleValue1);
	}
	BOOST_CHECK_EQUAL(m_deletedHandles.size(), 1u);
	BOOST_CHECK_EQUAL(m_deletedHandles[0], handleValue1);

	m_deletedHandles.clear();

	// Test that ScopedHandle will throw exception when passed handle is invalid
	// and destruction should do no harm
	{
		const int invalidHandle = 200;
		BOOST_CHECK_THROW(
			ScopedHandle<int> handle(
				invalidHandle,
				boost::bind(&ScopedHandleFixture::deleteMe, this, _1),
				"my description 2",
				boost::bind(&ScopedHandleFixture::verifyMe, this, _1)),
			Exception);
	}
	BOOST_CHECK_EQUAL(m_deletedHandles.size(), 0u);
}

BOOST_AUTO_TEST_SUITE_END()

} // namespace shark {
