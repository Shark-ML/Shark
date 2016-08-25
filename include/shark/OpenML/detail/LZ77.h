//===========================================================================
/*!
 * 
 *
 * \brief       Decoding of LZ77 (Lempel-Ziv 1977) encoded streams.
 * 
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2016
 *
 *
 * \par Copyright 1995-2016 Shark Development Team
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

#ifndef SHARK_OPENML_DETAIL_LZ77_H
#define SHARK_OPENML_DETAIL_LZ77_H


#include <shark/Core/DLLSupport.h>

#include <cstdint>
#include <cstddef>
#include <cassert>
#include <stdexcept>
#include <utility>
#include <string>
#include <vector>


namespace shark {
namespace openML {
namespace detail {


class CRC32
{
public:
	CRC32()
	: m_checksum(0xffffffff)
	{ }

	std::uint32_t checksum() const
	{ return (m_checksum ^ 0xffffffff); }

	template <typename Iterator>
	void process(Iterator begin, Iterator end)
	{
		for (; begin != end; ++begin)
		{
			std::uint32_t tmp = m_checksum;
			tmp ^= (std::uint8_t)(*begin);
			m_checksum = (m_checksum >> 8) ^ m_tab[tmp & 255];
		}
	}

	template <typename Container>
	void process(Container const& container)
	{ process(container.begin(), container.end()); }

private:
	std::uint32_t m_checksum;
	static const std::uint32_t m_tab[256];
};


class Adler32
{
public:
	Adler32()
	: m_s1(1)
	, m_s2(0)
	{ }

	std::uint32_t checksum() const
	{ return m_s1 | (m_s2 << 16); }

	template <typename Iterator>
	void process(Iterator begin, Iterator end)
	{
		for (; begin != end; ++begin)
		{
			m_s1 = m_s1 + (std::uint8_t)(*begin);
			if (m_s1 >= 65521) m_s1 -= 65521;
			m_s2 = (m_s1 + m_s2);
			if (m_s2 >= 65521) m_s2 -= 65521;
		}
	}

	template <typename Container>
	void process(Container const& container)
	{ process(container.begin(), container.end()); }

private:
	std::uint32_t m_s1;
	std::uint32_t m_s2;
};


// decode Huffman codes quickly with a lookup table
class HuffmanDecoder
{
public:
	// initialization: define new tree, call defineCode for each code afterwards
	void reset(unsigned int maximalCodeLength)
	{
		m_lookup.clear();
		m_lookup.resize(1 << maximalCodeLength);
	}

	// initialization: define the code of a literal
	void defineCode(unsigned int value, unsigned int code, unsigned int bits)
	{
		unsigned int step = 1 << bits;
		unsigned int mask = step - 1;
		Value v(bits, value);
		for (unsigned int i = (code & mask); i<m_lookup.size(); i+=step) m_lookup[i] = v;
	}

	// decode a single literal
	void decode(unsigned int code, unsigned int& bits, unsigned int& value)
	{
		unsigned int mask = m_lookup.size() - 1;
		Value const& val = m_lookup[code & mask];
		bits = val.bits;
		value = val.value;
	}

private:
	struct Value
	{
		Value(unsigned int bits_ = 0, unsigned int value_ = 0)
		: bits(bits_)
		, value(value_)
		{ }

		unsigned int bits;
		unsigned int value;
	};
	std::vector<Value> m_lookup;
};


// The StreamDecoderRFC1951 class supports decoding of arbitrarily long
// RFC-1951 (LZ77) streams with bounded buffers. In case of an invalid
// encoding it throws an exception.
SHARK_EXPORT_SYMBOL class StreamDecoderRFC1951
{
public:
	StreamDecoderRFC1951()
	: m_mode(emBlockType)
	, m_bLastBlock(false)
	, m_uncompressed(0)
	, m_bits(0)
	, m_numberOfBits(0)
	, m_bufferSize(0)
	, m_bufferEnd(0)
	{ }

	// The decode function reads bytes from the input stream and writes
	// bytes to the output stream. It stops if the input buffer is fully
	// processed or if the output buffer is full, or if the next atomic
	// operation would exceed one of the buffers. The function returns
	// the number of bytes processed for both streams.
	std::pair<std::size_t, std::size_t> decode(
			const char* inputbuffer,
			std::size_t inputsize,
			char* outputbuffer,
			std::size_t outputsize);

	// Check whether the end marker has been encountered.
	bool eof() const
	{ return (m_mode == emEOF); }

private:
	enum Mode
	{
		// decoder states
		emBlockType,                            // block type (3 bits)
		emUncompressedHeader,                   // read header size (2x 16 bits)
		emUncompressed,                         // copy next m_uncompressed bytes to output
		emHuffmanTree,                          // Huffman tree definition
		emCompressed,                           // literal or (length, distance)-pair
		emEOF,                                  // the stream is at its end
	};
	Mode m_mode;                                // what does the algorithm expect to read next?
	bool m_bLastBlock;                          // true if this is the last gzip block
	unsigned int m_uncompressed;                // number of remaining uncompressed bytes

	HuffmanDecoder m_literal;                   // Huffman tree for decoding literals/lengths
	HuffmanDecoder m_distance;                  // Huffman tree for decoding distances

	std::uint64_t m_bits;                       // buffer for input bits so that we always have >= 56 bits available
	unsigned int m_numberOfBits;                // number of bits in the input buffer

	unsigned char m_buffer[32768];              // 32K output buffer for lookups
	unsigned int m_bufferSize;                  // current size of the output buffer
	unsigned int m_bufferEnd;                   // position where to extend the circular buffer
};


// The decodeRFC1951 function is a simplified frontend for decompressing
// fixed.size (i.e., non-streaming) content.
SHARK_EXPORT_SYMBOL template <typename ContainerType, typename ReturnType = ContainerType>
ReturnType decodeRFC1951(ContainerType const& compressed)
{
	static_assert(sizeof(typename ContainerType::value_type) == 1, "[decodeRFC1951] The input container must contain bytes.");
	static constexpr std::size_t buffersize = 16384;
	char buffer[buffersize];
	ReturnType ret;
	StreamDecoderRFC1951 decoder;
	std::size_t pos = 0;
	while (! decoder.eof())
	{
		assert(pos <= compressed.size());
		std::pair<uint64_t, uint64_t> io = decoder.decode((const char*)&compressed[pos], compressed.size() - pos, buffer, buffersize);
		pos += io.first;
		ret.insert(ret.end(), buffer, buffer + io.second);
	}
	if (pos != compressed.size()) throw std::runtime_error("[decodeRFC1951] excess input bytes");
	return ret;
}


// The StreamDecoderRFC1950 class supports decoding of arbitrarily long
// RFC-1950 ("deflate") streams with bounded buffers. In case of an
// invalid encoding it throws an exception.
SHARK_EXPORT_SYMBOL class StreamDecoderRFC1950
{
public:
	StreamDecoderRFC1950()
	: m_mode(emHeader)
	{ }

	// The decode function reads bytes from the input stream and writes
	// bytes to the output stream. It stops if the input buffer is fully
	// processed or if the output buffer is full, or if the next atomic
	// operation would exceed one of the buffers. The function returns
	// the number of bytes processed for both streams.
	std::pair<std::size_t, std::size_t> decode(
			const char* inputbuffer,
			std::size_t inputsize,
			char* outputbuffer,
			std::size_t outputsize);

	// Check whether the input stream is at its end.
	bool eof() const
	{ return (m_mode == emEOF); }

private:
	enum Mode
	{
		// decoder states
		emHeader,                               // ridiculous RFC-1950 header
		em1951,                                 // redirect processing to RFC-1951 decoder
		emAdler32,                              // Adler32 checksum
		emEOF,                                  // the stream is at its end
	};
	Mode m_mode;                                // what does the algorithm expect to read next?
	StreamDecoderRFC1951 m_decoder;                // RFC-1951 decoder to do the actual work
	Adler32 m_checksum;                         // Adler32 checksum for uncompressed content
};


// The decode function is a simplified frontend for decompressing
// fixed.size (i.e., non-streaming) content. Only very weak guarantees
// can be made on the size of the output string.
SHARK_EXPORT_SYMBOL template <typename ContainerType, typename ReturnType = ContainerType>
ReturnType decodeRFC1950(ContainerType const& compressed)
{
	static constexpr std::size_t buffersize = 16384;
	char buffer[buffersize];
	ReturnType ret;
	StreamDecoderRFC1950 decoder;
	std::size_t pos = 0;
	while (! decoder.eof())
	{
		std::pair<uint64_t, uint64_t> io = decoder.decode((const char*)&compressed[pos], compressed.size() - pos, buffer, buffersize);
		pos += io.first;
		ret.insert(ret.end(), buffer, buffer + io.second);
	}
	if (pos != compressed.size()) throw std::runtime_error("[decodeRFC1950] excess input bytes");
	return ret;
}


// This function introduces the alias name "deflate" for "decodeRFC1950".
SHARK_EXPORT_SYMBOL template <typename ContainerType, typename ReturnType = ContainerType>
ReturnType deflate(ContainerType const& compressed)
{ return decodeRFC1950<ContainerType, ReturnType>(compressed); }


// The StreamDecoderRFC1952 class supports decoding of arbitrarily long
// RFC-1952 ("gzip") streams with bounded buffers. In case of an invalid
// encoding it throws an exception.
SHARK_EXPORT_SYMBOL class StreamDecoderRFC1952
{
public:
	StreamDecoderRFC1952()
	: m_mode(emHeader)
	, m_size(0)
	{ }

	// The decode function reads bytes from the input stream and writes
	// bytes to the output stream. It stops if the input buffer is fully
	// processed or if the output buffer is full, or if the next atomic
	// operation would exceed one of the buffers. The function returns
	// the number of bytes processed for both streams.
	std::pair<std::size_t, std::size_t> decode(
			const char* inputbuffer,
			std::size_t inputsize,
			char* outputbuffer,
			std::size_t outputsize);

	// Check whether the input stream is at its end.
	bool eof() const
	{ return (m_mode == emEOF); }

private:
	enum Mode
	{
		// decoder states
		emHeader,                               // RFC-1952 header
		emExtra,                                // RFC-1952 header "extra" field
		emName,                                 // RFC-1952 header filename field
		emComment,                              // RFC-1952 header comment field
		emCRC16,                                // RFC-1952 header CRC16 checksum
		em1951,                                 // redirect processing to RFC-1951 decoder
		emTrailer,                              // trailer consisting of CRC-32 and size fields
		emEOF,                                  // the stream is at its end
	};
	Mode m_mode;                                // what does the algorithm expect to read next?
	unsigned char m_flg;                        // FLG (flags) field
	std::size_t m_extra;                        // size of the "extra" field
	StreamDecoderRFC1951 m_decoder;             // RFC-1951 decoder to do the actual work
	CRC32 m_checksum;                           // CRC32 checksum for uncompressed content
	std::uint32_t m_size;                       // size of the uncompressed data
};


// The decode function is a simplified frontend for decompressing
// fixed.size (i.e., non-streaming) content. Only very weak guarantees
// can be made on the size of the output string.
SHARK_EXPORT_SYMBOL template <typename ContainerType, typename ReturnType = ContainerType>
ReturnType decodeRFC1952(ContainerType const& compressed)
{
	static constexpr std::size_t buffersize = 16384;
	char buffer[buffersize];
	ReturnType ret;
	StreamDecoderRFC1952 decoder;
	std::size_t pos = 0;
	while (! decoder.eof())
	{
		std::pair<uint64_t, uint64_t> io = decoder.decode((const char*)&compressed[pos], compressed.size() - pos, buffer, buffersize);
		pos += io.first;
		ret.insert(ret.end(), buffer, buffer + io.second);
	}
	if (pos != compressed.size()) throw std::runtime_error("[decodeRFC1952] excess input bytes");
	return ret;
}


// This function introduces the alias name "unzip" for "decodeRFC1952".
SHARK_EXPORT_SYMBOL template <typename ContainerType, typename ReturnType = ContainerType>
ReturnType unzip(ContainerType const& compressed)
{ return decodeRFC1952<ContainerType, ReturnType>(compressed); }


}}} // namespaces
#endif
