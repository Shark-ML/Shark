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
 * \date        2017
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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

#include <shark/OpenML/detail/LZ77.h>


using namespace std;


namespace shark {
namespace openML {
namespace detail {


// static
const std::uint32_t CRC32::m_tab[256] =
{
	0x00000000L, 0x77073096L, 0xEE0E612CL, 0x990951BAL,
	0x076DC419L, 0x706AF48FL, 0xE963A535L, 0x9E6495A3L,
	0x0EDB8832L, 0x79DCB8A4L, 0xE0D5E91EL, 0x97D2D988L,
	0x09B64C2BL, 0x7EB17CBDL, 0xE7B82D07L, 0x90BF1D91L,
	0x1DB71064L, 0x6AB020F2L, 0xF3B97148L, 0x84BE41DEL,
	0x1ADAD47DL, 0x6DDDE4EBL, 0xF4D4B551L, 0x83D385C7L,
	0x136C9856L, 0x646BA8C0L, 0xFD62F97AL, 0x8A65C9ECL,
	0x14015C4FL, 0x63066CD9L, 0xFA0F3D63L, 0x8D080DF5L,
	0x3B6E20C8L, 0x4C69105EL, 0xD56041E4L, 0xA2677172L,
	0x3C03E4D1L, 0x4B04D447L, 0xD20D85FDL, 0xA50AB56BL,
	0x35B5A8FAL, 0x42B2986CL, 0xDBBBC9D6L, 0xACBCF940L,
	0x32D86CE3L, 0x45DF5C75L, 0xDCD60DCFL, 0xABD13D59L,
	0x26D930ACL, 0x51DE003AL, 0xC8D75180L, 0xBFD06116L,
	0x21B4F4B5L, 0x56B3C423L, 0xCFBA9599L, 0xB8BDA50FL,
	0x2802B89EL, 0x5F058808L, 0xC60CD9B2L, 0xB10BE924L,
	0x2F6F7C87L, 0x58684C11L, 0xC1611DABL, 0xB6662D3DL,
	0x76DC4190L, 0x01DB7106L, 0x98D220BCL, 0xEFD5102AL,
	0x71B18589L, 0x06B6B51FL, 0x9FBFE4A5L, 0xE8B8D433L,
	0x7807C9A2L, 0x0F00F934L, 0x9609A88EL, 0xE10E9818L,
	0x7F6A0DBBL, 0x086D3D2DL, 0x91646C97L, 0xE6635C01L,
	0x6B6B51F4L, 0x1C6C6162L, 0x856530D8L, 0xF262004EL,
	0x6C0695EDL, 0x1B01A57BL, 0x8208F4C1L, 0xF50FC457L,
	0x65B0D9C6L, 0x12B7E950L, 0x8BBEB8EAL, 0xFCB9887CL,
	0x62DD1DDFL, 0x15DA2D49L, 0x8CD37CF3L, 0xFBD44C65L,
	0x4DB26158L, 0x3AB551CEL, 0xA3BC0074L, 0xD4BB30E2L,
	0x4ADFA541L, 0x3DD895D7L, 0xA4D1C46DL, 0xD3D6F4FBL,
	0x4369E96AL, 0x346ED9FCL, 0xAD678846L, 0xDA60B8D0L,
	0x44042D73L, 0x33031DE5L, 0xAA0A4C5FL, 0xDD0D7CC9L,
	0x5005713CL, 0x270241AAL, 0xBE0B1010L, 0xC90C2086L,
	0x5768B525L, 0x206F85B3L, 0xB966D409L, 0xCE61E49FL,
	0x5EDEF90EL, 0x29D9C998L, 0xB0D09822L, 0xC7D7A8B4L,
	0x59B33D17L, 0x2EB40D81L, 0xB7BD5C3BL, 0xC0BA6CADL,
	0xEDB88320L, 0x9ABFB3B6L, 0x03B6E20CL, 0x74B1D29AL,
	0xEAD54739L, 0x9DD277AFL, 0x04DB2615L, 0x73DC1683L,
	0xE3630B12L, 0x94643B84L, 0x0D6D6A3EL, 0x7A6A5AA8L,
	0xE40ECF0BL, 0x9309FF9DL, 0x0A00AE27L, 0x7D079EB1L,
	0xF00F9344L, 0x8708A3D2L, 0x1E01F268L, 0x6906C2FEL,
	0xF762575DL, 0x806567CBL, 0x196C3671L, 0x6E6B06E7L,
	0xFED41B76L, 0x89D32BE0L, 0x10DA7A5AL, 0x67DD4ACCL,
	0xF9B9DF6FL, 0x8EBEEFF9L, 0x17B7BE43L, 0x60B08ED5L,
	0xD6D6A3E8L, 0xA1D1937EL, 0x38D8C2C4L, 0x4FDFF252L,
	0xD1BB67F1L, 0xA6BC5767L, 0x3FB506DDL, 0x48B2364BL,
	0xD80D2BDAL, 0xAF0A1B4CL, 0x36034AF6L, 0x41047A60L,
	0xDF60EFC3L, 0xA867DF55L, 0x316E8EEFL, 0x4669BE79L,
	0xCB61B38CL, 0xBC66831AL, 0x256FD2A0L, 0x5268E236L,
	0xCC0C7795L, 0xBB0B4703L, 0x220216B9L, 0x5505262FL,
	0xC5BA3BBEL, 0xB2BD0B28L, 0x2BB45A92L, 0x5CB36A04L,
	0xC2D7FFA7L, 0xB5D0CF31L, 0x2CD99E8BL, 0x5BDEAE1DL,
	0x9B64C2B0L, 0xEC63F226L, 0x756AA39CL, 0x026D930AL,
	0x9C0906A9L, 0xEB0E363FL, 0x72076785L, 0x05005713L,
	0x95BF4A82L, 0xE2B87A14L, 0x7BB12BAEL, 0x0CB61B38L,
	0x92D28E9BL, 0xE5D5BE0DL, 0x7CDCEFB7L, 0x0BDBDF21L,
	0x86D3D2D4L, 0xF1D4E242L, 0x68DDB3F8L, 0x1FDA836EL,
	0x81BE16CDL, 0xF6B9265BL, 0x6FB077E1L, 0x18B74777L,
	0x88085AE6L, 0xFF0F6A70L, 0x66063BCAL, 0x11010B5CL,
	0x8F659EFFL, 0xF862AE69L, 0x616BFFD3L, 0x166CCF45L,
	0xA00AE278L, 0xD70DD2EEL, 0x4E048354L, 0x3903B3C2L,
	0xA7672661L, 0xD06016F7L, 0x4969474DL, 0x3E6E77DBL,
	0xAED16A4AL, 0xD9D65ADCL, 0x40DF0B66L, 0x37D83BF0L,
	0xA9BCAE53L, 0xDEBB9EC5L, 0x47B2CF7FL, 0x30B5FFE9L,
	0xBDBDF21CL, 0xCABAC28AL, 0x53B39330L, 0x24B4A3A6L,
	0xBAD03605L, 0xCDD70693L, 0x54DE5729L, 0x23D967BFL,
	0xB3667A2EL, 0xC4614AB8L, 0x5D681B02L, 0x2A6F2B94L,
	0xB40BBE37L, 0xC30C8EA1L, 0x5A05DF1BL, 0x2D02EF8DL
};


// encoding of the length alphabet with base value and extra bits
static const unsigned int table_length_base [29] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
static const unsigned int table_length_extra[29] = {0, 0, 0, 0, 0, 0, 0,  0,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3,  4,  4,  4,   4,   5,   5,   5,   5,   0};

static const unsigned int table_distance_base [30] = {1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
static const unsigned int table_distance_extra[30] = {0, 0, 0, 0, 1, 1, 2,  2,  3,  3,  4,  4,  5,  5,   6,   6,   7,   7,   8,   8,    9,    9,   10,   10,   11,   11,   12,    12,    13,    13};

// code length order
static const unsigned int codeLengthOrder[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};


unsigned int reverseBits(unsigned int value, unsigned int bits)
{
	unsigned int ret = 0;
	unsigned int bi = 1;
	unsigned int bo = 1 << (bits - 1);
	for (unsigned int i=0; i<bits; i++)
	{
		ret |= (value & bi) ? bo: 0;
		bi <<= 1;
		bo >>= 1;
	}
	return ret;
}


// The decode function reads bytes from the input stream and writes
// bytes to the output stream. It stops if the input buffer is fully
// processed or if the output buffer is full, or if the next atomic
// operation would exceed one of the buffers. The function returns
// the number of bytes processed for both streams.
pair<size_t, size_t> StreamDecoderRFC1951::decode(
		const char* inputbuffer,
		size_t inputsize,
		char* outputbuffer,
		size_t outputsize)
{
	// "functions"
	#define error throw SHARKEXCEPTION("[decode] decompression failed")
	#define available (8 * inputsize + m_numberOfBits)
	#define bits(n) (m_bits & ((1 << n) - 1))
	#define discard(n) \
	{ \
		assert((n) <= m_numberOfBits); \
		m_numberOfBits -= (n); \
		m_bits >>= (n); \
		while (m_numberOfBits <= 56 && inputsize > 0) \
		{ \
			m_bits |= ((uint64_t)(unsigned char)(*inputbuffer)) << m_numberOfBits; \
			m_numberOfBits += 8; \
			inputbuffer++; \
			inputsize--; \
		} \
	}
	struct State
	{
		uint64_t bits;
		unsigned int numberOfBits;
		size_t inputsize;
	};
	#define saveState State state { m_bits, m_numberOfBits, inputsize };
	#define restoreState \
	{ \
		m_bits = state.bits; \
		m_numberOfBits = state.numberOfBits; \
		inputbuffer -= (state.inputsize - inputsize); \
		inputsize = state.inputsize; \
	}
	#define output(c) \
	{ \
		m_buffer[m_bufferEnd] = (c); \
		*outputbuffer = (c); \
		if (m_bufferSize < 32768) m_bufferSize++; \
		m_bufferEnd = (m_bufferEnd + 1) & 32767; \
		outputbuffer++; \
		outputsize--; \
	}
	size_t startInputsize = inputsize;
	size_t startOutputsize = outputsize;
	#define done return make_pair(startInputsize - inputsize, startOutputsize - outputsize);

	discard(0)   // initialize bits buffer

	// perform atomic operations until we run out of buffer space
	while (true)
	{
		if (m_mode == emBlockType)
		{
			// read a block type code
			if (m_numberOfBits < 3) done;
			m_bLastBlock = bits(1);
			discard(1);
			int mode = bits(2);
			discard(2);
			if (mode == 0) m_mode = emUncompressedHeader;
			else if (mode == 1)
			{
				m_mode = emCompressed;

				// initialize the default Huffman trees
				m_literal.reset(9);
				m_distance.reset(5);
				for (unsigned int i = 0; i < 144; i++) m_literal.defineCode(i, reverseBits(i + 48, 8), 8);
				for (unsigned int i = 144; i < 256; i++) m_literal.defineCode(i, reverseBits(i + 256, 9), 9);
				for (unsigned int i = 256; i < 280; i++) m_literal.defineCode(i, reverseBits(i - 256, 7), 7);
				for (unsigned int i = 280; i < 288; i++) m_literal.defineCode(i, reverseBits(i - 88, 8), 8);
				for (unsigned int i = 0; i < 32; i++) m_distance.defineCode(i, reverseBits(i, 5), 5);
			}
			else if (mode == 2) m_mode = emHuffmanTree;
			else error;
		}
		else if (m_mode == emUncompressedHeader)
		{
			unsigned int d = m_numberOfBits & 7;
			if (m_numberOfBits < 32 + d) done;
			discard(d);   // discard until byte boundary
			m_uncompressed = bits(16);
			discard(16);
			unsigned int complement = bits(16);
			discard(16);
			if (m_uncompressed + complement != 65535) error;
			m_mode = emUncompressed;
		}
		else if (m_mode == emUncompressed)
		{
			// copy uncompressed data to the output
			assert((m_numberOfBits & 7) == 0);
			while (m_uncompressed > 0 && outputsize > 0 && m_numberOfBits >= 8)
			{
				unsigned int byte = bits(8);
				output(byte);
				discard(8);
				m_uncompressed--;
			}
			if (m_uncompressed == 0) m_mode = emBlockType;
			if (outputsize == 0) done;
		}
		else if (m_mode == emHuffmanTree)
		{
			saveState;

			// read the header
			if (m_numberOfBits < 14) done;
			unsigned int hlit = bits(5) + 257;
			discard(5);
			unsigned int hdist = bits(5) + 1;
			discard(5);
			unsigned int hclen = bits(4) + 4;
			discard(4);

			// read and build the "code-encoding" Huffman tree
			HuffmanDecoder huff;
			{
				unsigned int codelen[19];
				unsigned int ncodes[8] = {0, 0, 0, 0, 0, 0, 0, 0};
				if (available < 3*hclen) { restoreState; done; }
				for (unsigned int i=0; i<hclen; i++)
				{
					unsigned int len = bits(3);
					discard(3);
					codelen[codeLengthOrder[i]] = len;
					ncodes[len]++;
				}
				for (unsigned int i=hclen; i<19; i++) codelen[codeLengthOrder[i]] = 0;
				huff.reset(7);
				unsigned int code = 0;
				for (unsigned int len=1; len<8; len++)
				{
					for (unsigned int value=0; value<19; value++)
					{
						if (codelen[value] == len)
						{
							if (code >= 128) error;
							huff.defineCode(value, reverseBits(code, 7), len);
							code += (1 << (7 - len));
						}
					}
				}
				if (code > 128) error;
			}

			// read and build the literal/length Huffman tree
			{
				unsigned int codelen[288];
				unsigned int ncodes[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
				for (unsigned int i=0; i<hlit; )
				{
					unsigned int len = 0;
					unsigned int length = 0;
					huff.decode(m_bits, len, length);
					if (len > m_numberOfBits) error;
					discard(len);

					if (length < 16)
					{
						codelen[i] = length;
						ncodes[length]++;
						i++;
					}
					else if (length == 16)
					{
						if (m_numberOfBits < 2) { restoreState; done; }
						unsigned int l = bits(2) + 3;
						discard(2);
						for (unsigned int j=0; j<l; j++) codelen[i+j] = codelen[i-1];
						i += l;
					}
					else if (length == 17)
					{
						if (m_numberOfBits < 3) { restoreState; done; }
						unsigned int l = bits(3) + 3;
						discard(3);
						for (unsigned int j=0; j<l; j++) codelen[i+j] = 0;
						i += l;
					}
					else if (length == 18)
					{
						if (m_numberOfBits < 7) { restoreState; done; }
						unsigned int l = bits(7) + 11;
						discard(7);
						for (unsigned int j=0; j<l; j++) codelen[i+j] = 0;
						i += l;
					}
					else error;

					if (i > hlit) error;
				}
				unsigned int maxlength = 0;
				for (unsigned int l=0; l<16; l++) if (ncodes[l] > 0) maxlength = l;
				m_literal.reset(maxlength);

				unsigned int code = 0;
				for (unsigned int len=1; len<16; len++)
				{
					for (unsigned int value=0; value<hlit; value++)
					{
						if (codelen[value] == len)
						{
							if (code >= 32768) error;
							m_literal.defineCode(value, reverseBits(code, 15), len);
							code += (1 << (15 - len));
						}
					}
				}
				if (code > 32768) error;
			}

			// read and build the distance Huffman tree
			{
				unsigned int codelen[288];
				unsigned int ncodes[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
				for (unsigned i=0; i<hdist; )
				{
					unsigned int len = 0;
					unsigned int length = 0;
					huff.decode(m_bits, len, length);
					if (len > m_numberOfBits) error;
					discard(len);

					if (length < 16)
					{
						codelen[i] = length;
						ncodes[length]++;
						i++;
					}
					else if (length == 16)
					{
						if (m_numberOfBits < 2) { restoreState; done; }
						unsigned int l = bits(2) + 3;
						discard(2);
						for (unsigned int j=0; j<l; j++) codelen[i+j] = codelen[i-1];
						i += l;
					}
					else if (length == 17)
					{
						if (m_numberOfBits < 3) { restoreState; done; }
						unsigned int l = bits(3) + 3;
						discard(3);
						for (unsigned int j=0; j<l; j++) codelen[i+j] = 0;
						i += l;
					}
					else if (length == 18)
					{
						if (m_numberOfBits < 7) { restoreState; done; }
						unsigned int l = bits(7) + 11;
						discard(7);
						for (unsigned int j=0; j<l; j++) codelen[i+j] = 0;
						i += l;
					}
					else error;

					if (i > hlit) error;
				}
				unsigned int maxlength = 0;
				for (unsigned int l=0; l<16; l++) if (ncodes[l] > 0) maxlength = l;
				m_distance.reset(maxlength);

				unsigned int code = 0;
				for (unsigned int len=1; len<16; len++)
				{
					for (unsigned int value=0; value<hdist; value++)
					{
						if (codelen[value] == len)
						{
							if (code >= 32768) { restoreState; done; }
							m_distance.defineCode(value, reverseBits(code, 15), len);
							code += (1 << (15 - len));
						}
					}
				}
				if (code > 32768) error;
			}

			m_mode = emCompressed;
		}
		else if (m_mode == emCompressed)
		{
			// read the literal, but don't discard it yet
			unsigned int literal, len1;
			m_literal.decode(m_bits, len1, literal);
			if (len1 > m_numberOfBits) done;

			if (literal < 256)
			{
				// character literal
				if (outputsize == 0) done;
				discard(len1);
				output(literal);
			}
			else if (literal == 256)
			{
				// "end of block"-literal
				discard(len1);
				if (m_bLastBlock)
				{
					// discard until byte boundary to "flush" the input stream,
					// put buffered bytes back into the input stream
					discard(m_numberOfBits & 7);
					inputbuffer -= m_numberOfBits / 8;
					inputsize += m_numberOfBits / 8;
					m_bits = 0;
					m_numberOfBits = 0;
					m_mode = emEOF;
					done;
				}
				else m_mode = emBlockType;
			}
			else if (literal <= 285)
			{
				// read the length, with extra bits
				unsigned int length = table_length_base[literal - 257];
				unsigned int extraBits1 = table_length_extra[literal - 257];
				if (len1 + extraBits1 > m_numberOfBits) done;
				length += (m_bits >> len1) & ((1 << extraBits1) - 1);
				if (length > outputsize) done;

				// read the distance
				unsigned int len2, distLiteral;
				m_distance.decode(m_bits >> (len1 + extraBits1), len2, distLiteral);
				if (len1 + extraBits1 + len2 > m_numberOfBits) done;
				if (distLiteral >= 30) error;
				unsigned int distance = table_distance_base[distLiteral];
				unsigned int extraBits2 = table_distance_extra[distLiteral];
				if (len1 + extraBits1 + len2 + extraBits2 > m_numberOfBits) done;
				distance += (m_bits >> (len1 + extraBits1 + len2)) & ((1 << extraBits2) - 1);

				// accept this atomic operation
				discard(len1 + extraBits1 + len2 + extraBits2);

				// copy bytes
				for (unsigned int i=0; i<length; i++)
				{
					unsigned int c = (unsigned int)(unsigned char)m_buffer[(m_bufferEnd - distance + 32768) & 32767];
					output(c);
				}
			}
			else error;
		}
		else if (m_mode == emEOF)
		{
			error;   // trying to read beyond the end of the stream
		}
		else error;
	}
}


// The decode function reads bytes from the input stream and writes
// bytes to the output stream. It stops if the input buffer is fully
// processed or if the output buffer is full, or if the next atomic
// operation would exceed one of the buffers. The function returns
// the number of bytes processed for both streams.
pair<size_t, size_t> StreamDecoderRFC1950::decode(
		const char* inputbuffer,
		size_t inputsize,
		char* outputbuffer,
		size_t outputsize)
{
	size_t startInputsize = inputsize;
	size_t startOutputsize = outputsize;
	#undef done
	#undef error
	#define done return make_pair(startInputsize - inputsize, startOutputsize - outputsize);
	#define error throw SHARKEXCEPTION("[decode] decompression failed")

	// perform atomic operations until we run out of buffer space
	while (true)
	{
		if (m_mode == emHeader)
		{
			// check consistency of the header
			if (inputsize < 2) done;
			if ((inputbuffer[0] & 15) != 8) error;
			if (((256 * (unsigned int)(unsigned char)inputbuffer[0] + (unsigned int)(unsigned char)inputbuffer[1]) % 31) != 0) error;
			if (inputbuffer[1] & 32) error;   // preset dictionary not supported
			inputbuffer += 2;
			inputsize -= 2;
			m_mode = em1951;
		}
		else if (m_mode == em1951)
		{
			// invoke base decoder
			pair<size_t, size_t> io = m_decoder.decode(inputbuffer, inputsize, outputbuffer, outputsize);
			if (io.first == 0 && io.second == 0) done;

			// update checksum
			m_checksum.process(outputbuffer, outputbuffer + io.second);

			// progress buffers
			inputbuffer += io.first;
			inputsize -= io.first;
			outputbuffer += io.second;
			outputsize -= io.second;

			// update mode on EOF
			if (m_decoder.eof()) m_mode = emAdler32;
		}
		else if (m_mode == emAdler32)
		{
			// check the Adler-32 checksum
			if (inputsize < 4) done;
			uint32_t a = 0;
			for (unsigned int i=0; i<4; i++)
			{
				a *= 256;
				a += (uint32_t)(unsigned char)inputbuffer[i];
			}
			if (a != m_checksum.checksum()) error;
			inputbuffer += 4;
			inputsize -= 4;
			m_mode = emEOF;
			done;
		}
		else if (m_mode == emEOF)
		{
			error;   // trying to read beyond the end of the stream
		}
		else error;
	}
}


// The decode function reads bytes from the input stream and writes
// bytes to the output stream. It stops if the input buffer is fully
// processed or if the output buffer is full, or if the next atomic
// operation would exceed one of the buffers. The function returns
// the number of bytes processed for both streams.
pair<size_t, size_t> StreamDecoderRFC1952::decode(
		const char* inputbuffer,
		size_t inputsize,
		char* outputbuffer,
		size_t outputsize)
{
	size_t startInputsize = inputsize;
	size_t startOutputsize = outputsize;
	#undef done
	#undef error
	#define done return make_pair(startInputsize - inputsize, startOutputsize - outputsize);
	#define error throw SHARKEXCEPTION("[lz77::gzip] decompression failed")

	// perform atomic operations until we run out of buffer space
	while (true)
	{
		if (m_mode == emHeader)
		{
			// check for EOF
			if (inputsize == 0)
			{
				m_mode = emEOF;
				done;
			}

			// check consistency of the header
			if (inputsize < 10) done;
			if (((unsigned char)inputbuffer[0]) != 0x1f) error;
			if (((unsigned char)inputbuffer[1]) != 0x8b) error;
			if (((unsigned char)inputbuffer[2]) != 8) error;
			m_flg = inputbuffer[3];
			// ignore fields MTIME, XFL, and OS
			inputbuffer += 10;
			inputsize -= 10;
			m_extra = 0;
			if (m_flg & 2)
			{
				m_extra = (unsigned char)inputbuffer[0];
				m_extra += 256 * (unsigned int)(unsigned char)inputbuffer[1];
				inputbuffer += 2;
				inputsize -= 2;
				m_mode = emExtra;
			}
			else if (m_flg & 8) m_mode = emName;
			else if (m_flg & 16) m_mode = emComment;
			else if (m_flg & 1) m_mode = emCRC16;
			else m_mode = em1951;
		}
		else if (m_mode == emExtra)
		{
			// ignore extra field
			size_t n = std::min(m_extra, inputsize);
			inputbuffer += n;
			inputsize -= n;
			m_extra -= n;
			if (m_extra == 0)
			{
				if (m_flg & 8) m_mode = emName;
				else if (m_flg & 16) m_mode = emComment;
				else if (m_flg & 1) m_mode = emCRC16;
				else m_mode = em1951;
			}
		}
		else if (m_mode == emName)
		{
			// ignore name field
			bool terminated = false;
			while (inputsize > 0)
			{
				char c = inputbuffer[0];
				inputsize--;
				if (c == 0) { terminated = true; break; }
			}
			if (terminated)
			{
				if (m_flg & 16) m_mode = emComment;
				else if (m_flg & 1) m_mode = emCRC16;
				else m_mode = em1951;
			}
		}
		else if (m_mode == emComment)
		{
			// ignore comment field
			bool terminated = false;
			while (inputsize > 0)
			{
				char c = inputbuffer[0];
				inputsize--;
				if (c == 0) { terminated = true; break; }
			}
			if (terminated)
			{
				if (m_flg & 1) m_mode = emCRC16;
				else m_mode = em1951;
			}
		}
		else if (m_mode == emCRC16)
		{
			// ignore CRC16 field
			if (inputsize < 2) done;
			inputbuffer += 2;
			inputsize -= 2;
			m_mode = em1951;
		}
		else if (m_mode == em1951)
		{
			// invoke base decoder
			pair<size_t, size_t> io = m_decoder.decode(inputbuffer, inputsize, outputbuffer, outputsize);
			if (io.first == 0 && io.second == 0) done;

			// update checksum
			m_checksum.process(outputbuffer, outputbuffer + io.second);
			m_size += io.second;

			// progress buffers
			inputbuffer += io.first;
			inputsize -= io.first;
			outputbuffer += io.second;
			outputsize -= io.second;

			// update mode on EOF
			if (m_decoder.eof()) m_mode = emTrailer;
		}
		else if (m_mode == emTrailer)
		{
			// check the CRC-32 checksum and the size of the output
			if (inputsize < 8) done;
			std::uint32_t a = 0;
			std::uint32_t b = 0;
			for (unsigned int i=0; i<4; i++)
			{
				a |= (1 << (8*i)) * (std::uint32_t)(unsigned char)inputbuffer[i];
			}
			for (unsigned int i=4; i<8; i++)
			{
				b |= (1 << (8*i)) * (std::uint32_t)(unsigned char)inputbuffer[i];
			}
			if (a != m_checksum.checksum()) error;
			if (b != m_size) error;
			inputbuffer += 8;
			inputsize -= 8;
			m_mode = emHeader;
			done;
		}
		else if (m_mode == emEOF)
		{
			error;   // trying to read beyond the end of the stream
		}
		else error;
	}
}


}}} // namespaces
