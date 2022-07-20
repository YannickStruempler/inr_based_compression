'''Compression application using static arithmetic coding adapted from
https://www.nayuki.io/page/reference-arithmetic-coding
https://github.com/nayuki/Reference-arithmetic-coding.
'''

import contextlib
import copy
import numpy as np
from lib.arithmetic_coding import arithmeticcoding


class AE:

	def compress_bytes(self, byte_stream, out, bits, write_freq=True):
		self.NUM_FREQ = int(2 ** bits)
		input = copy.deepcopy(byte_stream)
		freqs = arithmeticcoding.SimpleFrequencyTable([0] * (self.NUM_FREQ + 1))

		while True:
			b = input.read(1)
			if len(b) == 0:
				break
			freqs.increment(b[0])
		freqs.increment(self.NUM_FREQ)  # EOF symbol gets a frequency of 1
		self.FREQ_TABLE_BITS =  int(np.log2(max(freqs.frequencies))) + 1
		# Read input file again, compress with arithmetic coding, and write output file
		with contextlib.closing(arithmeticcoding.BitOutputStream(out)) as bitout:

			if write_freq:
				self.write_int(bitout, 4, bits) #log2(frequency table length)
				self.write_int(bitout, 4, self.FREQ_TABLE_BITS) #bits used to encode frequencies
				self.write_frequencies(bitout, freqs)
			self.compress(freqs, byte_stream, bitout)
			return bitout.output.tell(), freqs.frequencies
	# Returns a frequency table based on the bytes in the given file.
	def get_frequencies(self, filepath):
		freqs = arithmeticcoding.SimpleFrequencyTable([0] * self.NUM_FREQ)
		with open(filepath, "rb") as input:
			while True:
				b = input.read(1)
				if len(b) == 0:
					break
				freqs.increment(b[0])
		return freqs


	def write_frequencies(self, bitout, freqs):
		for i in range(self.NUM_FREQ):
			self.write_int(bitout, self.FREQ_TABLE_BITS, freqs.get(i))


	def compress(self, freqs, inp, bitout):
		enc = arithmeticcoding.ArithmeticEncoder(32, bitout)
		while True:
			symbol = inp.read(1)
			if len(symbol) == 0:
				break
			enc.write(freqs, symbol[0])
		enc.write(freqs, self.NUM_FREQ)  # EOF
		enc.finish()  # Flush remaining code bits


	# Writes an unsigned integer of the given bit width to the given stream.
	def write_int(self, bitout, numbits, value):
		for i in reversed(range(numbits)):
			bitout.write((value >> i) & 1)  # Big endian


