
'''Decompression application using static arithmetic coding adapted from
https://www.nayuki.io/page/reference-arithmetic-coding
https://github.com/nayuki/Reference-arithmetic-coding.
'''

from lib.arithmetic_coding import arithmeticcoding


class AD:

    def decompress(self, inp, out):
        bitin = arithmeticcoding.BitInputStream(inp)
        freqs = self.read_frequencies(bitin)
        dec = arithmeticcoding.ArithmeticDecoder(32, bitin)
        while True:
            symbol = dec.read(freqs)
            if symbol == self.NUM_FREQ:  # EOF symbol
                break
            out.write(bytes((symbol,)))

    def read_frequencies(self, bitin):
        def read_int(n):
            result = 0
            for _ in range(n):
                result = (result << 1) | bitin.read_no_eof()  # Big endian
            return result

        self.BITWIDTH = read_int(4)  # log2(frequency table length)
        self.FREQ_TABLE_BITS = read_int(4)
        self.NUM_FREQ = int(2 ** self.BITWIDTH)

        freqs = [read_int(self.FREQ_TABLE_BITS) for _ in range(self.NUM_FREQ)]
        freqs.append(1)  # EOF symbol
        return arithmeticcoding.SimpleFrequencyTable(freqs)
