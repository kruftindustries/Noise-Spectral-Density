import unittest
import math
import cmath
import os
import tempfile
import signal as ps

class TestStatisticalFunctions(unittest.TestCase):
    """Test basic statistical functions"""
    
    def test_mean(self):
        """Test the mean function"""
        self.assertEqual(ps.mean([1, 2, 3, 4, 5]), 3.0)
        self.assertEqual(ps.mean([]), 0)
        self.assertEqual(ps.mean([-5, 5]), 0.0)
        self.assertAlmostEqual(ps.mean([0.1, 0.2, 0.3]), 0.2)
    
    def test_std(self):
        """Test the standard deviation function"""
        self.assertAlmostEqual(ps.std([1, 2, 3, 4, 5]), 1.4142135623730951)
        self.assertEqual(ps.std([]), 0)
        self.assertAlmostEqual(ps.std([2, 2, 2, 2]), 0.0)
        
    def test_min_max(self):
        """Test the min_max function"""
        self.assertEqual(ps.min_max([1, 2, 3, 4, 5]), (1, 5))
        self.assertEqual(ps.min_max([]), (0, 0))
        self.assertEqual(ps.min_max([7]), (7, 7))
        
    def test_linspace(self):
        """Test the linspace function"""
        self.assertEqual(ps.linspace(0, 1, 5), [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertEqual(ps.linspace(2, 3, 3), [2.0, 2.5, 3.0])
        self.assertEqual(ps.linspace(0, 0, 5), [0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(ps.linspace(5, 1, 1), [5])
        
    def test_arange(self):
        """Test the arange function"""
        self.assertEqual(ps.arange(5), [0, 1, 2, 3, 4])
        self.assertEqual(ps.arange(2, 5), [2, 3, 4])
        self.assertEqual(ps.arange(0, 10, 2), [0, 2, 4, 6, 8])
        self.assertEqual(ps.arange(0, 1, 0.5), [0, 0.5])
        
    def test_zeros(self):
        """Test the zeros function"""
        self.assertEqual(ps.zeros(3), [0.0, 0.0, 0.0])
        self.assertEqual(ps.zeros(0), [])
        
    def test_complex_zeros(self):
        """Test the complex_zeros function"""
        self.assertEqual(ps.complex_zeros(2), [0.0+0.0j, 0.0+0.0j])
        
    def test_array_operations(self):
        """Test array operations"""
        # Test power
        self.assertEqual(ps.power([1, 2, 3], 2), [1, 4, 9])
        
        # Test multiply
        self.assertEqual(ps.multiply([1, 2, 3], [2, 3, 4]), [2, 6, 12])
        
        # Test add
        self.assertEqual(ps.add([1, 2, 3], [4, 5, 6]), [5, 7, 9])
        
        # Test subtract
        self.assertEqual(ps.subtract([5, 6, 7], [1, 2, 3]), [4, 4, 4])
        
        # Test abs_list
        self.assertEqual(ps.abs_list([-1, 2, -3]), [1, 2, 3])
        
    def test_log10(self):
        """Test log10 function"""
        # Test with list of values
        data = [10, 100, 1000]
        log_values = ps.log10(data)
        self.assertEqual(len(log_values), 3)
        self.assertAlmostEqual(log_values[0], 1.0)
        self.assertAlmostEqual(log_values[1], 2.0)
        self.assertAlmostEqual(log_values[2], 3.0)
        
        # Test with a value close to zero (should return log10 of small value)
        small_values = [0, 1e-30]
        log_small = ps.log10(small_values)
        self.assertEqual(len(log_small), 2)
        self.assertTrue(log_small[0] < -10)  # log10(1e-20) = -20
        self.assertTrue(log_small[1] < -10)
        
    def test_polyfit_polyval(self):
        """Test polyfit and polyval functions"""
        # Test with a line: y = 2x
        x = [0, 1, 2, 3, 4]
        y = [0, 2, 4, 6, 8]  # y = 2*x
        
        coefs = ps.polyfit(x, y, 1)
        self.assertAlmostEqual(coefs[0], 0.0, places=10)  # intercept
        self.assertAlmostEqual(coefs[1], 2.0, places=10)  # slope
        
        y_pred = ps.polyval(coefs, x)
        for i, yi in enumerate(y):
            self.assertAlmostEqual(y_pred[i], yi, places=10)
            
        # Test with a line with intercept: y = 2x + 3
        x2 = [0, 1, 2, 3, 4]
        y2 = [3, 5, 7, 9, 11]
        
        coefs2 = ps.polyfit(x2, y2, 1)
        self.assertAlmostEqual(coefs2[0], 3.0, places=10)  # intercept
        self.assertAlmostEqual(coefs2[1], 2.0, places=10)  # slope
        
        y_pred2 = ps.polyval(coefs2, x2)
        for i, yi in enumerate(y2):
            self.assertAlmostEqual(y_pred2[i], yi, places=10)
            
    def test_detrend(self):
        """Test detrend function"""
        # Test constant detrend
        x = [1, 2, 3, 4, 5]
        detrended = ps.detrend(x, 'constant')
        self.assertAlmostEqual(ps.mean(detrended), 0.0, places=10)
        
        # Test with list of floats to ensure it works with all numeric types
        x_float = [1.0, 2.0, 3.0, 4.0, 5.0]
        detrended_float = ps.detrend(x_float, 'constant')
        self.assertAlmostEqual(ps.mean(detrended_float), 0.0, places=10)
        
        # Test linear detrend
        x = [0, 1, 2, 3, 4]
        y = [0, 2, 4, 6, 8]  # y = 2*x, perfect linear relationship
        detrended = ps.detrend(y, 'linear')
        
        # After linear detrending, all values should be close to zero
        for val in detrended:
            self.assertAlmostEqual(val, 0.0, places=10)
            
        # Test with a different linear relationship
        x = [0, 1, 2, 3, 4, 5]
        y = [3, 5, 7, 9, 11, 13]  # y = 2*x + 3
        detrended = ps.detrend(y, 'linear')
        
        # After linear detrending, all values should be close to zero
        for val in detrended:
            self.assertAlmostEqual(val, 0.0, places=10)
                
        # Test when detrend type is 'none', should return a copy
        x = [1, 2, 3]
        result = ps.detrend(x, 'none')
        self.assertEqual(result, x)
        # Verify it's a copy, not the original
        result[0] = 999
        self.assertNotEqual(result[0], x[0])
        
        # Test when detrend type is not recognized, should return a copy
        x = [1, 2, 3]
        result = ps.detrend(x, 'unknown')
        self.assertEqual(result, x)
        # Verify it's a copy, not the original
        result[0] = 999
        self.assertNotEqual(result[0], x[0])


class TestWindowFunctions(unittest.TestCase):
    """Test window functions"""
    
    def test_hann(self):
        """Test Hann window function"""
        # Test empty and single-point cases
        self.assertEqual(ps.hann(0), [])
        self.assertEqual(ps.hann(1), [1.0])
        
        # Test a simple case and verify symmetry
        window = ps.hann(5)
        self.assertEqual(len(window), 5)
        self.assertAlmostEqual(window[0], 0.0, places=10)
        self.assertAlmostEqual(window[4], 0.0, places=10)
        self.assertAlmostEqual(window[2], 1.0, places=10)
        
        # Verify symmetry
        self.assertAlmostEqual(window[1], window[3], places=10)
        
    def test_hamming(self):
        """Test Hamming window function"""
        # Test empty and single-point cases
        self.assertEqual(ps.hamming(0), [])
        self.assertEqual(ps.hamming(1), [1.0])
        
        # Test a simple case
        window = ps.hamming(5)
        self.assertEqual(len(window), 5)
        self.assertAlmostEqual(window[0], 0.08, places=2)
        self.assertAlmostEqual(window[4], 0.08, places=2)
        self.assertAlmostEqual(window[2], 1.0, places=2)
        
        # Verify symmetry
        self.assertAlmostEqual(window[1], window[3], places=10)
        
    def test_get_window(self):
        """Test get_window function"""
        # Test that each window type is correctly dispatched
        self.assertEqual(ps.get_window('hann', 3), ps.hann(3))
        self.assertEqual(ps.get_window('hamming', 3), ps.hamming(3))
        self.assertEqual(ps.get_window('blackman', 3), ps.blackman(3))
        self.assertEqual(ps.get_window('bartlett', 3), ps.bartlett(3))
        self.assertEqual(ps.get_window('boxcar', 3), ps.boxcar(3))
        self.assertEqual(ps.get_window('flattop', 3), ps.flattop(3))
        
        # Test default for unknown window
        self.assertEqual(ps.get_window('unknown', 3), ps.hann(3))


class TestFFT(unittest.TestCase):
    """Test FFT and related functions"""
    
    def test_fft_simple(self):
        """Test FFT on simple signals"""
        # Test with a basic signal (DC component only)
        x = [1.0, 1.0, 1.0, 1.0]
        result = ps.fft(x)
        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(result[0], 4.0+0j, places=8)  # DC component
        self.assertAlmostEqual(abs(result[1]), 0.0, places=8)
        self.assertAlmostEqual(abs(result[2]), 0.0, places=8)
        self.assertAlmostEqual(abs(result[3]), 0.0, places=8)
        
        # Test with a sine wave
        N = 8
        x = [math.sin(2 * math.pi * 1 * i / N) for i in range(N)]  # Sine wave with 1 cycle
        result = ps.fft(x)
        # DC component should be ~0
        self.assertAlmostEqual(abs(result[0]), 0.0, places=8)
        # Energy should be at frequency bin 1 and N-1 (for sine wave)
        self.assertTrue(abs(result[1]) > 0.1)
        self.assertTrue(abs(result[N-1]) > 0.1)
    
    def test_improved_fft(self):
        """Test the improved FFT implementation"""
        # Test with a basic signal (DC component only)
        x = [1.0, 1.0, 1.0, 1.0]
        result = ps.improved_fft(x)
        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(result[0], 4.0+0j, places=8)  # DC component
        self.assertAlmostEqual(abs(result[1]), 0.0, places=8)
        self.assertAlmostEqual(abs(result[2]), 0.0, places=8)
        self.assertAlmostEqual(abs(result[3]), 0.0, places=8)
        
        # Test with a sine wave
        N = 32  # Using a power of 2 for this test
        x = [math.sin(2 * math.pi * 4 * i / N) for i in range(N)]  # Sine wave with 4 cycles
        result = ps.improved_fft(x)
        # DC component should be ~0
        self.assertAlmostEqual(abs(result[0]), 0.0, places=8)
        # Energy should be at frequency bin 4 and N-4 for 4 cycles
        self.assertTrue(abs(result[4]) > 0.1)
        self.assertTrue(abs(result[N-4]) > 0.1)
        
    def test_ifft(self):
        """Test inverse FFT"""
        # Test IFFT of FFT (should get back original signal)
        x = [1.0, 2.0, 3.0, 4.0]
        fft_result = ps.fft(x)
        ifft_result = ps.ifft(fft_result)
        
        for i in range(len(x)):
            self.assertAlmostEqual(ifft_result[i].real, x[i], places=8)
            self.assertAlmostEqual(ifft_result[i].imag, 0.0, places=8)
            
    def test_rfft_irfft(self):
        """Test real FFT and inverse real FFT"""
        # Test with a real signal
        x = [1.0, 2.0, 3.0, 4.0]
        rfft_result = ps.rfft(x)
        # should have N//2+1 elements for real input
        self.assertEqual(len(rfft_result), len(x)//2 + 1)
        
        # Test rfftfreq
        freqs = ps.rfftfreq(len(x), 1.0)
        self.assertEqual(len(freqs), len(x)//2 + 1)
        self.assertEqual(freqs[0], 0.0)  # DC component
        
        # Test irfft with even length
        irfft_result = ps.irfft(rfft_result, len(x))
        self.assertEqual(len(irfft_result), len(x))
        
        # Values should be approximately equal to original signal
        for i in range(len(x)):
            self.assertAlmostEqual(irfft_result[i], x[i], places=6)
            
        # Test irfft with odd length
        x_odd = [1.0, 2.0, 3.0, 4.0, 5.0]
        rfft_odd = ps.rfft(x_odd)
        irfft_odd = ps.irfft(rfft_odd, len(x_odd))
        self.assertEqual(len(irfft_odd), len(x_odd))
        
        # Test explicit length specification
        x_custom = [1.0, 2.0, 3.0, 4.0]
        rfft_custom = ps.rfft(x_custom)
        # Request a specific length
        irfft_custom = ps.irfft(rfft_custom, 6)  # Request 6 elements output
        self.assertEqual(len(irfft_custom), 6)  # Should return exactly 6 elements
    
    def test_simplified_fft(self):
        """Test simplified FFT for large arrays"""
        # Test with a small array first to verify correct behavior
        x = [1.0, 2.0, 3.0, 4.0]
        result_standard = ps.fft(x)
        result_simplified = ps.simplified_fft(x)
        
        # They should be essentially the same for small arrays
        for i in range(len(x)):
            self.assertAlmostEqual(abs(result_standard[i]), abs(result_simplified[i]), places=6)
        
        # Test with a medium array
        N = 512
        x = [math.sin(2 * math.pi * 10 * i / N) for i in range(N)]  # Sine wave with 10 cycles
        result = ps.simplified_fft(x)
        
        # Check overall length
        self.assertEqual(len(result), N)
        
        # Check for peak at the right frequency
        magnitudes = [abs(val) for val in result]
        max_idx = magnitudes.index(max(magnitudes[:N//2]))  # Find peak in first half
        self.assertTrue(max_idx >= 9 and max_idx <= 11)  # Should be around bin 10
        
    def test_simplified_rfft(self):
        """Test simplified RFFT for large arrays"""
        # Test with a small array first
        x = [1.0, 2.0, 3.0, 4.0]
        result_standard = ps.rfft(x)
        result_simplified = ps.simplified_rfft(x)
        
        # For small arrays they should be very similar
        for i in range(len(result_standard)):
            self.assertAlmostEqual(abs(result_standard[i]), abs(result_simplified[i]), places=6)
        
        # Test with a medium array
        N = 512
        x = [math.sin(2 * math.pi * 10 * i / N) for i in range(N)]  # Sine wave with 10 cycles
        result = ps.simplified_rfft(x)
        
        # Check overall length - should be N//2 + 1
        self.assertEqual(len(result), N//2 + 1)
        
        # Check for peak at the right frequency
        magnitudes = [abs(val) for val in result]
        max_idx = magnitudes.index(max(magnitudes[1:]))  # Skip DC component
        self.assertTrue(max_idx >= 9 and max_idx <= 11)  # Should be around bin 10


class TestSpectralFunctions(unittest.TestCase):
    """Test spectral analysis functions"""
    
    def test_periodogram_simple_peaks(self):
        """Test periodogram peak detection with simple signals"""
        
        # Test with a pure sine wave
        fs = 1000.0  # 1000 Hz sampling rate (higher to better resolve frequencies)
        N = 1000     # 1000 samples
        t = [i / fs for i in range(N)]
        
        # Create a 50 Hz sine wave (easier to detect with standard FFT bins)
        f_sine = 50.0
        x_sine = [math.sin(2 * math.pi * f_sine * ti) for ti in t]
        
        # Calculate periodogram
        freqs, psd = ps.periodogram(x_sine, fs=fs, window='boxcar', detrend_type='none')
        
        # Find the peak frequency
        peak_idx = psd.index(max(psd))
        peak_freq = freqs[peak_idx]
        
        # Check that the peak is near the expected frequency
        # Use a more generous tolerance to account for FFT bin spacing
        frequency_resolution = fs/N  # The minimum resolution we can expect
        self.assertAlmostEqual(peak_freq, f_sine, delta=frequency_resolution*2)
        
        # Test with multiple sine waves
        f_sine1 = 50.0
        f_sine2 = 150.0
        x_mixed = [math.sin(2 * math.pi * f_sine1 * ti) + 
                   math.sin(2 * math.pi * f_sine2 * ti) for ti in t]
        
        # Calculate periodogram
        freqs_mixed, psd_mixed = ps.periodogram(x_mixed, fs=fs, window='boxcar', detrend_type='none')
        
        # Find the two highest peaks
        top_indices = sorted(range(len(psd_mixed)), key=lambda i: psd_mixed[i], reverse=True)[:2]
        top_freqs = sorted([freqs_mixed[i] for i in top_indices])
        
        # Check that both peaks are detected correctly
        self.assertAlmostEqual(top_freqs[0], f_sine1, delta=frequency_resolution*2)
        self.assertAlmostEqual(top_freqs[1], f_sine2, delta=frequency_resolution*2)
        
    def test_sine_peak_detection(self):
        """Test detection of a sinusoidal peak with periodogram"""
        
        # Use a high sampling rate
        fs = 1000.0  # 1000 Hz
        
        # Use a nice round number of samples (power of 2 helps FFT)
        N = 1024
        
        # Create time points
        t = [i / fs for i in range(N)]
        
        # Use a frequency that will align well with FFT bins
        f_sine = 100.0  # 100 Hz
        
        # Create a pure sine wave
        x = [math.sin(2 * math.pi * f_sine * ti) for ti in t]
        
        # Compute periodogram
        freqs, psd = ps.periodogram(x, fs=fs, window='boxcar', nfft=N, 
                                 detrend_type='none', scaling='spectrum')
        
        # Find peak
        peak_idx = psd.index(max(psd))
        peak_freq = freqs[peak_idx]
        
        # The peak should be at the sine frequency
        self.assertAlmostEqual(peak_freq, f_sine, delta=fs/N)
        
        # Also check some specific frequency values
        self.assertEqual(freqs[0], 0.0)  # DC component
        self.assertAlmostEqual(freqs[-1], fs/2, places=8)  # Nyquist frequency
        
    def test_welch_basic(self):
        """Test Welch's method with basic signals"""
        # Test with DC signal
        x = [5.0] * 100  # Using non-zero constant to ensure clear DC component
        freqs, psd = ps.welch(x, fs=1.0, nperseg=50, detrend_type='none', window='boxcar')
        
        self.assertGreater(psd[0], 0.1)  # DC component should have significant power
        
        # Check that other frequencies have relatively small power
        # Some spectral leakage is normal, so we're less strict
        dc_power = psd[0]
        for i in range(1, len(psd)):
            self.assertLess(psd[i], dc_power * 0.01)  # Non-DC components should be much smaller than DC
            
        # Test with sine wave
        fs = 100
        T = 1.0 / fs
        N = 1000
        t = [i * T for i in range(N)]
        freq = 5.0  # 5 Hz sine wave
        x = [math.sin(2 * math.pi * freq * ti) for ti in t]
        
        freqs, psd = ps.welch(x, fs=fs, nperseg=256, detrend_type='constant')
        
        # Find the peak frequency
        peak_idx = psd.index(max(psd))
        peak_freq = freqs[peak_idx]
        
        # The peak should be around the sine wave frequency
        self.assertTrue(abs(peak_freq - freq) < 1.0)
    
    def test_large_signal_processing(self):
        """Test processing of large signals to ensure efficiency"""
        # Generate a large signal
        fs = 1000.0  # 1000 Hz sampling rate
        N = 10000     # 10,000 samples - large enough to potentially cause slowdowns
        t = [i / fs for i in range(N)]
        
        # Create a sine wave with multiple frequency components
        f1, f2, f3 = 50.0, 120.0, 210.0
        x = [math.sin(2 * math.pi * f1 * ti) + 
             0.5 * math.sin(2 * math.pi * f2 * ti) + 
             0.25 * math.sin(2 * math.pi * f3 * ti) for ti in t]
        
        # Test periodogram with large signal
        freqs, psd = ps.periodogram(x, fs=fs, window='hann', detrend_type='constant')
        
        # Check that we get the right number of frequency bins
        self.assertEqual(len(freqs), N//2 + 1)
        
        # Test Welch method with large signal
        freqs, psd = ps.welch(x, fs=fs, nperseg=1024, noverlap=512, window='hann')
        
        # Check for peaks at our input frequencies
        psd_peaks = []
        for i in range(1, len(psd)-1):
            if psd[i] > psd[i-1] and psd[i] > psd[i+1]:
                psd_peaks.append((freqs[i], psd[i]))
        
        # Sort peaks by power
        psd_peaks.sort(key=lambda x: x[1], reverse=True)
        
        # Check that our top 3 peaks correspond to the input frequencies
        top_freqs = [p[0] for p in psd_peaks[:3]]
        
        # Each peak should be close to one of our input frequencies
        for f in [f1, f2, f3]:
            self.assertTrue(any(abs(peak_f - f) < 2.0 for peak_f in top_freqs))


class TestCSVFunctions(unittest.TestCase):
    """Test CSV handling functions"""
    
    def test_csv_to_list(self):
        """Test csv_to_list function"""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("1.0\n2.0\n3.0\n4.0\n5.0\n")
            temp_filename = f.name
        
        try:
            # Test reading the file
            data = ps.csv_to_list(temp_filename)
            self.assertEqual(data, [1.0, 2.0, 3.0, 4.0, 5.0])
            
            # Test with scientific notation
            with open(temp_filename, 'w') as f:
                f.write("1.0E-6\n2.5E-6\n-3.4E-6\n")
            
            data = ps.csv_to_list(temp_filename)
            self.assertEqual(len(data), 3)
            self.assertAlmostEqual(data[0], 1.0E-6)
            self.assertAlmostEqual(data[1], 2.5E-6)
            self.assertAlmostEqual(data[2], -3.4E-6)
            
            # Test with invalid lines
            with open(temp_filename, 'w') as f:
                f.write("1.0\nNot a number\n3.0\n")
            
            data = ps.csv_to_list(temp_filename)
            self.assertEqual(data, [1.0, 3.0])  # Invalid line should be skipped
            
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)


if __name__ == '__main__':
    unittest.main()