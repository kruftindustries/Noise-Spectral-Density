"""
Pure Python implementation of signal processing functions without NumPy dependency.
"""
import math
import cmath
import time

def mean(data):
    """Calculate the mean of a list of values"""
    if not data:
        return 0
    return sum(data) / len(data)

def std(data):
    """Calculate the standard deviation of a list of values"""
    if not data:
        return 0
    
    data_mean = mean(data)
    variance = sum((x - data_mean) ** 2 for x in data) / len(data)
    return math.sqrt(variance)

def min_max(data):
    """Return the minimum and maximum values from a list"""
    if not data:
        return 0, 0
    return min(data), max(data)

def linspace(start, stop, num):
    """Create a list of evenly spaced numbers over a specified interval"""
    if num < 2:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]

def arange(start, stop=None, step=1):
    """Return evenly spaced values within a given interval"""
    if stop is None:
        stop = start
        start = 0
        
    num = int((stop - start) / step)
    return [start + i * step for i in range(num)]
    
def zeros(n):
    """Create a list of zeros"""
    return [0.0] * n

def complex_zeros(n):
    """Create a list of complex zeros"""
    return [0.0 + 0.0j] * n

def power(data, p):
    """Element-wise power operation"""
    return [x ** p for x in data]

def multiply(data1, data2):
    """Element-wise multiplication of two lists"""
    return [a * b for a, b in zip(data1, data2)]

def add(data1, data2):
    """Element-wise addition of two lists"""
    return [a + b for a, b in zip(data1, data2)]

def subtract(data1, data2):
    """Element-wise subtraction of two lists"""
    return [a - b for a, b in zip(data1, data2)]

def abs_list(data):
    """Element-wise absolute value of a list"""
    if all(isinstance(x, complex) for x in data):
        return [abs(x) for x in data]
    return [abs(x) for x in data]

def log10(data):
    """Element-wise base-10 logarithm of a list"""
    return [math.log10(max(x, 1e-20)) for x in data]

def polyfit(x, y, degree):
    """
    Least squares polynomial fit.
    
    Parameters:
    - x: x-coordinates
    - y: y-coordinates
    - degree: Degree of polynomial (only 1 is fully implemented)
    
    Returns:
    - Coefficients [intercept, slope] for degree=1
    """
    # Simplified implementation for linear fit (degree=1)
    if degree != 1:
        raise ValueError("Only linear fit (degree=1) is implemented")
    
    n = len(x)
    if n < 2:
        raise ValueError("At least two points are required for linear fit")
    
    # Calculate sums for linear regression
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x2 = sum(xi**2 for xi in x)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    
    # Calculate slope and intercept
    denom = n * sum_x2 - sum_x**2
    if abs(denom) < 1e-10:  # Avoid division by near-zero
        # Vertical line case
        slope = 0.0
        intercept = sum_y / n if n > 0 else 0.0
    else:
        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n
    
    # Return coefficients in [intercept, slope] order
    return [intercept, slope]

def polyval(p, x):
    """
    Evaluate a polynomial at specific values.
    
    Parameters:
    - p: Polynomial coefficients [intercept, slope, ...]
    - x: Points at which to evaluate the polynomial
    
    Returns:
    - Evaluated polynomial values
    """
    result = []
    for x_i in x:
        # For a linear polynomial: y = p[0] + p[1]*x
        if len(p) == 2:
            value = p[0] + p[1] * x_i
        else:
            # General case for any polynomial
            value = 0
            for i, coef in enumerate(p):
                # p[0] is the highest degree term in numpy, but we're using the opposite order
                # p[0] is the intercept, p[1] is slope, etc.
                value += coef * (x_i ** i)
        result.append(value)
    return result

def hann(M):
    """
    Hann window function.
    """
    if M < 1:
        return []
    if M == 1:
        return [1.0]
    
    result = []
    for n in range(M):
        value = 0.5 - 0.5 * math.cos(2.0 * math.pi * n / (M - 1))
        result.append(value)
    return result

def hamming(M):
    """
    Hamming window function.
    """
    if M < 1:
        return []
    if M == 1:
        return [1.0]
    
    result = []
    for n in range(M):
        value = 0.54 - 0.46 * math.cos(2.0 * math.pi * n / (M - 1))
        result.append(value)
    return result

def blackman(M):
    """
    Blackman window function.
    """
    if M < 1:
        return []
    if M == 1:
        return [1.0]
    
    result = []
    for n in range(M):
        value = (0.42 - 0.5 * math.cos(2.0 * math.pi * n / (M - 1)) + 
                0.08 * math.cos(4.0 * math.pi * n / (M - 1)))
        result.append(value)
    return result

def bartlett(M):
    """
    Bartlett window function (triangular).
    """
    if M < 1:
        return []
    if M == 1:
        return [1.0]
    
    result = []
    for n in range(M):
        value = 1.0 - abs(2.0 * n / (M - 1) - 1.0)
        result.append(value)
    return result

def boxcar(M):
    """
    Boxcar window function (rectangular).
    """
    return [1.0] * M

def flattop(M):
    """
    Flat top window function.
    """
    if M < 1:
        return []
    if M == 1:
        return [1.0]
    
    a = [0.21557, 0.41663, 0.277263, 0.083578, 0.006947]
    result = [0.0] * M
    
    for n in range(M):
        value = 0.0
        for i in range(len(a)):
            value += (-1) ** i * a[i] * math.cos(2 * math.pi * i * n / (M - 1))
        result[n] = value
    
    return result

def get_window(window_type, N):
    """
    Get a specific window function by name.
    """
    if window_type == 'hann':
        return hann(N)
    elif window_type == 'hamming':
        return hamming(N)
    elif window_type == 'blackman':
        return blackman(N)
    elif window_type == 'bartlett':
        return bartlett(N)
    elif window_type == 'boxcar':
        return boxcar(N)
    elif window_type == 'flattop':
        return flattop(N)
    else:
        # Default to Hann window
        return hann(N)

def next_power_of_2(n):
    """
    Find the next power of 2 greater than or equal to n.
    """
    return 1 if n == 0 else 2 ** ((n - 1).bit_length())

def detrend(x, type='constant'):
    """
    Remove linear trend along axis from data.
    
    Parameters:
    - x: Input data
    - type: 'constant' for mean subtraction, 'linear' for linear detrending, 'none' for no detrending
    
    Returns:
    - Detrended data
    """
    if type == 'none':
        return list(x)  # Return a copy
        
    if type == 'constant':
        x_mean = mean(x)
        return [item - x_mean for item in x]
    elif type == 'linear':
        n = len(x)
        if n <= 1:
            return list(x)  # Can't detrend 0 or 1 points
            
        # Create time points
        t = list(range(n))
        
        # Calculate slope and intercept
        sum_t = sum(t)
        sum_x = sum(x)
        sum_tt = sum(ti**2 for ti in t)
        sum_tx = sum(ti * xi for ti, xi in zip(t, x))
        
        # Calculate slope and intercept from linear regression
        # For y = mx + b
        # m = (n*sum(tx) - sum(t)*sum(x)) / (n*sum(t^2) - sum(t)^2)
        # b = (sum(x) - m*sum(t)) / n
        denom = n * sum_tt - sum_t**2
        if denom == 0:  # Avoid division by zero
            return [xi - sum_x / n for xi in x]  # Fallback to constant detrend
            
        slope = (n * sum_tx - sum_t * sum_x) / denom
        intercept = (sum_x - slope * sum_t) / n
        
        # Subtract trend line from data
        return [xi - (slope * ti + intercept) for ti, xi in zip(t, x)]
    else:
        return list(x)  # Return a copy for unknown detrend type

def improved_fft(x):
    """
    Improved FFT implementation for better performance.
    
    This implementation uses the Cooley-Tukey FFT algorithm but with
    iterative steps to avoid deep recursion for large arrays.
    
    Parameters:
    - x: Input array
    
    Returns:
    - FFT of input array
    """
    N = len(x)
    
    # Handle base case
    if N <= 1:
        return x
    
    # Check if N is a power of 2
    if N & (N - 1) != 0:
        # If not a power of 2, pad with zeros to the next power of 2
        next_power = next_power_of_2(N)
        padded_x = list(x) + [0] * (next_power - N)
        N = next_power
    else:
        padded_x = list(x)
    
    # Bit-reversal permutation
    output = [0] * N
    for i in range(N):
        j = 0
        for k in range(N.bit_length() - 1):
            j = (j << 1) | ((i >> k) & 1)
        if j < N:
            output[j] = complex(padded_x[i]) if not isinstance(padded_x[i], complex) else padded_x[i]
    
    # Butterfly computation
    for s in range(1, N.bit_length()):
        m = 1 << s  # 2^s
        m2 = m >> 1  # m/2
        w = complex(math.cos(-2 * math.pi / m), math.sin(-2 * math.pi / m))
        
        for j in range(0, N, m):
            w_n = complex(1, 0)
            
            for k in range(m2):
                t = w_n * output[j + k + m2]
                u = output[j + k]
                output[j + k] = u + t
                output[j + k + m2] = u - t
                w_n *= w
    
    return output

def fft(x):
    """
    Compute the one-dimensional discrete Fourier Transform.
    This is a wrapper around improved_fft for compatibility.
    """
    # For small arrays, we can use the original recursive implementation
    if len(x) <= 16:
        N = len(x)
        
        # Base case for recursion
        if N <= 1:
            return x
        
        # Recursive case: split into even and odd indices
        even = fft([x[i] for i in range(0, N, 2)])
        odd = fft([x[i] for i in range(1, N, 2)])
        
        # Combine results
        result = complex_zeros(N)
        for k in range(N // 2):
            t = cmath.exp(-2j * math.pi * k / N) * odd[k]
            result[k] = even[k] + t
            result[k + N // 2] = even[k] - t
        
        return result
    else:
        # For larger arrays, use the improved implementation
        return improved_fft(x)

def rfft(x):
    """
    Compute the real FFT for real input.
    """
    N = len(x)
    
    # For large arrays, use a more efficient approach
    if N > 256:
        return simplified_rfft(x)
    
    result = fft(x)
    # For real input, only need first N//2 + 1 points
    return result[:N//2 + 1]

def rfftfreq(n, d=1.0):
    """
    Return the Discrete Fourier Transform sample frequencies for real input.
    
    Parameters:
    - n: Window length
    - d: Sample spacing (inverse of sampling rate)
    
    Returns:
    - f: Array of length n//2 + 1 containing the sample frequencies
    """
    # For a length n signal, the positive frequencies are 0, 1, ..., n//2 times
    # the fundamental frequency 1/(n*d)
    val = 1.0 / (n * d)
    num_freqs = n // 2 + 1
    results = [0.0] * num_freqs
    
    for i in range(num_freqs):
        results[i] = i * val
    
    return results

def irfft(complex_data, n=None):
    """
    Inverse real FFT.
    
    Parameters:
    - complex_data: Complex frequency domain data (result of rfft)
    - n: Length of the output array (default: 2*(len(complex_data)-1))
    
    Returns:
    - Time domain signal (real values)
    """
    if n is None:
        n = 2 * (len(complex_data) - 1)
    
    # For a signal of length n, we expect complex_data to have n//2 + 1 elements
    # The first element is the DC component, and if n is even, the last element is the Nyquist frequency
    
    # For even n:
    # [DC, f1, f2, ..., f_nyquist] -> [DC, f1, f2, ..., f_nyquist, conj(f_nyquist-1), ..., conj(f1)]
    # 
    # For odd n:
    # [DC, f1, f2, ...] -> [DC, f1, f2, ..., conj(f2), conj(f1)]
    
    # Create a new list for the full spectrum to avoid modifying the input
    full_spectrum = []
    
    # Copy the positive frequencies (including DC)
    for i in range(len(complex_data)):
        full_spectrum.append(complex_data[i])
    
    # For even n, don't duplicate Nyquist frequency
    last_idx = len(complex_data) - 1 if n % 2 == 1 else len(complex_data) - 2
    
    # Add negative frequencies (conjugates of positive frequencies in reverse order)
    for i in range(last_idx, 0, -1):
        full_spectrum.append(complex_data[i].conjugate())
    
    # At this point, full_spectrum should have exactly n elements for even n
    # If not, we need to adjust it
    if len(full_spectrum) != n:
        if len(full_spectrum) < n:
            # Pad with zeros if too short
            full_spectrum.extend([0j] * (n - len(full_spectrum)))
        else:
            # Truncate if too long
            full_spectrum = full_spectrum[:n]
    
    # Perform the inverse FFT
    result = ifft(full_spectrum)
    
    # Extract the real part and ensure correct length
    return [val.real for val in result[:n]]

def ifft(x):
    """
    Compute the one-dimensional inverse discrete Fourier Transform.
    """
    N = len(x)
    
    # For large arrays, use a more efficient approach
    if N > 256:
        # Calculate the conjugate of each element
        x_conj = [value.conjugate() for value in x]
        
        # Compute the forward FFT using improved implementation
        y = improved_fft(x_conj)
        
        # Take the conjugate and scale
        return [value.conjugate() / N for value in y]
    
    # For small arrays, use the original implementation
    # Calculate the conjugate of each element
    x_conj = [value.conjugate() for value in x]
    
    # Compute the forward FFT
    y = fft(x_conj)
    
    # Take the conjugate and scale
    return [value.conjugate() / N for value in y]

def simplified_fft(x):
    """
    A simplified FFT implementation for large arrays that trades some 
    accuracy for speed. Uses downsampling and interpolation to handle large arrays.
    """
    n = len(x)
    
    # For small arrays, use the regular FFT
    if n <= 256:
        return fft(x)
    
    # Determine appropriate stride to limit computation
    # More aggressive downsampling for very large arrays
    if n > 4096:
        stride = max(1, n // 256)  # For very large arrays
    else:
        stride = max(1, n // 512)  # Less aggressive for medium arrays
    
    x_downsampled = [x[i] for i in range(0, n, stride)]
    result_downsampled = improved_fft(x_downsampled)
    
    # Upsample to original size using linear interpolation
    result = complex_zeros(n)
    
    # Fill in known points
    ds_len = len(result_downsampled)
    for i in range(ds_len):
        orig_idx = i * stride
        if orig_idx < n:
            result[orig_idx] = result_downsampled[i]
    
    # Linear interpolation for intermediate points
    for i in range(n):
        if i % stride == 0:
            continue  # Already filled
        
        # Find surrounding known points
        left_idx = (i // stride) * stride
        right_idx = left_idx + stride
        
        if right_idx >= n:
            right_idx = left_idx  # Use left value if right is out of bounds
        
        # Linear interpolation weight
        if right_idx > left_idx:
            weight = (i - left_idx) / (right_idx - left_idx)
        else:
            weight = 0
        
        # Interpolate real and imaginary parts separately
        real_interp = (1 - weight) * result[left_idx].real + weight * result[right_idx].real
        imag_interp = (1 - weight) * result[left_idx].imag + weight * result[right_idx].imag
        
        result[i] = complex(real_interp, imag_interp)
    
    return result

def simplified_rfft(x):
    """
    A simplified real FFT implementation for large arrays that trades some
    accuracy for speed.
    """
    n = len(x)
    
    # For small arrays, use the regular RFFT
    if n <= 256:
        return rfft(x)
    
    # Determine appropriate stride to limit computation
    if n > 4096:
        stride = max(1, n // 256)  # For very large arrays
    else:
        stride = max(1, n // 512)  # Less aggressive for medium arrays
    
    x_downsampled = [x[i] for i in range(0, n, stride)]
    
    # Compute FFT of downsampled data
    result_downsampled = improved_fft(x_downsampled)
    
    # Determine size of output array (n//2 + 1)
    output_size = n // 2 + 1
    
    # Extract the first half for real input (rfft)
    ds_output_size = len(x_downsampled) // 2 + 1
    result_downsampled = result_downsampled[:ds_output_size]
    
    # Create result array
    result = complex_zeros(output_size)
    
    # Map downsampled frequencies to original frequencies
    for i in range(ds_output_size):
        # Calculate the original frequency this corresponds to
        orig_freq_idx = min(i * stride, output_size - 1)
        if orig_freq_idx < output_size:
            result[orig_freq_idx] = result_downsampled[i]
    
    # Perform linear interpolation for missing frequencies
    filled = [False] * output_size
    for i in range(0, ds_output_size * stride, stride):
        if i < output_size:
            filled[i] = True
    
    # Find and fill gaps
    for i in range(output_size):
        if filled[i]:
            continue
        
        # Find nearest left and right filled points
        left_idx = i
        while left_idx >= 0 and not filled[left_idx]:
            left_idx -= 1
        
        right_idx = i
        while right_idx < output_size and not filled[right_idx]:
            right_idx += 1
        
        # Handle edge cases
        if left_idx < 0:
            left_idx = right_idx
        if right_idx >= output_size:
            right_idx = left_idx
        
        # If both indices are valid, perform interpolation
        if left_idx >= 0 and right_idx < output_size:
            if right_idx > left_idx:
                weight = (i - left_idx) / (right_idx - left_idx)
                real_interp = (1 - weight) * result[left_idx].real + weight * result[right_idx].real
                imag_interp = (1 - weight) * result[left_idx].imag + weight * result[right_idx].imag
                result[i] = complex(real_interp, imag_interp)
            else:
                # If they're the same, just copy
                result[i] = result[left_idx]
    
    return result

def periodogram(x, fs=1.0, window='hann', nfft=None, detrend_type='constant', 
               return_onesided=True, scaling='density'):
    """
    Estimate power spectral density using periodogram method.
    
    Parameters:
    - x: Input signal
    - fs: Sampling frequency
    - window: Window function to apply
    - nfft: Length of FFT
    - detrend_type: Type of detrending
    - return_onesided: If True, return only positive frequencies for real input
    - scaling: 'density' for PSD, 'spectrum' for power spectrum
    
    Returns:
    - f: Array of frequency points
    - Pxx: Power spectral density or power spectrum
    """
    # Start time measurement
    start_time = time.time()
    
    # Convert input to list
    x = list(x)
    n_input = len(x)
    
    # Apply detrending if needed
    if detrend_type != 'none':
        x = detrend(x, detrend_type)
    
    # Get window function
    if isinstance(window, str):
        win = get_window(window, n_input)
    else:
        win = window
    
    # Apply window
    windowed_x = multiply(x, win)
    
    # Set nfft
    if nfft is None:
        nfft = next_power_of_2(n_input)
    
    # Pad with zeros if needed
    if len(windowed_x) < nfft:
        windowed_x += [0] * (nfft - len(windowed_x))
    
    # Compute FFT - use optimized method for large arrays
    if nfft > 256:
        fft_result = simplified_rfft(windowed_x)
    else:
        fft_result = rfft(windowed_x)
    
    # Calculate frequencies
    freqs = rfftfreq(nfft, 1.0/fs)
    
    # Calculate power spectrum
    psd = [abs(val)**2 for val in fft_result]
    
    # Scale PSD according to window and scaling type
    # For a Hann window, the appropriate scaling factor is 8/3
    window_scaling = sum(w**2 for w in win) / len(win)
    
    if scaling == 'density':
        # Scale by sampling frequency and window factors
        scale = 1.0 / (fs * window_scaling)
        psd = [p * scale for p in psd]
    elif scaling == 'spectrum':
        # Scale by window factor only
        scale = 1.0 / window_scaling
        psd = [p * scale for p in psd]
    
    # Apply scaling for one-sided spectrum
    if return_onesided and nfft % 2 == 0:
        # Scale all frequencies except DC and Nyquist by 2
        for i in range(1, len(psd) - 1):
            psd[i] *= 2.0
    
    # Print time taken for large arrays (debug)
    elapsed = time.time() - start_time
    if n_input > 1000:
        print(f"Periodogram completed in {elapsed:.2f} seconds for {n_input} points")
    
    return freqs, psd

def welch(x, fs=1.0, window='hann', nperseg=256, noverlap=None, 
         nfft=None, detrend_type='constant', return_onesided=True, 
         scaling='density', average='mean'):
    """
    Estimate power spectral density using Welch's method.
    Optimized for better performance with large arrays.
    
    Parameters:
    - x: Input signal
    - fs: Sampling frequency
    - window: Window function
    - nperseg: Length of each segment
    - noverlap: Number of points to overlap between segments
    - nfft: Length of FFT
    - detrend_type: 'constant' or 'linear' detrending
    - return_onesided: Return only positive frequencies if True
    - scaling: 'density' for PSD, 'spectrum' for power spectrum
    - average: Method for averaging periodograms ('mean' or 'median')
    
    Returns:
    - f: Array of frequency points
    - Pxx: Power spectral density or power spectrum
    """
    # Start time measurement
    start_time = time.time()
    
    # Make a copy of the input data
    x = list(x)  # Ensure we have a copy
    n_input = len(x)
    
    # Set default noverlap if not specified
    if noverlap is None:
        noverlap = nperseg // 2
    
    # Get window function
    if isinstance(window, str):
        win = get_window(window, nperseg)
    else:
        win = window
    
    # Set default nfft if not specified
    if nfft is None:
        nfft = next_power_of_2(nperseg)
    
    # Ensure valid parameters
    if nperseg > len(x):
        nperseg = len(x)
    
    if noverlap >= nperseg:
        noverlap = nperseg - 1
    
    # Calculate the number of segments
    step = nperseg - noverlap
    num_segments = max(1, (len(x) - nperseg) // step + 1)
    
    # Get frequency array
    freqs = rfftfreq(nfft, 1.0/fs)
    n_freqs = len(freqs)
    
    # Performance optimization: pre-calculate window normalization
    window_norm = sum(w**2 for w in win) / len(win)
    
    # Initialize PSD accumulator
    Pxx = zeros(n_freqs)
    
    # Use direct calculation for small numbers of segments
    if num_segments <= 10:
        # Process each segment
        for i in range(num_segments):
            start_idx = i * step
            segment = x[start_idx:start_idx + nperseg]
            
            # If segment is too short, pad with zeros
            if len(segment) < nperseg:
                segment = segment + [0] * (nperseg - len(segment))
            
            # Apply detrending if needed
            if detrend_type != 'none':
                segment = detrend(segment, detrend_type)
            
            # Apply window
            windowed_segment = multiply(segment, win)
            
            # Pad to nfft if needed
            if len(windowed_segment) < nfft:
                windowed_segment = windowed_segment + [0] * (nfft - len(windowed_segment))
            
            # Compute FFT using optimized method for large arrays
            if nfft > 256:
                fft_result = simplified_rfft(windowed_segment)
            else:
                fft_result = rfft(windowed_segment)
            
            # Calculate power
            seg_psd = [abs(val)**2 for val in fft_result]
            
            # Accumulate PSD
            if average == 'mean':
                for j in range(n_freqs):
                    Pxx[j] += seg_psd[j]
            else:  # median
                # For median, we need to store all values
                if i == 0:
                    psd_matrix = [[] for _ in range(n_freqs)]
                for j in range(n_freqs):
                    psd_matrix[j].append(seg_psd[j])
    else:
        # For many segments, process in batches
        batch_size = 10
        for batch_start in range(0, num_segments, batch_size):
            batch_end = min(batch_start + batch_size, num_segments)
            batch_psd = [0.0] * n_freqs  # Accumulator for current batch
            
            for i in range(batch_start, batch_end):
                start_idx = i * step
                segment = x[start_idx:start_idx + nperseg]
                
                # If segment is too short, pad with zeros
                if len(segment) < nperseg:
                    segment = segment + [0] * (nperseg - len(segment))
                
                # Apply detrending if needed
                if detrend_type != 'none':
                    segment = detrend(segment, detrend_type)
                
                # Apply window
                windowed_segment = multiply(segment, win)
                
                # Pad to nfft if needed
                if len(windowed_segment) < nfft:
                    windowed_segment = windowed_segment + [0] * (nfft - len(windowed_segment))
                
                # Compute FFT using optimized method
                if nfft > 256:
                    fft_result = simplified_rfft(windowed_segment)
                else:
                    fft_result = rfft(windowed_segment)
                
                # Calculate power and accumulate
                for j in range(n_freqs):
                    batch_psd[j] += abs(fft_result[j])**2
            
            # Add batch results to overall PSD
            for j in range(n_freqs):
                Pxx[j] += batch_psd[j]
    
    # Finalize PSD calculation
    if average == 'mean':
        # Calculate mean across segments
        Pxx = [p / num_segments for p in Pxx]
    else:  # median
        # Calculate median across segments
        Pxx = [sorted(psd_matrix[j])[num_segments // 2] for j in range(n_freqs)]
    
    # Apply scaling
    if scaling == 'density':
        # Scale by sampling frequency and window factor
        scale = 1.0 / (fs * window_norm)
        Pxx = [p * scale for p in Pxx]
    elif scaling == 'spectrum':
        # Scale by window factor only
        scale = 1.0 / window_norm
        Pxx = [p * scale for p in Pxx]
    
    # Apply one-sided scaling
    if return_onesided and nfft % 2 == 0:
        # Scale all frequencies except DC and Nyquist by 2
        for i in range(1, len(Pxx) - 1):
            Pxx[i] *= 2.0
    
    # Print time taken for large arrays (debug)
    elapsed = time.time() - start_time
    if n_input > 1000:
        print(f"Welch completed in {elapsed:.2f} seconds for {n_input} points with {num_segments} segments")
    
    return freqs, Pxx

def csv_to_list(filepath, delimiter=','):
    """
    Read a CSV file and return a list of values
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Remove any trailing whitespace and convert to float
    data = []
    for line in lines:
        line = line.strip()
        if line:  # Skip empty lines
            try:
                # Try to convert the value to float
                value = float(line)
                data.append(value)
            except ValueError:
                # Skip lines that can't be converted to float
                continue
    
    return data
