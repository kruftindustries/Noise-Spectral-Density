"""
Generate test data for the Noise Spectral Density application,
implemented in pure Python without any external dependencies.
"""
import math
import random
import cmath

def random_normal(mean=0, std=1):
    """
    Generate random numbers from a normal distribution using the Box-Muller transform.
    """
    u1 = random.random()
    u2 = random.random()
    
    z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    
    return mean + z0 * std

def generate_white_noise(num_samples, noise_level, seed=None):
    """
    Generate Gaussian white noise.
    
    Parameters:
    - num_samples: Number of samples to generate
    - noise_level: Standard deviation of the noise
    - seed: Random seed for reproducibility
    
    Returns:
    - List of noise samples
    """
    if seed is not None:
        random.seed(seed)
    
    return [random_normal(0, noise_level) for _ in range(num_samples)]

def generate_pink_noise(num_samples, fs, noise_level, seed=None):
    """
    Generate pink noise (1/f noise) in the time domain.
    
    Parameters:
    - num_samples: Number of samples to generate
    - fs: Sampling frequency
    - noise_level: Noise level (standard deviation)
    - seed: Random seed for reproducibility
    
    Returns:
    - List of pink noise samples
    """
    if seed is not None:
        random.seed(seed)
    
    # Number of FFT points (use next power of 2)
    nfft = 1
    while nfft < num_samples:
        nfft *= 2
    
    # Generate white noise in frequency domain
    # Use complex values with random phase
    X = [0.0] * nfft
    
    # DC component is real
    X[0] = 0.0
    
    # Generate frequency domain samples
    for i in range(1, nfft // 2):
        # Magnitude decreases with 1/sqrt(f)
        magnitude = 1.0 / math.sqrt(i)
        
        # Random phase
        phase = random.uniform(0, 2 * math.pi)
        
        # Complex value with given magnitude and phase
        X[i] = magnitude * complex(math.cos(phase), math.sin(phase))
        
        # Make sure we have conjugate symmetry for real output
        X[nfft - i] = X[i].conjugate()
    
    # Nyquist frequency component is real
    if nfft % 2 == 0:
        X[nfft // 2] = complex(random.uniform(-1, 1), 0)
    
    # Convert to time domain using inverse FFT
    x = inverse_fft(X)
    
    # Take only the real part and the required number of samples
    result = [sample.real for sample in x[:num_samples]]
    
    # Scale to desired noise level
    scale = noise_level / math.sqrt(sum(v**2 for v in result) / len(result))
    result = [sample * scale for sample in result]
    
    return result

def fft(x):
    """
    Compute the Fast Fourier Transform (FFT).
    
    Parameters:
    - x: Input signal (complex values)
    
    Returns:
    - FFT result (complex values)
    """
    n = len(x)
    if n <= 1:
        return x
    
    # Split into even and odd indices
    even = fft([x[i] for i in range(0, n, 2)])
    odd = fft([x[i] for i in range(1, n, 2)])
    
    # Combine
    result = [0j] * n
    for k in range(n // 2):
        t = cmath.exp(-2j * math.pi * k / n) * odd[k]
        result[k] = even[k] + t
        result[k + n // 2] = even[k] - t
    
    return result

def inverse_fft(X):
    """
    Compute the Inverse Fast Fourier Transform (IFFT).
    
    Parameters:
    - X: Frequency domain signal (complex values)
    
    Returns:
    - Time domain signal (complex values)
    """
    n = len(X)
    
    # Take complex conjugate
    X_conj = [x.conjugate() for x in X]
    
    # Compute forward FFT
    result = fft(X_conj)
    
    # Take conjugate and scale
    return [x.conjugate() / n for x in result]

def generate_sine_wave(time, frequency, amplitude=1.0, phase=0.0):
    """
    Generate a sine wave.
    
    Parameters:
    - time: List of time points
    - frequency: Frequency in Hz
    - amplitude: Amplitude of the sine wave
    - phase: Phase offset in radians
    
    Returns:
    - List of sine wave values
    """
    return [amplitude * math.sin(2 * math.pi * frequency * t + phase) for t in time]

def generate_adc_noise(num_samples, fs, noise_level, seed=None):
    """
    Generate simulated ADC noise with 1/f characteristic plus white noise.
    
    Parameters:
    - num_samples: Number of time-domain samples to generate
    - fs: Sampling frequency in Hz
    - noise_level: Base level of white noise (standard deviation)
    - seed: Random seed for reproducibility
    
    Returns:
    - time: Time array in seconds
    - noise: Noise signal array
    """
    if seed is not None:
        random.seed(seed)
    
    # Time array
    time = [i / fs for i in range(num_samples)]
    
    # Generate white noise component
    white_noise = generate_white_noise(num_samples, noise_level, seed)
    
    # Generate 1/f (pink) noise component
    pink_noise = generate_pink_noise(num_samples, fs, noise_level, seed)
    
    # Combine both noise types
    noise = [w + p for w, p in zip(white_noise, pink_noise)]
    
    return time, noise

def add_periodic_interference(time, noise, frequencies, amplitudes):
    """
    Add periodic interference at specific frequencies.
    
    Parameters:
    - time: Time array
    - noise: Base noise signal
    - frequencies: List of interference frequencies in Hz
    - amplitudes: List of interference amplitudes
    
    Returns:
    - Noise signal with added periodic interference
    """
    result = noise.copy()
    
    for freq, amp in zip(frequencies, amplitudes):
        sine_wave = generate_sine_wave(time, freq, amp)
        result = [n + s for n, s in zip(result, sine_wave)]
    
    return result

def write_csv(filename, data):
    """
    Write data to a CSV file.
    
    Parameters:
    - filename: Output filename
    - data: List of values to write
    """
    with open(filename, 'w') as f:
        for value in data:
            f.write(f"{value:.14e}\n")

def main():
    # Parameters
    fs = 20000  # 20 kHz sampling rate
    duration = 10  # 10 seconds
    num_samples = int(fs * duration)
    noise_level = 1e-6  # 1 µV base noise level
    
    # Generate base noise
    time, noise = generate_adc_noise(num_samples, fs, noise_level, seed=42)
    
    # Add some periodic interference (e.g., from power lines, etc.)
    interference_freqs = [50, 150, 1000]  # Hz (e.g., 50 Hz power line, harmonics, etc.)
    interference_amps = [5e-7, 2e-7, 1e-7]  # Amplitudes
    
    signal = add_periodic_interference(time, noise, interference_freqs, interference_amps)
    
    # Save to CSV for NSD app
    write_csv('test_noise_data.csv', signal)
    print(f"Saved {num_samples} samples to test_noise_data.csv")
    
    # Without matplotlib, we can't create a plot visualization.
    # If matplotlib is available, you could uncomment the following code:
    
    """
    try:
        import matplotlib.pyplot as plt
        # Plot a small section of the signal
        plt.figure(figsize=(10, 6))
        
        # Plot first 1000 samples
        plt.plot(time[:1000], signal[:1000])
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (V)')
        plt.title('Generated Noise Signal (First 1000 Samples)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('test_noise_plot.png')
        plt.show()
    except ImportError:
        print("Matplotlib not available for plotting. Data has been saved to CSV.")
    """
    
    print("Test data generation complete. Use this file with the NSD app.")
    print("Remember to set the sample rate to:", fs, "Hz")

if __name__ == "__main__":
    main()
