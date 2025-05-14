# Pure Python Noise Spectral Density (NSD) App

A cross-platform application for estimating noise spectral density (NSD) from time-series data, implemented in Pure Python with PySide6. This app is a recreation of the functionality from [macaba's NSD tool](https://github.com/macaba/NSD/tree/main) with zero external dependencies except for PySide6.

## Features

- Load single-column CSV files with time-series data
- Calculate noise spectral density using various windowing methods
- Adjustable sampling rate, window type, and segment length
- Interactive plotting with QtCharts
- Statistics display with mean, standard deviation, and min/max values
- **Zero dependencies** except for PySide6 (no NumPy or SciPy required)

## Installation

1. Clone this repository or download the source code
2. Install only PySide6:

```bash
pip install PySide6
```

3. Run the application:

```bash
python nsd_app.py
```

![Image](https://github.com/user-attachments/assets/81461585-b3c9-459d-b6ff-ccda84c5a8a1)

## Input File Format

The application expects CSV files with a single column of numeric data, with no header. Example:

```
-5.259715387375001E-07
-4.895393397810999E-07
-5.413877378806E-07
-5.731182876255E-07
-5.228528194452E-07
```

Multiple numeric formats are supported, including scientific notation.

## Usage Instructions

1. Click "Load CSV" to select your time-series data file
2. Set the correct sample rate for your data (in Hz)
3. Adjust the analysis parameters if needed:
   - Window Type: Select the windowing function (Hann, Hamming, etc.)
   - PSD Method: Choose between Welch's method or simple periodogram
   - Segment Length: Set the number of points for each segment in Welch's method
4. Click "Calculate NSD" to process the data and view the results
5. The plot shows frequency on the x-axis (logarithmic scale) and NSD in dB/Hz on the y-axis

## Test Data Generation

You can generate test data using the included `pure_generate_test_data.py` script:

```bash
python generate_test_data.py
```

This will create a CSV file with simulated ADC noise that includes both white noise and 1/f (pink) noise components, plus some periodic interference.

## Project Structure

- `nsd_app.py` - Main application with PySide6 GUI
- `signal.py` - Pure Python implementation of signal processing functions
- `generate_test_data.py` - Utility to generate test data without external dependencies

## Implementation Details

This version implements all signal processing functions in pure Python, including:

- FFT (Fast Fourier Transform) and inverse FFT
- Window functions (Hann, Hamming, Blackman, Bartlett, etc.)
- Welch's method for power spectral density estimation
- Periodogram for direct spectral estimation
- Basic statistical functions

## Performance Note

Due to the pure Python implementation of FFT and other algorithms, this version will be slower than the NumPy/SciPy version for large datasets. For production use with large datasets, consider using the version with NumPy/SciPy dependencies.

## Requirements

- Python 3.7+
- PySide6 (including QtCharts module)

## License

This project is released under the MIT License.
