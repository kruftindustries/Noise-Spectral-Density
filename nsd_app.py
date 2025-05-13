import sys, os, math
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QLabel, QFileDialog, QComboBox,
                             QSpinBox, QDoubleSpinBox, QCheckBox, QGridLayout, QGroupBox,
                             QProgressBar, QMessageBox)
from PySide6.QtCore import Qt, Slot, Signal, QPointF, QThread
from PySide6.QtGui import QIcon, QFont, QPainter, QPen
from PySide6.QtCharts import (QChart, QChartView, QLineSeries, QValueAxis, 
                             QLogValueAxis, QScatterSeries)

# Import our custom pure Python signal processing functions
import signal as ps

class PSDWorker(QThread):
    """Worker thread for calculating PSDs to prevent UI freezing"""
    finished = Signal(list, list)  # Signals: frequencies, psd
    progress = Signal(int)  # Progress signal (0-100)
    
    def __init__(self, data, fs, method, window, nperseg, noverlap, detrend):
        super().__init__()
        self.data = data
        self.fs = fs
        self.method = method
        self.window = window
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.detrend = detrend
        
    def run(self):
        """Run the PSD calculation in a separate thread"""
        try:
            # Emit starting progress
            self.progress.emit(5)
            
            # Using a smaller test sample can help identify issues
            # Let's use the first 2000 points for testing
            test_size = min(2000, len(self.data))
            test_data = self.data[:test_size]
            
            # Do a quick test calculation with a small subset
            # This helps catch formatting issues before full calculation
            if self.method == 'welch':
                try:
                    test_freq, test_psd = ps.welch(
                        test_data,
                        fs=self.fs,
                        window=self.window,
                        nperseg=min(self.nperseg, test_size),
                        noverlap=min(self.noverlap or 0, test_size-1),
                        detrend_type=self.detrend,
                        scaling='density'
                    )
                    self.progress.emit(10)
                except Exception as e:
                    print(f"Test welch calculation failed: {str(e)}")
                    # Continue anyway to the full calculation
            
            # Perform the actual calculation with full dataset
            if self.method == 'welch':
                # Welch's method
                frequencies, psd = ps.welch(
                    self.data, 
                    fs=self.fs, 
                    window=self.window,
                    nperseg=self.nperseg,
                    noverlap=self.noverlap,
                    detrend_type=self.detrend,
                    scaling='density'  # Return the power spectral density
                )
            else:
                # Simple periodogram
                frequencies, psd = ps.periodogram(
                    self.data,
                    fs=self.fs,
                    window=self.window,
                    detrend_type=self.detrend,
                    scaling='density'
                )
            
            # Emit finished signal with results
            self.progress.emit(100)
            self.finished.emit(frequencies, psd)
            
        except Exception as e:
            # Handle exceptions
            print(f"Error in worker thread: {str(e)}")
            self.progress.emit(0)  # Signal that something went wrong


class ChartView(QChartView):
    """Qt chart view class for displaying NSD plots"""
    def __init__(self, parent=None):
        chart = QChart()
        chart.setTitle("NSD estimation")
        chart.setAnimationOptions(QChart.SeriesAnimations)
        
        # Create logarithmic x-axis
        self.axis_x = QLogValueAxis()
        self.axis_x.setTitleText("Frequency (Hz)")
        self.axis_x.setLabelFormat("%.3g")
        self.axis_x.setBase(10.0)
        self.axis_x.setMinorTickCount(9)  # Add minor grid lines
        
        # Create logarithmic y-axis for nV/√Hz
        self.axis_y = QLogValueAxis()
        self.axis_y.setTitleText("Noise (nV/√Hz)")
        self.axis_y.setLabelFormat("%.0f")
        self.axis_y.setBase(10.0)
        self.axis_y.setMinorTickCount(9)  # Add minor grid lines
        
        chart.addAxis(self.axis_x, Qt.AlignBottom)
        chart.addAxis(self.axis_y, Qt.AlignLeft)
        
        # Set up the chart
        chart.legend().hide()
        
        # Set the chart theme to a cleaner look
        chart.setTheme(QChart.ChartThemeLight)
        chart.setBackgroundVisible(False)
        chart.setPlotAreaBackgroundVisible(True)
        chart.setDropShadowEnabled(False)
        
        # Initialize with empty series
        self.series = QLineSeries()
        self.series.setPen(QPen(Qt.blue, 1.5))  # Thicker line for better visibility
        chart.addSeries(self.series)
        self.series.attachAxis(self.axis_x)
        self.series.attachAxis(self.axis_y)
        
        super(ChartView, self).__init__(chart)
        self.setRenderHint(QPainter.Antialiasing)
        
    def set_data(self, x, y):
        """Update the chart with new data"""
        # Clear existing data
        self.series.clear()
        
        # Add new data points - ensure sorted by frequency for a cleaner plot
        points = [(x[i], y[i]) for i in range(len(x)) if x[i] > 0]  # Ensure positive values for log axis
        points.sort(key=lambda p: p[0])  # Sort by frequency
        
        # Add points to series
        for freq, amp in points:
            self.series.append(freq, amp)
                
        # Set axis ranges
        if points:
            # Find min and max values
            x_vals = [p[0] for p in points]
            y_vals = [p[1] for p in points]
            
            min_x = min(x_vals)
            max_x = max(x_vals)
            min_y = min(y_vals)
            max_y = max(y_vals)
            
            # Set logarithmic axis ranges with appropriate padding
            self.axis_x.setRange(min_x * 0.8, max_x * 1.2)
            
            # For log Y-axis, use appropriate multipliers
            self.axis_y.setRange(min_y * 0.8, max_y * 1.2)
            
            # Get the chart for customization
            chart = self.chart()
            
            # Adjust grid lines for better visibility
            chart.setPlotAreaBackgroundBrush(Qt.white)
            chart.setPlotAreaBackgroundVisible(True)
        
    def add_stats_text(self, stats_text):
        """Add statistics text to the chart"""
        # Create a new chart for displaying text
        stats_chart = self.chart()
        
        # Create a label with the statistics
        stats_label = QLabel(stats_text)
        stats_label.setStyleSheet("""
            QLabel { 
                background-color: rgba(255, 255, 255, 180); 
                padding: 8px; 
                border-radius: 5px;
                border: 1px solid #cccccc;
                font-size: 9pt;
                font-family: Arial;
            }
        """)
        
        # Add the label as a custom item to the chart
        stats_chart.scene().addWidget(stats_label)
        
        # Position the label in the top-left corner with some margin
        stats_label.move(60, 50)
        
    def clear_stats_text(self):
        """Clear any statistics labels from the chart"""
        stats_chart = self.chart()
        # Remove custom items (labels) from the chart
        for item in stats_chart.scene().items():
            if isinstance(item, QLabel):
                stats_chart.scene().removeItem(item)


class NoiseSpectralDensityApp(QMainWindow):
    """Main application window for Noise Spectral Density analysis"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Noise Spectral Density (NSD) Analyzer - nV/√Hz")
        self.setGeometry(100, 100, 1000, 600)
        
        # Data attributes
        self.data = None
        self.sample_rate = 1.0  # Default sample rate in Hz
        self.window_type = 'hann'
        self.psd_method = 'welch'
        self.nperseg = 1024  # Default segment length
        self.noverlap = None  # Default overlap (50% of nperseg)
        self.averaging = 'mean'  # Default averaging method
        self.detrend = 'constant'  # Default detrending
        
        # Worker thread
        self.worker = None
        
        # Setup UI
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface"""
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Top controls layout
        top_layout = QHBoxLayout()
        
        # File selection
        file_group = QGroupBox("Input Data")
        file_layout = QVBoxLayout()
        
        file_btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load CSV")
        self.load_btn.clicked.connect(self.load_csv)
        file_btn_layout.addWidget(self.load_btn)
        
        self.file_label = QLabel("No file loaded")
        
        file_layout.addLayout(file_btn_layout)
        file_layout.addWidget(self.file_label)
        file_group.setLayout(file_layout)
        
        # Analysis parameters
        param_group = QGroupBox("Analysis Parameters")
        param_layout = QGridLayout()
        
        # Sample rate
        param_layout.addWidget(QLabel("Sample Rate (Hz):"), 0, 0)
        self.sample_rate_spin = QDoubleSpinBox()
        self.sample_rate_spin.setRange(0.001, 1e12)
        self.sample_rate_spin.setValue(1.0)
        self.sample_rate_spin.setDecimals(6)
        self.sample_rate_spin.setSingleStep(1000)
        self.sample_rate_spin.valueChanged.connect(self.update_sample_rate)
        param_layout.addWidget(self.sample_rate_spin, 0, 1)
        
        # Window type
        param_layout.addWidget(QLabel("Window Type:"), 1, 0)
        self.window_combo = QComboBox()
        self.window_combo.addItems(["hann", "hamming", "blackman", "bartlett", "boxcar", "flattop"])
        self.window_combo.currentTextChanged.connect(self.update_window_type)
        param_layout.addWidget(self.window_combo, 1, 1)
        
        # PSD Method
        param_layout.addWidget(QLabel("PSD Method:"), 2, 0)
        self.psd_combo = QComboBox()
        self.psd_combo.addItems(["welch", "periodogram"])
        self.psd_combo.currentTextChanged.connect(self.update_psd_method)
        param_layout.addWidget(self.psd_combo, 2, 1)
        
        # Segment length
        param_layout.addWidget(QLabel("Segment Length:"), 3, 0)
        self.nperseg_spin = QSpinBox()
        self.nperseg_spin.setRange(16, 1048576)
        self.nperseg_spin.setValue(1024)
        self.nperseg_spin.setSingleStep(256)
        self.nperseg_spin.valueChanged.connect(self.update_nperseg)
        param_layout.addWidget(self.nperseg_spin, 3, 1)
        
        # Detrend options
        param_layout.addWidget(QLabel("Detrend:"), 4, 0)
        self.detrend_combo = QComboBox()
        self.detrend_combo.addItems(["constant", "linear", "none"])
        self.detrend_combo.currentTextChanged.connect(self.update_detrend)
        param_layout.addWidget(self.detrend_combo, 4, 1)
        
        # Smoothing option
        param_layout.addWidget(QLabel("Smoothing:"), 5, 0)
        self.smoothing_spin = QSpinBox()
        self.smoothing_spin.setRange(0, 100)
        self.smoothing_spin.setValue(0)
        self.smoothing_spin.setSingleStep(5)
        self.smoothing_spin.setToolTip("Apply moving average smoothing to the plot (0 = none)")
        self.smoothing_spin.valueChanged.connect(self.update_smoothing)
        param_layout.addWidget(self.smoothing_spin, 5, 1)
        
        # Actions
        param_layout.addWidget(QLabel("Actions:"), 6, 0)
        self.calculate_btn = QPushButton("Calculate NSD")
        self.calculate_btn.clicked.connect(self.calculate_nsd)
        self.calculate_btn.setEnabled(False)
        param_layout.addWidget(self.calculate_btn, 6, 1)
        
        param_group.setLayout(param_layout)
        
        # Add groups to top layout
        top_layout.addWidget(file_group, 1)
        top_layout.addWidget(param_group, 2)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)  # Hide initially
        
        # Chart view for plotting
        self.chart_view = ChartView(self)
        
        # Add all elements to main layout
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.chart_view)
        
        # Status bar for messages
        self.statusBar().showMessage("Ready")
        
        # Set the main widget
        self.setCentralWidget(main_widget)
        
        # Initialize smoothing parameter
        self.smoothing_level = 0
    
    @Slot()
    def load_csv(self):
        """Load time-series data from a CSV file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                # Show loading status
                self.statusBar().showMessage("Loading data...")
                QApplication.processEvents()  # Update UI
                
                # Use our custom CSV reader function
                data = ps.csv_to_list(file_path)
                
                if not data:
                    QMessageBox.warning(self, "Warning", "The CSV file is empty or contains invalid data.")
                    self.statusBar().showMessage("Failed to load data")
                    return
                
                self.data = data
                self.file_label.setText(os.path.basename(file_path))
                self.statusBar().showMessage(f"Loaded {len(data)} data points")
                self.calculate_btn.setEnabled(True)
                
                # Reset the chart
                self.chart_view.series.clear()
                self.chart_view.clear_stats_text()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading file: {str(e)}")
                self.statusBar().showMessage(f"Error loading file: {str(e)}")
    
    @Slot(float)
    def update_sample_rate(self, value):
        """Update the sample rate value"""
        self.sample_rate = value
    
    @Slot(str)
    def update_window_type(self, window_type):
        """Update the window type"""
        self.window_type = window_type
    
    @Slot(str)
    def update_psd_method(self, method):
        """Update the PSD calculation method"""
        self.psd_method = method
    
    @Slot(int)
    def update_nperseg(self, value):
        """Update the segment length"""
        self.nperseg = value
        # Default overlap is 50% of segment length
        self.noverlap = value // 2
    
    @Slot(str)
    def update_detrend(self, detrend_type):
        """Update the detrending method"""
        self.detrend = detrend_type
        
    @Slot(int)
    def update_smoothing(self, value):
        """Update the smoothing level"""
        self.smoothing_level = value
        
    def smooth_data(self, frequencies, amplitudes, window_size):
        """Apply moving average smoothing to the data"""
        if window_size <= 1:
            return frequencies, amplitudes
        
        # Sort data by frequency first
        data_points = sorted(zip(frequencies, amplitudes), key=lambda p: p[0])
        sorted_frequencies = [p[0] for p in data_points]
        sorted_amplitudes = [p[1] for p in data_points]
        
        # Apply moving average to amplitudes
        smoothed_amplitudes = []
        n = len(sorted_amplitudes)
        
        for i in range(n):
            # Calculate window bounds
            half_window = window_size // 2
            window_start = max(0, i - half_window)
            window_end = min(n, i + half_window + 1)
            
            # Calculate moving average
            window_values = sorted_amplitudes[window_start:window_end]
            if window_values:
                smoothed_amplitudes.append(sum(window_values) / len(window_values))
            else:
                smoothed_amplitudes.append(sorted_amplitudes[i])
        
        return sorted_frequencies, smoothed_amplitudes
    
    @Slot()
    def calculate_nsd(self):
        """Calculate and plot the noise spectral density"""
        if self.data is None:
            self.statusBar().showMessage("No data loaded")
            return
        
        # Disable UI while calculating
        self.calculate_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.statusBar().showMessage(f"Calculating {self.psd_method}...")
        QApplication.processEvents()  # Update UI
        
        try:
            # Create worker thread for calculation
            self.worker = PSDWorker(
                self.data, 
                self.sample_rate, 
                self.psd_method,
                self.window_type,
                self.nperseg,
                self.noverlap,
                self.detrend
            )
            
            # Connect signals
            self.worker.finished.connect(self.update_plot)
            self.worker.progress.connect(self.update_progress)
            
            # Start calculation
            self.worker.start()
            
        except Exception as e:
            # Handle any setup errors
            QMessageBox.critical(self, "Error", f"Error starting calculation: {str(e)}")
            self.statusBar().showMessage(f"Error: {str(e)}")
            self.progress_bar.setVisible(False)
            self.calculate_btn.setEnabled(True)
            self.load_btn.setEnabled(True)
    
    @Slot(int)
    def update_progress(self, value):
        """Update progress bar with current calculation progress"""
        self.progress_bar.setValue(value)
        QApplication.processEvents()  # Update UI
    
    @Slot(list, list)
    def update_plot(self, frequencies, psd):
        """Update the plot with calculated PSD results"""
        try:
            # Convert PSD to amplitude spectral density in nV/√Hz
            # PSD is in V²/Hz, take square root to get V/√Hz, then convert to nV/√Hz
            nsd_nv = [math.sqrt(max(p, 1e-30)) * 1e9 for p in psd]  # 1e9 converts V to nV
            
            # Apply smoothing if requested
            if self.smoothing_level > 0:
                frequencies, nsd_nv = self.smooth_data(frequencies, nsd_nv, self.smoothing_level)
            
            # Update the chart with new data
            self.chart_view.set_data(frequencies, nsd_nv)
            
            # Calculate statistics
            data_mean = ps.mean(self.data)
            data_std = ps.std(self.data)
            nsd_min, nsd_max = ps.min_max(nsd_nv)
            
            # Calculate RMS noise (if this makes sense for the data)
            noise_rms = data_std * 1e9  # Convert to nV
            
            # Add statistics to the chart
            stats_text = (
                f"Mean: {data_mean:.6e} V<br>"
                f"Std Dev: {data_std:.6e} V<br>"
                f"RMS Noise: {noise_rms:.2f} nV<br>"
                f"Min: {nsd_min:.2f} nV/√Hz<br>"
                f"Max: {nsd_max:.2f} nV/√Hz<br>"
                f"Method: {self.psd_method}"
            )
            
            # Clear any existing stats and add new ones
            self.chart_view.clear_stats_text()
            self.chart_view.add_stats_text(stats_text)
            
            # Clean up UI
            self.statusBar().showMessage("Calculation complete")
            self.progress_bar.setVisible(False)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error updating plot: {str(e)}")
            self.statusBar().showMessage(f"Error: {str(e)}")
        
        finally:
            # Re-enable controls
            self.calculate_btn.setEnabled(True)
            self.load_btn.setEnabled(True)
    
    def closeEvent(self, event):
        """Handle application close event"""
        # Stop worker thread if running
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = NoiseSpectralDensityApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
