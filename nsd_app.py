import sys
import os
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QLabel, QFileDialog, QComboBox,
                             QSpinBox, QDoubleSpinBox, QCheckBox, QGridLayout, QGroupBox)
from PySide6.QtCore import Qt, Slot, Signal, QPointF
from PySide6.QtGui import QIcon, QFont, QPainter
from PySide6.QtCharts import (QChart, QChartView, QLineSeries, QValueAxis, 
                             QLogValueAxis, QScatterSeries)

# Import our custom pure Python signal processing functions
import signal as ps

class ChartView(QChartView):
    """Qt chart view class for displaying NSD plots"""
    def __init__(self, parent=None):
        chart = QChart()
        chart.setTitle("Noise Spectral Density")
        chart.setAnimationOptions(QChart.SeriesAnimations)
        
        # Create logarithmic x-axis
        self.axis_x = QLogValueAxis()
        self.axis_x.setTitleText("Frequency (Hz)")
        self.axis_x.setLabelFormat("%.1e")
        self.axis_x.setBase(10.0)
        
        # Create y-axis for dB
        self.axis_y = QValueAxis()
        self.axis_y.setTitleText("NSD (dB/Hz)")
        self.axis_y.setLabelFormat("%.1f")
        
        chart.addAxis(self.axis_x, Qt.AlignBottom)
        chart.addAxis(self.axis_y, Qt.AlignLeft)
        
        # Set up the chart
        chart.legend().hide()
        
        # Initialize with empty series
        self.series = QLineSeries()
        chart.addSeries(self.series)
        self.series.attachAxis(self.axis_x)
        self.series.attachAxis(self.axis_y)
        
        super(ChartView, self).__init__(chart)
        self.setRenderHint(QPainter.Antialiasing)
        
    def set_data(self, x, y):
        """Update the chart with new data"""
        # Clear existing data
        self.series.clear()
        
        # Add new data points
        for i in range(len(x)):
            if x[i] > 0:  # Ensure positive values for log axis
                self.series.append(x[i], y[i])
                
        # Set axis ranges
        # Find min and max of valid x values (positive for log scale)
        valid_x = [val for val in x if val > 0]
        if valid_x:
            min_x = min(valid_x)
            max_x = max(valid_x)
            min_y = min(y)
            max_y = max(y)
            
            # Set axis ranges with some padding
            self.axis_x.setRange(min_x * 0.9, max_x * 1.1)
            self.axis_y.setRange(min_y - 5, max_y + 5)
        
    def add_stats_text(self, stats_text):
        """Add statistics text to the chart"""
        # Create a new chart for displaying text
        # This is a simple way to add persistent annotations
        stats_chart = self.chart()
        
        # Create a label with the statistics
        stats_label = QLabel(stats_text)
        stats_label.setStyleSheet("""
            QLabel { 
                background-color: rgba(245, 222, 179, 200); 
                padding: 5px; 
                border-radius: 5px;
                font-size: 10pt;
            }
        """)
        
        # Add the label as a custom item to the chart
        stats_chart.scene().addWidget(stats_label)
        
        # Position the label in the top-left corner with some margin
        stats_label.move(50, 50)

class NoiseSpectralDensityApp(QMainWindow):
    """Main application window for Noise Spectral Density analysis"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Noise Spectral Density (NSD) Analyzer")
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
        
        # Actions
        param_layout.addWidget(QLabel("Actions:"), 4, 0)
        self.calculate_btn = QPushButton("Calculate NSD")
        self.calculate_btn.clicked.connect(self.calculate_nsd)
        self.calculate_btn.setEnabled(False)
        param_layout.addWidget(self.calculate_btn, 4, 1)
        
        param_group.setLayout(param_layout)
        
        # Add groups to top layout
        top_layout.addWidget(file_group, 1)
        top_layout.addWidget(param_group, 2)
        
        # Chart view for plotting
        self.chart_view = ChartView(self)
        
        # Add all elements to main layout
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.chart_view)
        
        # Status bar for messages
        self.statusBar().showMessage("Ready")
        
        # Set the main widget
        self.setCentralWidget(main_widget)
    
    @Slot()
    def load_csv(self):
        """Load time-series data from a CSV file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                # Use our custom CSV reader function
                data = ps.csv_to_list(file_path)
                
                self.data = data
                self.file_label.setText(os.path.basename(file_path))
                self.statusBar().showMessage(f"Loaded {len(data)} data points")
                self.calculate_btn.setEnabled(True)
                
                # Reset the chart
                self.chart_view.series.clear()
                
            except Exception as e:
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
    
    @Slot()
    def calculate_nsd(self):
        """Calculate and plot the noise spectral density"""
        if self.data is None:
            self.statusBar().showMessage("No data loaded")
            return
        
        try:
            # Calculate power spectral density using our pure Python functions
            if self.psd_method == 'welch':
                # Welch's method
                frequencies, psd = ps.welch(
                    self.data, 
                    fs=self.sample_rate, 
                    window=self.window_type,
                    nperseg=self.nperseg,
                    noverlap=self.noverlap,
                    scaling='density'  # Return the power spectral density
                )
            else:
                # Simple periodogram
                frequencies, psd = ps.periodogram(
                    self.data,
                    fs=self.sample_rate,
                    window=self.window_type,
                    scaling='density'
                )
            
            # Convert to dB/Hz
            # Assuming the data is in voltage, so PSD is in V^2/Hz
            psd_db = [10 * ps.log10([p + 1e-20])[0] for p in psd]  # Add small value to avoid log(0)
            
            # Update the chart with new data
            self.chart_view.set_data(frequencies, psd_db)
            
            # Calculate statistics
            data_mean = ps.mean(self.data)
            data_std = ps.std(self.data)
            psd_min, psd_max = ps.min_max(psd_db)
            
            # Add statistics to the chart
            stats_text = (
                f"Mean: {data_mean:.6e}<br>"
                f"Std Dev: {data_std:.6e}<br>"
                f"Min: {psd_min:.2f} dB/Hz<br>"
                f"Max: {psd_max:.2f} dB/Hz"
            )
            
            # Create and position statistics text label
            self.chart_view.add_stats_text(stats_text)
            
            self.statusBar().showMessage("Calculation complete")
            
        except Exception as e:
            self.statusBar().showMessage(f"Error calculating NSD: {str(e)}")

def main():
    app = QApplication(sys.argv)
    window = NoiseSpectralDensityApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
