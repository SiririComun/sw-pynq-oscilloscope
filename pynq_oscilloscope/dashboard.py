import time
import threading
import numpy as np
import ipywidgets as widgets
from IPython.display import display
import plotly.graph_objects as go

from pynq_oscilloscope.xadc_dma import StreamingXADC
from pynq_oscilloscope.ad3_wavegen import AD3SignalGenerator

class OscilloscopeDashboard:
    """
    Unified, interactive real-time Plotly + IPywidgets Oscilloscope Dashboard.
    Integrates high-speed XADC DMA data acquisition with AD3 signal generation.
    """

    def __init__(self, overlay=None, packet_size: int = 1024):
        self.overlay = overlay
        self.packet_size = packet_size
        
        self._is_running = False
        self._thread = None
        
        # Instantiate Hardware Drivers
        self.ad3 = AD3SignalGenerator()
        self.xadc = None
        
        # Build UI Controls & Plotly Canvas
        self._build_ui()
        self._build_plot()

    def _build_ui(self):
        """Construct IPywidgets Control Panel."""
        # 1. Action Buttons
        self.start_btn = widgets.Button(description="Start", button_style="success", icon="play")
        self.stop_btn = widgets.Button(description="Stop", button_style="danger", icon="stop")
        
        self.start_btn.on_click(self._on_start_clicked)
        self.stop_btn.on_click(self._on_stop_clicked)

        # 2. Waveform Shape Dropdown
        self.shape_dropdown = widgets.Dropdown(
            options=["Sine", "Triangle", "Square"],
            value="Sine",
            description="Waveform:"
        )

        # 3. Frequency Controls with Zero-Latency Client-Side Link
        self.freq_slider = widgets.IntSlider(
            value=10000,
            min=100,
            max=1000000,
            step=1,
            description="Freq Slider:",
            continuous_update=False,
            layout=widgets.Layout(width="400px")
        )
        self.freq_input = widgets.BoundedIntText(
            value=10000,
            min=100,
            max=1000000,
            step=1,
            description="Exact Freq:"
        )
        widgets.jslink((self.freq_slider, "value"), (self.freq_input, "value"))

        # 4. Amplitude Controls with Zero-Latency Client-Side Link
        self.amp_slider = widgets.FloatSlider(
            value=1.5,
            min=0.1,
            max=1.5,
            step=0.1,
            description="Amp Slider:",
            continuous_update=False,
            layout=widgets.Layout(width="300px")
        )
        self.amp_input = widgets.BoundedFloatText(
            value=1.5,
            min=0.1,
            max=1.5,
            step=0.1,
            description="Exact Amp:"
        )
        widgets.jslink((self.amp_slider, "value"), (self.amp_input, "value"))

        # 5. Toggles
        self.trigger_toggle = widgets.ToggleButton(
            value=True,
            description="Trigger",
            button_style="info",
            tooltip="Activate rising-edge software trigger (Stationary Mode)"
        )
        self.autorange_toggle = widgets.ToggleButton(
            value=True,
            description="Auto-Range",
            button_style="info",
            tooltip="Activate adaptive time and voltage grid scaling"
        )

        # 6. Digital Vpp Readout
        self.readout = widgets.HTML(
            value="<h3 style='color: #00FFCC; padding-left: 10px; font-family: monospace;'>Live Vpp: 0.00 V</h3>"
        )

    def _build_plot(self):
        """Construct Plotly FigureWidget Canvas."""
        self.fig = go.FigureWidget()
        self.fig.add_scatter(
            y=[0] * self.packet_size,
            mode="lines",
            line=dict(color="#00FFCC", width=2),
            name="XADC Channel"
        )
        self.fig.update_layout(
            title="<b>Real-Time 1 MSPS DMA Oscilloscope</b>",
            xaxis_title="Samples / Timebase",
            yaxis_title="Voltage (V)",
            yaxis_range=[0, 3.5],
            template="plotly_dark",
            height=450,
            margin=dict(l=40, r=20, t=50, b=40),
            uirevision="oscilloscope_view_state"  # Preserves user zoom & pan
        )

    def _on_start_clicked(self, b):
        self.start()

    def _on_stop_clicked(self, b):
        self.stop()

    def _update_loop(self):
        """Background thread execution loop for acquisition & UI updates."""
        try:
            # Initialize hardware acquisition if overlay provided
            if self.overlay and self.xadc is None:
                self.xadc = StreamingXADC(self.overlay, default_packet_size=self.packet_size)

            # Start AD3 Signal Generation
            self.ad3.start(
                shape=self.shape_dropdown.value,
                frequency=float(self.freq_slider.value),
                amplitude=float(self.amp_slider.value)
            )

            while self._is_running:
                # Update AD3 parameters dynamically if changed via UI
                self.ad3.update_parameters(
                    shape=self.shape_dropdown.value,
                    frequency=float(self.freq_slider.value),
                    amplitude=float(self.amp_slider.value)
                )

                # Capture frame from XADC DMA (if overlay available)
                if self.xadc:
                    voltages = self.xadc.capture(crop_startup_samples=8)
                else:
                    # Simulation mode if no hardware overlay provided
                    t = np.linspace(0, 0.001, self.packet_size)
                    voltages = 1.65 + (self.amp_slider.value / 2.0) * np.sin(2 * np.pi * self.freq_slider.value * t)

                # --- FEATURE 1: SOFTWARE RISING-EDGE TRIGGER ---
                trigger_idx = 0
                if self.trigger_toggle.value and len(voltages) > 1:
                    threshold = 1.65  # Midpoint threshold for 0V-3.3V range
                    crossings = np.where((voltages[:-1] <= threshold) & (voltages[1:] > threshold))[0]
                    if len(crossings) > 0:
                        trigger_idx = crossings[0]

                # --- FEATURE 2: ADAPTIVE AXIS SCALING (AUTO-RANGE) ---
                if self.autorange_toggle.value:
                    freq = float(self.freq_slider.value)
                    period_us = 1e6 / freq if freq > 0 else 1000
                    show_samples = int(5 * period_us)
                    show_samples = max(40, min(show_samples, len(voltages) - trigger_idx))
                    
                    self.fig.layout.xaxis.range = [0, show_samples]
                    margin = 0.2
                    amp = float(self.amp_slider.value)
                    self.fig.layout.yaxis.range = [max(0.0, 1.65 - (amp + margin)), min(3.5, 1.65 + (amp + margin))]
                    
                    plot_voltages = voltages[trigger_idx : trigger_idx + show_samples]
                else:
                    plot_voltages = voltages[trigger_idx:]

                # Update Canvas & Live Readout
                self.fig.data[0].y = plot_voltages
                
                if len(voltages) > 0:
                    vpp = np.max(voltages) - np.min(voltages)
                    self.readout.value = f"<h3 style='color: #00FFCC; padding-left: 10px; font-family: monospace;'>Live Vpp: {vpp:.2f} V</h3>"

                # Throttle update rate to ~25 FPS to prevent browser lag
                time.sleep(0.04)

        except Exception as e:
            print(f"[Dashboard] Error in update loop: {e}")
        finally:
            self._is_running = False
            self.ad3.stop()
            if self.xadc:
                self.xadc.close()
                self.xadc = None
            print("[Dashboard] Stopped cleanly.")

    def start(self):
        """Launch dashboard execution in background thread."""
        if self._is_running:
            print("[Dashboard] Dashboard is already running.")
            return

        self._is_running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Halt dashboard background updates."""
        self._is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def display(self):
        """Render the complete UI Dashboard in Jupyter."""
        row1 = widgets.HBox([self.start_btn, self.stop_btn, self.trigger_toggle, self.autorange_toggle, self.readout])
        row2 = widgets.HBox([self.shape_dropdown, self.amp_slider, self.amp_input])
        row3 = widgets.HBox([self.freq_slider, self.freq_input])
        
        ui_layout = widgets.VBox([row1, row2, row3, self.fig])
        display(ui_layout)