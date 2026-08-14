import time
import threading
import numpy as np
import ipywidgets as widgets
from IPython.display import display
import plotly.graph_objects as go

from pynq_oscilloscope.xadc_dma import StreamingXADC
from pynq_oscilloscope.hw_trigger import HardwareTrigger
from pynq_oscilloscope.ad3_wavegen import AD3SignalGenerator

class OscilloscopeDashboard:
    """
    Unified, interactive real-time Plotly + IPywidgets Oscilloscope Dashboard.
    Integrates FPGA Hardware Triggering + 1 MSPS DMA acquisition with AD3 signal generation.
    """

    def __init__(self, overlay=None, packet_size: int = 16384, display_window: int = 1024):
        self.overlay = overlay
        self.packet_size = packet_size
        self.display_window = display_window
        
        self._is_running = False
        self._thread = None
        
        # Attach or instantiate sub-drivers
        if self.overlay and hasattr(self.overlay, "wavegen"):
            self.ad3 = self.overlay.wavegen
        else:
            self.ad3 = AD3SignalGenerator()

        if self.overlay and hasattr(self.overlay, "trigger"):
            self.trigger = self.overlay.trigger
        elif self.overlay:
            self.trigger = HardwareTrigger(self.overlay)
        else:
            self.trigger = None

        if self.overlay and hasattr(self.overlay, "xadc"):
            self.xadc = self.overlay.xadc
        elif self.overlay:
            self.xadc = StreamingXADC(self.overlay, default_packet_size=self.packet_size)
        else:
            self.xadc = None
        
        # Build UI Controls & Plotly Canvas
        self._build_ui()
        self._build_plot()

    def _build_ui(self):
        """Construct IPywidgets Control Panel."""
        # 1. Action Buttons
        self.start_btn = widgets.Button(description="Start", button_style="success", icon="play")
        self.stop_btn = widgets.Button(description="Stop", button_style="danger", icon="stop")
        self.force_trig_btn = widgets.Button(description="Force / Arm", button_style="warning", icon="bolt")
        
        self.start_btn.on_click(self._on_start_clicked)
        self.stop_btn.on_click(self._on_stop_clicked)
        self.force_trig_btn.on_click(self._on_force_trig_clicked)

        # 2. AD3 Waveform Controls
        self.shape_dropdown = widgets.Dropdown(
            options=["Sine", "Triangle", "Square"],
            value="Sine",
            description="Waveform:",
            layout=widgets.Layout(width="200px")
        )
        self.freq_slider = widgets.IntSlider(
            value=10000, min=100, max=1000000, step=1,
            description="Freq Slider:", continuous_update=False,
            layout=widgets.Layout(width="350px")
        )
        self.freq_input = widgets.BoundedIntText(
            value=10000, min=100, max=1000000, step=1,
            description="Exact Freq:", layout=widgets.Layout(width="180px")
        )
        widgets.jslink((self.freq_slider, "value"), (self.freq_input, "value"))

        self.amp_slider = widgets.FloatSlider(
            value=1.5, min=0.1, max=1.5, step=0.1,
            description="Amp Slider:", continuous_update=False,
            layout=widgets.Layout(width="280px")
        )
        self.amp_input = widgets.BoundedFloatText(
            value=1.5, min=0.1, max=1.5, step=0.1,
            description="Exact Amp:", layout=widgets.Layout(width="150px")
        )
        widgets.jslink((self.amp_slider, "value"), (self.amp_input, "value"))

        # 3. Hardware Trigger Controls
        self.trig_mode_dropdown = widgets.Dropdown(
            options=["Auto", "Normal", "Single"],
            value="Auto",
            description="Trig Mode:",
            layout=widgets.Layout(width="180px")
        )
        self.trig_edge_dropdown = widgets.Dropdown(
            options=["Rising", "Falling"],
            value="Rising",
            description="Trig Edge:",
            layout=widgets.Layout(width="180px")
        )
        self.trig_level_slider = widgets.FloatSlider(
            value=1.65, min=0.0, max=3.3, step=0.05,
            description="Trig Level:", continuous_update=False,
            layout=widgets.Layout(width="280px")
        )
        self.trig_level_input = widgets.BoundedFloatText(
            value=1.65, min=0.0, max=3.3, step=0.05,
            description="Exact Level:", layout=widgets.Layout(width="150px")
        )
        widgets.jslink((self.trig_level_slider, "value"), (self.trig_level_input, "value"))

        # 4. Display Toggles
        self.autorange_toggle = widgets.ToggleButton(
            value=True, description="Auto-Range",
            button_style="info", tooltip="Activate adaptive time and voltage grid scaling"
        )

        # 5. Digital Vpp Readout
        self.readout = widgets.HTML(
            value="<h3 style='color: #00FFCC; margin: 0; font-family: monospace;'>Live Vpp: 0.00 V</h3>"
        )

    def _build_plot(self):
        """Construct Plotly FigureWidget Canvas with live Trigger Level Line."""
        self.fig = go.FigureWidget()
        
        # Trace 0: Analog Channel Data
        self.fig.add_scatter(
            x=list(range(self.display_window)),
            y=[0] * self.display_window,
            mode="lines",
            line=dict(color="#00FFCC", width=2),
            name="A0 (Analog In)"
        )
        
        # Trace 1: Visual Trigger Threshold Level (Dashed Orange Line)
        self.fig.add_scatter(
            x=[0, self.display_window],
            y=[1.65, 1.65],
            mode="lines",
            line=dict(color="#FFA500", width=1.5, dash="dash"),
            name="Trigger Level"
        )

        self.fig.update_layout(
            title="<b>Real-Time 1 MSPS Hardware-Triggered Oscilloscope</b>",
            xaxis_title="Time (Microseconds / Samples @ 1 MSPS)",
            yaxis_title="Voltage (V)",
            yaxis_range=[0, 3.5],
            template="plotly_dark",
            height=460,
            margin=dict(l=40, r=20, t=50, b=40),
            uirevision="oscilloscope_view_state"
        )

    def _on_start_clicked(self, b):
        self.start()

    def _on_stop_clicked(self, b):
        self.stop()

    def _on_force_trig_clicked(self, b):
        if self.trigger:
            self.trigger.force_trigger()

    def _update_loop(self):
        """Background thread execution loop for acquisition & UI updates."""
        try:
            # Configure initial hardware trigger registers
            if self.trigger:
                self.trigger.configure(
                    mode=self.trig_mode_dropdown.value,
                    edge=self.trig_edge_dropdown.value,
                    threshold_volts=float(self.trig_level_slider.value),
                    timeout_ms=50.0
                )

            # Start AD3 Signal Generation
            self.ad3.start(
                shape=self.shape_dropdown.value,
                frequency=float(self.freq_slider.value),
                amplitude=float(self.amp_slider.value)
            )

            # Wait for AD3 USB hardware readiness
            wait_start = time.time()
            while self._is_running and not self.ad3.is_ready:
                time.sleep(0.05)
                if time.time() - wait_start > 4.0:
                    break

            while self._is_running:
                # 1. Update AD3 parameters dynamically
                self.ad3.update_parameters(
                    shape=self.shape_dropdown.value,
                    frequency=float(self.freq_slider.value),
                    amplitude=float(self.amp_slider.value)
                )

                # 2. Update Hardware Trigger Registers dynamically
                if self.trigger:
                    self.trigger.set_mode(self.trig_mode_dropdown.value)
                    self.trigger.set_edge(self.trig_edge_dropdown.value)
                    self.trigger.set_threshold(float(self.trig_level_slider.value))

                # 3. Capture Hardware Triggered Frame from DMA
                if self.xadc:
                    voltages = self.xadc.capture(crop_startup_samples=0)
                else:
                    # Simulation mode fallback if no FPGA overlay connected
                    t = np.linspace(0, 0.016, self.packet_size)
                    voltages = 1.65 + (self.amp_slider.value / 2.0) * np.sin(2 * np.pi * self.freq_slider.value * t)

                # 4. Adaptive Window Slicing (Sample 0 is ALREADY the hardware trigger point!)
                if self.autorange_toggle.value:
                    freq = float(self.freq_slider.value)
                    period_us = 1e6 / freq if freq > 0 else 1000
                    show_samples = int(5 * period_us)
                    show_samples = max(40, min(show_samples, self.display_window, len(voltages)))
                    plot_voltages = voltages[:show_samples]
                else:
                    max_idx = min(self.display_window, len(voltages))
                    plot_voltages = voltages[:max_idx]

                time_axis = np.arange(len(plot_voltages))
                trig_v = float(self.trig_level_slider.value)

                # 5. Atomic Canvas Batch Update
                with self.fig.batch_update():
                    # Update waveform
                    self.fig.data[0].x = time_axis
                    self.fig.data[0].y = plot_voltages
                    # Update visual trigger line
                    self.fig.data[1].x = [0, len(plot_voltages)]
                    self.fig.data[1].y = [trig_v, trig_v]

                    if self.autorange_toggle.value:
                        amp = float(self.amp_slider.value)
                        margin = 0.25
                        self.fig.layout.xaxis.range = [0, len(plot_voltages)]
                        self.fig.layout.yaxis.range = [max(0.0, 1.65 - (amp + margin)), min(3.5, 1.65 + (amp + margin))]

                # 6. Update Live Vpp
                if len(voltages) > 0:
                    vpp = np.max(voltages) - np.min(voltages)
                    self.readout.value = f"<h3 style='color: #00FFCC; margin: 0; font-family: monospace;'>Live Vpp: {vpp:.2f} V</h3>"

                time.sleep(0.035)

        except Exception as e:
            print(f"[Dashboard] Error in update loop: {e}")
        finally:
            self._is_running = False
            self.ad3.stop()
            print("[Dashboard] Stopped cleanly.")

    def start(self):
        """Launch dashboard execution in background thread."""
        if self._is_running:
            print("[Dashboard] Already running.")
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
        row1 = widgets.HBox([self.start_btn, self.stop_btn, self.force_trig_btn, self.autorange_toggle, self.readout],
                            layout=widgets.Layout(align_items="center", margin="0 0 10px 0"))
        row2 = widgets.HBox([self.shape_dropdown, self.amp_slider, self.amp_input])
        row3 = widgets.HBox([self.freq_slider, self.freq_input])
        row4 = widgets.HBox([self.trig_mode_dropdown, self.trig_edge_dropdown, self.trig_level_slider, self.trig_level_input])
        
        ui_layout = widgets.VBox([row1, row4, row2, row3, self.fig])
        display(ui_layout)