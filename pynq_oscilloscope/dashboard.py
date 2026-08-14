import time
import threading
from typing import Optional
import numpy as np
import ipywidgets as widgets
from IPython.display import display
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pynq_oscilloscope.xadc_dma import StreamingXADC
from pynq_oscilloscope.fft_dma import StreamingFFT
from pynq_oscilloscope.hw_trigger import HardwareTrigger
from pynq_oscilloscope.ad3_wavegen import AD3SignalGenerator

class OscilloscopeDashboard:
    """
    Unified, interactive real-time Plotly + IPywidgets Oscilloscope & Spectrum Analyzer Dashboard.
    Integrates FPGA Hardware Triggering, 1 MSPS DMA acquisition, 2048-pt PL FFT, and AD3 wavegen.
    """

    def __init__(
        self,
        overlay=None,
        packet_size: int = 16384,
        fft_points: int = 2048,
        display_window: int = 1024
    ):
        self.overlay = overlay
        self.packet_size = packet_size
        self.fft_points = fft_points
        self.display_window = display_window
        
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        
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

        if self.overlay and hasattr(self.overlay, "fft"):
            self.fft = self.overlay.fft
        elif self.overlay:
            self.fft = StreamingFFT(self.overlay, fft_points=self.fft_points)
        else:
            self.fft = None
        
        # Build UI Controls & Plotly Canvases
        self._build_ui()
        self._build_plots()

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
            value=10000, min=100, max=500000, step=100,
            description="Freq Slider:", continuous_update=False,
            layout=widgets.Layout(width="350px")
        )
        self.freq_input = widgets.BoundedIntText(
            value=10000, min=100, max=500000, step=100,
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

        # 4. FFT / Spectrum Analyzer Controls
        self.fft_unit_dropdown = widgets.Dropdown(
            options=["dBV", "dBFS", "Linear"],
            value="dBV",
            description="FFT Unit:",
            layout=widgets.Layout(width="170px")
        )
        self.fft_span_dropdown = widgets.Dropdown(
            options=[("Full (500 kHz)", 500000), ("100 kHz", 100000), ("20 kHz (Audio)", 20000)],
            value=500000,
            description="Span / Zoom:",
            layout=widgets.Layout(width="210px")
        )

        # 5. Display Toggles & Readouts
        self.autorange_toggle = widgets.ToggleButton(
            value=True, description="Auto-Range",
            button_style="info", tooltip="Activate adaptive time and voltage grid scaling"
        )
        self.readout_vpp = widgets.HTML(
            value="<span style='color: #00FFCC; font-family: monospace; font-size: 15px; font-weight: bold;'>Live Vpp: 0.00 V</span>"
        )
        self.readout_peak_freq = widgets.HTML(
            value="<span style='color: #FF007F; font-family: monospace; font-size: 15px; font-weight: bold;'>Peak f0: 0.0 kHz</span>"
        )

    def _build_plots(self):
        """Construct Plotly FigureWidgets for Tab 1 (Scope), Tab 2 (Spectrum), and Tab 3 (Dual)."""
        # ==========================================
        # 1. Scope Figure
        # ==========================================
        self.fig_scope = go.FigureWidget()
        self.fig_scope.add_scatter(
            x=list(range(self.display_window)),
            y=[0] * self.display_window,
            mode="lines",
            line=dict(color="#00FFCC", width=2),
            name="A0 (Time Domain)"
        )
        self.fig_scope.add_scatter(
            x=[0, self.display_window],
            y=[1.65, 1.65],
            mode="lines",
            line=dict(color="#FFA500", width=1.5, dash="dash"),
            name="Trigger Threshold"
        )
        self.fig_scope.update_layout(
            title="<b>Real-Time 1 MSPS Hardware-Triggered Oscilloscope</b>",
            xaxis_title="Time (Microseconds / Samples @ 1 MSPS)",
            yaxis_title="Voltage (V)",
            yaxis_range=[0, 3.5],
            template="plotly_dark",
            height=430,
            margin=dict(l=40, r=20, t=50, b=40),
            uirevision="scope_state"
        )

        # ==========================================
        # 2. Spectrum Analyzer Figure
        # ==========================================
        num_bins = self.fft_points // 2
        initial_freqs = np.linspace(0, 500000, num_bins)
        self.fig_spectrum = go.FigureWidget()
        self.fig_spectrum.add_scatter(
            x=initial_freqs,
            y=[-100] * num_bins,
            mode="lines",
            line=dict(color="#FF007F", width=1.8),
            name="PL FFT Spectrum"
        )
        self.fig_spectrum.add_scatter(
            x=[10000],
            y=[0],
            mode="markers+text",
            marker=dict(color="#00FFCC", size=9, symbol="diamond"),
            text=["Peak f0"],
            textposition="top center",
            name="Peak Marker"
        )
        self.fig_spectrum.update_layout(
            title="<b>Real-Time PL-Accelerated Spectrum Analyzer (2048-pt FFT)</b>",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Magnitude (dBV)",
            xaxis_range=[0, 500000],
            yaxis_range=[-80, 20],
            template="plotly_dark",
            height=430,
            margin=dict(l=40, r=20, t=50, b=40),
            uirevision="spectrum_state"
        )

        # ==========================================
        # 3. Dual View Figure (Stacked)
        # ==========================================
        self.fig_dual = make_subplots(
            rows=2, cols=1,
            shared_xaxes=False,
            vertical_spacing=0.15,
            subplot_titles=("<b>Oscilloscope (Time Domain)</b>", "<b>Spectrum Analyzer (Frequency Domain)</b>")
        )
        self.fig_dual = go.FigureWidget(self.fig_dual)
        
        # Dual Scope trace
        self.fig_dual.add_scatter(
            x=list(range(self.display_window)),
            y=[0] * self.display_window,
            mode="lines",
            line=dict(color="#00FFCC", width=1.8),
            name="Time Signal",
            row=1, col=1
        )
        # Dual Spectrum trace
        self.fig_dual.add_scatter(
            x=initial_freqs,
            y=[-100] * num_bins,
            mode="lines",
            line=dict(color="#FF007F", width=1.6),
            name="Spectrum",
            row=2, col=1
        )
        self.fig_dual.update_layout(
            template="plotly_dark",
            height=540,
            showlegend=False,
            margin=dict(l=40, r=20, t=50, b=40),
            uirevision="dual_state"
        )
        self.fig_dual.update_yaxes(title_text="Voltage (V)", range=[0, 3.5], row=1, col=1)
        self.fig_dual.update_yaxes(title_text="Mag (dBV)", range=[-80, 20], row=2, col=1)
        self.fig_dual.update_xaxes(title_text="Time (µs)", row=1, col=1)
        self.fig_dual.update_xaxes(title_text="Frequency (Hz)", range=[0, 500000], row=2, col=1)

    def _on_start_clicked(self, b):
        self.start()

    def _on_stop_clicked(self, b):
        self.stop()

    def _on_force_trig_clicked(self, b):
        if self.trigger:
            self.trigger.force_trigger()

    def _update_loop(self):
        """Background thread execution loop for concurrent Time & FFT acquisition."""
        try:
            # 1. Initialize Hardware Trigger Registers
            if self.trigger:
                self.trigger.configure(
                    mode=self.trig_mode_dropdown.value,
                    edge=self.trig_edge_dropdown.value,
                    threshold_volts=float(self.trig_level_slider.value),
                    timeout_ms=50.0
                )

            # 2. Start AD3 Signal Generation
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
                # Dynamic updates to wavegen and trigger
                self.ad3.update_parameters(
                    shape=self.shape_dropdown.value,
                    frequency=float(self.freq_slider.value),
                    amplitude=float(self.amp_slider.value)
                )

                if self.trigger:
                    self.trigger.set_mode(self.trig_mode_dropdown.value)
                    self.trigger.set_edge(self.trig_edge_dropdown.value)
                    self.trigger.set_threshold(float(self.trig_level_slider.value))

                # 3. Synchronized Dual DMA Capture (Prevents Broadcaster Deadlock)
                unit_name = self.fft_unit_dropdown.value
                if self.overlay and hasattr(self.overlay, "capture_both"):
                    voltages, freqs, mags = self.overlay.capture_both(unit=unit_name)
                elif self.xadc and self.fft:
                    self.xadc.dma.recvchannel.transfer(self.xadc._buffer)
                    self.fft.dma.recvchannel.transfer(self.fft._buffer)
                    self.xadc.dma.recvchannel.wait()
                    self.fft.dma.recvchannel.wait()
                    raw_time = np.array(self.xadc._buffer)
                    voltages = (raw_time >> 4) * (3.3 / 4095.0)
                    freqs, mags = self.fft.capture_spectrum(unit=unit_name)
                else:
                    # Simulation fallback
                    t = np.linspace(0, 0.016, self.packet_size)
                    voltages = 1.65 + (self.amp_slider.value / 2.0) * np.sin(2 * np.pi * self.freq_slider.value * t)
                    freqs = np.linspace(0, 500000, 1024)
                    mags = -60.0 + 40.0 * np.exp(-((freqs - self.freq_slider.value) / 1500.0) ** 2)

                # Peak frequency tracking
                peak_f, peak_m = StreamingFFT.get_peak_frequency(freqs, mags, min_freq_hz=500.0)

                # 4. Time Domain Window Slicing
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
                max_freq_span = float(self.fft_span_dropdown.value)

                # 5. Atomic Canvas Batch Updates
                # Update Tab 1: Scope
                with self.fig_scope.batch_update():
                    self.fig_scope.data[0].x = time_axis
                    self.fig_scope.data[0].y = plot_voltages
                    self.fig_scope.data[1].x = [0, len(plot_voltages)]
                    self.fig_scope.data[1].y = [trig_v, trig_v]
                    if self.autorange_toggle.value:
                        amp = float(self.amp_slider.value)
                        margin = 0.25
                        self.fig_scope.layout.xaxis.range = [0, len(plot_voltages)]
                        self.fig_scope.layout.yaxis.range = [max(0.0, 1.65 - (amp + margin)), min(3.5, 1.65 + (amp + margin))]

                # Update Tab 2: Spectrum Analyzer
                with self.fig_spectrum.batch_update():
                    self.fig_spectrum.data[0].x = freqs
                    self.fig_spectrum.data[0].y = mags
                    self.fig_spectrum.data[1].x = [peak_f]
                    self.fig_spectrum.data[1].y = [peak_m]
                    self.fig_spectrum.data[1].text = [f" {peak_f/1e3:.1f} kHz ({peak_m:.1f} {unit_name})"]
                    self.fig_spectrum.layout.xaxis.range = [0, max_freq_span]
                    self.fig_spectrum.layout.yaxis.title = f"Magnitude ({unit_name})"
                    if unit_name == "Linear":
                        self.fig_spectrum.layout.yaxis.range = [0, 3.5]
                    else:
                        self.fig_spectrum.layout.yaxis.range = [-80, 20]

                # Update Tab 3: Dual View
                with self.fig_dual.batch_update():
                    self.fig_dual.data[0].x = time_axis
                    self.fig_dual.data[0].y = plot_voltages
                    self.fig_dual.data[1].x = freqs
                    self.fig_dual.data[1].y = mags
                    self.fig_dual.layout.xaxis2.range = [0, max_freq_span]
                    self.fig_dual.layout.yaxis2.title = f"Mag ({unit_name})"

                # 6. Update Live Readouts
                if len(voltages) > 0:
                    vpp = np.max(voltages) - np.min(voltages)
                    self.readout_vpp.value = f"<span style='color: #00FFCC; font-family: monospace; font-size: 15px; font-weight: bold;'>Live Vpp: {vpp:.2f} V</span>"
                
                self.readout_peak_freq.value = f"<span style='color: #FF007F; font-family: monospace; font-size: 15px; font-weight: bold;'>Peak f0: {peak_f/1e3:.2f} kHz</span>"

                time.sleep(0.035)

        except Exception as e:
            print(f"[Dashboard] Error in update loop: {e}")
        finally:
            self._is_running = False
            self.ad3.stop()
            print("[Dashboard] Stopped cleanly.")

    def start(self):
        """Launch dashboard acquisition in background thread."""
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
        """Render the complete Multi-Tab UI Dashboard in Jupyter."""
        # Action & Status Row
        row1 = widgets.HBox(
            [self.start_btn, self.stop_btn, self.force_trig_btn, self.autorange_toggle, self.readout_vpp, self.readout_peak_freq],
            layout=widgets.Layout(align_items="center", gap="15px", margin="0 0 10px 0")
        )
        # Hardware Trigger Controls
        row_trig = widgets.HBox([self.trig_mode_dropdown, self.trig_edge_dropdown, self.trig_level_slider, self.trig_level_input])
        # AD3 Signal Generator Controls
        row_wave = widgets.HBox([self.shape_dropdown, self.amp_slider, self.amp_input])
        row_freq = widgets.HBox([self.freq_slider, self.freq_input])
        # FFT Controls
        row_fft = widgets.HBox([self.fft_unit_dropdown, self.fft_span_dropdown])

        control_panel = widgets.VBox([row1, row_trig, row_wave, row_freq, row_fft], layout=widgets.Layout(margin="0 0 15px 0"))

        # Build Multi-Tab Layout
        tabs = widgets.Tab()
        tabs.children = [self.fig_scope, self.fig_spectrum, self.fig_dual]
        tabs.set_title(0, "📈 Oscilloscope")
        tabs.set_title(1, "📊 Spectrum Analyzer")
        tabs.set_title(2, "🔀 Dual View")

        ui_layout = widgets.VBox([control_panel, tabs])
        display(ui_layout)