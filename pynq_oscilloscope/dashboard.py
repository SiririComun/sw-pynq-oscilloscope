import time
import threading
from typing import Optional
from IPython.display import clear_output, display
import numpy as np
import ipywidgets as widgets
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pynq_oscilloscope.xadc_dma import StreamingXADC
from pynq_oscilloscope.fft_dma import StreamingFFT
from pynq_oscilloscope.hw_trigger import HardwareTrigger
from pynq_oscilloscope.ad3_wavegen import AD3SignalGenerator

class OscilloscopeDashboard:
    """
    High-Performance Multi-Tab Oscilloscope & Spectrum Analyzer Dashboard.
    Integrates FPGA Hardware Triggering, 1 MSPS DMA acquisition, 2048-pt PL FFT, and AD3 wavegen.
    """

    def __init__(
        self,
        overlay=None,
        packet_size: int = 2048,
        fft_points: int = 2048,
        display_window: int = 1024
    ):
        self.overlay = overlay
        self.packet_size = packet_size
        self.fft_points = fft_points
        self.display_window = display_window
        
        self._is_running = False
        self._single_done = False
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
        self._setup_callbacks()

    def _build_ui(self):
        # 1. Action Buttons
        self.start_btn = widgets.Button(description="Start", button_style="success", icon="play")
        self.stop_btn = widgets.Button(description="Stop", button_style="danger", icon="stop")
        self.force_btn = widgets.Button(description="Force / Arm", button_style="warning", icon="bolt")
        self.clear_log_btn = widgets.Button(description="Clear Log", button_style="", icon="trash")
        self.autorange_toggle = widgets.ToggleButton(value=True, description="Auto-Range", button_style="info")

        self.readout_vpp = widgets.HTML("<span style='color:#00FFCC; font-family:monospace; font-size:15px; font-weight:bold;'>Live Vpp: 0.00 V</span>")
        self.readout_f0 = widgets.HTML("<span style='color:#FF007F; font-family:monospace; font-size:15px; font-weight:bold;'>Peak f0: 0.0 kHz</span>")

        # 2. AD3 Waveform Controls
        self.shape_dd = widgets.Dropdown(options=["Sine", "Triangle", "Square"], value="Sine", description="Waveform:", layout=widgets.Layout(width="200px"))
        self.freq_slider = widgets.IntSlider(value=10000, min=100, max=250000, step=100, description="Freq Slider:", continuous_update=False, layout=widgets.Layout(width="350px"))
        self.freq_input = widgets.BoundedIntText(value=10000, min=100, max=250000, step=100, description="Exact Freq:", layout=widgets.Layout(width="180px"))
        widgets.jslink((self.freq_slider, "value"), (self.freq_input, "value"))

        self.amp_slider = widgets.FloatSlider(value=1.5, min=0.1, max=1.5, step=0.1, description="Amp Slider:", continuous_update=False, layout=widgets.Layout(width="280px"))
        self.amp_input = widgets.BoundedFloatText(value=1.5, min=0.1, max=1.5, step=0.1, description="Exact Amp:", layout=widgets.Layout(width="150px"))
        widgets.jslink((self.amp_slider, "value"), (self.amp_input, "value"))

        # 3. Trigger Controls
        self.trig_mode_dd = widgets.Dropdown(options=["Auto", "Normal", "Single"], value="Auto", description="Trig Mode:", layout=widgets.Layout(width="180px"))
        self.trig_edge_dd = widgets.Dropdown(options=["Rising", "Falling"], value="Rising", description="Trig Edge:", layout=widgets.Layout(width="180px"))
        self.trig_level_slider = widgets.FloatSlider(value=1.65, min=0.0, max=3.3, step=0.05, description="Trig Level:", continuous_update=False, layout=widgets.Layout(width="280px"))
        self.trig_level_input = widgets.BoundedFloatText(value=1.65, min=0.0, max=3.3, step=0.05, description="Exact Level:", layout=widgets.Layout(width="150px"))
        widgets.jslink((self.trig_level_slider, "value"), (self.trig_level_input, "value"))

        # 4. FFT Controls
        self.fft_unit_dd = widgets.Dropdown(options=["dBV", "dBFS", "Linear"], value="dBV", description="FFT Unit:", layout=widgets.Layout(width="170px"))
        self.fft_span_dd = widgets.Dropdown(options=[("Full (500 kHz)", 500000), ("100 kHz", 100000), ("20 kHz", 20000)], value=100000, description="Span / Zoom:", layout=widgets.Layout(width="210px"))

    def _build_plots(self):
        # 1. Scope Figure (Tab 1)
        self.fig_scope = go.FigureWidget()
        self.fig_scope.add_scatter(x=list(range(500)), y=[1.65]*500, mode="lines", line=dict(color="#00FFCC", width=2), name="A0 (Time)")
        self.fig_scope.add_scatter(x=[0, 500], y=[1.65, 1.65], mode="lines", line=dict(color="#FFA500", width=1.5, dash="dash"), name="Threshold")
        self.fig_scope.update_layout(title="<b>Real-Time 1 MSPS Oscilloscope</b>", template="plotly_dark", height=420, margin=dict(l=40,r=20,t=45,b=35), uirevision="scope")
        self.fig_scope.update_yaxes(range=[0, 3.5], title="Voltage (V)")
        self.fig_scope.update_xaxes(title="Time (µs @ 1 MSPS)")

        # 2. Spectrum Figure (Tab 2)
        initial_freqs = self.fft.freq_axis if self.fft else np.linspace(0, 500000, 1024)
        self.fig_spectrum = go.FigureWidget()
        self.fig_spectrum.add_scatter(x=initial_freqs, y=[-100]*len(initial_freqs), mode="lines", line=dict(color="#FF007F", width=1.8), name="PL FFT Spectrum")
        self.fig_spectrum.add_scatter(x=[10000], y=[-40], mode="markers+text", marker=dict(color="#00FFCC", size=8, symbol="diamond"), text=["Peak"], textposition="top center", name="Peak")
        self.fig_spectrum.update_layout(title="<b>Real-Time PL-Accelerated Spectrum Analyzer (2048-pt FFT)</b>", template="plotly_dark", height=420, margin=dict(l=40,r=20,t=45,b=35), uirevision="spectrum")
        self.fig_spectrum.update_yaxes(range=[-110, 0], title="Magnitude (dBV)")
        self.fig_spectrum.update_xaxes(range=[0, 100000], title="Frequency (Hz)")

        # 3. Dual View Figure (Tab 3)
        self.fig_dual = make_subplots(rows=2, cols=1, vertical_spacing=0.15, subplot_titles=("<b>Oscilloscope (Time Domain)</b>", "<b>Spectrum Analyzer (Frequency Domain)</b>"))
        self.fig_dual = go.FigureWidget(self.fig_dual)
        self.fig_dual.add_scatter(x=list(range(500)), y=[1.65]*500, mode="lines", line=dict(color="#00FFCC", width=1.8), row=1, col=1)
        self.fig_dual.add_scatter(x=[0, 500], y=[1.65, 1.65], mode="lines", line=dict(color="#FFA500", width=1.2, dash="dash"), row=1, col=1)
        self.fig_dual.add_scatter(x=initial_freqs, y=[-100]*len(initial_freqs), mode="lines", line=dict(color="#FF007F", width=1.6), row=2, col=1)
        self.fig_dual.update_layout(template="plotly_dark", height=530, showlegend=False, margin=dict(l=40,r=20,t=45,b=35), uirevision="dual")
        self.fig_dual.update_yaxes(range=[0, 3.5], title="Voltage (V)", row=1, col=1)
        self.fig_dual.update_yaxes(range=[-110, 0], title="Mag (dBV)", row=2, col=1)
        self.fig_dual.update_xaxes(title="Time (µs)", row=1, col=1)
        self.fig_dual.update_xaxes(range=[0, 100000], title="Frequency (Hz)", row=2, col=1)

        # Tab Container
        self.tabs = widgets.Tab(children=[self.fig_scope, self.fig_spectrum, self.fig_dual])
        self.tabs.set_title(0, "📈 Oscilloscope")
        self.tabs.set_title(1, "📊 Spectrum Analyzer")
        self.tabs.set_title(2, "🔀 Dual View")

    def _setup_callbacks(self):
        self.start_btn.on_click(self._on_start_clicked)
        self.stop_btn.on_click(self._on_stop_clicked)
        self.force_btn.on_click(self._on_force_clicked)
        self.clear_log_btn.on_click(self._on_clear_log_clicked)

        self.shape_dd.observe(lambda _: self._update_wavegen_params(), names="value")
        self.freq_slider.observe(lambda _: self._update_wavegen_params(), names="value")
        self.amp_slider.observe(lambda _: self._update_wavegen_params(), names="value")
        self.trig_level_slider.observe(lambda _: self._update_hardware_threshold(), names="value")
        self.trig_mode_dd.observe(self._on_mode_or_edge_change, names="value")
        self.trig_edge_dd.observe(self._on_mode_or_edge_change, names="value")

    def _update_wavegen_params(self):
        self.ad3.update_parameters(shape=self.shape_dd.value, frequency=float(self.freq_slider.value), amplitude=float(self.amp_slider.value))

    def _update_hardware_threshold(self):
        if self.trigger:
            self.trigger.set_threshold(float(self.trig_level_slider.value))

    def _get_arm_control_word(self):
        is_falling = (self.trig_edge_dd.value == "Falling")
        mode = self.trig_mode_dd.value
        ctrl = (1 << 0) | (1 << 3)  # Bit 0: ARM, Bit 3: SINGLE SHOT
        if is_falling:
            ctrl |= (1 << 2)        # Bit 2: FALLING EDGE
        if mode == "Auto":
            ctrl |= (1 << 1)        # Bit 1: AUTO TIMEOUT ENABLED
        return ctrl

    def _on_mode_or_edge_change(self, _):
        if self.trig_mode_dd.value != "Single":
            self._single_done = False
        if self._is_running and self.trigger:
            self.trigger.mmio.write(0x00, self._get_arm_control_word())

    def _on_start_clicked(self, _):
        self.start()

    def _on_stop_clicked(self, _):
        self.stop()

    def _on_force_clicked(self, _):
        if self.trig_mode_dd.value == "Single":
            self._single_done = False
        if self.trigger:
            self.trigger.force_trigger()

    def _on_clear_log_clicked(self, _):
        clear_output(wait=True)
        display(widgets.VBox([self.control_panel, self.tabs]))

    def _update_loop(self):
        """Universal Arm-on-Demand Acquisition Loop."""
        dma_armed = False
        try:
            # Configure 50ms Timeout and Threshold
            if self.trigger:
                self.trigger.mmio.write(0x0C, 5000000)
                self._update_hardware_threshold()

            # Start AD3 Wavegen
            self.ad3.start(shape=self.shape_dd.value, frequency=float(self.freq_slider.value), amplitude=float(self.amp_slider.value), offset=1.65)
            wait_start = time.time()
            while self._is_running and not self.ad3.is_ready:
                time.sleep(0.05)
                if time.time() - wait_start > 3.0:
                    break

            print(f"[Dashboard] Acquisition active | Mode: {self.trig_mode_dd.value} | Signal: {self.shape_dd.value} @ {self.freq_slider.value} Hz")

            while self._is_running:
                mode = self.trig_mode_dd.value

                # Single-Shot hold
                if mode == "Single" and self._single_done:
                    time.sleep(0.02)
                    continue

                # 1. Arm both DMAs exactly ONCE per frame
                if not dma_armed:
                    self.xadc.dma.recvchannel.transfer(self.xadc._buffer)
                    self.fft.dma.recvchannel.transfer(self.fft._buffer)
                    self.trigger.mmio.write(0x00, self._get_arm_control_word())
                    dma_armed = True

                # 2. Check if hardware frame completed
                if self.xadc.dma.recvchannel.idle and self.fft.dma.recvchannel.idle:
                    dma_armed = False
                    
                    # Process Time Data
                    raw_time = np.array(self.xadc._buffer)
                    voltages = (raw_time >> 4) * (3.3 / 4095.0)

                    # Process FFT Data
                    raw_fft = np.array(self.fft._buffer, copy=True)[:1024].astype(np.float64)
                    unit_mode = self.fft_unit_dd.value
                    if unit_mode == "Linear":
                        mags = (raw_fft / 2048.0) * (3.3 / 4095.0) * 1000.0
                    elif unit_mode == "dBFS":
                        mags = 20.0 * np.log10(np.maximum(raw_fft, 1.0) / 65535.0)
                    else:
                        linear_v = (raw_fft / 2048.0) * (3.3 / 4095.0)
                        mags = 20.0 * np.log10(np.maximum(linear_v, 1e-6))

                    # Peak tracking
                    peak_idx = np.argmax(mags[10:]) + 10
                    peak_f = self.fft.freq_axis[peak_idx]
                    peak_m = mags[peak_idx]

                    # Auto-Range Slicing
                    vpp = float(np.max(voltages) - np.min(voltages))
                    freq_val = float(self.freq_slider.value)
                    
                    if self.autorange_toggle.value:
                        period_us = 1e6 / freq_val if freq_val > 0 else 1000
                        show_pts = int(5 * period_us) if vpp > 0.1 else 500
                        show_pts = max(40, min(show_pts, 1024, len(voltages)))
                    else:
                        show_pts = 500

                    plot_v = voltages[:show_pts]
                    time_x = np.arange(len(plot_v))
                    trig_v = float(self.trig_level_slider.value)
                    active_tab = self.tabs.selected_index
                    max_span = float(self.fft_span_dd.value)

                    # Update Active Tab
                    if active_tab == 0:
                        with self.fig_scope.batch_update():
                            self.fig_scope.data[0].x = time_x
                            self.fig_scope.data[0].y = plot_v
                            self.fig_scope.data[1].x = [0, len(plot_v)]
                            self.fig_scope.data[1].y = [trig_v, trig_v]
                            if self.autorange_toggle.value and vpp > 0.1:
                                amp = float(self.amp_slider.value)
                                self.fig_scope.layout.xaxis.range = [0, len(plot_v)]
                                self.fig_scope.layout.yaxis.range = [max(0.0, 1.65 - (amp + 0.25)), min(3.5, 1.65 + (amp + 0.25))]
                            elif self.autorange_toggle.value:
                                self.fig_scope.layout.yaxis.range = [0, 3.5]

                    elif active_tab == 1:
                        with self.fig_spectrum.batch_update():
                            self.fig_spectrum.data[0].x = self.fft.freq_axis
                            self.fig_spectrum.data[0].y = mags
                            self.fig_spectrum.data[1].x = [peak_f]
                            self.fig_spectrum.data[1].y = [peak_m]
                            self.fig_spectrum.data[1].text = [f" {peak_f/1e3:.1f} kHz ({peak_m:.1f} {unit_mode})"]
                            self.fig_spectrum.layout.xaxis.range = [0, max_span]
                            self.fig_spectrum.layout.yaxis.title = f"Magnitude ({unit_mode})"

                    elif active_tab == 2:
                        with self.fig_dual.batch_update():
                            self.fig_dual.data[0].x = time_x
                            self.fig_dual.data[0].y = plot_v
                            self.fig_dual.data[1].x = [0, len(plot_v)]
                            self.fig_dual.data[1].y = [trig_v, trig_v]
                            self.fig_dual.data[2].x = self.fft.freq_axis
                            self.fig_dual.data[2].y = mags
                            self.fig_dual.layout.xaxis2.range = [0, max_span]

                    # Digital Readouts
                    mode_tag = " <span style='color:#FFA500;'>(LOCKED)</span>" if mode == "Single" else ""
                    self.readout_vpp.value = f"<span style='color:#00FFCC; font-family:monospace; font-size:15px; font-weight:bold;'>Live Vpp: {vpp:.2f} V{mode_tag}</span>"
                    self.readout_f0.value = f"<span style='color:#FF007F; font-family:monospace; font-size:15px; font-weight:bold;'>Peak f0: {peak_f/1e3:.2f} kHz</span>"

                    if mode == "Single":
                        self._single_done = True

                    time.sleep(0.033)

                else:
                    time.sleep(0.01)

        except Exception as e:
            print(f"[Dashboard] Error: {e}")
        finally:
            self._is_running = False
            if dma_armed:
                try:
                    self.xadc.dma.mmio.write(0x30, 0x04)
                    self.fft.dma.mmio.write(0x30, 0x04)
                except Exception:
                    pass
            self.ad3.stop()
            print("[Dashboard] Stopped cleanly.")

    def start(self):
        if self._is_running:
            return
        
        print("[Dashboard] Starting acquisition...")
        self._single_done = False
        self.ad3.stop()
        try:
            self.xadc.dma.mmio.write(0x30, 0x04)
            self.fft.dma.mmio.write(0x30, 0x04)
            time.sleep(0.01)
            self.xadc.dma.recvchannel.start()
            self.fft.dma.recvchannel.start()
        except Exception:
            pass

        self._is_running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._is_running = False
        self._single_done = False

    def display(self):
        ctrl_row1 = widgets.HBox([self.start_btn, self.stop_btn, self.force_btn, self.clear_log_btn, self.autorange_toggle, self.readout_vpp, self.readout_f0], layout=widgets.Layout(align_items="center", gap="10px", margin="0 0 10px 0"))
        ctrl_row2 = widgets.HBox([self.trig_mode_dd, self.trig_edge_dd, self.trig_level_slider, self.trig_level_input])
        ctrl_row3 = widgets.HBox([self.shape_dd, self.amp_slider, self.amp_input])
        ctrl_row4 = widgets.HBox([self.freq_slider, self.freq_input])
        ctrl_row5 = widgets.HBox([self.fft_unit_dd, self.fft_span_dd])

        self.control_panel = widgets.VBox([ctrl_row1, ctrl_row2, ctrl_row3, ctrl_row4, ctrl_row5], layout=widgets.Layout(margin="0 0 15px 0"))
        display(widgets.VBox([self.control_panel, self.tabs]))