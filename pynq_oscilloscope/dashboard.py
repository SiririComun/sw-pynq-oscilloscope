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
    High-Performance Multi-Tab Dual-Channel Oscilloscope & Spectrum Analyzer Dashboard.
    Integrates FPGA Hardware Triggering, 1 MSPS Simultaneous Dual DMA, 2048-pt PL FFT, and Dual AD3 wavegen.
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
        
        # Attach sub-drivers from overlay
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
        # 1. Action Buttons & Status Readouts
        self.start_btn = widgets.Button(description="Start", button_style="success", icon="play", layout=widgets.Layout(width="100px"))
        self.stop_btn = widgets.Button(description="Stop", button_style="danger", icon="stop", layout=widgets.Layout(width="100px"))
        self.force_btn = widgets.Button(description="Force / Arm", button_style="warning", icon="bolt", layout=widgets.Layout(width="120px"))
        self.clear_log_btn = widgets.Button(description="Clear Log", button_style="", icon="trash", layout=widgets.Layout(width="100px"))
        self.autorange_toggle = widgets.ToggleButton(value=True, description="Auto-Range", button_style="info", layout=widgets.Layout(width="110px"))

        self.readout_vpp1 = widgets.HTML("<span style='color:#00FFCC; font-family:monospace; font-size:14px; font-weight:bold;'>CH1 Vpp: 0.00 V</span>")
        self.readout_vpp2 = widgets.HTML("<span style='color:#FF007F; font-family:monospace; font-size:14px; font-weight:bold;'>CH2 Vpp: 0.00 V</span>")
        self.readout_f0 = widgets.HTML("<span style='color:#FFA500; font-family:monospace; font-size:14px; font-weight:bold;'>Peak f0: 0.0 kHz</span>")

        # 2. Hardware Trigger Controls
        self.trig_mode_dd = widgets.Dropdown(options=["Auto", "Normal", "Single"], value="Auto", description="Trig Mode:", layout=widgets.Layout(width="180px"))
        self.trig_edge_dd = widgets.Dropdown(options=["Rising", "Falling"], value="Rising", description="Trig Edge:", layout=widgets.Layout(width="180px"))
        self.trig_level_slider = widgets.FloatSlider(value=1.65, min=0.0, max=3.3, step=0.05, description="Trig Level:", continuous_update=False, layout=widgets.Layout(width="280px"))
        self.trig_level_input = widgets.BoundedFloatText(value=1.65, min=0.0, max=3.3, step=0.05, description="Exact Level:", layout=widgets.Layout(width="150px"))
        widgets.jslink((self.trig_level_slider, "value"), (self.trig_level_input, "value"))

        # 3. Channel 1 Wavegen Controls (W1 -> A0)
        self.ch1_shape_dd = widgets.Dropdown(options=["Sine", "Triangle", "Square"], value="Sine", description="CH1 Shape:", layout=widgets.Layout(width="190px"))
        self.ch1_freq_slider = widgets.IntSlider(value=1000, min=50, max=250000, step=50, description="CH1 Freq:", continuous_update=False, layout=widgets.Layout(width="320px"))
        self.ch1_freq_input = widgets.BoundedIntText(value=1000, min=50, max=250000, step=50, layout=widgets.Layout(width="120px"))
        widgets.jslink((self.ch1_freq_slider, "value"), (self.ch1_freq_input, "value"))
        self.ch1_amp_slider = widgets.FloatSlider(value=1.0, min=0.1, max=1.5, step=0.1, description="CH1 Amp:", continuous_update=False, layout=widgets.Layout(width="260px"))

        # 4. Channel 2 Wavegen Controls (W2 -> A1)
        self.ch2_shape_dd = widgets.Dropdown(options=["Sine", "Triangle", "Square"], value="Square", description="CH2 Shape:", layout=widgets.Layout(width="190px"))
        self.ch2_freq_slider = widgets.IntSlider(value=5000, min=50, max=250000, step=50, description="CH2 Freq:", continuous_update=False, layout=widgets.Layout(width="320px"))
        self.ch2_freq_input = widgets.BoundedIntText(value=5000, min=50, max=250000, step=50, layout=widgets.Layout(width="120px"))
        widgets.jslink((self.ch2_freq_slider, "value"), (self.ch2_freq_input, "value"))
        self.ch2_amp_slider = widgets.FloatSlider(value=1.0, min=0.1, max=1.5, step=0.1, description="CH2 Amp:", continuous_update=False, layout=widgets.Layout(width="260px"))
        self.ch2_enable_chk = widgets.Checkbox(value=True, description="Enable CH2 (W2)", layout=widgets.Layout(width="160px"))

        # 5. FFT Controls
        self.fft_unit_dd = widgets.Dropdown(options=["dBV", "dBFS", "Linear"], value="dBV", description="FFT Unit:", layout=widgets.Layout(width="170px"))
        self.fft_span_dd = widgets.Dropdown(options=[("Full (500 kHz)", 500000), ("100 kHz", 100000), ("20 kHz", 20000)], value=100000, description="Span / Zoom:", layout=widgets.Layout(width="210px"))

    def _build_plots(self):
        # 1. Dual-Channel Scope Figure (Tab 1)
        self.fig_scope = go.FigureWidget()
        self.fig_scope.add_scatter(x=list(range(500)), y=[1.65]*500, mode="lines", line=dict(color="#00FFCC", width=2), name="CH1: A0 (Sine)")
        self.fig_scope.add_scatter(x=list(range(500)), y=[1.65]*500, mode="lines", line=dict(color="#FF007F", width=2), name="CH2: A1 (Square)")
        self.fig_scope.add_scatter(x=[0, 500], y=[1.65, 1.65], mode="lines", line=dict(color="#FFA500", width=1.5, dash="dash"), name="Trigger (1.65V)")
        self.fig_scope.update_layout(title="<b>Real-Time Dual-Channel 1 MSPS Oscilloscope</b>", template="plotly_dark", height=430, margin=dict(l=40,r=20,t=45,b=35), uirevision="scope")
        self.fig_scope.update_yaxes(range=[0, 3.3], title="Voltage (V)")
        self.fig_scope.update_xaxes(title="Time (µs @ 1 MSPS)")

        # 2. Spectrum Figure (Tab 2)
        initial_freqs = self.fft.freq_axis if self.fft else np.linspace(0, 500000, 1024)
        self.fig_spectrum = go.FigureWidget()
        self.fig_spectrum.add_scatter(x=initial_freqs, y=[-100]*len(initial_freqs), mode="lines", line=dict(color="#00FFCC", width=1.8), name="CH1 FFT Spectrum")
        self.fig_spectrum.add_scatter(x=[1000], y=[-40], mode="markers+text", marker=dict(color="#FFA500", size=8, symbol="diamond"), text=["Peak"], textposition="top center", name="Peak")
        self.fig_spectrum.update_layout(title="<b>Real-Time PL-Accelerated Spectrum Analyzer (2048-pt FFT)</b>", template="plotly_dark", height=430, margin=dict(l=40,r=20,t=45,b=35), uirevision="spectrum")
        self.fig_spectrum.update_yaxes(range=[-110, 0], title="Magnitude (dBV)")
        self.fig_spectrum.update_xaxes(range=[0, 100000], title="Frequency (Hz)")

        # 3. Dual View Figure (Tab 3)
        self.fig_dual = make_subplots(rows=2, cols=1, vertical_spacing=0.15, subplot_titles=("<b>Oscilloscope (Dual-Channel Time Domain)</b>", "<b>Spectrum Analyzer (CH1 Frequency Domain)</b>"))
        self.fig_dual = go.FigureWidget(self.fig_dual)
        self.fig_dual.add_scatter(x=list(range(500)), y=[1.65]*500, mode="lines", line=dict(color="#00FFCC", width=1.8), name="CH1 (A0)", row=1, col=1)
        self.fig_dual.add_scatter(x=list(range(500)), y=[1.65]*500, mode="lines", line=dict(color="#FF007F", width=1.8), name="CH2 (A1)", row=1, col=1)
        self.fig_dual.add_scatter(x=[0, 500], y=[1.65, 1.65], mode="lines", line=dict(color="#FFA500", width=1.2, dash="dash"), name="Trigger", row=1, col=1)
        self.fig_dual.add_scatter(x=initial_freqs, y=[-100]*len(initial_freqs), mode="lines", line=dict(color="#00FFCC", width=1.6), name="CH1 Spectrum", row=2, col=1)
        self.fig_dual.update_layout(template="plotly_dark", height=540, showlegend=True, margin=dict(l=40,r=20,t=45,b=35), uirevision="dual")
        self.fig_dual.update_yaxes(range=[0, 3.3], title="Voltage (V)", row=1, col=1)
        self.fig_dual.update_yaxes(range=[-110, 0], title="Mag (dBV)", row=2, col=1)
        self.fig_dual.update_xaxes(title="Time (µs)", row=1, col=1)
        self.fig_dual.update_xaxes(range=[0, 100000], title="Frequency (Hz)", row=2, col=1)

        # Tab Container
        self.tabs = widgets.Tab(children=[self.fig_scope, self.fig_spectrum, self.fig_dual])
        self.tabs.set_title(0, "📈 Dual Oscilloscope")
        self.tabs.set_title(1, "📊 Spectrum Analyzer")
        self.tabs.set_title(2, "🔀 Dual View")

    def _setup_callbacks(self):
        self.start_btn.on_click(self._on_start_clicked)
        self.stop_btn.on_click(self._on_stop_clicked)
        self.force_btn.on_click(self._on_force_clicked)
        self.clear_log_btn.on_click(self._on_clear_log_clicked)

        # Channel 1 callbacks
        self.ch1_shape_dd.observe(lambda _: self._update_wavegen_params(), names="value")
        self.ch1_freq_slider.observe(lambda _: self._update_wavegen_params(), names="value")
        self.ch1_amp_slider.observe(lambda _: self._update_wavegen_params(), names="value")

        # Channel 2 callbacks
        self.ch2_shape_dd.observe(lambda _: self._update_wavegen_params(), names="value")
        self.ch2_freq_slider.observe(lambda _: self._update_wavegen_params(), names="value")
        self.ch2_amp_slider.observe(lambda _: self._update_wavegen_params(), names="value")
        self.ch2_enable_chk.observe(lambda _: self._update_wavegen_params(), names="value")

        # Trigger callbacks
        self.trig_level_slider.observe(lambda _: self._update_hardware_threshold(), names="value")
        self.trig_mode_dd.observe(self._on_mode_or_edge_change, names="value")
        self.trig_edge_dd.observe(self._on_mode_or_edge_change, names="value")

    def _update_wavegen_params(self):
        if self.ad3:
            self.ad3.update_ch1(
                shape=self.ch1_shape_dd.value,
                frequency=float(self.ch1_freq_slider.value),
                amplitude=float(self.ch1_amp_slider.value),
                offset=1.65
            )
            self.ad3.update_ch2(
                shape=self.ch2_shape_dd.value,
                frequency=float(self.ch2_freq_slider.value),
                amplitude=float(self.ch2_amp_slider.value),
                offset=1.65,
                enabled=self.ch2_enable_chk.value
            )

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
        """High-Performance Synchronized Dual-DMA Acquisition Loop."""
        dma_armed = False
        try:
            # 1. Initialize XADC continuous sequencer
            if self.overlay and hasattr(self.overlay, "xadc_wiz_0"):
                xadc = self.overlay.xadc_wiz_0.mmio
                xadc.write(0x304, 0x2000)  # DRP 0x41 = 0x2000 (Continuous Sequence Mode)
                xadc.write(0x320, 0x0000)  # DRP 0x48 = Disable internal channels
                xadc.write(0x324, 0x0202)  # DRP 0x49 = Enable Vaux1 (bit 1) and Vaux9 (bit 9)

            # 2. Configure Trigger
            if self.trigger:
                self.trigger.mmio.write(0x0C, 5000000)  # 50ms Auto-Timeout
                self._update_hardware_threshold()

            # 3. Start AD3 Dual Wavegen
            self.ad3.start(
                shape=self.ch1_shape_dd.value,
                frequency=float(self.ch1_freq_slider.value),
                amplitude=float(self.ch1_amp_slider.value),
                offset=1.65,
                ch2_shape=self.ch2_shape_dd.value,
                ch2_frequency=float(self.ch2_freq_slider.value),
                ch2_amplitude=float(self.ch2_amp_slider.value),
                ch2_offset=1.65,
                enable_ch2=self.ch2_enable_chk.value
            )
            wait_start = time.time()
            while self._is_running and not self.ad3.is_ready:
                time.sleep(0.05)
                if time.time() - wait_start > 3.0:
                    break

            print(f"[Dashboard] Dual Acquisition active | Mode: {self.trig_mode_dd.value} | CH1: {self.ch1_shape_dd.value} @ {self.ch1_freq_slider.value} Hz | CH2: {self.ch2_shape_dd.value} @ {self.ch2_freq_slider.value} Hz")

            while self._is_running:
                mode = self.trig_mode_dd.value

                if mode == "Single" and self._single_done:
                    time.sleep(0.02)
                    continue

                # 4. Queue BOTH DMAs FIRST (Prevents Broadcaster Deadlock)
                if not dma_armed:
                    self.xadc.dma.recvchannel.transfer(self.xadc._buffer)
                    self.fft.dma.recvchannel.transfer(self.fft._buffer)
                    self.trigger.mmio.write(0x00, self._get_arm_control_word())
                    dma_armed = True

                # 5. Check if hardware frame completed
                if self.xadc.dma.recvchannel.idle and self.fft.dma.recvchannel.idle:
                    dma_armed = False
                    
                    # De-interleave Time Data: Even = A0 (Ch1), Odd = A1 (Ch2)
                    raw_time = np.array(self.xadc._buffer)
                    v_ch1 = (raw_time[0::2] >> 4) * (3.3 / 4095.0)
                    v_ch2 = (raw_time[1::2] >> 4) * (3.3 / 4095.0)

                    # Process FFT Data (Channel 1 Spectrum)
                    raw_fft = np.array(self.fft._buffer, copy=True)[:1024].astype(np.float64)
                    unit_mode = self.fft_unit_dd.value
                    if unit_mode == "Linear":
                        mags = (raw_fft / 2048.0) * (3.3 / 4095.0) * 1000.0
                    elif unit_mode == "dBFS":
                        mags = 20.0 * np.log10(np.maximum(raw_fft, 1.0) / 65535.0)
                    else:
                        linear_v = (raw_fft / 2048.0) * (3.3 / 4095.0)
                        mags = 20.0 * np.log10(np.maximum(linear_v, 1e-6))

                    # Peak tracking (CH1)
                    peak_idx = np.argmax(mags[10:]) + 10
                    peak_f = self.fft.freq_axis[peak_idx]
                    peak_m = mags[peak_idx]

                    # Auto-Range Slicing
                    vpp1 = float(np.max(v_ch1) - np.min(v_ch1))
                    vpp2 = float(np.max(v_ch2) - np.min(v_ch2))
                    freq_val = float(self.ch1_freq_slider.value)
                    
                    if self.autorange_toggle.value:
                        period_us = 1e6 / freq_val if freq_val > 0 else 1000
                        show_pts = int(5 * period_us) if vpp1 > 0.1 else 500
                        show_pts = max(40, min(show_pts, 1024, len(v_ch1)))
                    else:
                        show_pts = 500

                    plot_v1 = v_ch1[:show_pts]
                    plot_v2 = v_ch2[:show_pts]
                    time_x = np.arange(len(plot_v1))
                    trig_v = float(self.trig_level_slider.value)
                    active_tab = self.tabs.selected_index
                    max_span = float(self.fft_span_dd.value)

                    # Update Active Tab
                    if active_tab == 0:
                        with self.fig_scope.batch_update():
                            self.fig_scope.data[0].x = time_x
                            self.fig_scope.data[0].y = plot_v1
                            self.fig_scope.data[1].x = time_x
                            self.fig_scope.data[1].y = plot_v2
                            self.fig_scope.data[2].x = [0, len(plot_v1)]
                            self.fig_scope.data[2].y = [trig_v, trig_v]
                            if self.autorange_toggle.value:
                                self.fig_scope.layout.xaxis.range = [0, len(plot_v1)]
                                self.fig_scope.layout.yaxis.range = [0.0, 3.3]

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
                            self.fig_dual.data[0].y = plot_v1
                            self.fig_dual.data[1].x = time_x
                            self.fig_dual.data[1].y = plot_v2
                            self.fig_dual.data[2].x = [0, len(plot_v1)]
                            self.fig_dual.data[2].y = [trig_v, trig_v]
                            self.fig_dual.data[3].x = self.fft.freq_axis
                            self.fig_dual.data[3].y = mags
                            self.fig_dual.layout.xaxis2.range = [0, max_span]

                    # Status Bar Readouts
                    mode_tag = " <span style='color:#FFA500;'>(LOCKED)</span>" if mode == "Single" else ""
                    self.readout_vpp1.value = f"<span style='color:#00FFCC; font-family:monospace; font-size:14px; font-weight:bold;'>CH1 Vpp: {vpp1:.2f} V{mode_tag}</span>"
                    self.readout_vpp2.value = f"<span style='color:#FF007F; font-family:monospace; font-size:14px; font-weight:bold;'>CH2 Vpp: {vpp2:.2f} V</span>"
                    self.readout_f0.value = f"<span style='color:#FFA500; font-family:monospace; font-size:14px; font-weight:bold;'>Peak f0: {peak_f/1e3:.2f} kHz</span>"

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
        
        print("[Dashboard] Starting simultaneous dual acquisition...")
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
        ctrl_row1 = widgets.HBox([self.start_btn, self.stop_btn, self.force_btn, self.clear_log_btn, self.autorange_toggle, self.readout_vpp1, self.readout_vpp2, self.readout_f0], layout=widgets.Layout(align_items="center", gap="10px", margin="0 0 10px 0"))
        ctrl_row2 = widgets.HBox([self.trig_mode_dd, self.trig_edge_dd, self.trig_level_slider, self.trig_level_input])
        ctrl_row3 = widgets.HBox([self.ch1_shape_dd, self.ch1_amp_slider, self.ch1_freq_slider, self.ch1_freq_input])
        ctrl_row4 = widgets.HBox([self.ch2_shape_dd, self.ch2_amp_slider, self.ch2_freq_slider, self.ch2_freq_input, self.ch2_enable_chk])
        ctrl_row5 = widgets.HBox([self.fft_unit_dd, self.fft_span_dd])

        self.control_panel = widgets.VBox([ctrl_row1, ctrl_row2, ctrl_row3, ctrl_row4, ctrl_row5], layout=widgets.Layout(margin="0 0 15px 0"))
        display(widgets.VBox([self.control_panel, self.tabs]))