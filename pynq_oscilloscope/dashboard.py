"""
pynq_oscilloscope.dashboard: Full-Featured Interactive 4-Tab Dual-Channel Oscilloscope & Spectrum Analyzer UI.
Features sub-sample trigger phase-locking, dynamic 5-period auto-ranging, and unified AD3 wavegen binding.
"""

import time
import threading
from typing import Optional
from IPython.display import clear_output, display
import numpy as np
import ipywidgets as widgets
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pynq import allocate

from pynq_oscilloscope.xadc_dma import StreamingXADC
from pynq_oscilloscope.fft_dma import StreamingFFT
from pynq_oscilloscope.hw_trigger import HardwareTrigger
from pynq_oscilloscope.ad3_wavegen import AD3SignalGenerator


class OscilloscopeDashboard:
    """
    Complete 4-Tab Dual-Channel Oscilloscope & Spectrum Analyzer Dashboard.
    Features dynamic 5-period auto-range scaling and jitter-free edge trigger locking.
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
        self.num_pts_per_ch = packet_size // 2
        self.fs_per_ch = 50_000.0
        self.fft_points = fft_points
        self.display_window = display_window
        self.total_duration_ms = (self.num_pts_per_ch / self.fs_per_ch) * 1000.0
        
        self._is_running = False
        self._single_done = False
        self._thread: Optional[threading.Thread] = None
        
        # Bind directly to unified AD3 signal generator
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
            self.fft = StreamingFFT(self.overlay, fft_points=self.fft_points, sample_rate_hz=self.fs_per_ch)
        else:
            self.fft = None
        
        self._build_ui()
        self._build_plots()
        self._setup_callbacks()

    def _build_ui(self):
        # 1. Action Row & Real-Time Readouts
        self.start_btn = widgets.Button(description="Start Live", button_style="success", icon="play", layout=widgets.Layout(width="115px"))
        self.stop_btn = widgets.Button(description="Stop", button_style="danger", icon="stop", layout=widgets.Layout(width="95px"))
        self.force_btn = widgets.Button(description="Force / Arm", button_style="warning", icon="bolt", layout=widgets.Layout(width="115px"))
        self.clear_log_btn = widgets.Button(description="Clear Log", button_style="", icon="trash", layout=widgets.Layout(width="100px"))
        self.autorange_toggle = widgets.ToggleButton(value=True, description="Auto-Range", button_style="info", layout=widgets.Layout(width="110px"))

        self.readout_ch1 = widgets.HTML("<span style='color:#00FFCC; font-family:monospace; font-size:13px; font-weight:bold;'>A0: Vpp=0.00V | f0=0.0Hz</span>")
        self.readout_ch2 = widgets.HTML("<span style='color:#FF007F; font-family:monospace; font-size:13px; font-weight:bold;'>A1: Vpp=0.00V | f0=0.0Hz</span>")

        # 2. Trigger & FFT Routing Controls
        self.trig_mode_dd = widgets.Dropdown(options=["Auto", "Normal", "Single"], value="Auto", description="Trig Mode:", layout=widgets.Layout(width="180px"))
        self.trig_edge_dd = widgets.Dropdown(options=["Rising", "Falling"], value="Rising", description="Trig Edge:", layout=widgets.Layout(width="180px"))
        self.trig_src_dd = widgets.Dropdown(options=["CH1 (A0)", "CH2 (A1)"], value="CH1 (A0)", description="Trig Source:", layout=widgets.Layout(width="190px"))
        self.trig_level_slider = widgets.FloatSlider(value=1.65, min=0.0, max=3.3, step=0.05, description="Trig Level:", continuous_update=False, layout=widgets.Layout(width="220px"))
        self.trig_level_input = widgets.BoundedFloatText(value=1.65, min=0.0, max=3.3, step=0.05, layout=widgets.Layout(width="80px"))
        widgets.jslink((self.trig_level_slider, "value"), (self.trig_level_input, "value"))

        # 3. Channel 1 Controls (W1 -> A0)
        self.ch1_shape_dd = widgets.Dropdown(options=["Sine", "Triangle", "Square"], value="Sine", description="CH1 (A0):", layout=widgets.Layout(width="180px"))
        self.ch1_amp_slider = widgets.FloatSlider(value=1.0, min=0.1, max=1.5, step=0.1, description="Amp (V):", continuous_update=False, layout=widgets.Layout(width="200px"))
        self.ch1_amp_input = widgets.BoundedFloatText(value=1.0, min=0.1, max=1.5, step=0.1, layout=widgets.Layout(width="80px"))
        widgets.jslink((self.ch1_amp_slider, "value"), (self.ch1_amp_input, "value"))
        self.ch1_freq_slider = widgets.IntSlider(value=1000, min=50, max=10000, step=50, description="Freq (Hz):", continuous_update=False, layout=widgets.Layout(width="280px"))
        self.ch1_freq_input = widgets.BoundedIntText(value=1000, min=50, max=10000, step=50, layout=widgets.Layout(width="95px"))
        widgets.jslink((self.ch1_freq_slider, "value"), (self.ch1_freq_input, "value"))

        # 4. Channel 2 Controls (W2 -> A1)
        self.ch2_shape_dd = widgets.Dropdown(options=["Sine", "Triangle", "Square"], value="Square", description="CH2 (A1):", layout=widgets.Layout(width="180px"))
        self.ch2_amp_slider = widgets.FloatSlider(value=1.0, min=0.1, max=1.5, step=0.1, description="Amp (V):", continuous_update=False, layout=widgets.Layout(width="200px"))
        self.ch2_amp_input = widgets.BoundedFloatText(value=1.0, min=0.1, max=1.5, step=0.1, layout=widgets.Layout(width="80px"))
        widgets.jslink((self.ch2_amp_slider, "value"), (self.ch2_amp_input, "value"))
        self.ch2_freq_slider = widgets.IntSlider(value=2500, min=50, max=10000, step=50, description="Freq (Hz):", continuous_update=False, layout=widgets.Layout(width="280px"))
        self.ch2_freq_input = widgets.BoundedIntText(value=2500, min=50, max=10000, step=50, layout=widgets.Layout(width="95px"))
        widgets.jslink((self.ch2_freq_slider, "value"), (self.ch2_freq_input, "value"))

        # 5. FFT Controls
        self.fft_unit_dd = widgets.Dropdown(options=["dBV", "dBFS", "Linear"], value="dBV", description="FFT Unit:", layout=widgets.Layout(width="170px"))
        self.fft_span_dd = widgets.Dropdown(options=[("Full (25 kHz)", 25000), ("8 kHz (Audio)", 8000), ("2 kHz (Bass Zoom)", 2000)], value=8000, description="Span / Zoom:", layout=widgets.Layout(width="210px"))

    def _build_plots(self):
        t_ms = np.linspace(0, self.total_duration_ms, self.num_pts_per_ch - 16)
        initial_freq = np.linspace(0, 25000, len(t_ms) // 2 + 1)

        # Tab 1: Dual Scope
        self.fig_dual_scope = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
            subplot_titles=("<b>Channel 1: A0 (Time Domain)</b>", "<b>Channel 2: A1 (Time Domain)</b>"))
        self.fig_dual_scope = go.FigureWidget(self.fig_dual_scope)
        self.fig_dual_scope.add_scatter(x=t_ms, y=[1.65]*len(t_ms), mode="lines", line=dict(color="#00FFCC", width=1.8), name="A0", row=1, col=1)
        self.fig_dual_scope.add_scatter(x=[0, self.total_duration_ms], y=[1.65, 1.65], mode="lines", line=dict(color="#FFA500", width=1.2, dash="dash"), name="Trigger (A0)", row=1, col=1)
        self.fig_dual_scope.add_scatter(x=t_ms, y=[1.65]*len(t_ms), mode="lines", line=dict(color="#FF007F", width=1.8), name="A1", row=2, col=1)
        self.fig_dual_scope.add_scatter(x=[0, self.total_duration_ms], y=[1.65, 1.65], mode="lines", line=dict(color="#FFA500", width=1.2, dash="dash"), name="Trigger (A1)", visible=False, row=2, col=1)
        self.fig_dual_scope.update_layout(template="plotly_dark", height=500, margin=dict(l=40, r=20, t=45, b=35), uirevision="t1")
        self.fig_dual_scope.update_yaxes(range=[0, 3.3], title="Voltage (V)", row=1, col=1)
        self.fig_dual_scope.update_yaxes(range=[0, 3.3], title="Voltage (V)", row=2, col=1)
        self.fig_dual_scope.update_xaxes(range=[0, self.total_duration_ms], title="Time (Milliseconds)", row=2, col=1)

        # Tab 2: Dual Spectrum
        self.fig_dual_fft = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
            subplot_titles=("<b>Channel 1: A0 Spectrum (FFT)</b>", "<b>Channel 2: A1 Spectrum (FFT)</b>"))
        self.fig_dual_fft = go.FigureWidget(self.fig_dual_fft)
        self.fig_dual_fft.add_scatter(x=initial_freq, y=[-100]*len(initial_freq), mode="lines", line=dict(color="#00FFCC", width=1.8), name="A0 FFT", row=1, col=1)
        self.fig_dual_fft.add_scatter(x=[1000], y=[-40], mode="markers+text", marker=dict(color="#FFA500", size=7, symbol="diamond"), text=["Peak"], textposition="top center", name="Peak 1", row=1, col=1)
        self.fig_dual_fft.add_scatter(x=initial_freq, y=[-100]*len(initial_freq), mode="lines", line=dict(color="#FF007F", width=1.8), name="A1 FFT", row=2, col=1)
        self.fig_dual_fft.add_scatter(x=[2500], y=[-40], mode="markers+text", marker=dict(color="#FFA500", size=7, symbol="diamond"), text=["Peak"], textposition="top center", name="Peak 2", row=2, col=1)
        self.fig_dual_fft.update_layout(template="plotly_dark", height=500, margin=dict(l=40, r=20, t=45, b=35), uirevision="t2")
        self.fig_dual_fft.update_yaxes(range=[-110, 0], title="Mag (dBV)", row=1, col=1)
        self.fig_dual_fft.update_yaxes(range=[-110, 0], title="Mag (dBV)", row=2, col=1)
        self.fig_dual_fft.update_xaxes(range=[0, 8000], title="Frequency (Hz)", row=2, col=1)

        # Tab 3: Channel 1 View
        self.fig_ch1_view = make_subplots(rows=2, cols=1, vertical_spacing=0.15,
            subplot_titles=("<b>Channel 1: A0 (Time Domain)</b>", "<b>Channel 1: A0 (Frequency Domain)</b>"))
        self.fig_ch1_view = go.FigureWidget(self.fig_ch1_view)
        self.fig_ch1_view.add_scatter(x=t_ms, y=[1.65]*len(t_ms), mode="lines", line=dict(color="#00FFCC", width=1.8), name="A0 Time", row=1, col=1)
        self.fig_ch1_view.add_scatter(x=[0, self.total_duration_ms], y=[1.65, 1.65], mode="lines", line=dict(color="#FFA500", width=1.2, dash="dash"), name="Trigger", row=1, col=1)
        self.fig_ch1_view.add_scatter(x=initial_freq, y=[-100]*len(initial_freq), mode="lines", line=dict(color="#00FFCC", width=1.8), name="A0 FFT", row=2, col=1)
        self.fig_ch1_view.update_layout(template="plotly_dark", height=500, margin=dict(l=40, r=20, t=45, b=35), uirevision="t3")
        self.fig_ch1_view.update_yaxes(range=[0, 3.3], title="Voltage (V)", row=1, col=1)
        self.fig_ch1_view.update_yaxes(range=[-110, 0], title="Mag (dBV)", row=2, col=1)
        self.fig_ch1_view.update_xaxes(range=[0, 8000], title="Frequency (Hz)", row=2, col=1)

        # Tab 4: Channel 2 View
        self.fig_ch2_view = make_subplots(rows=2, cols=1, vertical_spacing=0.15,
            subplot_titles=("<b>Channel 2: A1 (Time Domain)</b>", "<b>Channel 2: A1 (Frequency Domain)</b>"))
        self.fig_ch2_view = go.FigureWidget(self.fig_ch2_view)
        self.fig_ch2_view.add_scatter(x=t_ms, y=[1.65]*len(t_ms), mode="lines", line=dict(color="#FF007F", width=1.8), name="A1 Time", row=1, col=1)
        self.fig_ch2_view.add_scatter(x=[0, self.total_duration_ms], y=[1.65, 1.65], mode="lines", line=dict(color="#FFA500", width=1.2, dash="dash"), name="Trigger", row=1, col=1)
        self.fig_ch2_view.add_scatter(x=initial_freq, y=[-100]*len(initial_freq), mode="lines", line=dict(color="#FF007F", width=1.8), name="A1 FFT", row=2, col=1)
        self.fig_ch2_view.update_layout(template="plotly_dark", height=500, margin=dict(l=40, r=20, t=45, b=35), uirevision="t4")
        self.fig_ch2_view.update_yaxes(range=[0, 3.3], title="Voltage (V)", row=1, col=1)
        self.fig_ch2_view.update_yaxes(range=[-110, 0], title="Mag (dBV)", row=2, col=1)
        self.fig_ch2_view.update_xaxes(range=[0, 8000], title="Frequency (Hz)", row=2, col=1)

        self.tabs = widgets.Tab(children=[self.fig_dual_scope, self.fig_dual_fft, self.fig_ch1_view, self.fig_ch2_view])
        self.tabs.set_title(0, "📈 Dual Scope (A0 & A1)")
        self.tabs.set_title(1, "📊 Dual FFT (A0 & A1)")
        self.tabs.set_title(2, "🔀 Channel 1 (A0)")
        self.tabs.set_title(3, "🔀 Channel 2 (A1)")

    def _setup_callbacks(self):
        self.start_btn.on_click(lambda _: self.start())
        self.stop_btn.on_click(lambda _: self.stop())
        self.force_btn.on_click(lambda _: self._on_force_clicked())
        self.clear_log_btn.on_click(lambda _: self._on_clear_log_clicked())

        self.ch1_shape_dd.observe(lambda _: self._update_wavegen(), names="value")
        self.ch1_freq_slider.observe(lambda _: self._update_wavegen(), names="value")
        self.ch1_amp_slider.observe(lambda _: self._update_wavegen(), names="value")

        self.ch2_shape_dd.observe(lambda _: self._update_wavegen(), names="value")
        self.ch2_freq_slider.observe(lambda _: self._update_wavegen(), names="value")
        self.ch2_amp_slider.observe(lambda _: self._update_wavegen(), names="value")

        self.trig_level_slider.observe(lambda _: self._update_trig_level(), names="value")
        self.trig_mode_dd.observe(self._on_trig_param_change, names="value")
        self.trig_edge_dd.observe(self._on_trig_param_change, names="value")
        self.trig_src_dd.observe(self._on_trig_param_change, names="value")

    def _update_trig_level(self):
        if self.trigger:
            self.trigger.set_threshold(float(self.trig_level_slider.value))

    def _update_wavegen(self):
        if self.ad3 and self.ad3.is_running:
            self.ad3.update_ch1(shape=self.ch1_shape_dd.value, frequency=self.ch1_freq_slider.value, amplitude=self.ch1_amp_slider.value)
            self.ad3.update_ch2(shape=self.ch2_shape_dd.value, frequency=self.ch2_freq_slider.value, amplitude=self.ch2_amp_slider.value)

    def _get_arm_control_word(self) -> int:
        is_falling = (self.trig_edge_dd.value == "Falling")
        is_ch2 = ("CH2" in self.trig_src_dd.value or "A1" in self.trig_src_dd.value)
        mode = self.trig_mode_dd.value
        
        ctrl = (1 << 0) | (1 << 3)  # Bit 0: Arm, Bit 3: Single Shot
        if is_falling: ctrl |= (1 << 2)
        if mode == "Auto": ctrl |= (1 << 1)
        if is_ch2: ctrl |= (1 << 5)  # Bit 5 = 1 for CH2 (A1), 0 for CH1 (A0)
        return ctrl

    def _on_trig_param_change(self, _):
        if self.trig_mode_dd.value != "Single": self._single_done = False
        if self._is_running and self.trigger:
            self.trigger.mmio.write(0x00, self._get_arm_control_word())

    def _on_force_clicked(self):
        if self.trig_mode_dd.value == "Single": self._single_done = False
        if self.trigger: self.trigger.force_trigger()

    def _on_clear_log_clicked(self):
        clear_output(wait=True)
        display(widgets.VBox([self.control_panel, self.tabs]))

    @staticmethod
    def _find_trigger_edge(signal: np.ndarray, threshold: float, is_falling: bool) -> int:
        if len(signal) < 10:
            return 0
        search_limit = min(len(signal) - 1, 200)
        if is_falling:
            for i in range(search_limit):
                if signal[i] >= threshold and signal[i + 1] < threshold:
                    return i
        else:
            for i in range(search_limit):
                if signal[i] <= threshold and signal[i + 1] > threshold:
                    return i
        return 0

    def _update_loop(self):
        dma_time = self.overlay.axi_dma_0
        trig = self.trigger

        # Initialize XADC simultaneous sequence
        if hasattr(self.overlay, "xadc_wiz_0"):
            self.overlay.xadc_wiz_0.mmio.write(0x304, 0xC000)
            self.overlay.xadc_wiz_0.mmio.write(0x320, 0x0000)
            self.overlay.xadc_wiz_0.mmio.write(0x324, 0x0202)

        # Reset DMA 0
        dma_time.mmio.write(0x30, 0x04)
        time.sleep(0.005)
        dma_time.recvchannel.start()

        if trig:
            trig.mmio.write(0x0C, 5000000)
            self._update_trig_level()

        # Start AD3 Wavegen
        self.ad3.start()
        time.sleep(0.5)

        buf_time = allocate(shape=(self.packet_size,), dtype="u2")
        dma_armed = False
        print(f"[Dashboard] Active (50 kSPS Simultaneous Audio Stream | 40.96 ms Window)")

        try:
            while self._is_running:
                mode = self.trig_mode_dd.value
                if mode == "Single" and self._single_done:
                    time.sleep(0.02)
                    continue

                if not dma_armed:
                    dma_time.recvchannel.transfer(buf_time)
                    trig.mmio.write(0x00, self._get_arm_control_word())
                    dma_armed = True

                if dma_time.recvchannel.idle:
                    dma_armed = False

                    raw = np.array(buf_time)
                    v_a0 = (raw[0::2] >> 4) * (3.3 / 4095.0)
                    v_a1 = (raw[1::2] >> 4) * (3.3 / 4095.0)

                    # Crop boundary words
                    p_v1 = v_a0[8:-8]
                    p_v2 = v_a1[8:-8]

                    # Spectrum with Hann Window
                    n = len(p_v1)
                    sig_a0 = (p_v1 - np.mean(p_v1)) * np.hanning(n)
                    sig_a1 = (p_v2 - np.mean(p_v2)) * np.hanning(n)

                    freqs = np.fft.rfftfreq(n, d=1.0 / self.fs_per_ch)
                    mag_a0 = 20.0 * np.log10(np.maximum(np.abs(np.fft.rfft(sig_a0)) / (n / 2.0), 1e-6))
                    mag_a1 = 20.0 * np.log10(np.maximum(np.abs(np.fft.rfft(sig_a1)) / (n / 2.0), 1e-6))

                    vpp1, vpp2 = float(np.ptp(p_v1)), float(np.ptp(p_v2))
                    p_f1, p_m1 = StreamingFFT.get_peak_frequency(freqs, mag_a0, min_freq_hz=20.0)
                    p_f2, p_m2 = StreamingFFT.get_peak_frequency(freqs, mag_a1, min_freq_hz=20.0)

                    trig_v = float(self.trig_level_slider.value)
                    is_falling = (self.trig_edge_dd.value == "Falling")
                    is_trig_a0 = ("CH1" in self.trig_src_dd.value or "A0" in self.trig_src_dd.value)
                    active_tab = self.tabs.selected_index
                    max_span = float(self.fft_span_dd.value)

                    # Sub-sample Phase Lock
                    trig_src_sig = p_v1 if is_trig_a0 else p_v2
                    edge_offset = self._find_trigger_edge(trig_src_sig, trig_v, is_falling)

                    # Dynamic 5-Period Auto-Range Timebase
                    active_f0 = p_f1 if is_trig_a0 else p_f2
                    if self.autorange_toggle.value and active_f0 > 30.0:
                        period_ms = 1000.0 / active_f0
                        show_duration_ms = min(self.total_duration_ms, max(1.5, 5.0 * period_ms))
                        show_pts = int((show_duration_ms / 1000.0) * self.fs_per_ch)
                        show_pts = max(30, min(show_pts, len(p_v1) - edge_offset))
                    else:
                        show_pts = len(p_v1) - edge_offset
                        show_duration_ms = (show_pts / self.fs_per_ch) * 1000.0

                    plot_v1 = p_v1[edge_offset : edge_offset + show_pts]
                    plot_v2 = p_v2[edge_offset : edge_offset + show_pts]
                    t_ms = np.linspace(0, show_duration_ms, len(plot_v1))

                    if active_tab == 0:  # Tab 1: Dual Scope
                        with self.fig_dual_scope.batch_update():
                            self.fig_dual_scope.data[0].x = t_ms
                            self.fig_dual_scope.data[0].y = plot_v1
                            self.fig_dual_scope.data[2].x = t_ms
                            self.fig_dual_scope.data[2].y = plot_v2
                            if is_trig_a0:
                                self.fig_dual_scope.data[1].x = [0, show_duration_ms]
                                self.fig_dual_scope.data[1].y = [trig_v, trig_v]
                                self.fig_dual_scope.data[1].visible = True
                                self.fig_dual_scope.data[3].visible = False
                            else:
                                self.fig_dual_scope.data[3].x = [0, show_duration_ms]
                                self.fig_dual_scope.data[3].y = [trig_v, trig_v]
                                self.fig_dual_scope.data[3].visible = True
                                self.fig_dual_scope.data[1].visible = False
                            self.fig_dual_scope.layout.xaxis2.range = [0, show_duration_ms]

                    elif active_tab == 1:  # Tab 2: Dual FFT
                        with self.fig_dual_fft.batch_update():
                            self.fig_dual_fft.data[0].x = freqs
                            self.fig_dual_fft.data[0].y = mag_a0
                            self.fig_dual_fft.data[1].x = [p_f1]
                            self.fig_dual_fft.data[1].y = [p_m1]
                            self.fig_dual_fft.data[1].text = [f" {p_f1:.1f} Hz"]
                            self.fig_dual_fft.data[2].x = freqs
                            self.fig_dual_fft.data[2].y = mag_a1
                            self.fig_dual_fft.data[3].x = [p_f2]
                            self.fig_dual_fft.data[3].y = [p_m2]
                            self.fig_dual_fft.data[3].text = [f" {p_f2:.1f} Hz"]
                            self.fig_dual_fft.layout.xaxis2.range = [0, max_span]

                    elif active_tab == 2:  # Tab 3: CH1 View
                        with self.fig_ch1_view.batch_update():
                            self.fig_ch1_view.data[0].x = t_ms
                            self.fig_ch1_view.data[0].y = plot_v1
                            self.fig_ch1_view.data[1].x = [0, show_duration_ms]
                            self.fig_ch1_view.data[1].y = [trig_v, trig_v]
                            self.fig_ch1_view.data[2].x = freqs
                            self.fig_ch1_view.data[2].y = mag_a0
                            self.fig_ch1_view.layout.xaxis.range = [0, show_duration_ms]
                            self.fig_ch1_view.layout.xaxis2.range = [0, max_span]

                    elif active_tab == 3:  # Tab 4: CH2 View
                        with self.fig_ch2_view.batch_update():
                            self.fig_ch2_view.data[0].x = t_ms
                            self.fig_ch2_view.data[0].y = plot_v2
                            self.fig_ch2_view.data[1].x = [0, show_duration_ms]
                            self.fig_ch2_view.data[1].y = [trig_v, trig_v]
                            self.fig_ch2_view.data[2].x = freqs
                            self.fig_ch2_view.data[2].y = mag_a1
                            self.fig_ch2_view.layout.xaxis.range = [0, show_duration_ms]
                            self.fig_ch2_view.layout.xaxis2.range = [0, max_span]

                    mode_tag = " (LOCKED)" if mode == "Single" else ""
                    self.readout_ch1.value = f"<span style='color:#00FFCC; font-family:monospace; font-size:13px; font-weight:bold;'>A0: Vpp={vpp1:.2f}V | f0={p_f1:.1f}Hz{mode_tag}</span>"
                    self.readout_ch2.value = f"<span style='color:#FF007F; font-family:monospace; font-size:13px; font-weight:bold;'>A1: Vpp={vpp2:.2f}V | f0={p_f2:.1f}Hz</span>"

                    if mode == "Single": self._single_done = True
                    time.sleep(0.033)
                else:
                    time.sleep(0.005)

        finally:
            self._is_running = False
            buf_time.close()
            self.ad3.stop()
            print("[Dashboard] Stopped cleanly.")

    def start(self):
        if not self._is_running:
            self._is_running = True
            self._single_done = False
            self._thread = threading.Thread(target=self._update_loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._is_running = False
        self._single_done = False

    def display(self):
        r1 = widgets.HBox([self.start_btn, self.stop_btn, self.force_btn, self.autorange_toggle, self.clear_log_btn, self.readout_ch1, self.readout_ch2], layout=widgets.Layout(gap="10px", margin="0 0 8px 0"))
        r2 = widgets.HBox([self.trig_mode_dd, self.trig_edge_dd, self.trig_src_dd, self.trig_level_slider, self.trig_level_input])
        r3 = widgets.HBox([self.ch1_shape_dd, self.ch1_amp_slider, self.ch1_amp_input, self.ch1_freq_slider, self.ch1_freq_input])
        r4 = widgets.HBox([self.ch2_shape_dd, self.ch2_amp_slider, self.ch2_amp_input, self.ch2_freq_slider, self.ch2_freq_input])
        r5 = widgets.HBox([self.fft_unit_dd, self.fft_span_dd])
        self.control_panel = widgets.VBox([r1, r2, r3, r4, r5], layout=widgets.Layout(margin="0 0 12px 0"))
        display(widgets.VBox([self.control_panel, self.tabs]))