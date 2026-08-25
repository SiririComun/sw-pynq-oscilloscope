"""
pynq_oscilloscope.audio_dashboard: Dedicated Real-Time Audio & Passive Microphone Dashboard.
Runs completely independently without requiring an Analog Discovery 3.
Features real-time VU meters, clipping alerts, sub-bin pitch tracking, and Hann-windowed FFT.
"""

import time
import threading
from typing import Optional, Tuple
from IPython.display import clear_output, display
import numpy as np
import ipywidgets as widgets
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pynq import allocate

from pynq_oscilloscope.xadc_dma import StreamingXADC
from pynq_oscilloscope.fft_dma import StreamingFFT
from pynq_oscilloscope.hw_trigger import HardwareTrigger


class AudioDashboard:
    """
    Dedicated Passive Microphone Dashboard (MAX4466 / MAX9814 on Pins A0 & A1).
    Features 4-tab synchronized view, live VU meter bars, saturation warnings,
    and Hann-windowed harmonic overtone tracking.
    """

    def __init__(
        self,
        overlay=None,
        packet_size: int = 2048,
        fs_per_ch: float = 50_000.0,
        display_window: int = 1024,
        **kwargs
    ):
        self.overlay = overlay
        self.packet_size = packet_size
        self.num_pts_per_ch = packet_size // 2
        self.fs_per_ch = fs_per_ch
        self.display_window = display_window
        self.total_duration_ms = (self.num_pts_per_ch / self.fs_per_ch) * 1000.0

        self._is_running = False
        self._single_done = False
        self._thread: Optional[threading.Thread] = None

        # Bind sub-drivers from overlay
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
            self.fft = StreamingFFT(self.overlay, fft_points=self.packet_size, sample_rate_hz=self.fs_per_ch)
        else:
            self.fft = None

        self._build_ui()
        self._build_plots()
        self._setup_callbacks()

    def _build_ui(self):
        # 1. Action Row & Real-Time Readouts
        self.start_btn = widgets.Button(description="Start Live", button_style="success", icon="play", layout=widgets.Layout(width="115px"))
        self.stop_btn = widgets.Button(description="Stop", button_style="danger", icon="stop", layout=widgets.Layout(width="95px"))
        self.force_btn = widgets.Button(description="Force Trig", button_style="warning", icon="bolt", layout=widgets.Layout(width="115px"))
        self.clear_log_btn = widgets.Button(description="Clear Log", button_style="", icon="trash", layout=widgets.Layout(width="100px"))
        self.autorange_toggle = widgets.ToggleButton(value=True, description="Auto-Range", button_style="info", layout=widgets.Layout(width="110px"))

        self.readout_vu = widgets.HTML("<span style='color:#00FFCC; font-family:monospace; font-size:13px; font-weight:bold;'>Mic 1 (A0): [          ] 0.00V | Mic 2 (A1): [          ] 0.00V | Pitch f0: 0.0 Hz</span>")

        # 2. Controls Row
        self.trig_mode_dd = widgets.Dropdown(options=["Auto", "Normal", "Single"], value="Auto", description="Trig Mode:", layout=widgets.Layout(width="180px"))
        self.trig_edge_dd = widgets.Dropdown(options=["Rising", "Falling"], value="Rising", description="Trig Edge:", layout=widgets.Layout(width="180px"))
        self.trig_src_dd = widgets.Dropdown(options=["Mic 1 (A0)", "Mic 2 (A1)"], value="Mic 1 (A0)", description="Trig Src:", layout=widgets.Layout(width="190px"))
        self.trig_level_slider = widgets.FloatSlider(value=1.65, min=0.0, max=3.3, step=0.05, description="Trig Level:", continuous_update=False, layout=widgets.Layout(width="220px"))
        self.trig_level_input = widgets.BoundedFloatText(value=1.65, min=0.0, max=3.3, step=0.05, layout=widgets.Layout(width="80px"))
        widgets.jslink((self.trig_level_slider, "value"), (self.trig_level_input, "value"))

        # 3. FFT Controls
        self.fft_unit_dd = widgets.Dropdown(options=["dBV", "dBFS", "Linear"], value="dBV", description="FFT Unit:", layout=widgets.Layout(width="170px"))
        self.fft_span_dd = widgets.Dropdown(
            options=[("Audio Spectrum (25 kHz)", 25000), ("Vocal / Speech (10 kHz)", 10000), ("Bass Sub-Band (2 kHz)", 2000)],
            value=25000,
            description="FFT Span:",
            layout=widgets.Layout(width="230px")
        )

    def _build_plots(self):
        t_ms = np.linspace(0, self.total_duration_ms, self.num_pts_per_ch - 16)
        initial_freq = np.linspace(0, self.fs_per_ch / 2.0, len(t_ms) // 2 + 1)

        # Tab 1: Dual Audio Scope
        self.fig_dual_scope = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
            subplot_titles=("<b>Microphone 1: Header A0 (Time Domain)</b>", "<b>Microphone 2: Header A1 (Time Domain)</b>"))
        self.fig_dual_scope = go.FigureWidget(self.fig_dual_scope)
        self.fig_dual_scope.add_scatter(x=t_ms, y=[1.65]*len(t_ms), mode="lines", line=dict(color="#00FFCC", width=1.6), name="Mic 1", row=1, col=1)
        self.fig_dual_scope.add_scatter(x=[0, self.total_duration_ms], y=[1.65, 1.65], mode="lines", line=dict(color="#FFA500", width=1.2, dash="dash"), name="Trigger (A0)", row=1, col=1)
        self.fig_dual_scope.add_scatter(x=t_ms, y=[1.65]*len(t_ms), mode="lines", line=dict(color="#FF007F", width=1.6), name="Mic 2", row=2, col=1)
        self.fig_dual_scope.add_scatter(x=[0, self.total_duration_ms], y=[1.65, 1.65], mode="lines", line=dict(color="#FFA500", width=1.2, dash="dash"), name="Trigger (A1)", visible=False, row=2, col=1)
        self.fig_dual_scope.update_layout(template="plotly_dark", height=500, margin=dict(l=40, r=20, t=45, b=35), uirevision="t1")
        self.fig_dual_scope.update_yaxes(range=[0, 3.3], title="Voltage (V)", row=1, col=1)
        self.fig_dual_scope.update_yaxes(range=[0, 3.3], title="Voltage (V)", row=2, col=1)
        self.fig_dual_scope.update_xaxes(range=[0, self.total_duration_ms], title="Time (Milliseconds)", row=2, col=1)

        # Tab 2: Dual Audio Spectrum
        self.fig_dual_fft = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
            subplot_titles=("<b>Microphone 1: A0 Spectrum (FFT)</b>", "<b>Microphone 2: A1 Spectrum (FFT)</b>"))
        self.fig_dual_fft = go.FigureWidget(self.fig_dual_fft)
        self.fig_dual_fft.add_scatter(x=initial_freq, y=[-100]*len(initial_freq), mode="lines", line=dict(color="#00FFCC", width=1.6), name="Mic 1 FFT", row=1, col=1)
        self.fig_dual_fft.add_scatter(x=[1000], y=[-40], mode="markers+text", marker=dict(color="#FFA500", size=7, symbol="diamond"), text=["Peak"], textposition="top center", name="Peak 1", row=1, col=1)
        self.fig_dual_fft.add_scatter(x=initial_freq, y=[-100]*len(initial_freq), mode="lines", line=dict(color="#FF007F", width=1.6), name="Mic 2 FFT", row=2, col=1)
        self.fig_dual_fft.add_scatter(x=[1000], y=[-40], mode="markers+text", marker=dict(color="#FFA500", size=7, symbol="diamond"), text=["Peak"], textposition="top center", name="Peak 2", row=2, col=1)
        self.fig_dual_fft.update_layout(template="plotly_dark", height=500, margin=dict(l=40, r=20, t=45, b=35), uirevision="t2")
        self.fig_dual_fft.update_yaxes(range=[-100, 5], title="Mag (dBV)", row=1, col=1)
        self.fig_dual_fft.update_yaxes(range=[-100, 5], title="Mag (dBV)", row=2, col=1)
        self.fig_dual_fft.update_xaxes(range=[0, 25000], title="Frequency (Hz)", row=2, col=1)

        # Tab 3: Dedicated Mic 1 View
        self.fig_mic1_view = make_subplots(rows=2, cols=1, vertical_spacing=0.15,
            subplot_titles=("<b>Microphone 1: A0 (Time Domain)</b>", "<b>Microphone 1: A0 (Frequency Spectrum)</b>"))
        self.fig_mic1_view = go.FigureWidget(self.fig_mic1_view)
        self.fig_mic1_view.add_scatter(x=t_ms, y=[1.65]*len(t_ms), mode="lines", line=dict(color="#00FFCC", width=1.6), name="Mic 1 Time", row=1, col=1)
        self.fig_mic1_view.add_scatter(x=[0, self.total_duration_ms], y=[1.65, 1.65], mode="lines", line=dict(color="#FFA500", width=1.2, dash="dash"), name="Trigger", row=1, col=1)
        self.fig_mic1_view.add_scatter(x=initial_freq, y=[-100]*len(initial_freq), mode="lines", line=dict(color="#00FFCC", width=1.6), name="Mic 1 FFT", row=2, col=1)
        self.fig_mic1_view.update_layout(template="plotly_dark", height=500, margin=dict(l=40, r=20, t=45, b=35), uirevision="t3")
        self.fig_mic1_view.update_yaxes(range=[0, 3.3], title="Voltage (V)", row=1, col=1)
        self.fig_mic1_view.update_yaxes(range=[-100, 5], title="Mag (dBV)", row=2, col=1)
        self.fig_mic1_view.update_xaxes(range=[0, 25000], title="Frequency (Hz)", row=2, col=1)

        # Tab 4: Dedicated Mic 2 View
        self.fig_mic2_view = make_subplots(rows=2, cols=1, vertical_spacing=0.15,
            subplot_titles=("<b>Microphone 2: A1 (Time Domain)</b>", "<b>Microphone 2: A1 (Frequency Spectrum)</b>"))
        self.fig_mic2_view = go.FigureWidget(self.fig_mic2_view)
        self.fig_mic2_view.add_scatter(x=t_ms, y=[1.65]*len(t_ms), mode="lines", line=dict(color="#FF007F", width=1.6), name="Mic 2 Time", row=1, col=1)
        self.fig_mic2_view.add_scatter(x=[0, self.total_duration_ms], y=[1.65, 1.65], mode="lines", line=dict(color="#FFA500", width=1.2, dash="dash"), name="Trigger", row=1, col=1)
        self.fig_mic2_view.add_scatter(x=initial_freq, y=[-100]*len(initial_freq), mode="lines", line=dict(color="#FF007F", width=1.6), name="Mic 2 FFT", row=2, col=1)
        self.fig_mic2_view.update_layout(template="plotly_dark", height=500, margin=dict(l=40, r=20, t=45, b=35), uirevision="t4")
        self.fig_mic2_view.update_yaxes(range=[0, 3.3], title="Voltage (V)", row=1, col=1)
        self.fig_mic2_view.update_yaxes(range=[-100, 5], title="Mag (dBV)", row=2, col=1)
        self.fig_mic2_view.update_xaxes(range=[0, 25000], title="Frequency (Hz)", row=2, col=1)

        self.tabs = widgets.Tab(children=[self.fig_dual_scope, self.fig_dual_fft, self.fig_mic1_view, self.fig_mic2_view])
        self.tabs.set_title(0, "📈 Dual Audio Scope")
        self.tabs.set_title(1, "📊 Dual Audio FFT")
        self.tabs.set_title(2, "🎙 Mic 1 (A0)")
        self.tabs.set_title(3, "🎙 Mic 2 (A1)")

    def _setup_callbacks(self):
        self.start_btn.on_click(lambda _: self.start())
        self.stop_btn.on_click(lambda _: self.stop())
        self.force_btn.on_click(lambda _: self._on_force_clicked())
        self.clear_log_btn.on_click(lambda _: self._on_clear_log_clicked())

        self.trig_level_slider.observe(lambda _: self._update_trig_level(), names="value")
        self.trig_mode_dd.observe(self._on_trig_param_change, names="value")
        self.trig_edge_dd.observe(self._on_trig_param_change, names="value")
        self.trig_src_dd.observe(self._on_trig_param_change, names="value")

    def _update_trig_level(self):
        if self.trigger:
            self.trigger.set_threshold(float(self.trig_level_slider.value))

    def _get_arm_control_word(self) -> int:
        is_falling = (self.trig_edge_dd.value == "Falling")
        is_ch2 = ("Mic 2" in self.trig_src_dd.value or "A1" in self.trig_src_dd.value)
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

    @staticmethod
    def _vu_bar(vpp: float, vmin: float, vmax: float) -> Tuple[str, bool]:
        bars = int(min(10, max(0, vpp / 0.25)))
        is_clipped = (vmin < 0.10 or vmax > 3.10)
        bar_char = "█" if not is_clipped else "!"
        return "[" + bar_char * bars + " " * (10 - bars) + "]", is_clipped

    def _update_loop(self):
        dma_time = self.overlay.axi_dma_0
        trig = self.trigger

        # Initialize XADC continuous sequencer for audio mode
        if hasattr(self.overlay, "xadc_wiz_0"):
            self.overlay.xadc_wiz_0.mmio.write(0x304, 0x2000)
            self.overlay.xadc_wiz_0.mmio.write(0x320, 0x0000)
            self.overlay.xadc_wiz_0.mmio.write(0x324, 0x0202)

        # Set Audio Profile (M=10, 50 kSPS)
        if self.overlay and hasattr(self.overlay, "set_profile"):
            self.overlay.set_profile("audio")
        elif trig:
            trig.set_decimation(10)

        # Reset DMA 0
        dma_time.mmio.write(0x30, 0x04)
        time.sleep(0.005)
        dma_time.recvchannel.start()

        if trig:
            trig.mmio.write(0x0C, 5000000)
            self._update_trig_level()

        buf_time = allocate(shape=(self.packet_size,), dtype="u2")
        dma_armed = False
        print(f"[AudioDashboard] Passive Microphone Instrument Active (50 kSPS | {self.total_duration_ms:.2f} ms Timebase)")

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
                    n = len(p_v1)

                    vpp1, vpp2 = float(np.ptp(p_v1)), float(np.ptp(p_v2))
                    vu1_str, clip1 = self._vu_bar(vpp1, np.min(p_v1), np.max(p_v1))
                    vu2_str, clip2 = self._vu_bar(vpp2, np.min(p_v2), np.max(p_v2))

                    # Compute Hann-windowed FFT & Peak Pitch
                    sig_a0 = (p_v1 - np.mean(p_v1)) * np.hanning(n)
                    sig_a1 = (p_v2 - np.mean(p_v2)) * np.hanning(n)
                    freqs = np.fft.rfftfreq(n, d=1.0 / self.fs_per_ch)
                    mag_a0 = 20.0 * np.log10(np.maximum(np.abs(np.fft.rfft(sig_a0)) / (n / 2.0), 1e-6))
                    mag_a1 = 20.0 * np.log10(np.maximum(np.abs(np.fft.rfft(sig_a1)) / (n / 2.0), 1e-6))

                    p_f1, p_m1 = StreamingFFT.get_peak_frequency(freqs, mag_a0, min_freq_hz=30.0)
                    p_f2, p_m2 = StreamingFFT.get_peak_frequency(freqs, mag_a1, min_freq_hz=30.0)

                    trig_v = float(self.trig_level_slider.value)
                    is_falling = (self.trig_edge_dd.value == "Falling")
                    is_trig_a0 = ("Mic 1" in self.trig_src_dd.value or "A0" in self.trig_src_dd.value)
                    active_tab = self.tabs.selected_index
                    max_span = float(self.fft_span_dd.value)

                    # Sub-sample Phase Locking on active trigger channel
                    trig_src_sig = p_v1 if is_trig_a0 else p_v2
                    edge_offset = self._find_trigger_edge(trig_src_sig, trig_v, is_falling)

                    # Dynamic 5-Period Auto-Range Timebase
                    active_f0 = p_f1 if is_trig_a0 else p_f2
                    if self.autorange_toggle.value and active_f0 > 50.0:
                        period_ms = 1000.0 / active_f0
                        show_duration_ms = min(self.total_duration_ms, max(0.01, 5.0 * period_ms))
                        show_pts = int((show_duration_ms / 1000.0) * self.fs_per_ch)
                        show_pts = max(16, min(show_pts, len(p_v1) - edge_offset))
                    else:
                        show_pts = len(p_v1) - edge_offset
                        show_duration_ms = (show_pts / self.fs_per_ch) * 1000.0

                    plot_v1 = p_v1[edge_offset : edge_offset + show_pts]
                    plot_v2 = p_v2[edge_offset : edge_offset + show_pts]
                    t_ms = np.linspace(0, show_duration_ms, len(plot_v1))

                    if active_tab == 0:  # Tab 1: Dual Audio Scope
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

                    elif active_tab == 1:  # Tab 2: Dual Audio FFT
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

                    elif active_tab == 2:  # Tab 3: Mic 1 View
                        with self.fig_mic1_view.batch_update():
                            self.fig_mic1_view.data[0].x = t_ms
                            self.fig_mic1_view.data[0].y = plot_v1
                            self.fig_mic1_view.data[1].x = [0, show_duration_ms]
                            self.fig_mic1_view.data[1].y = [trig_v, trig_v]
                            self.fig_mic1_view.data[2].x = freqs
                            self.fig_mic1_view.data[2].y = mag_a0
                            self.fig_mic1_view.layout.xaxis.range = [0, show_duration_ms]
                            self.fig_mic1_view.layout.xaxis2.range = [0, max_span]

                    elif active_tab == 3:  # Tab 4: Mic 2 View
                        with self.fig_mic2_view.batch_update():
                            self.fig_mic2_view.data[0].x = t_ms
                            self.fig_mic2_view.data[0].y = plot_v2
                            self.fig_mic2_view.data[1].x = [0, show_duration_ms]
                            self.fig_mic2_view.data[1].y = [trig_v, trig_v]
                            self.fig_mic2_view.data[2].x = freqs
                            self.fig_mic2_view.data[2].y = mag_a1
                            self.fig_mic2_view.layout.xaxis.range = [0, show_duration_ms]
                            self.fig_mic2_view.layout.xaxis2.range = [0, max_span]

                    # Status VU String with Clipping indicator
                    c1_tag = " <span style='color:#FF0000;'>(CLIP)</span>" if clip1 else ""
                    c2_tag = " <span style='color:#FF0000;'>(CLIP)</span>" if clip2 else ""
                    mode_tag = " (LOCKED)" if mode == "Single" else ""

                    self.readout_vu.value = (
                        f"<span style='color:#00FFCC; font-family:monospace; font-size:13px; font-weight:bold;'>"
                        f"Mic 1 (A0): {vu1_str} {vpp1:.2f}V{c1_tag} | Mic 2 (A1): {vu2_str} {vpp2:.2f}V{c2_tag} | Pitch f0: {p_f1:.1f} Hz{mode_tag}"
                        f"</span>"
                    )

                    if mode == "Single": self._single_done = True
                    time.sleep(0.033)
                else:
                    time.sleep(0.005)

        finally:
            self._is_running = False
            buf_time.close()
            print("[AudioDashboard] Stopped cleanly.")

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
        r1 = widgets.HBox([self.start_btn, self.stop_btn, self.force_btn, self.autorange_toggle, self.clear_log_btn, self.readout_vu], layout=widgets.Layout(gap="10px", margin="0 0 8px 0"))
        r2 = widgets.HBox([self.trig_mode_dd, self.trig_edge_dd, self.trig_src_dd, self.trig_level_slider, self.trig_level_input])
        r3 = widgets.HBox([self.fft_unit_dd, self.fft_span_dd])
        self.control_panel = widgets.VBox([r1, r2, r3], layout=widgets.Layout(margin="0 0 12px 0"))
        display(widgets.VBox([self.control_panel, self.tabs]))