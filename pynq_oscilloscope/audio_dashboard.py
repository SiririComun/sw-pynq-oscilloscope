"""
pynq_oscilloscope.audio_dashboard: Interactive Real-Time Multi-Regime Audio & Microphone Instrument.
Features dynamic on-the-fly switching between Full Audio (50 kSPS), Speech (25 kSPS),
Deep Bass Zoom (10 kSPS, Δf = 4.88 Hz), and Wideband Scope (500 kSPS).
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


class AudioDashboard:
    """
    Dedicated Multi-Regime Audio & Microphone Instrument for PYNQ-Z2.
    Supports dynamic decimation (M=1, 10, 20, 50) and multi-window spectral analysis.
    """

    def __init__(
        self,
        overlay=None,
        packet_size: int = 2048,
        fft_points: int = 2048
    ):
        self.overlay = overlay
        self.packet_size = packet_size
        self.fft_points = fft_points
        self.fs_per_ch = 50_000.0  # Default Audio (50 kSPS)
        self.num_pts_per_ch = packet_size // 2
        self.total_duration_ms = (self.num_pts_per_ch / self.fs_per_ch) * 1000.0
        
        self._is_running = False
        self._single_done = False
        self._thread: Optional[threading.Thread] = None

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
        # 1. Action Row & Status Metrics
        self.start_btn = widgets.Button(
            description="Start Audio", button_style="success", icon="play", layout=widgets.Layout(width="120px")
        )
        self.stop_btn = widgets.Button(
            description="Stop", button_style="danger", icon="stop", layout=widgets.Layout(width="90px")
        )
        self.force_btn = widgets.Button(
            description="Force / Arm", button_style="warning", icon="bolt", layout=widgets.Layout(width="115px")
        )
        self.clear_log_btn = widgets.Button(
            description="Clear Log", button_style="", icon="trash", layout=widgets.Layout(width="100px")
        )

        self.readout_ch1 = widgets.HTML(
            "<span style='color:#00FFCC; font-family:monospace; font-size:13px; font-weight:bold;'>"
            "A0 (Mic 1): Vpp=0.00V | f0=0.0Hz [OK]</span>"
        )
        self.readout_ch2 = widgets.HTML(
            "<span style='color:#FF007F; font-family:monospace; font-size:13px; font-weight:bold;'>"
            "A1 (Mic 2): Vpp=0.00V | f0=0.0Hz [OK]</span>"
        )

        # 2. Operating Regime & Multi-Window Controls
        self.profile_dd = widgets.Dropdown(
            options=[
                ("🎙 Full-Band Audio (50 kSPS)", "audio"),
                ("🗣 Speech / Vocal (25 kSPS)", "speech"),
                ("🎸 Deep Bass Zoom (10 kSPS, Δf=4.88Hz)", "bass_zoom"),
                ("📈 Wideband Scope (500 kSPS)", "oscilloscope")
            ],
            value="audio",
            description="Regime:",
            layout=widgets.Layout(width="300px")
        )

        self.window_dd = widgets.Dropdown(
            options=["Hann", "Hamming", "Blackman", "Flat-Top", "Rectangular"],
            value="Hann",
            description="Window:",
            layout=widgets.Layout(width="180px")
        )

        # 3. Trigger Controls
        self.trig_mode_dd = widgets.Dropdown(
            options=["Auto", "Normal", "Single"], value="Auto", description="Trig Mode:", layout=widgets.Layout(width="180px")
        )
        self.trig_edge_dd = widgets.Dropdown(
            options=["Rising", "Falling"], value="Rising", description="Trig Edge:", layout=widgets.Layout(width="180px")
        )
        self.trig_src_dd = widgets.Dropdown(
            options=["CH1 (A0)", "CH2 (A1)"], value="CH1 (A0)", description="Trig Source:", layout=widgets.Layout(width="190px")
        )
        self.trig_level_slider = widgets.FloatSlider(
            value=1.65, min=0.0, max=3.3, step=0.05, description="Trig Level:", continuous_update=False, layout=widgets.Layout(width="220px")
        )
        self.trig_level_input = widgets.BoundedFloatText(
            value=1.65, min=0.0, max=3.3, step=0.05, layout=widgets.Layout(width="80px")
        )
        widgets.jslink((self.trig_level_slider, "value"), (self.trig_level_input, "value"))

        # 4. Spectral Controls
        self.fft_unit_dd = widgets.Dropdown(
            options=["dBV", "dBFS", "Linear"], value="dBV", description="FFT Unit:", layout=widgets.Layout(width="170px")
        )
        self.fft_span_dd = widgets.Dropdown(
            options=[("Auto Full Span", 0), ("8 kHz (Speech)", 8000), ("2 kHz (Bass Zoom)", 2000), ("500 Hz (Sub-Bass)", 500)],
            value=0, description="Audio Span:", layout=widgets.Layout(width="220px")
        )

    def _build_plots(self):
        time_ms = np.linspace(0, self.total_duration_ms, self.num_pts_per_ch)
        initial_freq = np.linspace(0, 25000, self.num_pts_per_ch // 2 + 1)

        # Tab 1: Dual Audio Scope
        self.fig_dual_scope = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
            subplot_titles=("<b>Channel 1: A0 (Mic 1 Audio)</b>", "<b>Channel 2: A1 (Mic 2 Audio)</b>")
        )
        self.fig_dual_scope = go.FigureWidget(self.fig_dual_scope)
        self.fig_dual_scope.add_scatter(x=time_ms, y=[1.65]*len(time_ms), mode="lines", line=dict(color="#00FFCC", width=1.8), name="Mic 1 (A0)", row=1, col=1)
        self.fig_dual_scope.add_scatter(x=[0, self.total_duration_ms], y=[1.65, 1.65], mode="lines", line=dict(color="#FFA500", width=1.2, dash="dash"), name="Trigger (A0)", row=1, col=1)
        self.fig_dual_scope.add_scatter(x=time_ms, y=[1.65]*len(time_ms), mode="lines", line=dict(color="#FF007F", width=1.8), name="Mic 2 (A1)", row=2, col=1)
        self.fig_dual_scope.add_scatter(x=[0, self.total_duration_ms], y=[1.65, 1.65], mode="lines", line=dict(color="#FFA500", width=1.2, dash="dash"), name="Trigger (A1)", visible=False, row=2, col=1)
        self.fig_dual_scope.update_layout(template="plotly_dark", height=500, margin=dict(l=40, r=20, t=45, b=35), uirevision="a1")
        self.fig_dual_scope.update_yaxes(range=[0, 3.3], title="Voltage (V)", row=1, col=1)
        self.fig_dual_scope.update_yaxes(range=[0, 3.3], title="Voltage (V)", row=2, col=1)
        self.fig_dual_scope.update_xaxes(range=[0, self.total_duration_ms], title="Time (Milliseconds)", row=2, col=1)

        # Tab 2: Dual Audio Spectrum
        self.fig_dual_fft = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
            subplot_titles=("<b>Channel 1: A0 Audio Spectrum (FFT)</b>", "<b>Channel 2: A1 Audio Spectrum (FFT)</b>")
        )
        self.fig_dual_fft = go.FigureWidget(self.fig_dual_fft)
        self.fig_dual_fft.add_scatter(x=initial_freq, y=[-100]*len(initial_freq), mode="lines", line=dict(color="#00FFCC", width=1.8), name="A0 FFT", row=1, col=1)
        self.fig_dual_fft.add_scatter(x=[440], y=[-40], mode="markers+text", marker=dict(color="#FFA500", size=7, symbol="diamond"), text=["Peak"], textposition="top center", name="Peak 1", row=1, col=1)
        self.fig_dual_fft.add_scatter(x=initial_freq, y=[-100]*len(initial_freq), mode="lines", line=dict(color="#FF007F", width=1.8), name="A1 FFT", row=2, col=1)
        self.fig_dual_fft.add_scatter(x=[440], y=[-40], mode="markers+text", marker=dict(color="#FFA500", size=7, symbol="diamond"), text=["Peak"], textposition="top center", name="Peak 2", row=2, col=1)
        self.fig_dual_fft.update_layout(template="plotly_dark", height=500, margin=dict(l=40, r=20, t=45, b=35), uirevision="a2")
        self.fig_dual_fft.update_yaxes(range=[-100, 0], title="Mag (dBV)", row=1, col=1)
        self.fig_dual_fft.update_yaxes(range=[-100, 0], title="Mag (dBV)", row=2, col=1)
        self.fig_dual_fft.update_xaxes(range=[0, 25000], title="Frequency (Hz)", row=2, col=1)

        # Tab 3: Mic 1 View
        self.fig_ch1_view = make_subplots(
            rows=2, cols=1, vertical_spacing=0.15,
            subplot_titles=("<b>Mic 1: A0 (Time Domain)</b>", "<b>Mic 1: A0 (Frequency Domain)</b>")
        )
        self.fig_ch1_view = go.FigureWidget(self.fig_ch1_view)
        self.fig_ch1_view.add_scatter(x=time_ms, y=[1.65]*len(time_ms), mode="lines", line=dict(color="#00FFCC", width=1.8), name="Mic 1 Time", row=1, col=1)
        self.fig_ch1_view.add_scatter(x=[0, self.total_duration_ms], y=[1.65, 1.65], mode="lines", line=dict(color="#FFA500", width=1.2, dash="dash"), name="Trigger", row=1, col=1)
        self.fig_ch1_view.add_scatter(x=initial_freq, y=[-100]*len(initial_freq), mode="lines", line=dict(color="#00FFCC", width=1.8), name="Mic 1 FFT", row=2, col=1)
        self.fig_ch1_view.update_layout(template="plotly_dark", height=500, margin=dict(l=40, r=20, t=45, b=35), uirevision="a3")
        self.fig_ch1_view.update_yaxes(range=[0, 3.3], title="Voltage (V)", row=1, col=1)
        self.fig_ch1_view.update_yaxes(range=[-100, 0], title="Mag (dBV)", row=2, col=1)
        self.fig_ch1_view.update_xaxes(range=[0, self.total_duration_ms], title="Time (ms)", row=1, col=1)
        self.fig_ch1_view.update_xaxes(range=[0, 25000], title="Frequency (Hz)", row=2, col=1)

        # Tab 4: Mic 2 View
        self.fig_ch2_view = make_subplots(
            rows=2, cols=1, vertical_spacing=0.15,
            subplot_titles=("<b>Mic 2: A1 (Time Domain)</b>", "<b>Mic 2: A1 (Frequency Domain)</b>")
        )
        self.fig_ch2_view = go.FigureWidget(self.fig_ch2_view)
        self.fig_ch2_view.add_scatter(x=time_ms, y=[1.65]*len(time_ms), mode="lines", line=dict(color="#FF007F", width=1.8), name="Mic 2 Time", row=1, col=1)
        self.fig_ch2_view.add_scatter(x=[0, self.total_duration_ms], y=[1.65, 1.65], mode="lines", line=dict(color="#FFA500", width=1.2, dash="dash"), name="Trigger", row=1, col=1)
        self.fig_ch2_view.add_scatter(x=initial_freq, y=[-100]*len(initial_freq), mode="lines", line=dict(color="#FF007F", width=1.8), name="Mic 2 FFT", row=2, col=1)
        self.fig_ch2_view.update_layout(template="plotly_dark", height=500, margin=dict(l=40, r=20, t=45, b=35), uirevision="a4")
        self.fig_ch2_view.update_yaxes(range=[0, 3.3], title="Voltage (V)", row=1, col=1)
        self.fig_ch2_view.update_yaxes(range=[-100, 0], title="Mag (dBV)", row=2, col=1)
        self.fig_ch2_view.update_xaxes(range=[0, self.total_duration_ms], title="Time (ms)", row=1, col=1)
        self.fig_ch2_view.update_xaxes(range=[0, 25000], title="Frequency (Hz)", row=2, col=1)

        self.tabs = widgets.Tab(children=[self.fig_dual_scope, self.fig_dual_fft, self.fig_ch1_view, self.fig_ch2_view])
        self.tabs.set_title(0, "🎙 Dual Audio Scope (A0 & A1)")
        self.tabs.set_title(1, "📊 Dual Audio FFT (A0 & A1)")
        self.tabs.set_title(2, "🔀 Mic 1: A0")
        self.tabs.set_title(3, "🔀 Mic 2: A1")

    def _setup_callbacks(self):
        self.start_btn.on_click(lambda _: self.start())
        self.stop_btn.on_click(lambda _: self.stop())
        self.force_btn.on_click(lambda _: self._on_force_clicked())
        self.clear_log_btn.on_click(lambda _: self._on_clear_log_clicked())

        self.profile_dd.observe(self._on_profile_change, names="value")
        self.trig_level_slider.observe(lambda _: self._update_trig_level(), names="value")
        self.trig_mode_dd.observe(self._on_trig_param_change, names="value")
        self.trig_edge_dd.observe(self._on_trig_param_change, names="value")
        self.trig_src_dd.observe(self._on_trig_param_change, names="value")

    def _on_profile_change(self, _):
        if self.overlay:
            info = self.overlay.set_profile(mode=self.profile_dd.value)
            self.fs_per_ch = info["sample_rate_hz"]
            self.total_duration_ms = info["time_window_ms"]
            print(f"[AudioDashboard] Switched to profile '{info['mode'].upper()}': fs={self.fs_per_ch/1e3:.1f}kSPS, Δf={info['delta_f_hz']:.2f}Hz, Window={self.total_duration_ms:.1f}ms")

    def _update_trig_level(self):
        if self.trigger:
            self.trigger.set_threshold(float(self.trig_level_slider.value))

    def _get_arm_control_word(self) -> int:
        is_falling = (self.trig_edge_dd.value == "Falling")
        is_ch1 = ("CH1" in self.trig_src_dd.value or "A0" in self.trig_src_dd.value)
        mode = self.trig_mode_dd.value
        
        ctrl = (1 << 0) | (1 << 3)
        if is_falling:
            ctrl |= (1 << 2)
        if mode == "Auto":
            ctrl |= (1 << 1)
        if is_ch1:
            ctrl |= (1 << 5)
        return ctrl

    def _on_trig_param_change(self, _):
        if self.trig_mode_dd.value != "Single":
            self._single_done = False
        if self._is_running and self.trigger:
            self.trigger.mmio.write(0x00, self._get_arm_control_word())

    def _on_force_clicked(self):
        if self.trig_mode_dd.value == "Single":
            self._single_done = False
        if self.trigger:
            self.trigger.force_trigger()

    def _on_clear_log_clicked(self):
        clear_output(wait=True)
        display(widgets.VBox([self.control_panel, self.tabs]))

    def _update_loop(self):
        """Dedicated Multi-Regime Audio Acquisition Loop."""
        dma_time = self.overlay.axi_dma_0
        trig = self.trigger

        # 1. Initialize XADC continuous sequencer
        if hasattr(self.overlay, "xadc_wiz_0"):
            xadc = self.overlay.xadc_wiz_0.mmio
            xadc.write(0x304, 0x2000)
            xadc.write(0x320, 0x0000)
            xadc.write(0x324, 0x0202)

        # 2. Reset DMA 0
        dma_time.mmio.write(0x30, 0x04)
        time.sleep(0.005)
        dma_time.recvchannel.start()

        # 3. Configure Trigger
        if trig:
            trig.mmio.write(0x0C, 5000000)  # 50ms Auto-Timeout
            self._update_trig_level()

        # Apply active profile
        self._on_profile_change(None)

        buf_time = allocate(shape=(self.packet_size,), dtype="u2")
        dma_armed = False

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

                    # Crop edge boundary words
                    if len(v_a0) > 16:
                        p_v1 = v_a0[8:-8]
                        p_v2 = v_a1[8:-8]
                    else:
                        p_v1 = v_a0
                        p_v2 = v_a1

                    # Compute Multi-Window Spectrum
                    win_name = self.window_dd.value.lower().replace("-", "")
                    freqs, mag_a0 = self.fft.compute_spectrum(p_v1, unit=self.fft_unit_dd.value, window_type=win_name)
                    _, mag_a1     = self.fft.compute_spectrum(p_v2, unit=self.fft_unit_dd.value, window_type=win_name)

                    vpp1 = float(np.ptp(p_v1))
                    vpp2 = float(np.ptp(p_v2))

                    peak_f1, peak_m1 = StreamingFFT.get_peak_frequency(freqs, mag_a0, min_freq_hz=10.0)
                    peak_f2, peak_m2 = StreamingFFT.get_peak_frequency(freqs, mag_a1, min_freq_hz=10.0)

                    # Clipping Detection (<0.10V or >3.10V)
                    clip1 = (np.min(p_v1) < 0.10 or np.max(p_v1) > 3.10)
                    clip2 = (np.min(p_v2) < 0.10 or np.max(p_v2) > 3.10)
                    tag_clip1 = " <span style='color:#FF0000; font-weight:bold;'>[CLIP!]</span>" if clip1 else " <span style='color:#00FFCC;'>[OK]</span>"
                    tag_clip2 = " <span style='color:#FF0000; font-weight:bold;'>[CLIP!]</span>" if clip2 else " <span style='color:#FF007F;'>[OK]</span>"

                    trig_v = float(self.trig_level_slider.value)
                    active_tab = self.tabs.selected_index
                    
                    user_span = float(self.fft_span_dd.value)
                    max_span = user_span if user_span > 0 else (self.fs_per_ch / 2.0)
                    t_x = np.linspace(0, self.total_duration_ms, len(p_v1))
                    is_trig_a0 = ("CH1" in self.trig_src_dd.value or "A0" in self.trig_src_dd.value)

                    # Update Active Tab
                    if active_tab == 0:  # Tab 1: Scope
                        with self.fig_dual_scope.batch_update():
                            self.fig_dual_scope.data[0].x = t_x
                            self.fig_dual_scope.data[0].y = p_v1
                            self.fig_dual_scope.data[2].x = t_x
                            self.fig_dual_scope.data[2].y = p_v2
                            if is_trig_a0:
                                self.fig_dual_scope.data[1].x = [0, self.total_duration_ms]
                                self.fig_dual_scope.data[1].y = [trig_v, trig_v]
                                self.fig_dual_scope.data[1].visible = True
                                self.fig_dual_scope.data[3].visible = False
                            else:
                                self.fig_dual_scope.data[3].x = [0, self.total_duration_ms]
                                self.fig_dual_scope.data[3].y = [trig_v, trig_v]
                                self.fig_dual_scope.data[3].visible = True
                                self.fig_dual_scope.data[1].visible = False
                            self.fig_dual_scope.layout.xaxis2.range = [0, self.total_duration_ms]

                    elif active_tab == 1:  # Tab 2: FFT
                        with self.fig_dual_fft.batch_update():
                            self.fig_dual_fft.data[0].x = freqs
                            self.fig_dual_fft.data[0].y = mag_a0
                            self.fig_dual_fft.data[1].x = [peak_f1]
                            self.fig_dual_fft.data[1].y = [peak_m1]
                            self.fig_dual_fft.data[1].text = [f" {peak_f1:.1f} Hz"]
                            self.fig_dual_fft.data[2].x = freqs
                            self.fig_dual_fft.data[2].y = mag_a1
                            self.fig_dual_fft.data[3].x = [peak_f2]
                            self.fig_dual_fft.data[3].y = [peak_m2]
                            self.fig_dual_fft.data[3].text = [f" {peak_f2:.1f} Hz"]
                            self.fig_dual_fft.layout.xaxis2.range = [0, max_span]

                    elif active_tab == 2:  # Tab 3: Mic 1
                        with self.fig_ch1_view.batch_update():
                            self.fig_ch1_view.data[0].x = t_x
                            self.fig_ch1_view.data[0].y = p_v1
                            self.fig_ch1_view.data[1].x = [0, self.total_duration_ms]
                            self.fig_ch1_view.data[1].y = [trig_v, trig_v]
                            self.fig_ch1_view.data[2].x = freqs
                            self.fig_ch1_view.data[2].y = mag_a0
                            self.fig_ch1_view.layout.xaxis.range = [0, self.total_duration_ms]
                            self.fig_ch1_view.layout.xaxis2.range = [0, max_span]

                    elif active_tab == 3:  # Tab 4: Mic 2
                        with self.fig_ch2_view.batch_update():
                            self.fig_ch2_view.data[0].x = t_x
                            self.fig_ch2_view.data[0].y = p_v2
                            self.fig_ch2_view.data[1].x = [0, self.total_duration_ms]
                            self.fig_ch2_view.data[1].y = [trig_v, trig_v]
                            self.fig_ch2_view.data[2].x = freqs
                            self.fig_ch2_view.data[2].y = mag_a1
                            self.fig_ch2_view.layout.xaxis.range = [0, self.total_duration_ms]
                            self.fig_ch2_view.layout.xaxis2.range = [0, max_span]

                    # Metric Readouts
                    mode_tag = " (LOCKED)" if mode == "Single" else ""
                    self.readout_ch1.value = (
                        f"<span style='color:#00FFCC; font-family:monospace; font-size:13px; font-weight:bold;'>"
                        f"A0: Vpp={vpp1:.2f}V | f0={peak_f1:.1f}Hz{tag_clip1}{mode_tag}</span>"
                    )
                    self.readout_ch2.value = (
                        f"<span style='color:#FF007F; font-family:monospace; font-size:13px; font-weight:bold;'>"
                        f"A1: Vpp={vpp2:.2f}V | f0={peak_f2:.1f}Hz{tag_clip2}</span>"
                    )

                    if mode == "Single":
                        self._single_done = True

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
        r1 = widgets.HBox(
            [self.start_btn, self.stop_btn, self.force_btn, self.clear_log_btn, self.readout_ch1, self.readout_ch2],
            layout=widgets.Layout(gap="10px", margin="0 0 8px 0")
        )
        r2 = widgets.HBox([self.profile_dd, self.window_dd, self.fft_unit_dd, self.fft_span_dd])
        r3 = widgets.HBox([self.trig_mode_dd, self.trig_edge_dd, self.trig_src_dd, self.trig_level_slider, self.trig_level_input])
        self.control_panel = widgets.VBox([r1, r2, r3], layout=widgets.Layout(margin="0 0 12px 0"))
        display(widgets.VBox([self.control_panel, self.tabs]))