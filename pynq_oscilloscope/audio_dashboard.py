"""
pynq_oscilloscope.audio_dashboard: Interactive Real-Time Audio & Microphone Instrument.
Tailored for passive dual microphones (MAX4466), featuring 50 kSPS decimated streaming,
40.96 ms multi-cycle timebases, real-time clipping detection, and sub-Hertz peak tracking.
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
    Dedicated 4-Tab Audio & Microphone Instrument for PYNQ-Z2.
    Operates on the 50 kSPS decimated audio stream (40.96 ms observation window).
    """

    def __init__(
        self,
        overlay=None,
        packet_size: int = 2048,
        fft_points: int = 2048
    ):
        self.overlay = overlay
        self.packet_size = packet_size
        self.num_pts_per_ch = packet_size // 2  # 1024 samples per channel
        self.fs_per_ch = 50_000.0               # 50 kSPS decimated audio rate
        self.fft_points = fft_points
        self.total_duration_ms = (self.num_pts_per_ch / self.fs_per_ch) * 1000.0  # 40.96 ms
        
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
        # 1. Action Row & Metrics
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

        # 2. Hardware Trigger Controls (Defaults to MAX4466 1.65V baseline)
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

        # 3. Audio Spectral & Windowing Controls
        self.fft_unit_dd = widgets.Dropdown(
            options=["dBV", "dBFS", "Linear"], value="dBV", description="FFT Unit:", layout=widgets.Layout(width="170px")
        )
        self.fft_span_dd = widgets.Dropdown(
            options=[("Full Audio (25 kHz)", 25000), ("Speech (8 kHz)", 8000), ("Bass Zoom (2 kHz)", 2000), ("Sub-Bass (500 Hz)", 500)],
            value=8000, description="Audio Span:", layout=widgets.Layout(width="220px")
        )
        self.window_dd = widgets.Dropdown(
            options=["Hann", "Rectangular"], value="Hann", description="Windowing:", layout=widgets.Layout(width="180px")
        )

    def _build_plots(self):
        time_ms = np.linspace(0, self.total_duration_ms, self.num_pts_per_ch)
        initial_freq = np.linspace(0, 25000, self.num_pts_per_ch // 2 + 1)

        # Tab 1: Dual Audio Scope (A0 Top, A1 Bottom in Milliseconds)
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

        # Tab 2: Dual Audio Spectrum (0 to 25 kHz)
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
        self.fig_dual_fft.update_xaxes(range=[0, 8000], title="Frequency (Hz)", row=2, col=1)

        # Tab 3: Mic 1 View (A0 Time + Spectrum)
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
        self.fig_ch1_view.update_xaxes(range=[0, 8000], title="Frequency (Hz)", row=2, col=1)

        # Tab 4: Mic 2 View (A1 Time + Spectrum)
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
        self.fig_ch2_view.update_xaxes(range=[0, 8000], title="Frequency (Hz)", row=2, col=1)

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

        self.trig_level_slider.observe(lambda _: self._update_trig_level(), names="value")
        self.trig_mode_dd.observe(self._on_trig_param_change, names="value")
        self.trig_edge_dd.observe(self._on_trig_param_change, names="value")
        self.trig_src_dd.observe(self._on_trig_param_change, names="value")

    def _update_trig_level(self):
        if self.trigger:
            self.trigger.set_threshold(float(self.trig_level_slider.value))

    def _get_arm_control_word(self) -> int:
        is_falling = (self.trig_edge_dd.value == "Falling")
        is_ch1 = ("CH1" in self.trig_src_dd.value or "A0" in self.trig_src_dd.value)
        mode = self.trig_mode_dd.value
        
        ctrl = (1 << 0) | (1 << 3)  # Bit 0: Arm, Bit 3: Single Shot
        if is_falling:
            ctrl |= (1 << 2)        # Bit 2: Falling Edge
        if mode == "Auto":
            ctrl |= (1 << 1)        # Bit 1: Auto Timeout
        if is_ch1:
            ctrl |= (1 << 5)        # Bit 5: 1 = CH1 (A0), 0 = CH2 (A1)
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
        """Dedicated Passive Microphone Acquisition Loop (50 kSPS)."""
        dma_time = self.overlay.axi_dma_0
        dma_fft = self.overlay.axi_dma_1
        trig = self.trigger

        # 1. Initialize XADC continuous sequencer for Vaux1 (A0) and Vaux9 (A1)
        if hasattr(self.overlay, "xadc_wiz_0"):
            xadc = self.overlay.xadc_wiz_0.mmio
            xadc.write(0x304, 0x2000)  # DRP 0x41 = 0x2000 (Continuous Sequence Mode)
            xadc.write(0x320, 0x0000)  # DRP 0x48 = Disable internal channels
            xadc.write(0x324, 0x0202)  # DRP 0x49 = Enable Vaux1 (bit 1) and Vaux9 (bit 9)

        # 2. Reset DMAs
        dma_time.mmio.write(0x30, 0x04)
        dma_fft.mmio.write(0x30, 0x04)
        time.sleep(0.01)
        dma_time.recvchannel.start()
        dma_fft.recvchannel.start()

        # 3. Configure Trigger
        if trig:
            trig.mmio.write(0x0C, 5000000)  # 50ms Auto-Timeout
            self._update_trig_level()

        print(f"[AudioDashboard] Active (50 kSPS Audio Rate | Window: {self.total_duration_ms:.2f} ms)")

        buf_time = allocate(shape=(self.packet_size,), dtype="u2")
        buf_fft = allocate(shape=(self.packet_size,), dtype="u2")
        dma_armed = False

        time_ms = np.linspace(0, self.total_duration_ms, self.num_pts_per_ch)
        freqs = np.fft.rfftfreq(self.num_pts_per_ch, d=1.0 / self.fs_per_ch)

        try:
            while self._is_running:
                mode = self.trig_mode_dd.value

                if mode == "Single" and self._single_done:
                    time.sleep(0.02)
                    continue

                # 4. Queue BOTH DMAs FIRST
                if not dma_armed:
                    dma_time.recvchannel.transfer(buf_time)
                    dma_fft.recvchannel.transfer(buf_fft)
                    trig.mmio.write(0x00, self._get_arm_control_word())
                    dma_armed = True

                # 5. Check hardware completion
                if dma_time.recvchannel.idle and dma_fft.recvchannel.idle:
                    dma_armed = False

                    # De-interleave: Even = A0 (Mic 1), Odd = A1 (Mic 2)
                    raw_time = np.array(buf_time)
                    v_a0 = (raw_time[0::2] >> 4) * (3.3 / 4095.0)
                    v_a1 = (raw_time[1::2] >> 4) * (3.3 / 4095.0)

                    # DC Baseline Removal (Mean subtraction for pure zero-centered audio)
                    ac_a0 = v_a0 - np.mean(v_a0)
                    ac_a1 = v_a1 - np.mean(v_a1)

                    # Apply Windowing
                    if self.window_dd.value == "Hann":
                        win = np.hanning(self.num_pts_per_ch)
                        sig_a0 = ac_a0 * win
                        sig_a1 = ac_a1 * win
                    else:
                        sig_a0 = ac_a0
                        sig_a1 = ac_a1

                    # Compute Fast Spectrum
                    fft_a0 = np.abs(np.fft.rfft(sig_a0)) / (self.num_pts_per_ch / 2.0)
                    fft_a1 = np.abs(np.fft.rfft(sig_a1)) / (self.num_pts_per_ch / 2.0)

                    unit = self.fft_unit_dd.value
                    if unit == "dBV":
                        mag_a0 = 20.0 * np.log10(np.maximum(fft_a0, 1e-6))
                        mag_a1 = 20.0 * np.log10(np.maximum(fft_a1, 1e-6))
                    elif unit == "dBFS":
                        mag_a0 = 20.0 * np.log10(np.maximum(fft_a0 / 1.65, 1e-6))
                        mag_a1 = 20.0 * np.log10(np.maximum(fft_a1 / 1.65, 1e-6))
                    else:
                        mag_a0 = fft_a0 * 1000.0  # mV
                        mag_a1 = fft_a1 * 1000.0

                    vpp1 = float(np.max(v_a0) - np.min(v_a0))
                    vpp2 = float(np.max(v_a1) - np.min(v_a1))

                    # Sub-bin Quadratic Peak Interpolation
                    peak_f1, peak_m1 = StreamingFFT.get_peak_frequency(freqs, mag_a0, min_freq_hz=20.0)
                    peak_f2, peak_m2 = StreamingFFT.get_peak_frequency(freqs, mag_a1, min_freq_hz=20.0)

                    # Clipping / Saturation Detectors (MAX4466 rails: < 0.10V or > 3.10V)
                    clip1 = (np.min(v_a0) < 0.10 or np.max(v_a0) > 3.10)
                    clip2 = (np.min(v_a1) < 0.10 or np.max(v_a1) > 3.10)
                    tag_clip1 = " <span style='color:#FF0000; font-weight:bold;'>[CLIPPING / TRIM GAIN!]</span>" if clip1 else " <span style='color:#00FFCC;'>[OK]</span>"
                    tag_clip2 = " <span style='color:#FF0000; font-weight:bold;'>[CLIPPING / TRIM GAIN!]</span>" if clip2 else " <span style='color:#FF007F;'>[OK]</span>"

                    trig_v = float(self.trig_level_slider.value)
                    active_tab = self.tabs.selected_index
                    max_span = float(self.fft_span_dd.value)
                    is_trig_a0 = ("CH1" in self.trig_src_dd.value or "A0" in self.trig_src_dd.value)

                    # Update Active Tab
                    if active_tab == 0:  # Tab 1: Dual Audio Scope
                        with self.fig_dual_scope.batch_update():
                            self.fig_dual_scope.data[0].y = v_a0
                            self.fig_dual_scope.data[2].y = v_a1
                            if is_trig_a0:
                                self.fig_dual_scope.data[1].y = [trig_v, trig_v]
                                self.fig_dual_scope.data[1].visible = True
                                self.fig_dual_scope.data[3].visible = False
                            else:
                                self.fig_dual_scope.data[3].y = [trig_v, trig_v]
                                self.fig_dual_scope.data[3].visible = True
                                self.fig_dual_scope.data[1].visible = False

                    elif active_tab == 1:  # Tab 2: Dual Audio FFT
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

                    elif active_tab == 2:  # Tab 3: Mic 1 View
                        with self.fig_ch1_view.batch_update():
                            self.fig_ch1_view.data[0].y = v_a0
                            self.fig_ch1_view.data[1].y = [trig_v, trig_v]
                            self.fig_ch1_view.data[2].x = freqs
                            self.fig_ch1_view.data[2].y = mag_a0
                            self.fig_ch1_view.layout.xaxis2.range = [0, max_span]

                    elif active_tab == 3:  # Tab 4: Mic 2 View
                        with self.fig_ch2_view.batch_update():
                            self.fig_ch2_view.data[0].y = v_a1
                            self.fig_ch2_view.data[1].y = [trig_v, trig_v]
                            self.fig_ch2_view.data[2].x = freqs
                            self.fig_ch2_view.data[2].y = mag_a1
                            self.fig_ch2_view.layout.xaxis2.range = [0, max_span]

                    # Status Bar Readouts with Clip Indicators
                    mode_tag = " (LOCKED)" if mode == "Single" else ""
                    self.readout_ch1.value = (
                        f"<span style='color:#00FFCC; font-family:monospace; font-size:13px; font-weight:bold;'>"
                        f"A0 (Mic 1): Vpp={vpp1:.2f}V | f0={peak_f1:.1f}Hz{tag_clip1}{mode_tag}</span>"
                    )
                    self.readout_ch2.value = (
                        f"<span style='color:#FF007F; font-family:monospace; font-size:13px; font-weight:bold;'>"
                        f"A1 (Mic 2): Vpp={vpp2:.2f}V | f0={peak_f2:.1f}Hz{tag_clip2}</span>"
                    )

                    if mode == "Single":
                        self._single_done = True

                    time.sleep(0.033)  # Target ~30 FPS
                else:
                    time.sleep(0.005)

        finally:
            self._is_running = False
            buf_time.close()
            buf_fft.close()
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
        r2 = widgets.HBox([self.trig_mode_dd, self.trig_edge_dd, self.trig_src_dd, self.trig_level_slider, self.trig_level_input])
        r3 = widgets.HBox([self.fft_unit_dd, self.fft_span_dd, self.window_dd])
        self.control_panel = widgets.VBox([r1, r2, r3], layout=widgets.Layout(margin="0 0 12px 0"))
        display(widgets.VBox([self.control_panel, self.tabs]))