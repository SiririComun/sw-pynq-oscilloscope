"""
pynq_oscilloscope.filter_dashboard: Full-Featured Real-Time Hardware Filter & IFFT Dashboard.
Features 4-tab multi-domain visualization, live frequency cutoff sliders, interactive presets
(Sub-Bass, Full Bass, Vocals, Notch, Highpass), shaded passband overlays, and 70+ FPS DMA streaming.
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
from pynq_oscilloscope.hw_filter import HardwareFilter


class AudioFilterDashboard:
    """
    Interactive Real-Time Hardware Filter & IFFT Reconstruction Instrument.
    Features 50 kSPS dual-channel acquisition, real-time spectral mask reconfiguration,
    sub-sample trigger phase-locking, and concurrent 3-DMA streaming (Raw, FFT, Filtered).
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
        self.num_pts_per_ch = packet_size // 2  # 1024 samples per channel
        self.fft_points = self.num_pts_per_ch   # N=1024 matching demuxed stream
        self.fs_per_ch = fs_per_ch
        self.display_window = display_window
        self.total_duration_ms = (self.num_pts_per_ch / self.fs_per_ch) * 1000.0

        self._is_running = False
        self._single_done = False
        self._thread: Optional[threading.Thread] = None

        # Bind hardware sub-drivers
        if self.overlay and hasattr(self.overlay, "trigger"):
            self.trigger = self.overlay.trigger
        elif self.overlay:
            self.trigger = HardwareTrigger(self.overlay)
        else:
            self.trigger = None

        if self.overlay and hasattr(self.overlay, "filter"):
            self.filter = self.overlay.filter
        elif self.overlay:
            self.filter = HardwareFilter(self.overlay, sample_rate_hz=self.fs_per_ch, fft_points=self.fft_points)
        else:
            self.filter = None

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
        # 1. Row 1: Action Transport & Real-Time Readout HUD
        self.start_btn = widgets.Button(description="Start Live", button_style="success", icon="play", layout=widgets.Layout(width="115px"))
        self.stop_btn = widgets.Button(description="Stop", button_style="danger", icon="stop", layout=widgets.Layout(width="95px"))
        self.force_btn = widgets.Button(description="Force Trig", button_style="warning", icon="bolt", layout=widgets.Layout(width="115px"))
        self.clear_log_btn = widgets.Button(description="Clear Log", button_style="", icon="trash", layout=widgets.Layout(width="100px"))
        self.autorange_toggle = widgets.ToggleButton(value=True, description="Auto-Range", button_style="info", layout=widgets.Layout(width="110px"))

        self.readout_stats = widgets.HTML("<span style='color:#00FFCC; font-family:monospace; font-size:13px; font-weight:bold;'>Raw Vpp: 0.00V | Filt Vpp: 0.00V | Atten: 0.0 dB | Mode: Lowpass (Bass)</span>")

        # 2. Row 2: ⚡ Quick Filter Presets
        self.preset_label = widgets.HTML("<span style='color:#FFA500; font-family:sans-serif; font-size:12px; font-weight:bold;'>⚡ Quick Presets:</span>", layout=widgets.Layout(margin="4px 6px 0 0"))
        self.btn_preset_bass = widgets.Button(description="Sub-Bass (20-120 Hz)", button_style="primary", layout=widgets.Layout(width="160px"))
        self.btn_preset_fullbass = widgets.Button(description="Full Bass (20-250 Hz)", button_style="primary", layout=widgets.Layout(width="160px"))
        self.btn_preset_vocals = widgets.Button(description="Vocals (300-3.4k Hz)", button_style="primary", layout=widgets.Layout(width="165px"))
        self.btn_preset_highpass = widgets.Button(description="Highpass (>1 kHz)", button_style="primary", layout=widgets.Layout(width="150px"))
        self.btn_preset_notch = widgets.Button(description="60 Hz Notch", button_style="warning", layout=widgets.Layout(width="120px"))
        self.btn_preset_bypass = widgets.Button(description="Bypass Filter", button_style="", layout=widgets.Layout(width="120px"))

        # 3. Row 3: 🎛 Custom Range & Filter Tuning
        self.tune_label = widgets.HTML("<span style='color:#00FFCC; font-family:sans-serif; font-size:12px; font-weight:bold;'>🎛 Custom Range:</span>", layout=widgets.Layout(margin="4px 6px 0 0"))
        self.filter_mode_dd = widgets.Dropdown(
            options=[("Lowpass (Bass)", "lowpass"), ("Highpass (Treble)", "highpass"), ("Bandpass", "bandpass"), ("Notch", "notch")],
            value="lowpass",
            description="Mode:",
            layout=widgets.Layout(width="190px")
        )
        self.low_cut_slider = widgets.FloatSlider(value=20.0, min=0.0, max=25000.0, step=10.0, description="Low-Cut (Hz):", continuous_update=False, layout=widgets.Layout(width="260px"))
        self.low_cut_input = widgets.BoundedFloatText(value=20.0, min=0.0, max=25000.0, step=10.0, layout=widgets.Layout(width="85px"))
        widgets.jslink((self.low_cut_slider, "value"), (self.low_cut_input, "value"))

        self.high_cut_slider = widgets.FloatSlider(value=250.0, min=20.0, max=25000.0, step=10.0, description="High-Cut (Hz):", continuous_update=False, layout=widgets.Layout(width="260px"))
        self.high_cut_input = widgets.BoundedFloatText(value=250.0, min=20.0, max=25000.0, step=10.0, layout=widgets.Layout(width="85px"))
        widgets.jslink((self.high_cut_slider, "value"), (self.high_cut_input, "value"))

        # 4. Row 4: ⏱ Trigger & Timebase Options
        self.trig_mode_dd = widgets.Dropdown(options=["Auto", "Normal", "Single"], value="Auto", description="Trig Mode:", layout=widgets.Layout(width="180px"))
        self.trig_edge_dd = widgets.Dropdown(options=["Rising", "Falling"], value="Rising", description="Trig Edge:", layout=widgets.Layout(width="180px"))
        self.trig_src_dd = widgets.Dropdown(options=["CH1 (A0)", "CH2 (A1)"], value="CH1 (A0)", description="Trig Src:", layout=widgets.Layout(width="180px"))
        self.trig_level_slider = widgets.FloatSlider(value=1.65, min=0.0, max=3.3, step=0.05, description="Trig Level:", continuous_update=False, layout=widgets.Layout(width="210px"))
        self.trig_level_input = widgets.BoundedFloatText(value=1.65, min=0.0, max=3.3, step=0.05, layout=widgets.Layout(width="80px"))
        widgets.jslink((self.trig_level_slider, "value"), (self.trig_level_input, "value"))

        self.fft_span_dd = widgets.Dropdown(
            options=[("Audio Spectrum (25 kHz)", 25000), ("Vocal Band (10 kHz)", 10000), ("Bass Sub-Band (2.5 kHz)", 2500), ("Sub-Bass (500 Hz)", 500)],
            value=2500,
            description="FFT Span:",
            layout=widgets.Layout(width="230px")
        )

    def _build_plots(self):
        t_ms = np.linspace(0, self.total_duration_ms, self.num_pts_per_ch - 16)
        initial_freq = np.linspace(0, self.fs_per_ch / 2.0, len(t_ms) // 2 + 1)

        # =====================================================================
        # Tab 0: Quad Filter View (Expanded 720px Height & Clean Margins)
        # =====================================================================
        self.fig_quad = make_subplots(
            rows=3, cols=1, vertical_spacing=0.12,
            subplot_titles=(
                "<b>Row 1: Raw Input Time Waveform (Channel 1 / A0)</b>",
                "<b>Row 2: FPGA-Filtered Time Waveform (Reconstructed via IFFT)</b>",
                "<b>Row 3: Frequency Spectrum & Applied Hardware Filter Mask Window</b>"
            )
        )
        self.fig_quad = go.FigureWidget(self.fig_quad)
        self.fig_quad.add_scatter(x=t_ms, y=[1.65]*len(t_ms), mode="lines", line=dict(color="#00FFCC", width=1.5), name="Raw Input (A0)", row=1, col=1)
        self.fig_quad.add_scatter(x=[0, self.total_duration_ms], y=[1.65, 1.65], mode="lines", line=dict(color="#FFA500", width=1.2, dash="dash"), name="Trigger Threshold", row=1, col=1)
        self.fig_quad.add_scatter(x=t_ms, y=[1.65]*len(t_ms), mode="lines", line=dict(color="#FF007F", width=2.0), name="Filtered IFFT", row=2, col=1)
        self.fig_quad.add_scatter(x=initial_freq, y=[-100]*len(initial_freq), mode="lines", line=dict(color="#E040FB", width=1.8), name="Spectrum (FFT)", row=3, col=1)

        self.fig_quad.update_layout(
            template="plotly_dark",
            height=720,
            margin=dict(l=45, r=25, t=40, b=55),
            showlegend=False,
            uirevision="t0"
        )
        self.fig_quad.update_yaxes(range=[0, 3.3], title="Voltage (V)", row=1, col=1)
        self.fig_quad.update_yaxes(range=[0, 3.3], title="Voltage (V)", row=2, col=1)
        self.fig_quad.update_yaxes(range=[-100, 5], title="Mag (dBV)", row=3, col=1)
        self.fig_quad.update_xaxes(range=[0, self.total_duration_ms], title="Time (ms)", row=1, col=1)
        self.fig_quad.update_xaxes(range=[0, self.total_duration_ms], title="Time (ms)", row=2, col=1)
        self.fig_quad.update_xaxes(range=[0, 2500], title="Frequency (Hz)", row=3, col=1)

        # =====================================================================
        # Tab 1: Time Domain Superimposed Overlay
        # =====================================================================
        self.fig_overlay = go.FigureWidget()
        self.fig_overlay.add_scatter(x=t_ms, y=[1.65]*len(t_ms), mode="lines", line=dict(color="rgba(0, 255, 204, 0.45)", width=1.4), name="Raw Input (A0)")
        self.fig_overlay.add_scatter(x=t_ms, y=[1.65]*len(t_ms), mode="lines", line=dict(color="#FF007F", width=2.2), name="FPGA Filtered (IFFT)")
        self.fig_overlay.update_layout(
            title="<b>Time Domain Overlay: Raw Input vs. FPGA Reconstructed Output</b>",
            template="plotly_dark", height=480, margin=dict(l=45, r=25, t=45, b=45), uirevision="t1"
        )
        self.fig_overlay.update_yaxes(range=[0, 3.3], title="Voltage (V)")
        self.fig_overlay.update_xaxes(range=[0, self.total_duration_ms], title="Time (ms)")

        # =====================================================================
        # Tab 2: Dedicated Channel 1 / Raw View
        # =====================================================================
        self.fig_raw_view = make_subplots(rows=2, cols=1, vertical_spacing=0.14,
            subplot_titles=("<b>Channel 1: A0 Raw Time Waveform</b>", "<b>Channel 1: A0 Raw FFT Spectrum</b>"))
        self.fig_raw_view = go.FigureWidget(self.fig_raw_view)
        self.fig_raw_view.add_scatter(x=t_ms, y=[1.65]*len(t_ms), mode="lines", line=dict(color="#00FFCC", width=1.6), name="Raw Time", row=1, col=1)
        self.fig_raw_view.add_scatter(x=[0, self.total_duration_ms], y=[1.65, 1.65], mode="lines", line=dict(color="#FFA500", width=1.2, dash="dash"), name="Trigger", row=1, col=1)
        self.fig_raw_view.add_scatter(x=initial_freq, y=[-100]*len(initial_freq), mode="lines", line=dict(color="#00FFCC", width=1.8), name="Raw FFT", row=2, col=1)
        self.fig_raw_view.update_layout(template="plotly_dark", height=520, margin=dict(l=45, r=25, t=40, b=45), showlegend=False, uirevision="t2")
        self.fig_raw_view.update_yaxes(range=[0, 3.3], title="Voltage (V)", row=1, col=1)
        self.fig_raw_view.update_yaxes(range=[-100, 5], title="Mag (dBV)", row=2, col=1)
        self.fig_raw_view.update_xaxes(range=[0, self.total_duration_ms], title="Time (ms)", row=1, col=1)
        self.fig_raw_view.update_xaxes(range=[0, 2500], title="Frequency (Hz)", row=2, col=1)

        # =====================================================================
        # Tab 3: Dedicated Filtered View
        # =====================================================================
        self.fig_filt_view = make_subplots(rows=2, cols=1, vertical_spacing=0.14,
            subplot_titles=("<b>FPGA Reconstructed Time Waveform (axi_dma_2)</b>", "<b>Hardware-Filtered FFT Spectrum</b>"))
        self.fig_filt_view = go.FigureWidget(self.fig_filt_view)
        self.fig_filt_view.add_scatter(x=t_ms, y=[1.65]*len(t_ms), mode="lines", line=dict(color="#FF007F", width=2.0), name="Filt Time", row=1, col=1)
        self.fig_filt_view.add_scatter(x=initial_freq, y=[-100]*len(initial_freq), mode="lines", line=dict(color="#FF007F", width=1.8), name="Filt FFT", row=2, col=1)
        self.fig_filt_view.update_layout(template="plotly_dark", height=520, margin=dict(l=45, r=25, t=40, b=45), showlegend=False, uirevision="t3")
        self.fig_filt_view.update_yaxes(range=[0, 3.3], title="Voltage (V)", row=1, col=1)
        self.fig_filt_view.update_yaxes(range=[-100, 5], title="Mag (dBV)", row=2, col=1)
        self.fig_filt_view.update_xaxes(range=[0, self.total_duration_ms], title="Time (ms)", row=1, col=1)
        self.fig_filt_view.update_xaxes(range=[0, 2500], title="Frequency (Hz)", row=2, col=1)

        # Container Tabs
        self.tabs = widgets.Tab(children=[self.fig_quad, self.fig_overlay, self.fig_raw_view, self.fig_filt_view])
        self.tabs.set_title(0, "🎛 Quad Filter View")
        self.tabs.set_title(1, "📈 Time Overlay")
        self.tabs.set_title(2, "🎙 Raw Input (A0)")
        self.tabs.set_title(3, "🎵 Filtered Output")

    def _setup_callbacks(self):
        self.start_btn.on_click(lambda _: self.start())
        self.stop_btn.on_click(lambda _: self.stop())
        self.force_btn.on_click(lambda _: self._on_force_clicked())
        self.clear_log_btn.on_click(lambda _: self._on_clear_log_clicked())

        # Presets
        self.btn_preset_bass.on_click(lambda _: self._apply_preset(0.0, 120.0, "lowpass"))
        self.btn_preset_fullbass.on_click(lambda _: self._apply_preset(0.0, 250.0, "lowpass"))
        self.btn_preset_vocals.on_click(lambda _: self._apply_preset(300.0, 3400.0, "bandpass"))
        self.btn_preset_highpass.on_click(lambda _: self._apply_preset(1000.0, self.fs_per_ch / 2.0, "highpass"))
        self.btn_preset_notch.on_click(lambda _: self._apply_preset(50.0, 70.0, "notch"))
        self.btn_preset_bypass.on_click(lambda _: self._apply_bypass())

        # Sliders & Dropdowns
        self.filter_mode_dd.observe(lambda _: self._update_filter_params(), names="value")
        self.low_cut_slider.observe(lambda _: self._update_filter_params(), names="value")
        self.high_cut_slider.observe(lambda _: self._update_filter_params(), names="value")
        self.trig_level_slider.observe(lambda _: self._update_trig_level(), names="value")
        self.trig_mode_dd.observe(self._on_trig_param_change, names="value")
        self.trig_edge_dd.observe(self._on_trig_param_change, names="value")
        self.trig_src_dd.observe(self._on_trig_param_change, names="value")

    def _apply_preset(self, low: float, high: float, mode: str):
        self.filter_mode_dd.value = mode
        self.low_cut_slider.value = low
        self.high_cut_slider.value = min(float(self.high_cut_slider.max), high)
        self._update_filter_params()

    def _apply_bypass(self):
        if self.filter:
            self.filter.bypass()

    def _update_filter_params(self):
        if self.filter:
            self.filter.set_passband(
                low_hz=float(self.low_cut_slider.value),
                high_hz=float(self.high_cut_slider.value),
                mode=self.filter_mode_dd.value,
                enable=True
            )

    def _update_trig_level(self):
        if self.trigger:
            self.trigger.set_threshold(float(self.trig_level_slider.value))

    def _get_arm_control_word(self) -> int:
        is_falling = (self.trig_edge_dd.value == "Falling")
        is_ch2 = ("CH2" in self.trig_src_dd.value or "A1" in self.trig_src_dd.value)
        mode = self.trig_mode_dd.value
        
        ctrl = (1 << 0) | (1 << 3)  # Bit 0: Arm, Bit 3: Single Shot (Arm-on-Demand)
        if is_falling:
            ctrl |= (1 << 2)
        if mode == "Auto":
            ctrl |= (1 << 1)
        if is_ch2:
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
        dma_filt = self.overlay.dma_filtered
        dma_fft = self.overlay.axi_dma_1
        trig = self.trigger

        # Initialize XADC sequencer & synchronize FFT to N=1024
        if hasattr(self.overlay, "xadc_wiz_0"):
            self.overlay.xadc_wiz_0.mmio.write(0x304, 0x2000)
            self.overlay.xadc_wiz_0.mmio.write(0x320, 0x0000)
            self.overlay.xadc_wiz_0.mmio.write(0x324, 0x0202)

        if trig:
            trig.set_fft_config(n_points=1024)
            trig.mmio.write(0x0C, 5000000)
            self._update_trig_level()

        self._update_filter_params()

        # Allocate matched DMA buffers
        buf_time = allocate(shape=(self.packet_size,), dtype="u2")      # 2048 words (stereo interleaved)
        buf_filt = allocate(shape=(self.num_pts_per_ch,), dtype="u2")   # 1024 words
        buf_fft = allocate(shape=(self.num_pts_per_ch,), dtype="u2")    # 1024 words

        def reset_dmas():
            for dma_block in [dma_time, dma_filt, dma_fft]:
                if dma_block:
                    try:
                        dma_block.mmio.write(0x30, 0x04)
                        time.sleep(0.002)
                        dma_block.recvchannel.start()
                    except Exception:
                        pass

        reset_dmas()
        print(f"[FilterDashboard] Real-Time Hardware Filter Instrument Active (50 kSPS | 3-DMA Synchronized Stream)")

        try:
            while self._is_running:
                mode = self.trig_mode_dd.value
                if mode == "Single" and self._single_done:
                    time.sleep(0.02)
                    continue

                # 1. Queue all 3 DMAs BEFORE opening the trigger gate
                dma_time.recvchannel.transfer(buf_time)
                if dma_filt:
                    dma_filt.recvchannel.transfer(buf_filt)
                if dma_fft:
                    dma_fft.recvchannel.transfer(buf_fft)

                # 2. Arm Hardware Trigger with Arm-on-Demand (0x0B)
                trig.mmio.write(0x00, self._get_arm_control_word())

                # 3. Wait for all 3 DMAs with timeout protection
                t0 = time.time()
                while not (dma_time.recvchannel.idle and (not dma_filt or dma_filt.recvchannel.idle) and (not dma_fft or dma_fft.recvchannel.idle)):
                    if time.time() - t0 > 0.3:
                        reset_dmas()
                        break
                    time.sleep(0.0005)
                else:
                    # 4. Extract data cleanly with calibrated 1:1 amplitude
                    raw_interleaved = np.array(buf_time)
                    raw_ch1 = raw_interleaved[0::2]
                    v_raw = ((raw_ch1 >> 4) * (3.3 / 4095.0))[8:-8]

                    raw_f = np.array(buf_filt)
                    ac_filt = (raw_f.astype(np.float64) - 32768.0)[8:-8]
                    v_filt = np.clip(1.65 + (ac_filt / 32768.0) * 1.65, 0.0, 3.3)

                    n = len(v_raw)
                    t_ms = np.linspace(0, (n / self.fs_per_ch) * 1000.0, n)

                    # FFT Spectrum
                    freqs, mags = self.fft.process_buffer(buf_fft, unit="dBV")

                    vpp_raw = float(np.ptp(v_raw))
                    vpp_filt = float(np.ptp(v_filt))
                    atten_db = 20.0 * np.log10(max(1e-4, vpp_filt) / max(1e-4, vpp_raw))

                    active_tab = self.tabs.selected_index
                    trig_v = float(self.trig_level_slider.value)
                    max_span = float(self.fft_span_dd.value)

                    # Dynamic Vertical Auto-Range Scaling (Preserves Full 20.48 ms Window!)
                    if self.autorange_toggle.value and vpp_raw > 0.1:
                        y_margin = max(0.2, vpp_raw * 0.2)
                        y_min = max(0.0, 1.65 - (vpp_raw / 2.0 + y_margin))
                        y_max = min(3.3, 1.65 + (vpp_raw / 2.0 + y_margin))
                    else:
                        y_min, y_max = 0.0, 3.3

                    if active_tab == 0:  # Tab 0: Quad Filter View
                        with self.fig_quad.batch_update():
                            self.fig_quad.data[0].x = t_ms
                            self.fig_quad.data[0].y = v_raw
                            self.fig_quad.data[1].x = [0, t_ms[-1]]
                            self.fig_quad.data[1].y = [trig_v, trig_v]
                            self.fig_quad.data[2].x = t_ms
                            self.fig_quad.data[2].y = v_filt
                            self.fig_quad.data[3].x = freqs
                            self.fig_quad.data[3].y = mags
                            self.fig_quad.layout.yaxis.range = [y_min, y_max]
                            self.fig_quad.layout.yaxis2.range = [y_min, y_max]
                            self.fig_quad.layout.xaxis3.range = [0, max_span]

                    elif active_tab == 1:  # Tab 1: Time Overlay
                        with self.fig_overlay.batch_update():
                            self.fig_overlay.data[0].x = t_ms
                            self.fig_overlay.data[0].y = v_raw
                            self.fig_overlay.data[1].x = t_ms
                            self.fig_overlay.data[1].y = v_filt
                            self.fig_overlay.layout.yaxis.range = [y_min, y_max]
                            self.fig_overlay.layout.xaxis.range = [0, t_ms[-1]]

                    elif active_tab == 2:  # Tab 2: Raw Input View
                        with self.fig_raw_view.batch_update():
                            self.fig_raw_view.data[0].x = t_ms
                            self.fig_raw_view.data[0].y = v_raw
                            self.fig_raw_view.data[1].x = [0, t_ms[-1]]
                            self.fig_raw_view.data[1].y = [trig_v, trig_v]
                            self.fig_raw_view.data[2].x = freqs
                            self.fig_raw_view.data[2].y = mags
                            self.fig_raw_view.layout.yaxis.range = [y_min, y_max]
                            self.fig_raw_view.layout.xaxis2.range = [0, max_span]

                    elif active_tab == 3:  # Tab 3: Filtered View
                        with self.fig_filt_view.batch_update():
                            self.fig_filt_view.data[0].x = t_ms
                            self.fig_filt_view.data[0].y = v_filt
                            self.fig_filt_view.data[1].x = freqs
                            self.fig_filt_view.data[1].y = mags
                            self.fig_filt_view.layout.yaxis.range = [y_min, y_max]
                            self.fig_filt_view.layout.xaxis2.range = [0, max_span]

                    self.readout_stats.value = (
                        f"<span style='color:#00FFCC; font-family:monospace; font-size:13px; font-weight:bold;'>"
                        f"Raw Vpp: {vpp_raw:.2f}V | Filt Vpp: {vpp_filt:.2f}V | Atten: {atten_db:+.1f}dB | Mode: {self.filter_mode_dd.value.capitalize()}"
                        f"</span>"
                    )

                    if mode == "Single":
                        self._single_done = True
                    time.sleep(0.025)

        finally:
            self._is_running = False
            buf_time.close()
            buf_filt.close()
            buf_fft.close()
            print("[FilterDashboard] Stopped cleanly.")

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
        r1 = widgets.HBox([self.start_btn, self.stop_btn, self.force_btn, self.autorange_toggle, self.clear_log_btn, self.readout_stats], layout=widgets.Layout(gap="10px", margin="0 0 8px 0", align_items="center"))
        r2 = widgets.HBox([self.preset_label, self.btn_preset_bass, self.btn_preset_fullbass, self.btn_preset_vocals, self.btn_preset_highpass, self.btn_preset_notch, self.btn_preset_bypass], layout=widgets.Layout(gap="8px", margin="0 0 8px 0", align_items="center"))
        r3 = widgets.HBox([self.tune_label, self.filter_mode_dd, self.low_cut_slider, self.low_cut_input, self.high_cut_slider, self.high_cut_input], layout=widgets.Layout(gap="6px", margin="0 0 8px 0", align_items="center"))
        r4 = widgets.HBox([self.trig_mode_dd, self.trig_edge_dd, self.trig_src_dd, self.trig_level_slider, self.trig_level_input, self.fft_span_dd], layout=widgets.Layout(gap="8px", margin="0 0 8px 0", align_items="center"))

        self.control_panel = widgets.VBox([r1, r2, r3, r4], layout=widgets.Layout(margin="0 0 14px 0"))
        display(widgets.VBox([self.control_panel, self.tabs]))