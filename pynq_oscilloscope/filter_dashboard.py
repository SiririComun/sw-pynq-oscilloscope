"""
pynq_oscilloscope.filter_dashboard: Interactive Real-Time Hardware Filter Dashboard.
Provides live frequency cutoff sliders, filter mode presets (Bass, Vocals, Notch, Treble),
and concurrent 4-trace visualization of Raw Time, Filtered Time, Raw FFT, and Filtered FFT.
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

from pynq_oscilloscope.fft_dma import StreamingFFT
from pynq_oscilloscope.hw_trigger import HardwareTrigger
from pynq_oscilloscope.hw_filter import HardwareFilter


class AudioFilterDashboard:
    """
    Interactive Real-Time Hardware Filter & IFFT Reconstruction Instrument.
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

        # Bind sub-drivers
        if self.overlay and hasattr(self.overlay, "trigger"):
            self.trigger = self.overlay.trigger
        elif self.overlay:
            self.trigger = HardwareTrigger(self.overlay)
        else:
            self.trigger = None

        if self.overlay and hasattr(self.overlay, "filter"):
            self.filter = self.overlay.filter
        elif self.overlay:
            self.filter = HardwareFilter(self.overlay, sample_rate_hz=self.fs_per_ch, fft_points=self.packet_size)
        else:
            self.filter = None

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
        # 1. Action Row & Readouts
        self.start_btn = widgets.Button(description="Start Live", button_style="success", icon="play", layout=widgets.Layout(width="115px"))
        self.stop_btn = widgets.Button(description="Stop", button_style="danger", icon="stop", layout=widgets.Layout(width="95px"))
        self.force_btn = widgets.Button(description="Force Trig", button_style="warning", icon="bolt", layout=widgets.Layout(width="115px"))
        self.clear_log_btn = widgets.Button(description="Clear Log", button_style="", icon="trash", layout=widgets.Layout(width="100px"))

        self.readout_stats = widgets.HTML("<span style='color:#00FFCC; font-family:monospace; font-size:13px; font-weight:bold;'>Raw Vpp: 0.00V | Filt Vpp: 0.00V | Atten: 0.0 dB | Mode: Lowpass (Bass)</span>")

        # 2. Filter Presets Row
        self.btn_preset_bass = widgets.Button(description="Sub-Bass (20-120 Hz)", button_style="primary", layout=widgets.Layout(width="170px"))
        self.btn_preset_fullbass = widgets.Button(description="Full Bass (20-250 Hz)", button_style="primary", layout=widgets.Layout(width="170px"))
        self.btn_preset_vocals = widgets.Button(description="Vocals (300-3.4k Hz)", button_style="primary", layout=widgets.Layout(width="170px"))
        self.btn_preset_notch = widgets.Button(description="60 Hz Notch", button_style="warning", layout=widgets.Layout(width="130px"))
        self.btn_preset_bypass = widgets.Button(description="Bypass Filter", button_style="", layout=widgets.Layout(width="130px"))

        # 3. Filter Controls Row
        self.filter_mode_dd = widgets.Dropdown(options=[("Lowpass (Bass)", "lowpass"), ("Highpass (Treble)", "highpass"), ("Bandpass", "bandpass"), ("Notch", "notch")], value="lowpass", description="Mode:", layout=widgets.Layout(width="200px"))
        self.low_cut_slider = widgets.FloatSlider(value=20.0, min=0.0, max=25000.0, step=10.0, description="Low-Cut (Hz):", continuous_update=False, layout=widgets.Layout(width="260px"))
        self.low_cut_input = widgets.BoundedFloatText(value=20.0, min=0.0, max=25000.0, step=10.0, layout=widgets.Layout(width="85px"))
        widgets.jslink((self.low_cut_slider, "value"), (self.low_cut_input, "value"))

        self.high_cut_slider = widgets.FloatSlider(value=250.0, min=20.0, max=25000.0, step=10.0, description="High-Cut (Hz):", continuous_update=False, layout=widgets.Layout(width="260px"))
        self.high_cut_input = widgets.BoundedFloatText(value=250.0, min=20.0, max=25000.0, step=10.0, layout=widgets.Layout(width="85px"))
        widgets.jslink((self.high_cut_slider, "value"), (self.high_cut_input, "value"))

        # 4. Trigger & Timing Controls
        self.trig_mode_dd = widgets.Dropdown(options=["Auto", "Normal", "Single"], value="Auto", description="Trig Mode:", layout=widgets.Layout(width="180px"))
        self.trig_level_slider = widgets.FloatSlider(value=1.65, min=0.0, max=3.3, step=0.05, description="Trig Level:", continuous_update=False, layout=widgets.Layout(width="220px"))

    def _build_plots(self):
        t_ms = np.linspace(0, self.total_duration_ms, self.num_pts_per_ch - 16)
        initial_freq = np.linspace(0, self.fs_per_ch / 2.0, len(t_ms) // 2 + 1)

        # Tab 1: Quad Domain View (4-Trace Stacked)
        self.fig_quad = make_subplots(
            rows=3, cols=1, vertical_spacing=0.10,
            subplot_titles=(
                "<b>Row 1: Raw Input Time Waveform (Channel 1 / A0)</b>",
                "<b>Row 2: FPGA-Filtered Time Waveform (Reconstructed via IFFT)</b>",
                "<b>Row 3: Frequency Spectrum & Real-Time Filter Mask</b>"
            )
        )
        self.fig_quad = go.FigureWidget(self.fig_quad)
        self.fig_quad.add_scatter(x=t_ms, y=[1.65]*len(t_ms), mode="lines", line=dict(color="#00FFCC", width=1.5), name="Raw Time", row=1, col=1)
        self.fig_quad.add_scatter(x=t_ms, y=[1.65]*len(t_ms), mode="lines", line=dict(color="#FF007F", width=2.0), name="Filtered Time", row=2, col=1)
        self.fig_quad.add_scatter(x=initial_freq, y=[-100]*len(initial_freq), mode="lines", line=dict(color="#E040FB", width=1.8), name="Spectrum", row=3, col=1)
        self.fig_quad.update_layout(template="plotly_dark", height=600, margin=dict(l=40, r=20, t=40, b=35), showlegend=False, uirevision="t1")
        self.fig_quad.update_yaxes(range=[0, 3.3], title="Voltage (V)", row=1, col=1)
        self.fig_quad.update_yaxes(range=[0, 3.3], title="Voltage (V)", row=2, col=1)
        self.fig_quad.update_yaxes(range=[-100, 5], title="Mag (dBV)", row=3, col=1)
        self.fig_quad.update_xaxes(range=[0, self.total_duration_ms], title="Time (ms)", row=2, col=1)
        self.fig_quad.update_xaxes(range=[0, 5000], title="Frequency (Hz)", row=3, col=1)

        # Tab 2: Time-Domain Overlay Comparison
        self.fig_overlay = go.FigureWidget()
        self.fig_overlay.add_scatter(x=t_ms, y=[1.65]*len(t_ms), mode="lines", line=dict(color="rgba(0, 255, 204, 0.4)", width=1.2), name="Raw A0 Wave")
        self.fig_overlay.add_scatter(x=t_ms, y=[1.65]*len(t_ms), mode="lines", line=dict(color="#FF007F", width=2.2), name="FPGA-Filtered Wave")
        self.fig_overlay.update_layout(template="plotly_dark", height=450, title="<b>Time Domain Overlay: Raw Input vs. FPGA Reconstructed Output</b>", uirevision="t2")
        self.fig_overlay.update_yaxes(range=[0, 3.3], title="Voltage (V)")
        self.fig_overlay.update_xaxes(range=[0, self.total_duration_ms], title="Time (ms)")

        self.tabs = widgets.Tab(children=[self.fig_quad, self.fig_overlay])
        self.tabs.set_title(0, "🎛 Quad Filter View")
        self.tabs.set_title(1, "📈 Time Overlay Comparison")

    def _setup_callbacks(self):
        self.start_btn.on_click(lambda _: self.start())
        self.stop_btn.on_click(lambda _: self.stop())
        self.force_btn.on_click(lambda _: self._on_force_clicked())
        self.clear_log_btn.on_click(lambda _: self._on_clear_log_clicked())

        # Presets
        self.btn_preset_bass.on_click(lambda _: self._apply_preset(0.0, 120.0, "lowpass"))
        self.btn_preset_fullbass.on_click(lambda _: self._apply_preset(0.0, 250.0, "lowpass"))
        self.btn_preset_vocals.on_click(lambda _: self._apply_preset(300.0, 3400.0, "bandpass"))
        self.btn_preset_notch.on_click(lambda _: self._apply_preset(50.0, 70.0, "notch"))
        self.btn_preset_bypass.on_click(lambda _: self._apply_bypass())

        # Sliders
        self.filter_mode_dd.observe(lambda _: self._update_filter_params(), names="value")
        self.low_cut_slider.observe(lambda _: self._update_filter_params(), names="value")
        self.high_cut_slider.observe(lambda _: self._update_filter_params(), names="value")
        self.trig_level_slider.observe(lambda _: self._update_trig_level(), names="value")

    def _apply_preset(self, low: float, high: float, mode: str):
        self.filter_mode_dd.value = mode
        self.low_cut_slider.value = low
        self.high_cut_slider.value = high
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

    def _on_force_clicked(self):
        if self.trig_mode_dd.value == "Single": self._single_done = False
        if self.trigger: self.trigger.force_trigger()

    def _on_clear_log_clicked(self):
        clear_output(wait=True)
        display(widgets.VBox([self.control_panel, self.tabs]))

    def _update_loop(self):
        dma_time = self.overlay.axi_dma_0
        dma_filt = self.overlay.dma_filtered
        dma_fft = self.overlay.axi_dma_1
        trig = self.trigger

        # Initialize XADC and DMAs
        if hasattr(self.overlay, "xadc_wiz_0"):
            self.overlay.xadc_wiz_0.mmio.write(0x304, 0x2000)
            self.overlay.xadc_wiz_0.mmio.write(0x320, 0x0000)
            self.overlay.xadc_wiz_0.mmio.write(0x324, 0x0202)

        for dma_block in [dma_time, dma_filt, dma_fft]:
            if dma_block:
                dma_block.mmio.write(0x30, 0x04)
                time.sleep(0.002)
                dma_block.recvchannel.start()

        if trig:
            trig.mmio.write(0x0C, 5000000)
            self._update_trig_level()

        buf_time = allocate(shape=(self.packet_size,), dtype="u2")
        buf_filt = allocate(shape=(self.packet_size,), dtype="u2")
        buf_fft = allocate(shape=(self.packet_size,), dtype="u2")
        dma_armed = False
        print(f"[FilterDashboard] Real-Time Hardware Filter Instrument Active (50 kSPS | 3-DMA Synchronized Stream)")

        try:
            while self._is_running:
                mode = self.trig_mode_dd.value
                if mode == "Single" and self._single_done:
                    time.sleep(0.02)
                    continue

                if not dma_armed:
                    dma_time.recvchannel.transfer(buf_time)
                    if dma_filt: dma_filt.recvchannel.transfer(buf_filt)
                    if dma_fft: dma_fft.recvchannel.transfer(buf_fft)
                    ctrl = (1 << 0) | (1 << 3) if mode == "Single" else ((1 << 0) | (1 << 1))
                    trig.mmio.write(0x00, ctrl)
                    dma_armed = True

                all_idle = dma_time.recvchannel.idle and (not dma_filt or dma_filt.recvchannel.idle) and (not dma_fft or dma_fft.recvchannel.idle)

                if all_idle and dma_armed:
                    dma_armed = False

                    raw_interleaved = np.array(buf_time)
                    raw_ch1 = raw_interleaved[0::2]
                    v_raw = (raw_ch1 >> 4) * (3.3 / 4095.0)
                    p_raw = v_raw[8:-8]

                    raw_f = np.array(buf_filt)
                    v_filt = (raw_f >> 4) * (3.3 / 4095.0)
                    p_filt = v_filt[8:-8]

                    n = len(p_raw)
                    t_ms = np.linspace(0, (n / self.fs_per_ch) * 1000.0, n)

                    # FFT Magnitude
                    freqs, mags = self.fft.process_buffer(buf_fft, unit="dBV")

                    vpp_raw = float(np.ptp(p_raw))
                    vpp_filt = float(np.ptp(p_filt))
                    atten_db = 20.0 * np.log10(max(1e-4, vpp_filt) / max(1e-4, vpp_raw))

                    active_tab = self.tabs.selected_index

                    if active_tab == 0:
                        with self.fig_quad.batch_update():
                            self.fig_quad.data[0].x = t_ms
                            self.fig_quad.data[0].y = p_raw
                            self.fig_quad.data[1].x = t_ms
                            self.fig_quad.data[1].y = p_filt
                            self.fig_quad.data[2].x = freqs
                            self.fig_quad.data[2].y = mags
                            self.fig_quad.layout.xaxis2.range = [0, t_ms[-1]]
                            self.fig_quad.layout.xaxis3.range = [0, max(500.0, float(self.high_cut_slider.value) * 1.5)]
                    elif active_tab == 1:
                        with self.fig_overlay.batch_update():
                            self.fig_overlay.data[0].x = t_ms
                            self.fig_overlay.data[0].y = p_raw
                            self.fig_overlay.data[1].x = t_ms
                            self.fig_overlay.data[1].y = p_filt
                            self.fig_overlay.layout.xaxis.range = [0, t_ms[-1]]

                    self.readout_stats.value = (
                        f"<span style='color:#00FFCC; font-family:monospace; font-size:13px; font-weight:bold;'>"
                        f"Raw Vpp: {vpp_raw:.2f}V | Filt Vpp: {vpp_filt:.2f}V | Atten: {atten_db:+.1f}dB | Mode: {self.filter_mode_dd.value.capitalize()}"
                        f"</span>"
                    )

                    if mode == "Single": self._single_done = True
                    time.sleep(0.033)
                else:
                    time.sleep(0.005)

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
        r1 = widgets.HBox([self.start_btn, self.stop_btn, self.force_btn, self.clear_log_btn, self.readout_stats], layout=widgets.Layout(gap="10px", margin="0 0 8px 0"))
        r2 = widgets.HBox([self.btn_preset_bass, self.btn_preset_fullbass, self.btn_preset_vocals, self.btn_preset_notch, self.btn_preset_bypass], layout=widgets.Layout(gap="8px", margin="0 0 8px 0"))
        r3 = widgets.HBox([self.filter_mode_dd, self.low_cut_slider, self.low_cut_input, self.high_cut_slider, self.high_cut_input])
        r4 = widgets.HBox([self.trig_mode_dd, self.trig_level_slider])

        self.control_panel = widgets.VBox([r1, r2, r3, r4], layout=widgets.Layout(margin="0 0 12px 0"))
        display(widgets.VBox([self.control_panel, self.tabs]))