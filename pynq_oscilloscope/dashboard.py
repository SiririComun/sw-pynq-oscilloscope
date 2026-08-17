"""
pynq_oscilloscope.dashboard: Interactive Real-Time 4-Tab Dual-Channel Oscilloscope & Spectrum Analyzer UI.
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
    4-Tab Dual-Channel Oscilloscope & Spectrum Analyzer Dashboard.
    Features dynamic trigger line placement (moves to A0 or A1 based on trigger source),
    1 MSPS interleaved dual-DMA streaming, and dual-channel FFT analysis.
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
        self.num_pts_per_ch = packet_size // 2  # 1024 samples per channel
        self.fs_per_ch = 500_000.0              # 500 kSPS per channel
        self.fft_points = fft_points
        self.display_window = display_window
        
        self._is_running = False
        self._single_done = False
        self._thread: Optional[threading.Thread] = None
        
        # Stop any dangling wavegen handles
        if self.overlay and hasattr(self.overlay, "wavegen") and self.overlay.wavegen:
            try:
                self.overlay.wavegen.stop()
            except Exception:
                pass
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
        # 1. Action Row & Real-Time Metric Readouts
        self.start_btn = widgets.Button(
            description="Start Live", button_style="success", icon="play", layout=widgets.Layout(width="115px")
        )
        self.stop_btn = widgets.Button(
            description="Stop", button_style="danger", icon="stop", layout=widgets.Layout(width="95px")
        )
        self.force_btn = widgets.Button(
            description="Force / Arm", button_style="warning", icon="bolt", layout=widgets.Layout(width="115px")
        )
        self.clear_log_btn = widgets.Button(
            description="Clear Log", button_style="", icon="trash", layout=widgets.Layout(width="100px")
        )
        self.autorange_toggle = widgets.ToggleButton(
            value=True, description="Auto-Range", button_style="info", layout=widgets.Layout(width="110px")
        )

        self.readout_ch1 = widgets.HTML(
            "<span style='color:#00FFCC; font-family:monospace; font-size:13px; font-weight:bold;'>A0: Vpp=0.00V | f0=0.0kHz</span>"
        )
        self.readout_ch2 = widgets.HTML(
            "<span style='color:#FF007F; font-family:monospace; font-size:13px; font-weight:bold;'>A1: Vpp=0.00V | f0=0.0kHz</span>"
        )

        # 2. Trigger Controls
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

        # 3. Channel 1 Controls (W1 -> A0)
        self.ch1_shape_dd = widgets.Dropdown(
            options=["Sine", "Triangle", "Square"], value="Sine", description="CH1 (A0):", layout=widgets.Layout(width="180px")
        )
        self.ch1_amp_slider = widgets.FloatSlider(
            value=1.5, min=0.1, max=1.5, step=0.1, description="Amp (V):", continuous_update=False, layout=widgets.Layout(width="200px")
        )
        self.ch1_freq_slider = widgets.IntSlider(
            value=10000, min=50, max=100000, step=50, description="Freq (Hz):", continuous_update=False, layout=widgets.Layout(width="280px")
        )
        self.ch1_freq_input = widgets.BoundedIntText(
            value=10000, min=50, max=100000, step=50, layout=widgets.Layout(width="95px")
        )
        widgets.jslink((self.ch1_freq_slider, "value"), (self.ch1_freq_input, "value"))

        # 4. Channel 2 Controls (W2 -> A1) - Perfectly symmetric with Channel 1
        self.ch2_shape_dd = widgets.Dropdown(
            options=["Sine", "Triangle", "Square"], value="Square", description="CH2 (A1):", layout=widgets.Layout(width="180px")
        )
        self.ch2_amp_slider = widgets.FloatSlider(
            value=1.5, min=0.1, max=1.5, step=0.1, description="Amp (V):", continuous_update=False, layout=widgets.Layout(width="200px")
        )
        self.ch2_freq_slider = widgets.IntSlider(
            value=25000, min=50, max=100000, step=50, description="Freq (Hz):", continuous_update=False, layout=widgets.Layout(width="280px")
        )
        self.ch2_freq_input = widgets.BoundedIntText(
            value=25000, min=50, max=100000, step=50, layout=widgets.Layout(width="95px")
        )
        widgets.jslink((self.ch2_freq_slider, "value"), (self.ch2_freq_input, "value"))

        # 5. FFT Controls
        self.fft_unit_dd = widgets.Dropdown(
            options=["dBV", "dBFS", "Linear"], value="dBV", description="FFT Unit:", layout=widgets.Layout(width="170px")
        )
        self.fft_span_dd = widgets.Dropdown(
            options=[("Full (250 kHz)", 250000), ("100 kHz", 100000), ("20 kHz", 20000)],
            value=250000, description="Span / Zoom:", layout=widgets.Layout(width="210px")
        )

    def _build_plots(self):
        initial_time = np.arange(500)
        initial_freq = np.linspace(0, 250000, 513)

        # Tab 1: Dual Scope (A0 Top, A1 Bottom)
        # Trace 0: A0 (Row 1)
        # Trace 1: Trigger Line on A0 (Row 1)
        # Trace 2: A1 (Row 2)
        # Trace 3: Trigger Line on A1 (Row 2)
        self.fig_dual_scope = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
            subplot_titles=("<b>Channel 1: A0 (Time Domain)</b>", "<b>Channel 2: A1 (Time Domain)</b>")
        )
        self.fig_dual_scope = go.FigureWidget(self.fig_dual_scope)
        self.fig_dual_scope.add_scatter(x=initial_time, y=[1.65]*500, mode="lines", line=dict(color="#00FFCC", width=1.8), name="A0", row=1, col=1)
        self.fig_dual_scope.add_scatter(x=[0, 500], y=[1.65, 1.65], mode="lines", line=dict(color="#FFA500", width=1.2, dash="dash"), name="Trigger (A0)", row=1, col=1)
        self.fig_dual_scope.add_scatter(x=initial_time, y=[1.65]*500, mode="lines", line=dict(color="#FF007F", width=1.8), name="A1", row=2, col=1)
        self.fig_dual_scope.add_scatter(x=[0, 500], y=[1.65, 1.65], mode="lines", line=dict(color="#FFA500", width=1.2, dash="dash"), name="Trigger (A1)", visible=False, row=2, col=1)
        self.fig_dual_scope.update_layout(template="plotly_dark", height=500, margin=dict(l=40, r=20, t=45, b=35), uirevision="t1")
        self.fig_dual_scope.update_yaxes(range=[0, 3.3], title="Voltage (V)", row=1, col=1)
        self.fig_dual_scope.update_yaxes(range=[0, 3.3], title="Voltage (V)", row=2, col=1)
        self.fig_dual_scope.update_xaxes(title="Time (µs @ 500 kSPS)", row=2, col=1)

        # Tab 2: Dual Spectrum (FFT A0 Top, FFT A1 Bottom)
        self.fig_dual_fft = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
            subplot_titles=("<b>Channel 1: A0 Spectrum (FFT)</b>", "<b>Channel 2: A1 Spectrum (FFT)</b>")
        )
        self.fig_dual_fft = go.FigureWidget(self.fig_dual_fft)
        self.fig_dual_fft.add_scatter(x=initial_freq, y=[-100]*len(initial_freq), mode="lines", line=dict(color="#00FFCC", width=1.8), name="A0 FFT", row=1, col=1)
        self.fig_dual_fft.add_scatter(x=[1000], y=[-40], mode="markers+text", marker=dict(color="#FFA500", size=7, symbol="diamond"), text=["Peak"], textposition="top center", name="Peak 1", row=1, col=1)
        self.fig_dual_fft.add_scatter(x=initial_freq, y=[-100]*len(initial_freq), mode="lines", line=dict(color="#FF007F", width=1.8), name="A1 FFT", row=2, col=1)
        self.fig_dual_fft.add_scatter(x=[5000], y=[-40], mode="markers+text", marker=dict(color="#FFA500", size=7, symbol="diamond"), text=["Peak"], textposition="top center", name="Peak 2", row=2, col=1)
        self.fig_dual_fft.update_layout(template="plotly_dark", height=500, margin=dict(l=40, r=20, t=45, b=35), uirevision="t2")
        self.fig_dual_fft.update_yaxes(range=[-110, 0], title="Mag (dBV)", row=1, col=1)
        self.fig_dual_fft.update_yaxes(range=[-110, 0], title="Mag (dBV)", row=2, col=1)
        self.fig_dual_fft.update_xaxes(range=[0, 250000], title="Frequency (Hz)", row=2, col=1)

        # Tab 3: Channel 1 View (A0 Time Top, A0 FFT Bottom)
        self.fig_ch1_view = make_subplots(
            rows=2, cols=1, vertical_spacing=0.15,
            subplot_titles=("<b>Channel 1: A0 (Time Domain)</b>", "<b>Channel 1: A0 (Frequency Domain)</b>")
        )
        self.fig_ch1_view = go.FigureWidget(self.fig_ch1_view)
        self.fig_ch1_view.add_scatter(x=initial_time, y=[1.65]*500, mode="lines", line=dict(color="#00FFCC", width=1.8), name="A0 Time", row=1, col=1)
        self.fig_ch1_view.add_scatter(x=[0, 500], y=[1.65, 1.65], mode="lines", line=dict(color="#FFA500", width=1.2, dash="dash"), name="Trigger", row=1, col=1)
        self.fig_ch1_view.add_scatter(x=initial_freq, y=[-100]*len(initial_freq), mode="lines", line=dict(color="#00FFCC", width=1.8), name="A0 FFT", row=2, col=1)
        self.fig_ch1_view.update_layout(template="plotly_dark", height=500, margin=dict(l=40, r=20, t=45, b=35), uirevision="t3")
        self.fig_ch1_view.update_yaxes(range=[0, 3.3], title="Voltage (V)", row=1, col=1)
        self.fig_ch1_view.update_yaxes(range=[-110, 0], title="Mag (dBV)", row=2, col=1)
        self.fig_ch1_view.update_xaxes(title="Time (µs)", row=1, col=1)
        self.fig_ch1_view.update_xaxes(range=[0, 250000], title="Frequency (Hz)", row=2, col=1)

        # Tab 4: Channel 2 View (A1 Time Top with Trigger, A1 FFT Bottom)
        self.fig_ch2_view = make_subplots(
            rows=2, cols=1, vertical_spacing=0.15,
            subplot_titles=("<b>Channel 2: A1 (Time Domain)</b>", "<b>Channel 2: A1 (Frequency Domain)</b>")
        )
        self.fig_ch2_view = go.FigureWidget(self.fig_ch2_view)
        self.fig_ch2_view.add_scatter(x=initial_time, y=[1.65]*500, mode="lines", line=dict(color="#FF007F", width=1.8), name="A1 Time", row=1, col=1)
        self.fig_ch2_view.add_scatter(x=[0, 500], y=[1.65, 1.65], mode="lines", line=dict(color="#FFA500", width=1.2, dash="dash"), name="Trigger", row=1, col=1)
        self.fig_ch2_view.add_scatter(x=initial_freq, y=[-100]*len(initial_freq), mode="lines", line=dict(color="#FF007F", width=1.8), name="A1 FFT", row=2, col=1)
        self.fig_ch2_view.update_layout(template="plotly_dark", height=500, margin=dict(l=40, r=20, t=45, b=35), uirevision="t4")
        self.fig_ch2_view.update_yaxes(range=[0, 3.3], title="Voltage (V)", row=1, col=1)
        self.fig_ch2_view.update_yaxes(range=[-110, 0], title="Mag (dBV)", row=2, col=1)
        self.fig_ch2_view.update_xaxes(title="Time (µs)", row=1, col=1)
        self.fig_ch2_view.update_xaxes(range=[0, 250000], title="Frequency (Hz)", row=2, col=1)

        # Tab Container
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
                enabled=True
            )

    def _get_arm_control_word(self) -> int:
        """
        Computes the register control word.
        Bit 5 is mapped such that 'CH1 (A0)' triggers A0 and 'CH2 (A1)' triggers A1.
        """
        is_falling = (self.trig_edge_dd.value == "Falling")
        is_ch1 = ("CH1" in self.trig_src_dd.value or "A0" in self.trig_src_dd.value)
        mode = self.trig_mode_dd.value
        
        # Base: Bit 0 = ARM, Bit 3 = SINGLE SHOT
        ctrl = (1 << 0) | (1 << 3)
        if is_falling:
            ctrl |= (1 << 2)        # Bit 2: FALLING EDGE
        if mode == "Auto":
            ctrl |= (1 << 1)        # Bit 1: AUTO TIMEOUT
        if is_ch1:
            ctrl |= (1 << 5)        # Inverted hardware routing: 1 = CH1 (A0), 0 = CH2 (A1)
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
        """High-Performance Synchronized Dual-DMA Acquisition Loop."""
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
            trig.mmio.write(0x0C, 5000000)  # 50ms Auto-Timeout @ 100MHz clock
            self._update_trig_level()

        # 4. Start AD3 Dual Wavegen (Both W1 and W2 always enabled)
        self.ad3.start(
            shape=self.ch1_shape_dd.value,
            frequency=float(self.ch1_freq_slider.value),
            amplitude=float(self.ch1_amp_slider.value),
            offset=1.65,
            ch2_shape=self.ch2_shape_dd.value,
            ch2_frequency=float(self.ch2_freq_slider.value),
            ch2_amplitude=float(self.ch2_amp_slider.value),
            ch2_offset=1.65,
            enable_ch2=True
        )
        wait_start = time.time()
        while self._is_running and not self.ad3.is_ready:
            time.sleep(0.05)
            if time.time() - wait_start > 3.0:
                break

        print(f"[Dashboard] Dual Acquisition active | Source: {self.trig_src_dd.value} | Mode: {self.trig_mode_dd.value}")

        buf_time = allocate(shape=(self.packet_size,), dtype="u2")
        buf_fft = allocate(shape=(self.packet_size,), dtype="u2")
        dma_armed = False

        freqs = np.fft.rfftfreq(self.num_pts_per_ch, d=1.0 / self.fs_per_ch)

        try:
            while self._is_running:
                mode = self.trig_mode_dd.value

                if mode == "Single" and self._single_done:
                    time.sleep(0.02)
                    continue

                # 5. Queue BOTH DMAs FIRST (Prevents Broadcaster Deadlock)
                if not dma_armed:
                    dma_time.recvchannel.transfer(buf_time)
                    dma_fft.recvchannel.transfer(buf_fft)
                    trig.mmio.write(0x00, self._get_arm_control_word())
                    dma_armed = True

                # 6. Check hardware completion
                if dma_time.recvchannel.idle and dma_fft.recvchannel.idle:
                    dma_armed = False

                    # De-interleave: Even = A0 (Ch1), Odd = A1 (Ch2)
                    raw_time = np.array(buf_time)
                    v_a0 = (raw_time[0::2] >> 4) * (3.3 / 4095.0)
                    v_a1 = (raw_time[1::2] >> 4) * (3.3 / 4095.0)

                    # Compute Fast FFT for both channels
                    fft_a0 = np.abs(np.fft.rfft(v_a0 - np.mean(v_a0))) / (self.num_pts_per_ch / 2.0)
                    fft_a1 = np.abs(np.fft.rfft(v_a1 - np.mean(v_a1))) / (self.num_pts_per_ch / 2.0)

                    unit = self.fft_unit_dd.value
                    if unit == "dBV":
                        mag_a0 = 20.0 * np.log10(np.maximum(fft_a0, 1e-6))
                        mag_a1 = 20.0 * np.log10(np.maximum(fft_a1, 1e-6))
                    elif unit == "dBFS":
                        mag_a0 = 20.0 * np.log10(np.maximum(fft_a0 / 1.65, 1e-6))
                        mag_a1 = 20.0 * np.log10(np.maximum(fft_a1 / 1.65, 1e-6))
                    else:
                        mag_a0 = fft_a0 * 1000.0
                        mag_a1 = fft_a1 * 1000.0

                    vpp1 = float(np.max(v_a0) - np.min(v_a0))
                    vpp2 = float(np.max(v_a1) - np.min(v_a1))

                    p_idx1 = np.argmax(mag_a0[5:]) + 5
                    p_idx2 = np.argmax(mag_a1[5:]) + 5
                    peak_f1 = freqs[p_idx1]
                    peak_f2 = freqs[p_idx2]

                    # Auto-range window based on the selected trigger channel's frequency
                    is_trig_a0 = ("CH1" in self.trig_src_dd.value or "A0" in self.trig_src_dd.value)
                    if self.autorange_toggle.value:
                        f_ref = float(self.ch1_freq_slider.value) if is_trig_a0 else float(self.ch2_freq_slider.value)
                        period_pts = int(self.fs_per_ch / f_ref) if f_ref > 0 else 500
                        show_pts = max(40, min(5 * period_pts, 500, len(v_a0)))
                    else:
                        show_pts = 500

                    t_x = np.arange(show_pts) * (1e6 / self.fs_per_ch)  # Microseconds
                    p_v1 = v_a0[:show_pts]
                    p_v2 = v_a1[:show_pts]
                    trig_v = float(self.trig_level_slider.value)
                    active_tab = self.tabs.selected_index
                    max_span = float(self.fft_span_dd.value)

                    # Update Active Tab
                    if active_tab == 0:  # Tab 1: Dual Scope
                        with self.fig_dual_scope.batch_update():
                            self.fig_dual_scope.data[0].x = t_x
                            self.fig_dual_scope.data[0].y = p_v1
                            self.fig_dual_scope.data[2].x = t_x
                            self.fig_dual_scope.data[2].y = p_v2

                            # Move trigger line to A0 or A1 depending on selection
                            if is_trig_a0:
                                self.fig_dual_scope.data[1].x = [0, t_x[-1]]
                                self.fig_dual_scope.data[1].y = [trig_v, trig_v]
                                self.fig_dual_scope.data[1].visible = True
                                self.fig_dual_scope.data[3].visible = False
                            else:
                                self.fig_dual_scope.data[3].x = [0, t_x[-1]]
                                self.fig_dual_scope.data[3].y = [trig_v, trig_v]
                                self.fig_dual_scope.data[3].visible = True
                                self.fig_dual_scope.data[1].visible = False

                            self.fig_dual_scope.layout.xaxis2.range = [0, t_x[-1]]

                    elif active_tab == 1:  # Tab 2: Dual FFT
                        with self.fig_dual_fft.batch_update():
                            self.fig_dual_fft.data[0].x = freqs
                            self.fig_dual_fft.data[0].y = mag_a0
                            self.fig_dual_fft.data[1].x = [peak_f1]
                            self.fig_dual_fft.data[1].y = [mag_a0[p_idx1]]
                            self.fig_dual_fft.data[1].text = [f" {peak_f1/1e3:.1f} kHz"]
                            self.fig_dual_fft.data[2].x = freqs
                            self.fig_dual_fft.data[2].y = mag_a1
                            self.fig_dual_fft.data[3].x = [peak_f2]
                            self.fig_dual_fft.data[3].y = [mag_a1[p_idx2]]
                            self.fig_dual_fft.data[3].text = [f" {peak_f2/1e3:.1f} kHz"]
                            self.fig_dual_fft.layout.xaxis2.range = [0, max_span]

                    elif active_tab == 2:  # Tab 3: Channel 1 View (A0)
                        with self.fig_ch1_view.batch_update():
                            self.fig_ch1_view.data[0].x = t_x
                            self.fig_ch1_view.data[0].y = p_v1
                            self.fig_ch1_view.data[1].x = [0, t_x[-1]]
                            self.fig_ch1_view.data[1].y = [trig_v, trig_v]
                            self.fig_ch1_view.data[2].x = freqs
                            self.fig_ch1_view.data[2].y = mag_a0
                            self.fig_ch1_view.layout.xaxis2.range = [0, max_span]

                    elif active_tab == 3:  # Tab 4: Channel 2 View (A1)
                        with self.fig_ch2_view.batch_update():
                            self.fig_ch2_view.data[0].x = t_x
                            self.fig_ch2_view.data[0].y = p_v2
                            self.fig_ch2_view.data[1].x = [0, t_x[-1]]
                            self.fig_ch2_view.data[1].y = [trig_v, trig_v]
                            self.fig_ch2_view.data[2].x = freqs
                            self.fig_ch2_view.data[2].y = mag_a1
                            self.fig_ch2_view.layout.xaxis2.range = [0, max_span]

                    # Status Bar Readouts
                    mode_tag = " (LOCKED)" if mode == "Single" else ""
                    self.readout_ch1.value = (
                        f"<span style='color:#00FFCC; font-family:monospace; font-size:13px; font-weight:bold;'>"
                        f"A0: Vpp={vpp1:.2f}V | f0={peak_f1/1e3:.1f}kHz{mode_tag}</span>"
                    )
                    self.readout_ch2.value = (
                        f"<span style='color:#FF007F; font-family:monospace; font-size:13px; font-weight:bold;'>"
                        f"A1: Vpp={vpp2:.2f}V | f0={peak_f2/1e3:.1f}kHz</span>"
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
        r1 = widgets.HBox(
            [self.start_btn, self.stop_btn, self.force_btn, self.autorange_toggle, self.clear_log_btn, self.readout_ch1, self.readout_ch2],
            layout=widgets.Layout(gap="10px", margin="0 0 8px 0")
        )
        r2 = widgets.HBox([self.trig_mode_dd, self.trig_edge_dd, self.trig_src_dd, self.trig_level_slider, self.trig_level_input])
        r3 = widgets.HBox([self.ch1_shape_dd, self.ch1_amp_slider, self.ch1_freq_slider, self.ch1_freq_input])
        r4 = widgets.HBox([self.ch2_shape_dd, self.ch2_amp_slider, self.ch2_freq_slider, self.ch2_freq_input])
        r5 = widgets.HBox([self.fft_unit_dd, self.fft_span_dd])
        self.control_panel = widgets.VBox([r1, r2, r3, r4, r5], layout=widgets.Layout(margin="0 0 12px 0"))
        display(widgets.VBox([self.control_panel, self.tabs]))