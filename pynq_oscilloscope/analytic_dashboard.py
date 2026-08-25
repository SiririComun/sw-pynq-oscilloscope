"""
pynq_oscilloscope.analytic_dashboard: Professional Multi-Domain Acoustic Analytics Dashboard.
Features 3-strip stacked channel isolation, decoupled dual Y-axes, full-width STFT waterfall,
and live spatial energy balance monitoring.
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

from pynq_oscilloscope.analytics import AcousticAnalytics
from pynq_oscilloscope.hw_trigger import HardwareTrigger
from pynq_oscilloscope.ad3_wavegen import AD3SignalGenerator
from pynq_oscilloscope.audio_dashboard import AudioDashboard


class AcousticAnalyticDashboard:
    """
    Professional Multi-Domain Acoustic Diagnostic Dashboard.
    Implements 3-strip dedicated channel grids, dual-axis pitch/phase tracking,
    and full-width spectrogram rendering.
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

        self._build_ui()
        self._build_plots()
        self._setup_callbacks()

    def _build_ui(self):
        # 1. Action Row & Spatial Readouts
        self.start_btn = widgets.Button(description="Start Live", button_style="success", icon="play", layout=widgets.Layout(width="115px"))
        self.stop_btn = widgets.Button(description="Stop", button_style="danger", icon="stop", layout=widgets.Layout(width="95px"))
        self.force_btn = widgets.Button(description="Force Trig", button_style="warning", icon="bolt", layout=widgets.Layout(width="115px"))
        self.clear_log_btn = widgets.Button(description="Clear Log", button_style="", icon="trash", layout=widgets.Layout(width="100px"))

        self.readout_metrics = widgets.HTML("<span style='color:#00FFCC; font-family:monospace; font-size:13px; font-weight:bold;'>A0: 0.00V | A1: 0.00V | f0: 0.0Hz | ΔL: +0.0dB | Δϕ: +0.0° | [   ● C   ]</span>")

        # 2. Controls Row
        self.trig_mode_dd = widgets.Dropdown(options=["Auto", "Normal", "Single"], value="Auto", description="Trig Mode:", layout=widgets.Layout(width="180px"))
        self.trig_level_slider = widgets.FloatSlider(value=1.65, min=0.0, max=3.3, step=0.05, description="Trig Level:", continuous_update=False, layout=widgets.Layout(width="220px"))
        self.stft_window_dd = widgets.Dropdown(options=["blackmanharris", "hanning", "hamming"], value="blackmanharris", description="STFT Win:", layout=widgets.Layout(width="200px"))
        self.profile_dd = widgets.Dropdown(options=[("Audio (50 kSPS)", "audio"), ("Bass Zoom (10 kSPS)", "bass_zoom"), ("Scope (500 kSPS)", "oscilloscope")], value="audio", description="Regime:", layout=widgets.Layout(width="220px"))

    def _build_plots(self):
        t_ms = np.linspace(0, self.total_duration_ms, self.num_pts_per_ch - 16)
        initial_freq = np.linspace(0, self.fs_per_ch / 2.0, 128)

        # =====================================================================
        # Tab 1: 3-Strip Stacked Amplitude & Energy
        # =====================================================================
        self.fig_amp_time = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            subplot_titles=(
                "<b>Channel 1: A0 (Voltage & Symmetrical Envelope)</b>",
                "<b>Channel 2: A1 (Voltage & Symmetrical Envelope)</b>",
                "<b>Spatial Energy Balance: Inter-aural Level Difference ΔL(t) [dB]</b>"
            )
        )
        self.fig_amp_time = go.FigureWidget(self.fig_amp_time)
        # Strip 1: CH1 / A0
        self.fig_amp_time.add_scatter(x=t_ms, y=[1.65]*len(t_ms), mode="lines", line=dict(color="rgba(0, 255, 204, 0.35)", width=1.1), name="A0 Raw (V)", row=1, col=1)
        self.fig_amp_time.add_scatter(x=t_ms, y=[1.65]*len(t_ms), mode="lines", line=dict(color="#00FFCC", width=1.8), name="A0 +A(t)", row=1, col=1)
        self.fig_amp_time.add_scatter(x=t_ms, y=[1.65]*len(t_ms), mode="lines", line=dict(color="#00FFCC", width=1.8), name="A0 -A(t)", showlegend=False, row=1, col=1)
        # Strip 2: CH2 / A1
        self.fig_amp_time.add_scatter(x=t_ms, y=[1.65]*len(t_ms), mode="lines", line=dict(color="rgba(255, 0, 127, 0.35)", width=1.1), name="A1 Raw (V)", row=2, col=1)
        self.fig_amp_time.add_scatter(x=t_ms, y=[1.65]*len(t_ms), mode="lines", line=dict(color="#FF007F", width=1.8), name="A1 +A(t)", row=2, col=1)
        self.fig_amp_time.add_scatter(x=t_ms, y=[1.65]*len(t_ms), mode="lines", line=dict(color="#FF007F", width=1.8), name="A1 -A(t)", showlegend=False, row=2, col=1)
        # Strip 3: ILD Ratio
        self.fig_amp_time.add_scatter(x=t_ms, y=[0.0]*len(t_ms), mode="lines", line=dict(color="#FFA500", width=1.8), name="ILD ΔL(t) [dB]", row=3, col=1)

        self.fig_amp_time.update_layout(template="plotly_dark", height=560, margin=dict(l=45, r=25, t=40, b=35), uirevision="t1")
        self.fig_amp_time.update_yaxes(range=[0, 3.3], title="A0 (V)", row=1, col=1)
        self.fig_amp_time.update_yaxes(range=[0, 3.3], title="A1 (V)", row=2, col=1)
        self.fig_amp_time.update_yaxes(range=[-25, 25], title="ΔL (dB)", row=3, col=1)
        self.fig_amp_time.update_xaxes(range=[0, self.total_duration_ms], title="Time (Milliseconds)", row=3, col=1)

        # =====================================================================
        # Tab 2: Full-Width Spectrogram & Decoupled Pitch/Phase (Dual Y-Axes)
        # =====================================================================
        self.fig_freq_time = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
            specs=[[{}], [{"secondary_y": True}]],
            subplot_titles=(
                "<b>Rolling STFT Waterfall Spectrogram (Channel 1 / A0)</b>",
                "<b>Sub-Hertz Pitch Tracking f0(t) [Right Axis] & Phase Difference Δϕ(t) [Left Axis]</b>"
            )
        )
        self.fig_freq_time = go.FigureWidget(self.fig_freq_time)
        dummy_z = np.full((128, 30), -90.0)
        # Heatmap
        self.fig_freq_time.add_heatmap(
            z=dummy_z, x=np.linspace(0, self.total_duration_ms, 30), y=initial_freq,
            colorscale="Turbo", zmin=-85, zmax=0,
            colorbar=dict(title="dBV", len=0.48, y=0.78), row=1, col=1
        )
        # Left Y-Axis: Phase Difference (-180 to +180 deg)
        self.fig_freq_time.add_scatter(
            x=[0, self.total_duration_ms], y=[0, 0], mode="lines",
            line=dict(color="#FFA500", width=1.8, dash="dash"),
            name="Phase Diff Δϕ(t) [°]", secondary_y=False, row=2, col=1
        )
        # Right Y-Axis: Pitch f0 (0 to 25 kHz)
        self.fig_freq_time.add_scatter(
            x=[0, self.total_duration_ms], y=[1000, 1000], mode="lines+markers",
            line=dict(color="#00FFCC", width=2.0),
            name="Pitch f0(t) [Hz]", secondary_y=True, row=2, col=1
        )

        self.fig_freq_time.update_layout(template="plotly_dark", height=560, margin=dict(l=45, r=55, t=40, b=35), uirevision="t2")
        self.fig_freq_time.update_yaxes(range=[0, self.fs_per_ch / 2.0], title="Frequency (Hz)", row=1, col=1)
        self.fig_freq_time.update_yaxes(range=[-180, 180], title="Phase Δϕ (°)", secondary_y=False, row=2, col=1)
        self.fig_freq_time.update_yaxes(range=[0, self.fs_per_ch / 2.0], title="Pitch f0 (Hz)", secondary_y=True, row=2, col=1)
        self.fig_freq_time.update_xaxes(range=[0, self.total_duration_ms], title="Time (Milliseconds)", row=2, col=1)

        # =====================================================================
        # Tab 3: Dedicated Phase & Wave Timing (TDoA Zero-Crossings)
        # =====================================================================
        self.fig_phase_timing = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
            subplot_titles=(
                "<b>Zero-Centered Waveform Overlay (A0 vs A1 Timing Lead/Lag)</b>",
                "<b>Continuous Instantaneous Phase Difference Δϕ(t)</b>"
            )
        )
        self.fig_phase_timing = go.FigureWidget(self.fig_phase_timing)
        self.fig_phase_timing.add_scatter(x=t_ms, y=[0.0]*len(t_ms), mode="lines", line=dict(color="#00FFCC", width=1.8), name="A0 AC Wave (V)", row=1, col=1)
        self.fig_phase_timing.add_scatter(x=t_ms, y=[0.0]*len(t_ms), mode="lines", line=dict(color="#FF007F", width=1.8), name="A1 AC Wave (V)", row=1, col=1)
        self.fig_phase_timing.add_scatter(x=t_ms, y=[0.0]*len(t_ms), mode="lines", line=dict(color="#FFA500", width=1.8), name="Phase Δϕ(t) [°]", row=2, col=1)

        self.fig_phase_timing.update_layout(template="plotly_dark", height=540, margin=dict(l=45, r=25, t=40, b=35), uirevision="t3")
        self.fig_phase_timing.update_yaxes(range=[-1.65, 1.65], title="AC Voltage (V)", row=1, col=1)
        self.fig_phase_timing.update_yaxes(range=[-180, 180], title="Phase Δϕ (°)", row=2, col=1)
        self.fig_phase_timing.update_xaxes(range=[0, self.total_duration_ms], title="Time (Milliseconds)", row=2, col=1)

        # Tabs Container
        self.tabs = widgets.Tab(children=[self.fig_amp_time, self.fig_freq_time, self.fig_phase_timing])
        self.tabs.set_title(0, "📈 Amplitude & ILD")
        self.tabs.set_title(1, "📊 Spectrogram & Pitch")
        self.tabs.set_title(2, "⏱ Phase & TDoA")

    def _setup_callbacks(self):
        self.start_btn.on_click(lambda _: self.start())
        self.stop_btn.on_click(lambda _: self.stop())
        self.force_btn.on_click(lambda _: self._on_force_clicked())
        self.clear_log_btn.on_click(lambda _: self._on_clear_log_clicked())
        self.profile_dd.observe(self._on_profile_change, names="value")

    def _on_profile_change(self, change):
        if self.overlay and hasattr(self.overlay, "set_profile"):
            info = self.overlay.set_profile(change["new"])
            self.fs_per_ch = info["sample_rate_hz"]
            self.total_duration_ms = info["time_window_ms"]
            print(f"[Analytics] Regime set to '{change['new']}' ({self.fs_per_ch:.0f} Hz, {self.total_duration_ms:.2f} ms window)")

    def _on_force_clicked(self):
        if self.trig_mode_dd.value == "Single":
            self._single_done = False
        if self.trigger:
            self.trigger.force_trigger()

    def _on_clear_log_clicked(self):
        clear_output(wait=True)
        display(widgets.VBox([self.control_panel, self.tabs]))

    def _update_loop(self):
        dma_time = self.overlay.axi_dma_0
        trig = self.trigger

        # Initialize XADC into Continuous Sequence Mode (0x2000)
        if hasattr(self.overlay, "xadc_wiz_0"):
            self.overlay.xadc_wiz_0.mmio.write(0x304, 0x2000)
            self.overlay.xadc_wiz_0.mmio.write(0x320, 0x0000)
            self.overlay.xadc_wiz_0.mmio.write(0x324, 0x0202)

        # Reset DMA 0
        dma_time.mmio.write(0x30, 0x04)
        time.sleep(0.005)
        dma_time.recvchannel.start()

        if trig:
            trig.mmio.write(0x0C, 5000000)
            trig.set_threshold(float(self.trig_level_slider.value))

        buf_time = allocate(shape=(self.packet_size,), dtype="u2")
        dma_armed = False
        print(f"[AcousticAnalytics] High-Performance Diagnostic Engine Active ({self.fs_per_ch:.0f} Hz Synchronous Stream)")

        try:
            while self._is_running:
                mode = self.trig_mode_dd.value
                if mode == "Single" and self._single_done:
                    time.sleep(0.02)
                    continue

                if not dma_armed:
                    dma_time.recvchannel.transfer(buf_time)
                    ctrl = (1 << 0) | (1 << 3) if mode == "Single" else ((1 << 0) | (1 << 1))
                    trig.mmio.write(0x00, ctrl)
                    dma_armed = True

                if dma_time.recvchannel.idle:
                    dma_armed = False

                    raw = np.array(buf_time)
                    v_a0 = (raw[0::2] >> 4) * (3.3 / 4095.0)
                    v_a1 = (raw[1::2] >> 4) * (3.3 / 4095.0)

                    # Crop boundary words
                    p_v1 = v_a0[8:-8]
                    p_v2 = v_a1[8:-8]
                    n_pts = len(p_v1)
                    t_ms = np.linspace(0, (n_pts / self.fs_per_ch) * 1000.0, n_pts)

                    # 1. Analytic Amplitude Envelopes & Shrouds
                    env_a0 = AcousticAnalytics.extract_analytic_envelope(p_v1, remove_dc=True)
                    env_a1 = AcousticAnalytics.extract_analytic_envelope(p_v2, remove_dc=True)
                    v_mid0 = float(np.mean(p_v1))
                    v_mid1 = float(np.mean(p_v2))

                    # 2. Inter-aural Level Difference (ILD)
                    ild_db, ste_a0, ste_a1, ild_idx = AcousticAnalytics.compute_ild(p_v1, p_v2, window_len=64, hop_size=16)
                    t_ild_ms = (ild_idx / self.fs_per_ch) * 1000.0

                    # 3. Sliding STFT Spectrogram (Full width)
                    stft_times, stft_freqs, spec_mat = AcousticAnalytics.compute_sliding_stft(
                        p_v1, fs=self.fs_per_ch, nperseg=256, noverlap=192, window=self.stft_window_dd.value
                    )
                    stft_times_full = np.linspace(0, t_ms[-1], spec_mat.shape[1])

                    # 4. Instantaneous Phase Difference & Sub-Hertz Pitch
                    phase_diff_deg = AcousticAnalytics.compute_instantaneous_phase_diff(p_v1, p_v2)
                    mean_phase_deg = float(np.median(phase_diff_deg))

                    spec_full = 20.0 * np.log10(np.maximum(np.abs(np.fft.rfft(p_v1 - v_mid0)) / (n_pts / 2.0), 1e-6))
                    freq_axis_full = np.fft.rfftfreq(n_pts, d=1.0 / self.fs_per_ch)
                    p_f0, p_mag = AcousticAnalytics.track_sub_hertz_pitch(freq_axis_full, spec_full, min_freq_hz=30.0)

                    vpp1, vpp2 = float(np.ptp(p_v1)), float(np.ptp(p_v2))
                    mean_ild = float(np.mean(ild_db))

                    # Live Pan Balance Indicator
                    if mean_ild > 8.0:
                        pan_str = "[◀◀ L       ]"
                    elif mean_ild > 2.5:
                        pan_str = "[ ◀ L       ]"
                    elif mean_ild < -8.0:
                        pan_str = "[       R ▶▶]"
                    elif mean_ild < -2.5:
                        pan_str = "[       R ▶ ]"
                    else:
                        pan_str = "[    ● C    ]"

                    active_tab = self.tabs.selected_index

                    if active_tab == 0:  # Tab 1: 3-Strip Stacked Amplitude & ILD
                        with self.fig_amp_time.batch_update():
                            # Strip 1: A0
                            self.fig_amp_time.data[0].x = t_ms
                            self.fig_amp_time.data[0].y = p_v1
                            self.fig_amp_time.data[1].x = t_ms
                            self.fig_amp_time.data[1].y = v_mid0 + env_a0
                            self.fig_amp_time.data[2].x = t_ms
                            self.fig_amp_time.data[2].y = v_mid0 - env_a0
                            # Strip 2: A1
                            self.fig_amp_time.data[3].x = t_ms
                            self.fig_amp_time.data[3].y = p_v2
                            self.fig_amp_time.data[4].x = t_ms
                            self.fig_amp_time.data[4].y = v_mid1 + env_a1
                            self.fig_amp_time.data[5].x = t_ms
                            self.fig_amp_time.data[5].y = v_mid1 - env_a1
                            # Strip 3: ILD
                            self.fig_amp_time.data[6].x = t_ild_ms
                            self.fig_amp_time.data[6].y = ild_db
                            self.fig_amp_time.layout.xaxis3.range = [0, t_ms[-1]]

                    elif active_tab == 1:  # Tab 2: Spectrogram & Decoupled Pitch/Phase
                        with self.fig_freq_time.batch_update():
                            self.fig_freq_time.data[0].z = spec_mat
                            self.fig_freq_time.data[0].x = stft_times_full
                            self.fig_freq_time.data[0].y = stft_freqs
                            self.fig_freq_time.data[1].x = t_ms
                            self.fig_freq_time.data[1].y = phase_diff_deg
                            self.fig_freq_time.data[2].x = [0, t_ms[-1]]
                            self.fig_freq_time.data[2].y = [p_f0, p_f0]
                            self.fig_freq_time.data[2].text = [f"f0 = {p_f0:.1f} Hz"]
                            self.fig_freq_time.layout.xaxis2.range = [0, t_ms[-1]]

                    elif active_tab == 2:  # Tab 3: Dedicated Phase Alignment & TDoA
                        with self.fig_phase_timing.batch_update():
                            self.fig_phase_timing.data[0].x = t_ms
                            self.fig_phase_timing.data[0].y = p_v1 - v_mid0
                            self.fig_phase_timing.data[1].x = t_ms
                            self.fig_phase_timing.data[1].y = p_v2 - v_mid1
                            self.fig_phase_timing.data[2].x = t_ms
                            self.fig_phase_timing.data[2].y = phase_diff_deg
                            self.fig_phase_timing.layout.xaxis2.range = [0, t_ms[-1]]

                    self.readout_metrics.value = (
                        f"<span style='color:#00FFCC; font-family:monospace; font-size:13px; font-weight:bold;'>"
                        f"A0: {vpp1:.2f}V | A1: {vpp2:.2f}V | f0: {p_f0:.1f}Hz | ΔL: {mean_ild:+.1f}dB | Δϕ: {mean_phase_deg:+.1f}° | {pan_str}"
                        f"</span>"
                    )

                    if mode == "Single":
                        self._single_done = True
                    time.sleep(0.033)
                else:
                    time.sleep(0.005)

        except Exception as e:
            print(f"[AcousticAnalytics ERROR]: {e}")
        finally:
            self._is_running = False
            buf_time.close()
            print("[AcousticAnalytics] Dashboard stopped cleanly.")

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
        r1 = widgets.HBox([self.start_btn, self.stop_btn, self.force_btn, self.clear_log_btn, self.readout_metrics], layout=widgets.Layout(gap="10px", margin="0 0 8px 0"))
        r2 = widgets.HBox([self.trig_mode_dd, self.trig_level_slider, self.stft_window_dd, self.profile_dd])
        self.control_panel = widgets.VBox([r1, r2], layout=widgets.Layout(margin="0 0 12px 0"))
        display(widgets.VBox([self.control_panel, self.tabs]))