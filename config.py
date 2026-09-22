"""
Configuration for the concurrent-LCM experiments.

FR2 is the primary deployment; FR1 is the mechanism control (the B->P
coupling should weaken substantially without analog beam selection, so a
shrinking interaction term in FR1 is evidence the effect comes from the
beam-CSI dependency rather than from the harness).

Every stochastic element is addressed by an explicit seed so that the eight
forced-subset arms of the interaction experiment run on identical channel
realizations.
"""

from dataclasses import dataclass, asdict
import json


@dataclass(frozen=True)
class ChannelConfig:
    """TR 38.901 CDL configuration for one deployment."""

    name: str
    carrier_frequency: float          # [Hz]
    subcarrier_spacing: float         # [Hz]
    num_subcarriers: int              # N_f
    cdl_model: str                    # "A".."E"
    delay_spread: float               # [s]

    bs_num_rows: int
    bs_num_cols: int
    bs_polarization: str              # "single" | "dual"
    ut_num_rows: int
    ut_num_cols: int
    ut_polarization: str

    min_speed: float                  # [m/s]
    max_speed: float                  # [m/s]

    # Required in FR2: without it the best beam is near-constant across
    # realizations and beam prediction becomes degenerate.
    randomize_orientation: bool = True

    @property
    def slot_duration(self) -> float:
        return 1e-3 / (self.subcarrier_spacing / 15e3)

    @property
    def sampling_frequency(self) -> float:
        return 1.0 / self.slot_duration

    @property
    def num_bs_ant(self) -> int:
        pol = 2 if self.bs_polarization == "dual" else 1
        return self.bs_num_rows * self.bs_num_cols * pol

    @property
    def num_ut_ant(self) -> int:
        pol = 2 if self.ut_polarization == "dual" else 1
        return self.ut_num_rows * self.ut_num_cols * pol


FR2 = ChannelConfig(
    name="FR2", carrier_frequency=28e9, subcarrier_spacing=120e3,
    num_subcarriers=132, cdl_model="B", delay_spread=30e-9,
    bs_num_rows=4, bs_num_cols=8, bs_polarization="single",
    ut_num_rows=1, ut_num_cols=2, ut_polarization="single",
    min_speed=1.0, max_speed=15.0, randomize_orientation=True,
)

FR1 = ChannelConfig(
    name="FR1", carrier_frequency=3.5e9, subcarrier_spacing=30e3,
    num_subcarriers=132, cdl_model="C", delay_spread=300e-9,
    bs_num_rows=2, bs_num_cols=8, bs_polarization="single",
    ut_num_rows=1, ut_num_cols=2, ut_polarization="single",
    min_speed=1.0, max_speed=15.0, randomize_orientation=True,
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Episode structure and seeding."""

    episode_slots: int = 200          # T
    num_realizations: int = 512       # the n that carries the statistics
    base_seed: int = 20260907
    meas_noise_std_db: float = 1.0
    setB_fraction: float = 0.25
    cache_dir: str = "cache/realizations"


@dataclass(frozen=True)
class PipelineConfig:
    """Pipeline structure, feedback timing, and the utility model."""

    # CSI report granularity: subbands, not raw subcarriers.
    csi_subbands: int = 12

    # CSI prediction: observe W past reports, predict k slots ahead.
    pred_window: int = 8
    feedback_delay: int = 2           # k

    # CSI compression: latent dimension and bits per latent element.
    latent_dim: int = 16
    bits_per_latent: int = 4

    # Link adaptation
    snr_ref_db: float = 15.0          # reference SNR at unit channel gain
    la_margin_db: float = 1.0         # initial outer-loop margin

    # Outer-loop link adaptation. A FIXED margin makes the utility reward
    # noisy CSI: with 2 dB MCS steps, an optimistic quantization error
    # sometimes lands a higher MCS that still succeeds, and that gain can
    # outweigh the extra outages. A coarse 2-bit fallback then beats a
    # -27 dB autoencoder, which is nonsense. OLLA removes the bias by
    # driving every arm to the same BLER, so throughput differences reflect
    # CSI quality rather than which way the errors happen to lean.
    target_bler: float = 0.1
    olla_step_db: float = 0.1

    # Beam sweep overhead. The non-AI beam fallback is *accurate* but costly:
    # it measures the full codebook rather than Set B. This asymmetry is the
    # mechanism most likely to make the interaction term sub-additive, so it
    # must be modelled explicitly rather than folded into an accuracy penalty.
    sweep_period_slots: int = 20
    slots_per_beam_measurement: float = 0.05

    # Beam prediction mode. "spatial" = Rel-18 BM-Case1 (Set B -> Set A, same
    # slot). "temporal" = predict best beam at t+k. Spatial is the
    # conservative default; temporal strengthens the B->P coupling but invites
    # the objection that the coupling was engineered.
    beam_mode: str = "spatial"

    # Training
    train_realizations: int = 1024
    val_realizations: int = 128
    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3


DEFAULT_EXPERIMENT = ExperimentConfig()
DEFAULT_PIPELINE = PipelineConfig()


def dump(cfg) -> str:
    return json.dumps(asdict(cfg), indent=2, sort_keys=True)


if __name__ == "__main__":
    for c in (FR2, FR1):
        print(f"{c.name}: slot={c.slot_duration*1e6:.1f} us, "
              f"fs={c.sampling_frequency:.0f} Hz, "
              f"BS ant={c.num_bs_ant}, UT ant={c.num_ut_ant}")
