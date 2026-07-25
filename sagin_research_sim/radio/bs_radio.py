from sagin_research_sim.config.radio_config import RadioConfig, ResourceGridConfig


def make_bs_radio(grid=None):
    # Python 3.9 compatible: avoid Optional[ResourceGridConfig] type syntax.
    if grid is None:
        grid = ResourceGridConfig(num_freq_blocks=100, num_time_slots=10, subcarrier_spacing_hz=15e3)
    return RadioConfig(
        name="sub6_bs",
        carrier_freq_hz=3.5e9,
        bandwidth_hz=20e6,
        tx_power_dbm=23.0,
        noise_figure_db=7.0,
        max_spectral_efficiency_bps_hz=6.0,
        resource_grid=grid,
    )
