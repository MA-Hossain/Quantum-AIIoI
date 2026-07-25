from sagin_research_sim.config.radio_config import RadioConfig, ResourceGridConfig


def make_mmwave_radio(grid=None):
    if grid is None:
        grid = ResourceGridConfig(num_freq_blocks=275, num_time_slots=10, subcarrier_spacing_hz=120e3)
    return RadioConfig(
        name="mmwave_28ghz",
        carrier_freq_hz=28e9,
        bandwidth_hz=100e6,
        tx_power_dbm=24.0,
        noise_figure_db=8.0,
        max_spectral_efficiency_bps_hz=7.0,
        resource_grid=grid,
    )
