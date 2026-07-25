from sagin_research_sim.config.radio_config import RadioConfig, ResourceGridConfig


def make_satellite_radio(grid=None):
    if grid is None:
        grid = ResourceGridConfig(num_freq_blocks=200, num_time_slots=10, subcarrier_spacing_hz=30e3)
    return RadioConfig(
        name="ku_leo_satellite",
        carrier_freq_hz=12e9,
        bandwidth_hz=50e6,
        tx_power_dbm=30.0,
        noise_figure_db=9.0,
        max_spectral_efficiency_bps_hz=4.0,
        resource_grid=grid,
    )
