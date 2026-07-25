from sagin_research_sim.config.radio_config import RadioConfig, ResourceGridConfig


def make_wifi_radio(grid=None):
    if grid is None:
        grid = ResourceGridConfig(num_freq_blocks=64, num_time_slots=10, subcarrier_spacing_hz=312.5e3, subcarriers_per_rb=1)
    return RadioConfig(
        name="wifi6",
        carrier_freq_hz=5.2e9,
        bandwidth_hz=20e6,
        tx_power_dbm=20.0,
        noise_figure_db=6.0,
        max_spectral_efficiency_bps_hz=8.0,
        resource_grid=grid,
    )
