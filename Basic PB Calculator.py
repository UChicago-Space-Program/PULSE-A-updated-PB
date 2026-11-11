from pathlib import Path


class PowerBudgetCalculator:
    def __init__(self):
        # ==========================================
        # 1. HARDWARE DATABASE
        # ==========================================
        self.components = {
            # --- OBC & C&DH ---
            "obc_beaglebone": 2.20,
            "rtc": 0.00,
            "can_transceiver": 0.07,
            "watchdog": 0.01,
            "payload_controller": 0.10,  # Estimated

            # --- Power (PMCU) ---
            "pmcu_stm32": 0.01,
            "current_sensors": 0.08,
            "battery_heater": 8.80,  # Heater (8.8W when full on)

            # --- ADCS ---
            "magnetorquers": 0.45,
            "cubemag": 0.11,
            "adcs_core": 0.96,
            "rw_nominal": 0.45,
            "rw_peak": 2.68,
            "sun_sensor_nom": 0.10,
            "sun_sensor_peak": 0.18,
            "gps": 0.07,

            # --- Comms ---
            "radio_rx": 0.10,
            "radio_tx": 3.12,
            "antenna": 0.03,

            # --- Payload ---
            "fpga": 7.20,
            "edfa": 9.50,
            "seed_laser": 0.12,
            "beacon_laser": 0.73,
            "calib_laser": 0.50,
            "calib_camera": 0.89,
            "fsm_driver": 0.16,
        }

    def calculate_watts(self, state_name):
        """Sums up the power for a specific state."""
        p = 0.0

        # --- BASE LOAD (Always On) ---
        # "pmcu... battery heater (on for 5% of the time at all times)"
        p += (
            self.components["pmcu_stm32"]
            + self.components["watchdog"]
            + self.components["current_sensors"]
        )
        p += self.components["battery_heater"] * 0.05

        # --- STATE DEFINITIONS ---
        if state_name == "No Control":
            pass  # Only base load

        elif state_name == "Detumble":
            p += (
                self.components["obc_beaglebone"]
                + self.components["rtc"]
                + self.components["can_transceiver"]
                + self.components["watchdog"]
            )
            p += (
                self.components["magnetorquers"]
                + self.components["cubemag"]
                + self.components["adcs_core"]
            )
            p += self.components["radio_rx"] + (self.components["radio_tx"] * 0.15)
            p += self.components["gps"] + self.components["antenna"]

        elif "Sun-Pointing" in state_name or state_name == "Pointing Away":
            p += (
                self.components["obc_beaglebone"]
                + self.components["rtc"]
                + self.components["can_transceiver"]
                + self.components["watchdog"]
            )
            # ADCS weighted average
            p += (
                self.components["magnetorquers"]
                + self.components["cubemag"]
                + self.components["adcs_core"]
            )
            p += (self.components["rw_nominal"] * 0.95) + (
                self.components["rw_peak"] * 0.05
            )
            p += (self.components["sun_sensor_nom"] * 0.98) + (
                self.components["sun_sensor_peak"] * 0.02
            )
            p += self.components["radio_rx"] + self.components["gps"] + self.components["antenna"]

        elif state_name == "RF Pass":
            p += (
                self.components["obc_beaglebone"]
                + self.components["rtc"]
                + self.components["can_transceiver"]
                + self.components["watchdog"]
            )
            p += (
                self.components["magnetorquers"]
                + self.components["cubemag"]
                + self.components["adcs_core"]
            )
            p += (self.components["rw_nominal"] * 0.95) + (
                self.components["rw_peak"] * 0.05
            )
            p += (self.components["sun_sensor_nom"] * 0.98) + (
                self.components["sun_sensor_peak"] * 0.02
            )
            p += self.components["radio_rx"] + (self.components["radio_tx"] * 0.25)  # Tx 25%
            p += self.components["gps"] + self.components["antenna"]

        elif state_name == "Payload - Calibration":
            # Start with standard ops
            p += self.calculate_watts("Sun-Pointing")
            # Add payload controller
            p += self.components["payload_controller"]
            # Add calibration hardware
            p += (
                self.components["calib_laser"]
                + self.components["calib_camera"]
                + self.components["fpga"]
                + self.components["fsm_driver"]
            )

        elif state_name == "Payload - Acquisition":
            # Start with standard ops
            p += self.calculate_watts("Sun-Pointing")
            # Add payload controller
            p += self.components["payload_controller"]
            # Add acquisition hardware (high power)
            p += (
                self.components["edfa"]
                + self.components["seed_laser"]
                + self.components["beacon_laser"]
                + self.components["fsm_driver"]
                + self.components["fpga"]
            )

        return round(p, 2)

    def load_power_profile(self, filepath):
        """
        Load a whitespace-separated STK power file.

        Expected format:
        - Row 1: column names
        - Row 2: units row (ignored)
        - Row 3+: numeric rows
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Power profile not found: {path}")

        lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if len(lines) < 3:
            raise ValueError(f"Power profile has too few rows: {path}")

        headers = lines[0].split()
        data_rows = lines[2:]  # skip units row

        data = {col: [] for col in headers}
        for raw in data_rows:
            parts = raw.split()
            if len(parts) != len(headers):
                continue
            for col, value in zip(headers, parts):
                data[col].append(float(value))

        return data

    def compute_orbit_energy_wh(self, profile_data, power_col="MinTotalPower"):
        """
        Compute Wh per orbit from a profile column.

        Supports two data styles automatically:
        1) Instantaneous power samples (W): integrates using dt.
        2) Monotonic cumulative series: uses per-orbit delta (last - first).
        """
        elapsed_col = "PULSE.ElapsedTime"
        orbit_col = "PULSE.OrbitNumberBounded"

        elapsed_days = profile_data[elapsed_col]
        orbit_ids = profile_data[orbit_col]
        powers = profile_data[power_col]

        if len(elapsed_days) < 2:
            raise ValueError("Not enough profile samples to integrate energy")

        orbit_numbers = [int(round(v)) for v in orbit_ids]

        # Heuristic: if almost monotonic increasing, treat as cumulative quantity.
        nondecreasing_count = 0
        for prev, curr in zip(powers[:-1], powers[1:]):
            if curr >= prev:
                nondecreasing_count += 1
        monotonic_ratio = nondecreasing_count / max(1, len(powers) - 1)
        is_cumulative = monotonic_ratio > 0.995

        if is_cumulative:
            first_by_orbit = {}
            last_by_orbit = {}
            for orbit, value in zip(orbit_numbers, powers):
                if orbit not in first_by_orbit:
                    first_by_orbit[orbit] = value
                last_by_orbit[orbit] = value

            orbit_keys = sorted(first_by_orbit.keys())
            return [last_by_orbit[k] - first_by_orbit[k] for k in orbit_keys]

        dt_hours = (elapsed_days[1] - elapsed_days[0]) * 24.0
        energy_by_orbit = {}
        for orbit, power in zip(orbit_numbers, powers):
            energy_by_orbit[orbit] = energy_by_orbit.get(orbit, 0.0) + (power * dt_hours)

        orbit_keys = sorted(energy_by_orbit.keys())
        return [energy_by_orbit[k] for k in orbit_keys]


# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    calc = PowerBudgetCalculator()

    print(f"{'STATE NAME':<25} | {'POWER (Watts)':<15}")
    print("-" * 45)

    states = [
        "No Control",
        "Detumble",
        "Sun-Pointing",
        "Pointing Away",
        "RF Pass",
        "Payload - Calibration",
        "Payload - Acquisition",
    ]

    for s in states:
        watts = calc.calculate_watts(s)
        print(f"{s:<25} | {watts} W")
