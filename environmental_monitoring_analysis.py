"""
Environmental Monitoring Analysis Program

This program loads environmental sensor configuration data from a JSON file and
sensor readings from a CSV file. It validates each reading, classifies valid
readings as normal, warning, or critical, stores invalid entries separately,
creates alerts where thresholds are exceeded, calculates summary statistics,
and writes several output files for reporting and validation purposes.

Expected input files:
    - environmental_sensor_config.json
    - environmental_sensor_readings.csv

Generated output files:
    - valid_readings.csv
    - invalid_readings.csv
    - alerts.csv
    - environmental_monitoring_report.txt

The code is intentionally structured using object-oriented programming so that
sensor definitions, readings, validation, alert generation, statistical
analysis, and reporting are all separated into clear responsibilities.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any


@dataclass
class Sensor:
    """
    Represent a single environmental sensor and all threshold information used
    to validate and classify its readings.

    Attributes:
        sensor_id:
            Unique identifier used in the CSV file, such as 'temperature' or
            'humidity'.
        name:
            Human-readable sensor name used in reports.
        unit:
            Measurement unit for the sensor, such as 'C', '%', 'hPa', or 'lux'.
        min_valid:
            Minimum physically or operationally valid reading accepted by the
            program.
        max_valid:
            Maximum physically or operationally valid reading accepted by the
            program.
        warning_low:
            Lower warning threshold. If a value falls below this level but is
            still within the valid range, it is classified as a warning.
        warning_high:
            Upper warning threshold. If a value rises above this level but is
            still within the valid range, it is classified as a warning.
        critical_low:
            Lower critical threshold. If a value falls below this level and is
            still within the valid range, it is classified as critical.
        critical_high:
            Upper critical threshold. If a value rises above this level and is
            still within the valid range, it is classified as critical.
    """

    sensor_id: str
    name: str
    unit: str
    min_valid: float
    max_valid: float
    warning_low: float
    warning_high: float
    critical_low: float
    critical_high: float

    def validate_value(self, value: float) -> bool:
        """
        Check whether a reading falls inside the valid operating range.

        Args:
            value:
                Numeric reading value to be checked.

        Returns:
            True if the value is inside the valid range, otherwise False.
        """
        return self.min_valid <= value <= self.max_valid

    def classify_status(self, value: float) -> str:
        """
        Classify a valid reading as normal, warning, or critical.

        The classification logic assumes the value has already passed the
        validity check. Critical conditions take priority over warnings.

        Args:
            value:
                Numeric reading value to classify.

        Returns:
            A string equal to 'normal', 'warning', or 'critical'.
        """
        if value <= self.critical_low or value >= self.critical_high:
            return "critical"
        if value <= self.warning_low or value >= self.warning_high:
            return "warning"
        return "normal"

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the sensor configuration into a dictionary representation.

        Returns:
            Dictionary containing the sensor configuration fields.
        """
        return {
            "sensor_id": self.sensor_id,
            "name": self.name,
            "unit": self.unit,
            "min_valid": self.min_valid,
            "max_valid": self.max_valid,
            "warning_low": self.warning_low,
            "warning_high": self.warning_high,
            "critical_low": self.critical_low,
            "critical_high": self.critical_high,
        }


@dataclass
class Reading:
    """
    Represent one sensor reading imported from the CSV file.

    Attributes:
        timestamp:
            Parsed datetime value for the reading.
        sensor_id:
            Identifier linking the reading to a configured sensor.
        value:
            Numeric sensor reading.
        status:
            Classification after validation. Typical values are 'normal',
            'warning', 'critical', or 'invalid'.
        is_valid:
            Boolean showing whether the reading passed validation.
        error_message:
            Description of the validation failure when a reading is invalid.
    """

    timestamp: datetime | None
    sensor_id: str
    value: float | None
    status: str = "unprocessed"
    is_valid: bool = False
    error_message: str = ""

    def mark_valid(self, status: str) -> None:
        """
        Mark the reading as valid and assign a status classification.

        Args:
            status:
                Classification string such as 'normal', 'warning', or
                'critical'.
        """
        self.is_valid = True
        self.status = status
        self.error_message = ""

    def mark_invalid(self, error_message: str) -> None:
        """
        Mark the reading as invalid and store a reason.

        Args:
            error_message:
                Human-readable explanation of why the reading failed validation.
        """
        self.is_valid = False
        self.status = "invalid"
        self.error_message = error_message

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the reading into a dictionary for CSV export.

        Returns:
            Dictionary representation of the reading.
        """
        return {
            "timestamp": self.timestamp.isoformat() if self.timestamp else "",
            "sensor_id": self.sensor_id,
            "value": self.value if self.value is not None else "",
            "status": self.status,
            "is_valid": self.is_valid,
            "error_message": self.error_message,
        }


@dataclass
class Alert:
    """
    Represent an alert created when a reading reaches a warning or critical
    condition.

    Attributes:
        timestamp:
            Time associated with the alert.
        sensor_id:
            Sensor that triggered the alert.
        severity:
            Alert severity, typically 'warning' or 'critical'.
        message:
            Human-readable explanation of the alert.
        reading_value:
            Sensor value that triggered the alert.
    """

    timestamp: datetime
    sensor_id: str
    severity: str
    message: str
    reading_value: float

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the alert into a dictionary for CSV export.

        Returns:
            Dictionary representation of the alert.
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "sensor_id": self.sensor_id,
            "severity": self.severity,
            "message": self.message,
            "reading_value": self.reading_value,
        }


@dataclass
class AlertRule:
    """
    Represent a rule used to decide whether a validated reading should generate
    an alert.

    This class is kept explicit even though the logic is simple, because it
    makes the object-oriented design clearer and leaves room for more advanced
    rule handling in the future.

    Attributes:
        rule_id:
            Unique identifier for the rule.
        sensor_id:
            Sensor to which the rule applies.
        condition_type:
            Classification that should trigger the alert, such as 'warning' or
            'critical'.
        severity:
            Severity level assigned to the created alert.
        message:
            Alert message template.
    """

    rule_id: str
    sensor_id: str
    condition_type: str
    severity: str
    message: str

    def evaluate(self, reading: Reading) -> bool:
        """
        Determine whether the rule should trigger for a given reading.

        Args:
            reading:
                Reading object that has already been validated and classified.

        Returns:
            True if the rule applies, otherwise False.
        """
        return (
            reading.is_valid
            and reading.sensor_id == self.sensor_id
            and reading.status == self.condition_type
        )

    def build_alert(self, reading: Reading) -> Alert:
        """
        Create an Alert object from a reading that triggered the rule.

        Args:
            reading:
                Reading that satisfied this rule.

        Returns:
            Newly created Alert instance.

        Raises:
            ValueError:
                If the reading does not contain the required timestamp or value.
        """
        if reading.timestamp is None:
            raise ValueError("Cannot build alert from reading without a timestamp")

        if reading.value is None:
            raise ValueError("Cannot build alert from reading without a numeric value")

        return Alert(
            timestamp=reading.timestamp,
            sensor_id=reading.sensor_id,
            severity=self.severity,
            message=self.message,
            reading_value=reading.value,
        )

@dataclass
class DataSet:
    """
    Store processed readings in separate valid and invalid collections.

    This class provides a central place to keep imported readings once they
    have been classified by the validation logic.
    """

    valid_readings: list[Reading] = field(default_factory=list)
    invalid_readings: list[Reading] = field(default_factory=list)

    def add_reading(self, reading: Reading) -> None:
        """
        Store a reading in the appropriate internal collection.

        Args:
            reading:
                Reading object to be stored.
        """
        if reading.is_valid:
            self.valid_readings.append(reading)
        else:
            self.invalid_readings.append(reading)

    def get_by_sensor(self, sensor_id: str) -> list[Reading]:
        """
        Return all valid readings associated with a specific sensor.

        Args:
            sensor_id:
                Sensor identifier to filter by.

        Returns:
            List of valid readings belonging to the requested sensor.
        """
        return [reading for reading in self.valid_readings if reading.sensor_id == sensor_id]

    def get_valid_readings(self) -> list[Reading]:
        """
        Return all valid readings currently stored.

        Returns:
            List of valid readings.
        """
        return self.valid_readings

    def get_invalid_readings(self) -> list[Reading]:
        """
        Return all invalid readings currently stored.

        Returns:
            List of invalid readings.
        """
        return self.invalid_readings

    def count(self) -> int:
        """
        Return the total number of stored readings.

        Returns:
            Total number of valid and invalid readings combined.
        """
        return len(self.valid_readings) + len(self.invalid_readings)


class Validator:
    """
    Validate timestamps, numeric values, and sensor-specific reading ranges.

    This class separates checking logic from data storage and reporting so that
    the validation process can be tested independently.
    """

    @staticmethod
    def validate_timestamp(timestamp_str: str) -> datetime:
        """
        Parse and validate a timestamp string in ISO-like format.

        Accepted examples include formats such as:
            2026-05-08T09:00:00
            2026-05-08 09:00:00

        Args:
            timestamp_str:
                String representation of the timestamp.

        Returns:
            Parsed datetime object.

        Raises:
            ValueError:
                If the timestamp cannot be parsed.
        """
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(timestamp_str.strip(), fmt)
            except ValueError:
                continue
        raise ValueError("Invalid timestamp format")

    @staticmethod
    def validate_numeric(value: str) -> float:
        """
        Convert a raw CSV value into a float.

        Args:
            value:
                String value from the CSV file.

        Returns:
            Parsed float value.

        Raises:
            ValueError:
                If the value is blank or cannot be converted to float.
        """
        if value is None or str(value).strip() == "":
            raise ValueError("Reading value is blank")
        return float(value)

    def validate_reading(self, sensor: Sensor, reading: Reading) -> Reading:
        """
        Validate and classify a reading using the sensor configuration.

        Args:
            sensor:
                Sensor configuration object associated with the reading.
            reading:
                Reading object to validate.

        Returns:
            The same Reading object, updated with validity status, error
            message, and classification.
        """
        if reading.value is None:
            reading.mark_invalid("Reading has no numeric value")
            return reading

        if not sensor.validate_value(reading.value):
            reading.mark_invalid("Reading outside valid sensor range")
            return reading

        reading.mark_valid(sensor.classify_status(reading.value))
        return reading


class AlertManager:
    """
    Manage alert rules and store alerts generated during processing.
    """

    def __init__(self) -> None:
        """
        Initialise the alert manager with empty rule and alert collections.
        """
        self.rules: list[AlertRule] = []
        self.alerts: list[Alert] = []

    def add_rule(self, rule: AlertRule) -> None:
        """
        Add a new alert rule.

        Args:
            rule:
                AlertRule instance to be stored.
        """
        self.rules.append(rule)

    def evaluate_reading(self, reading: Reading) -> list[Alert]:
        """
        Evaluate a validated reading against all configured alert rules.

        Args:
            reading:
                Reading object already validated and classified.

        Returns:
            List of alerts generated from this reading. The list may be empty.
        """
        generated_alerts: list[Alert] = []

        for rule in self.rules:
            if rule.evaluate(reading):
                alert = rule.build_alert(reading)
                self.alerts.append(alert)
                generated_alerts.append(alert)

        return generated_alerts

    def get_active_alerts(self) -> list[Alert]:
        """
        Return all alerts created during the run.

        Returns:
            List of stored alerts.
        """
        return self.alerts

    def count_by_severity(self) -> dict[str, int]:
        """
        Count alerts by severity level.

        Returns:
            Dictionary mapping severity names to counts.
        """
        counts: dict[str, int] = {}
        for alert in self.alerts:
            counts[alert.severity] = counts.get(alert.severity, 0) + 1
        return counts


class StatisticsEngine:
    """
    Calculate summary statistics for each sensor based on valid readings.
    """

    @staticmethod
    def calculate_summary(readings: list[Reading]) -> dict[str, float | int | None]:
        """
        Generate a summary of basic statistics for a collection of readings.

        Args:
            readings:
                List of valid Reading objects for a single sensor.

        Returns:
            Dictionary containing count, minimum, maximum, mean, and median.
            If no readings are available, the numerical fields are returned as
            None and count is zero.
        """
        if not readings:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
            }

        values = [reading.value for reading in readings if reading.value is not None]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": mean(values),
            "median": median(values),
        }


class CSVLoader:
    """
    Load raw sensor readings from a CSV file.

    The loader focuses only on importing and parsing rows. Validation against
    sensor rules is handled later by the Validator class.
    """

    REQUIRED_COLUMNS = {"timestamp", "sensor_id", "value"}

    def __init__(self, file_path: str | Path) -> None:
        """
        Store the file path used for loading reading data.

        Args:
            file_path:
                Path to the CSV file containing sensor readings.
        """
        self.file_path = Path(file_path)

    def load_rows(self) -> list[dict[str, str]]:
        """
        Load all rows from the CSV file using DictReader.

        Returns:
            List of row dictionaries.

        Raises:
            FileNotFoundError:
                If the CSV file does not exist.
            ValueError:
                If required columns are missing.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.file_path}")

        with self.file_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)

            if reader.fieldnames is None:
                raise ValueError("CSV file is missing a header row")

            missing_columns = self.REQUIRED_COLUMNS - set(reader.fieldnames)
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise ValueError(f"CSV missing required columns: {missing}")

            return list(reader)

    def parse_row(self, row: dict[str, str]) -> Reading:
        """
        Parse a raw CSV row into a Reading object.

        Parsing errors are not raised directly to the caller. Instead, the
        returned Reading object is marked invalid so that processing can
        continue and the failure can be reported later.

        Args:
            row:
                Dictionary representing one CSV row.

        Returns:
            Reading object, either ready for validation or already marked
            invalid if parsing failed.
        """
        sensor_id = (row.get("sensor_id") or "").strip()

        try:
            timestamp = Validator.validate_timestamp(row.get("timestamp", ""))
        except ValueError as exc:
            reading = Reading(timestamp=None, sensor_id=sensor_id, value=None)
            reading.mark_invalid(str(exc))
            return reading

        try:
            value = Validator.validate_numeric(row.get("value", ""))
        except ValueError as exc:
            reading = Reading(timestamp=timestamp, sensor_id=sensor_id, value=None)
            reading.mark_invalid(str(exc))
            return reading

        return Reading(timestamp=timestamp, sensor_id=sensor_id, value=value)


class ReportGenerator:
    """
    Generate report files summarising processed readings, invalid entries,
    and alerts.
    """

    def __init__(self, output_directory: str | Path = ".") -> None:
        """
        Store the output directory used for generated files.

        Args:
            output_directory:
                Directory in which report files should be written.
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def generate_csv_output(
        self,
        file_name: str,
        rows: list[dict[str, Any]],
        fieldnames: list[str],
    ) -> None:
        """
        Write a list of dictionaries to a CSV file.

        Args:
            file_name:
                Name of the output CSV file.
            rows:
                Row dictionaries to be written.
            fieldnames:
                Ordered list of column names for the output file.
        """
        output_path = self.output_directory / file_name
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def generate_text_report(
        self,
        file_name: str,
        summary: dict[str, dict[str, float | int | None]],
        invalid_count: int,
        alert_counts: dict[str, int],
    ) -> None:
        """
        Write a plain-text summary report containing statistics and alert data.

        Args:
            file_name:
                Name of the output text file.
            summary:
                Per-sensor summary statistics dictionary.
            invalid_count:
                Number of invalid readings encountered.
            alert_counts:
                Count of alerts grouped by severity.
        """
        output_path = self.output_directory / file_name

        lines: list[str] = []
        lines.append("ENVIRONMENTAL MONITORING REPORT")
        lines.append("=" * 40)
        lines.append("")
        lines.append(f"Invalid readings: {invalid_count}")
        lines.append("Alert counts by severity:")

        if alert_counts:
            for severity, count in sorted(alert_counts.items()):
                lines.append(f"  - {severity}: {count}")
        else:
            lines.append("  - No alerts generated")

        lines.append("")
        lines.append("Per-sensor statistics:")
        lines.append("")

        for sensor_id, stats in summary.items():
            lines.append(f"{sensor_id}")
            lines.append(f"  Count : {stats['count']}")
            lines.append(f"  Min   : {stats['min']}")
            lines.append(f"  Max   : {stats['max']}")
            lines.append(f"  Mean  : {stats['mean']}")
            lines.append(f"  Median: {stats['median']}")
            lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")


class MonitorSystem:
    """
    Coordinate the complete environmental monitoring analysis workflow.

    This class acts as the main controller for the program. It loads sensor
    configuration, imports CSV data, validates and classifies readings,
    generates alerts, computes statistics, and writes final outputs.
    """

    def __init__(self) -> None:
        """
        Initialise all core objects used by the monitoring system.
        """
        self.sensors: dict[str, Sensor] = {}
        self.dataset = DataSet()
        self.validator = Validator()
        self.alert_manager = AlertManager()
        self.statistics_engine = StatisticsEngine()
        self.report_generator = ReportGenerator()

    def load_sensor_config(self, config_path: str | Path) -> None:
        """
        Load sensor definitions from a JSON configuration file.

        The JSON file must contain a top-level key called 'sensors' whose value
        is a list of sensor definitions.

        Args:
            config_path:
                Path to the configuration JSON file.

        Raises:
            FileNotFoundError:
                If the configuration file does not exist.
            ValueError:
                If the JSON structure is not valid for this program.
        """
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        with config_file.open("r", encoding="utf-8") as file:
            config_data = json.load(file)

        sensor_list = config_data.get("sensors")
        if not isinstance(sensor_list, list):
            raise ValueError("Configuration must contain a 'sensors' list")

        for item in sensor_list:
            sensor = Sensor(
                sensor_id=item["sensor_id"],
                name=item["name"],
                unit=item["unit"],
                min_valid=float(item["min_valid"]),
                max_valid=float(item["max_valid"]),
                warning_low=float(item["warning_low"]),
                warning_high=float(item["warning_high"]),
                critical_low=float(item["critical_low"]),
                critical_high=float(item["critical_high"]),
            )
            self.sensors[sensor.sensor_id] = sensor

            # Each sensor receives one warning rule and one critical rule.
            self.alert_manager.add_rule(
                AlertRule(
                    rule_id=f"{sensor.sensor_id}_warning",
                    sensor_id=sensor.sensor_id,
                    condition_type="warning",
                    severity="warning",
                    message=f"{sensor.name} reading has entered the warning range.",
                )
            )
            self.alert_manager.add_rule(
                AlertRule(
                    rule_id=f"{sensor.sensor_id}_critical",
                    sensor_id=sensor.sensor_id,
                    condition_type="critical",
                    severity="critical",
                    message=f"{sensor.name} reading has entered the critical range.",
                )
            )

    def process_reading(self, reading: Reading) -> None:
        """
        Validate, classify, store, and evaluate a single reading.

        Args:
            reading:
                Reading object parsed from the CSV file.
        """
        if reading.sensor_id not in self.sensors:
            reading.mark_invalid("Unknown sensor ID")
            self.dataset.add_reading(reading)
            return

        if reading.is_valid is False and reading.status == "invalid":
            self.dataset.add_reading(reading)
            return

        sensor = self.sensors[reading.sensor_id]
        validated_reading = self.validator.validate_reading(sensor, reading)
        self.dataset.add_reading(validated_reading)

        if validated_reading.is_valid:
            self.alert_manager.evaluate_reading(validated_reading)

    def import_csv(self, file_path: str | Path) -> None:
        """
        Load and process all rows from the input CSV file.

        Args:
            file_path:
                Path to the environmental readings CSV file.
        """
        loader = CSVLoader(file_path)
        rows = loader.load_rows()

        # Each row is parsed first, then sent through the processing pipeline.
        for row in rows:
            reading = loader.parse_row(row)
            self.process_reading(reading)

    def build_summary(self) -> dict[str, dict[str, float | int | None]]:
        """
        Build per-sensor statistics for all configured sensors.

        Returns:
            Dictionary mapping sensor IDs to summary statistics.
        """
        summary: dict[str, dict[str, float | int | None]] = {}
        for sensor_id in self.sensors:
            sensor_readings = self.dataset.get_by_sensor(sensor_id)
            summary[sensor_id] = self.statistics_engine.calculate_summary(sensor_readings)
        return summary

    def generate_outputs(self) -> None:
        """
        Write all output files required by the program.

        Generated files include:
            - valid_readings.csv
            - invalid_readings.csv
            - alerts.csv
            - environmental_monitoring_report.txt
        """
        valid_rows = [reading.to_dict() for reading in self.dataset.get_valid_readings()]
        invalid_rows = [reading.to_dict() for reading in self.dataset.get_invalid_readings()]
        alert_rows = [alert.to_dict() for alert in self.alert_manager.get_active_alerts()]
        summary = self.build_summary()
        alert_counts = self.alert_manager.count_by_severity()

        self.report_generator.generate_csv_output(
            file_name="valid_readings.csv",
            rows=valid_rows,
            fieldnames=["timestamp", "sensor_id", "value", "status", "is_valid", "error_message"],
        )

        self.report_generator.generate_csv_output(
            file_name="invalid_readings.csv",
            rows=invalid_rows,
            fieldnames=["timestamp", "sensor_id", "value", "status", "is_valid", "error_message"],
        )

        self.report_generator.generate_csv_output(
            file_name="alerts.csv",
            rows=alert_rows,
            fieldnames=["timestamp", "sensor_id", "severity", "message", "reading_value"],
        )

        self.report_generator.generate_text_report(
            file_name="environmental_monitoring_report.txt",
            summary=summary,
            invalid_count=len(self.dataset.get_invalid_readings()),
            alert_counts=alert_counts,
        )

    def run(self, config_path: str | Path, readings_path: str | Path) -> None:
        """
        Execute the complete workflow from configuration loading to reporting.

        Args:
            config_path:
                Path to the sensor configuration JSON file.
            readings_path:
                Path to the sensor readings CSV file.
        """
        self.load_sensor_config(config_path)
        self.import_csv(readings_path)
        self.generate_outputs()


def main() -> None:
    """
    Run the environmental monitoring analysis program using command-line
    arguments so different input files can be processed without editing
    the source code.
    """
    parser = argparse.ArgumentParser(
        description="Environmental monitoring analysis program."
    )

    parser.add_argument(
        "--config",
        default="environmental_sensor_config.json",
        help="Path to the sensor configuration JSON file."
    )

    parser.add_argument(
        "--readings",
        default="environmental_sensor_readings.csv",
        help="Path to the sensor readings CSV file."
    )

    args = parser.parse_args()

    system = MonitorSystem()
    system.run(args.config, args.readings)

    print("Environmental monitoring analysis completed successfully.")
    print(f"Configuration file used: {args.config}")
    print(f"Readings file used: {args.readings}")
    print("Generated files:")
    print(" - valid_readings.csv")
    print(" - invalid_readings.csv")
    print(" - alerts.csv")
    print(" - environmental_monitoring_report.txt")

if __name__ == "__main__":
    main()
