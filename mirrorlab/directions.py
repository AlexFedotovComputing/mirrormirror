"""Explicit physical routes and archive bindings; filenames carry no geometry."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Direction:
    laser_side: str
    ring_side: str
    injection_side: str
    legacy_ideal: str
    legacy_real: str
    survey_file: str
    local_incoming: tuple
    additional_screens_file: str | None = None

    def metadata(self):
        # source_side is retained as a compatibility alias for plot legends.
        return {**asdict(self), "source_side": self.laser_side}


ROUTES = {
    "west_to_east": Direction(
        "WEST", "WEST", "EAST", "из-к+_идеальная.py", "из-к+_реальная.py",
        "Геодезические измерения из-к+.xlsx", (0., 0., 1.)),
    "east_to_west": Direction(
        "EAST", "EAST", "WEST", "legacy/variants/из+к-_идеальная.txt", "из+к-_реальная.py",
        "Геодезические измерения из+к-.xlsx", (0., 0., -1.), "лавсан_из+к-.txt"),
}

# Original naming convention, independent of numerical survey coordinates.
SIDE_DESIGNATIONS = {"EAST": "+", "WEST": "−"}
DIRECTIONS = {key: route.metadata() for key, route in ROUTES.items()}
