"""I3Extractor class(es) for extracting I3Particle properties."""

from typing import TYPE_CHECKING, Dict

from graphnet.data.extractors.icecube import I3Extractor

if TYPE_CHECKING:
    from icecube import icetray  # pyright: reportMissingImports=false


class I3ParticleExtractor(I3Extractor):
    """Class for extracting I3Particle properties.

    Can be used to extract predictions from other algorithms for comparisons
    with GraphNeT.
    """

    def __init__(self, extractor_name: str):
        """Construct I3ParticleExtractor."""
        # Base class constructor
        super().__init__(extractor_name=extractor_name)

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, float]:
        """Extract I3Particle properties from I3Particle in frame."""
        output = {}
        name = self._extractor_name
        if name in frame:
            output.update(
                {
                    "zenith_" + 'EventGenerator': frame[name].dir.zenith,
                    "azimuth_" + 'EventGenerator': frame[name].dir.azimuth,
                    "dir_x_" + 'EventGenerator': frame[name].dir.x,
                    "dir_y_" + 'EventGenerator': frame[name].dir.y,
                    "dir_z_" + 'EventGenerator': frame[name].dir.z,
                    "pos_x_" + 'EventGenerator': frame[name].pos.x,
                    "pos_y_" + 'EventGenerator': frame[name].pos.y,
                    "pos_z_" + 'EventGenerator': frame[name].pos.z,
                    "time_" + 'EventGenerator': frame[name].time,
                    "speed_" + 'EventGenerator': frame[name].speed,
                    "energy_" + 'EventGenerator': frame[name].energy,
                }
            )

        return output
