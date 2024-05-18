"""I3Extractor class(es) for extracting I3Particle properties."""

from typing import TYPE_CHECKING, Dict

from graphnet.data.extractors.icecube import I3Extractor

#import torch

if TYPE_CHECKING:
    from icecube import icetray  # pyright: reportMissingImports=false


class I3MapExtractor(I3Extractor):
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
                    "EnergyVisible": frame[name]['EnergyVisible'],
                    "Length": frame[name]['Length'],
                    "LengthInDetector": frame[name]['LengthInDetector'],
                    "PrimaryAzimuth": frame[name]['PrimaryAzimuth'],
                    "PrimaryDirectionX": frame[name]['PrimaryDirectionX'],
                    "PrimaryDirectionY": frame[name]['PrimaryDirectionY'],
                    "PrimaryDirectionZ": frame[name]['PrimaryDirectionZ'],
                    "PrimaryEnergy": frame[name]['PrimaryEnergy'],
                    "PrimaryZenith": frame[name]['PrimaryZenith'],
                    "TotalDepositedEnergy": frame[name]['TotalDepositedEnergy'],
                    "VertexTime": frame[name]['VertexTime'],
                    "VertexX": frame[name]['VertexX'],
                    "VertexY": frame[name]['VertexY'],
                    "VertexZ": frame[name]['VertexZ'],
                    "leading_energy_rel_entry": frame[name]['leading_energy_rel_entry'],
                    "num_coincident_events": frame[name]['num_coincident_events'],
                    "num_muons_at_entry": frame[name]['num_muons_at_entry'],
                    "num_muons_at_entry_above_threshold": frame[name]['num_muons_at_entry_above_threshold'],
                    "p_entering": frame[name]['p_entering'],
                    "p_entering_muon_bundle": frame[name]['p_entering_muon_bundle'],
                    "p_entering_muon_single": frame[name]['p_entering_muon_single'],
                    "p_entering_muon_single_stopping": frame[name]['p_entering_muon_single_stopping'],
                    "p_is_track": frame[name]['p_is_track'],
                    "p_outside_cascade": frame[name]['p_outside_cascade'],
                    "p_starting": frame[name]['p_starting'],
                    "p_starting_300m": frame[name]['p_starting_300m'],
                    "p_starting_cc": frame[name]['p_starting_cc'],
                    "p_starting_cc_e": frame[name]['p_starting_cc_e'],
                    "p_starting_cc_mu": frame[name]['p_starting_cc_mu'],
                    "p_starting_cc_tau": frame[name]['p_starting_cc_tau'],
                    "p_starting_cc_tau_double_bang": frame[name]['p_starting_cc_tau_double_bang'],
                    "p_starting_cc_tau_muon_decay": frame[name]['p_starting_cc_tau_muon_decay'],
                    "p_starting_glashow": frame[name]['p_starting_glashow'],
                    "p_starting_nc": frame[name]['p_starting_nc'],
                }

            )

        return output