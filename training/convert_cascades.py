"""Example of converting I3-files to SQLite and Parquet."""

from graphnet.data.extractors.icecube import (
    I3FeatureExtractorIceCube86,
    I3TruthExtractor,
    I3ParticleExtractor,
    I3MapExtractor,
)
from graphnet.data.pre_configured import I3ToParquetConverter, I3ToSQLiteConverter
from graphnet.utilities.imports import has_icecube_package
from graphnet.utilities.logging import Logger



def main() -> None:
    """Convert IceCube-86 I3 files to intermediate `backend` format."""
    # Check(s)

    inputs = ["/scratch/users/allorana/parquet_small/separated_batches"]
    outdir = f"/scratch/users/allorana/parquet_really_small/"
    gcd_rescue = "/scratch/users/allorana/graphnet/data/tests/i3/DNNCascadeL5_NuGen_21430_00000993/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz"

    converter = I3ToParquetConverter(
        extractors=[
            #I3FeatureExtractorIceCube86("SRTInIcePulses"),
            I3FeatureExtractorIceCube86("InIceDSTPulses"),
            #I3FeatureExtractorIceCube86("SplitInIceDSTPulses"),
            I3TruthExtractor(),
            #I3ParticleExtractor('EventGeneratorSelectedRecoNN_I3Particle'),
            #I3MapExtractor('LabelsDeepLearning'),
            #I3MapExtractor('LabelsDeepLearning_p150'),
        ],
        outdir=outdir,
        gcd_rescue=gcd_rescue,
        num_workers=32,
    )
    converter(inputs)
    converter.merge_files(events_per_batch=5000)


if __name__ == "__main__":

    if not has_icecube_package():
        Logger(log_folder=None).error("IceTray not installed")
    else:
        main()
