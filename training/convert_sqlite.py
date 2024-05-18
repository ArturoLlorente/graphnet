"""Example of converting I3-files to SQLite and Parquet."""

from graphnet.data.extractors.internal import (
    ParquetExtractor
)
from graphnet.data.pre_configured import ParquetToSQLiteConverter



def main() -> None:
    """Convert IceCube-86 I3 files to intermediate `backend` format."""
    # Check(s)

    inputs = ["/scratch/users/allorana/sqlite_part33"]
    outdir = f"/scratch/users/allorana/merged_sqlite_1505/part33"
    gcd_rescue = "/scratch/users/allorana/graphnet/data/tests/i3/DNNCascadeL5_NuGen_21430_00000993/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz"

    converter = ParquetToSQLiteConverter(
        extractors=[
            ParquetExtractor("truth"),
            ParquetExtractor("InIceDSTPulses")
        ],
        outdir=outdir,
        num_workers=0,
    )
    converter(inputs)
    converter.merge_files()


if __name__ == "__main__":
    main()
