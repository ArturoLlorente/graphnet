"""Example of converting I3-files to SQLite and Parquet."""

from graphnet.data.extractors.internal import (
    ParquetExtractor
)
from graphnet.data.pre_configured import ParquetToSQLiteConverter



def main() -> None:
    """Convert IceCube-86 I3 files to intermediate `backend` format."""
    # Check(s)

    idx = 33
    inputs = [f"/scratch/users/allorana/parquet_separated_parts/sqlite_part{idx}"]
    outdir = f"/scratch/users/allorana/merged_sqlite_1505/part{idx}"

    converter = ParquetToSQLiteConverter(
        extractors=[
            ParquetExtractor("truth"),
            ParquetExtractor("InIceDSTPulses"),
            ParquetExtractor("EventGeneratorSelectedRecoNN_I3Particle"),
        ],
        outdir=outdir,
        num_workers=0,
    )
    converter(inputs)
    converter.merge_files()


if __name__ == "__main__":
    main()
