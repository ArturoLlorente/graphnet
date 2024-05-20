"""Example of converting I3-files to SQLite and Parquet."""

from graphnet.data.extractors.internal import (
    ParquetExtractor
)
from graphnet.data.pre_configured import ParquetToSQLiteConverter



def main() -> None:
    """Convert IceCube-86 I3 files to intermediate `backend` format."""
    # Check(s)

    inputs = ["/scratch/users/allorana/parquet_separated_parts/sqlite_part30",
              "/scratch/users/allorana/parquet_separated_parts/sqlite_part31",
              "/scratch/users/allorana/parquet_separated_parts/sqlite_part32",
              "/scratch/users/allorana/parquet_separated_parts/sqlite_part33",]
    outdir = "/scratch/users/allorana/merged_sqlite_1505/meta_test"

    converter = ParquetToSQLiteConverter(
        extractors=[
            ParquetExtractor("truth"),
            #ParquetExtractor("InIceDSTPulses"),
            ParquetExtractor("EventGeneratorSelectedRecoNN_I3Particle"),
        ],
        outdir=outdir,
        num_workers=4,
    )
    converter(inputs)
    converter.merge_files()


if __name__ == "__main__":
    main()
