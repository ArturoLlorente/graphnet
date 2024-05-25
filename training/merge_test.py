"""Example of converting I3-files to SQLite and Parquet."""

from graphnet.data.extractors.internal import (
    ParquetExtractor
)
from graphnet.data.pre_configured import ParquetToSQLiteConverter



def main() -> None:
    """Convert IceCube-86 I3 files to intermediate `backend` format."""
    # Check(s)
    for i in range(1,10):
        inputs = [f"/scratch/users/allorana/parquet_separated_parts/sqlite_part{i}",]
        outdir = f"/scratch/users/allorana/all_truth_parts/part{i}"

        converter = ParquetToSQLiteConverter(
            extractors=[
                ParquetExtractor("truth"),
                #ParquetExtractor("InIceDSTPulses"),
                #ParquetExtractor("EventGeneratorSelectedRecoNN_I3Particle"),
            ],
            outdir=outdir,
            num_workers=0,
        )
        converter(inputs)
        converter.merge_files()


if __name__ == "__main__":
    main()
