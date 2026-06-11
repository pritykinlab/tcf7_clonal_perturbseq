import sys
import gzip
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def trim_by_indices(sequence, start_index, end_index):
    # Keep only the part of the sequence between start_index and end_index
    return sequence[start_index:end_index]

def trim_quality(quality, start_index, end_index):
    # Keep only the part of the quality scores between start_index and end_index
    return quality[start_index:end_index]

def main(input_file, output_file, start_index, end_index):
    with gzip.open(input_file, "rt") as handle_in, gzip.open(output_file, "wt") as handle_out:
        for record in SeqIO.parse(handle_in, "fastq"):
            sequence = str(record.seq)
            quality = record.letter_annotations["phred_quality"]

            trimmed_sequence = trim_by_indices(sequence, start_index, end_index)
            trimmed_quality = trim_quality(quality, start_index, end_index)

            trimmed_record = SeqRecord(
                Seq(trimmed_sequence),
                id=record.id,
                name=record.name,
                description=record.description,
                letter_annotations={"phred_quality": trimmed_quality}
            )
            SeqIO.write(trimmed_record, handle_out, "fastq")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python trim.py <input_file> <output_file> <start_index> <end_index>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    start_index = int(sys.argv[3])
    end_index = int(sys.argv[4])

    main(input_file, output_file, start_index, end_index)
