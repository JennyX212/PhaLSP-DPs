import random
import pandas as pd
from Bio import SeqIO


def generate_fragments(fasta_file, fragment_length=1000, num_fragments=5, output_csv='../data/data2_contigs/fragments1000.csv'):
    records = SeqIO.parse(fasta_file, 'fasta')
    data = []
    contig_counter = 1

    for record in records:
        seq_id = record.id
        sequence = str(record.seq)
        seq_len = len(sequence)

        for _ in range(num_fragments):
            if seq_len <= fragment_length:
                fragment = sequence
            else:
                start = random.randint(0, seq_len - fragment_length)
                fragment = sequence[start:start + fragment_length]
            data.append([f'contig{contig_counter}', seq_id, fragment])
            contig_counter += 1

    df = pd.DataFrame(data, columns=['Contig', 'Phage ID', 'Fragment'])
    df.to_csv(output_csv, index=False)

fasta_file = '../data/phage1865.fasta'
generate_fragments(fasta_file)
