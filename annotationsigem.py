import os
from Bio import pairwise2
from Bio import SeqIO
import numpy as np
from multiprocessing import Pool, cpu_count

match_score = 2       # Score for a match
mismatch_score = -1   # Penalty for a mismatch
gap_open_penalty = -0.5  # Penalty for opening a gap
gap_extend_penalty = -0.1  # Penalty for extending a gap

def calculate_alignment_scores(long_seq, short_seqs, window_size):
    max_score = window_size * match_score
    step_size = 1  # For every position

    all_normalized_scores = []

    for short_name, short_seq in short_seqs.items():
        normalized_scores = []

        for i in range(0, len(long_seq) - window_size + 1, step_size):
            window = long_seq[i:i + window_size]
            local_alignments = pairwise2.align.localms(window, short_seq, match_score, mismatch_score, gap_open_penalty, gap_extend_penalty)

            best_alignment_score = float("-inf")
            best_alignment = None

            for alignment in local_alignments:
                alignment_score = alignment[2]
                normalized_score = alignment_score / max_score

                if alignment_score > best_alignment_score:
                    best_alignment_score = alignment_score
                    best_alignment = alignment

            if best_alignment:
                normalized_scores.append(normalized_score)

        # Handle the end of the sequence
        for i in range(len(long_seq) - window_size + 1, len(long_seq), step_size):
            window = long_seq[i:]
            current_window_size = len(window)
            local_alignments = pairwise2.align.localms(window, short_seq, match_score, mismatch_score, gap_open_penalty, gap_extend_penalty)

            best_alignment_score = float("-inf")
            best_alignment = None

            for alignment in local_alignments:
                alignment_score = alignment[2]
                normalized_score = alignment_score / (current_window_size * match_score)

                if alignment_score > best_alignment_score:
                    best_alignment_score = alignment_score
                    best_alignment = alignment

            if best_alignment:
                normalized_scores.append(normalized_score)

        all_normalized_scores.append(normalized_scores)

    return all_normalized_scores

def save_normalized_scores(normalized_scores, file_name):
    
    max_length = max(len(scores) for scores in normalized_scores)
    scores_array = np.full((len(normalized_scores), max_length), np.nan)
    
    for i, scores in enumerate(normalized_scores):
        scores_array[i, :len(scores)] = scores
    
    np.save(file_name, scores_array)
    print(f"Normalized scores saved to {file_name}")

def read_fasta(file_path):
    sequences = {}
    for record in SeqIO.parse(file_path, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences

def process_sequence(seq_id, sequence, blocks, window_size, output_dir):
    if seq_id in blocks:
        sequence_blocks = blocks[seq_id]
        normalized_scores = calculate_alignment_scores(sequence, sequence_blocks, window_size)
        output_file = os.path.join(output_dir, f"{seq_id}_prob.npy")
        save_normalized_scores(normalized_scores, output_file)

def main():
    sequences_file = r"sequences.fasta"
    blocks_file = r"blocks.fasta"  
    window_size = 15  # Change if needed
    output_dir = r"probabilities"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sequences = read_fasta(sequences_file)
    blocks = read_fasta(blocks_file)

    grouped_blocks = {}
    for block_name, block_seq in blocks.items():
        seq_id = block_name.split('.')[0]
        if seq_id not in grouped_blocks:
            grouped_blocks[seq_id] = {}
        grouped_blocks[seq_id][block_name] = block_seq

    num_processes = cpu_count()  
    with Pool(num_processes) as pool:
        pool.starmap(process_sequence, [(seq_id, sequence, grouped_blocks, window_size, output_dir) for seq_id, sequence in sequences.items()])

if __name__ == "__main__":
    main()
