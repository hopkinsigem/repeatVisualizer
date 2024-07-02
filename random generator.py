import random
import os

# List of amino acids
amino_acids = "ACDEFGHIKLMNPQRSTVWY"

def generate_random_block(min_length, max_length):
    """Generates a random block of amino acids with a length between min_length and max_length."""
    length = random.randint(min_length, max_length)
    return ''.join(random.choice(amino_acids) for _ in range(length))

def mutate_block(block, min_factor=0.8, max_factor=1.2):
    """Mutates a given block by changing its length and mutating some amino acids."""
    original_length = len(block)
    new_length = int(original_length * random.uniform(min_factor, max_factor))

    if new_length > original_length:
        block += ''.join(random.choice(amino_acids) for _ in range(new_length - original_length))
    else:
        block = block[:new_length]

    # Randomly change 2% to 20% of the amino acids
    mutation_percentage = random.uniform(0.02, 0.2)
    num_mutations = int(mutation_percentage * len(block))
    block = list(block)
    for _ in range(num_mutations):
        index = random.randint(0, len(block) - 1)
        block[index] = random.choice(amino_acids)

    return ''.join(block)

def generate_blocks(num_blocks):
    """Generates a dictionary of random blocks with specified length distribution."""
    blocks = {}
    for i in range(1, num_blocks + 1):
        block_name = f"block{i}"
        if random.random() < 0.2:  # 20% chance
            blocks[block_name] = generate_random_block(30, 60)
        else:  # 80% chance
            blocks[block_name] = generate_random_block(80, 120)
    return blocks

def generate_sequence(blocks, num_blocks_in_sequence):
    """Generates a random sequence of blocks with some variations and annotations."""
    block_names = list(blocks.keys())
    sequence = []
    annotations = []
    position = 0

    for _ in range(num_blocks_in_sequence):
        block_name = random.choice(block_names)
        mutated_block = mutate_block(blocks[block_name])
        block_length = len(mutated_block)
        sequence.append(mutated_block)
        annotations.append((block_name, position + 1, position + block_length + 1))
        position += block_length

    # Ensure 30% of the sequence consists of repeated blocks of one type
    num_repeats = int(0.3 * num_blocks_in_sequence)
    repeated_block_name = random.choice(block_names)
    for _ in range(num_repeats):
        mutated_block = mutate_block(blocks[repeated_block_name])
        block_length = len(mutated_block)
        sequence.append(mutated_block)
        annotations.append((repeated_block_name, position + 1, position + block_length + 1))
        position += block_length

    random.shuffle(sequence)
    return ''.join(sequence), annotations

def generate_synthetic_sequences(repeat=3):
    """Generates synthetic sequences multiple times and saves to files."""
    if not os.path.exists('annotations'):
        os.makedirs('annotations')

    sequences = []
    all_blocks = []
    all_annotations = []

    for seq_num in range(1, repeat + 1):
        num_blocks = random.randint(5, 10)
        blocks = generate_blocks(num_blocks)
        num_blocks_in_sequence = random.randint(60, 100)
        sequence, annotations = generate_sequence(blocks, num_blocks_in_sequence)

        sequences.append((seq_num, sequence))
        all_blocks.append((seq_num, blocks))
        all_annotations.append((seq_num, annotations))

        # Save annotations to text file
        with open(f'annotations/{seq_num}.txt', 'w') as f:
            f.write("Block Name,Start,End\n")
            for annotation in annotations:
                f.write(f"{annotation[0]},{annotation[1]},{annotation[2]}\n")

    # Save sequences to FASTA
    with open('sequences.fasta', 'w') as f:
        for seq_num, sequence in sequences:
            f.write(f">{seq_num}\n")
            f.write(sequence + "\n")

    # Save blocks to FASTA
    with open('blocks.fasta', 'w') as f:
        for seq_num, blocks in all_blocks:
            block_num = 1
            for block_name, block_seq in blocks.items():
                f.write(f">{seq_num}.{block_num}\n")
                f.write(block_seq + "\n")
                block_num += 1
def main():
    # Run the function to generate synthetic sequences and save to files
    generate_synthetic_sequences()

if __name__ == "__main__":
    main()
