# -------------------------------------------------------------------
# AUTHOR: Ivan Trinh
# FILENAME: SPIMI_Pipeline.py
# SPECIFICATION: This is an implementation of a SPIMI-based pipeline, 
# taking in a TSV file and creating block text files, then merging
# them together alphabetically in an overall inverted index
# FOR: CS 5180 - Assignment #2
# TIME SPENT: about like a few hours
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Importing some Python libraries
# -------------------------------------------------------------------
import csv
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import heapq
import os
# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
CORPUS_FILE = "corpus.tsv"
BLOCK_SIZE = 100
BLOCKS_DIR = "blocks"
OUTPUT_FILE = "index.txt"
os.makedirs(BLOCKS_DIR, exist_ok=True)
MAX_INPUT_BUFFER = 100
MAX_OUTPUT_BUFFER = 500
# -------------------------------------------------------------------
# Opening corpus.tsv
# -------------------------------------------------------------------
docs = []
with open(CORPUS_FILE, "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        doc_id = row[0].strip()
        text = row[1]
        docs.append((doc_id, text))
# -------------------------------------------------------------------
# Building tokenizer function
# -------------------------------------------------------------------
vectorizer = CountVectorizer(stop_words="english") 
vectorizer.fit([text for _, text in docs]) 
def tokenize(text): 
    return vectorizer.build_tokenizer()(text.lower())
# -------------------------------------------------------------------
# Building SPIMI blocks function
# -------------------------------------------------------------------
def build_block(block_docs, block_id):
    index = defaultdict(list)
    for doc_id, text in block_docs:
        tokens = tokenize(text)
        seen = set()
        for term in tokens:
            if term not in seen:
                index[term].append(doc_id)
                seen.add(term)

    sorted_terms = sorted(index.keys())

    filename = os.path.join(BLOCKS_DIR, f"block_{block_id}.txt")
    with open(filename, "w", encoding="utf-8") as out:
        for term in sorted_terms:
            postings = ",".join(str(d) for d in sorted(index[term]))
            out.write(f"{term}:{postings}\n")
# -------------------------------------------------------------------
# Reading next chunk of lines function
# -------------------------------------------------------------------
def read_chunk(f, size = MAX_INPUT_BUFFER):
    lines = []
    for _ in range(size):
        line = f.readline()
        if not line:
            break
        term, postings = line.strip().split(":")
        postings = postings.split(",") 
        lines.append((term, postings))
    return lines
# -------------------------------------------------------------
# Building SPIMI blocks from corpus file
# -------------------------------------------------------------
for i in range(10): 
    start = i * BLOCK_SIZE 
    end = start + BLOCK_SIZE 
    block_docs = docs[start:end] 
    build_block(block_docs, i + 1) 
    
print("SPIMI block construction complete")
# -------------------------------------------------------------
# Merging the blocks to make an inverted index
# -------------------------------------------------------------
block_files = [] 
buffers = [] 
pointers = []
heap = []
output_buffer = []

for i in range(1, 11):
    f = open(os.path.join(BLOCKS_DIR, f"block_{i}.txt"), "r", encoding="utf-8")
    block_files.append(f)

    buf = read_chunk(f)
    buffers.append(buf)
    pointers.append(0)

for i in range(10):
    if buffers[i]:
        term, postings = buffers[i][0]
        heapq.heappush(heap, (term, postings, i))

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    while heap:
        term, postings, block_id = heapq.heappop(heap)
        merged = postings[:]

        while heap and heap[0][0] == term:
            _, p2, b2 = heapq.heappop(heap)
            merged = sorted(set(merged).union(p2))

            pointers[b2] += 1
            if pointers[b2] >= len(buffers[b2]):
                buffers[b2] = read_chunk(block_files[b2]) 
                pointers[b2] = 0

            if buffers[b2]:
                t, p = buffers[b2][pointers[b2]]
                heapq.heappush(heap, (t, p, b2))
        
        output_buffer.append(f"{term}:{','.join(merged)}")

        if len(output_buffer) >= MAX_OUTPUT_BUFFER: 
            out.write("\n".join(output_buffer) + "\n") 
            output_buffer = [] 
        
        pointers[block_id] += 1
        if pointers[block_id] >= len(buffers[block_id]):
            buffers[block_id] = read_chunk(block_files[block_id])
            pointers[block_id] = 0
        
        if buffers[block_id]:
            t, p = buffers[block_id][pointers[block_id]]
            heapq.heappush(heap, (t, p, block_id))
    
    if output_buffer:
        out.write("\n".join(output_buffer) + "\n")
print("Merge complete and outputted to " + OUTPUT_FILE)