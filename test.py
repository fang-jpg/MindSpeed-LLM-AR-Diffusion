import re
import statistics
import sys

path = sys.argv[1]

with open(path, encoding="utf-8") as file:
    text = file.read()

lengths = [
    int(value)
    for value in re.findall(r"generated_tokens=(\d+)", text)
]

if not lengths:
    raise RuntimeError("No per-request generated_tokens found")

early = [length for length in lengths if length < 256]
full = [length for length in lengths if length == 256]

print("requests:", len(lengths))
print("average generated tokens:", statistics.mean(lengths))
print("EOS stopped:", len(early))
print("EOS stop rate:", len(early) / len(lengths))
print("generated full 256:", len(full))

if early:
    print("early-stop lengths:", sorted(early)) 