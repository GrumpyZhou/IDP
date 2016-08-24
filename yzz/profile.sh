#bin/bash
# Generate profile
python -m cProfile -o main.profile mnist.py

# Convert .profile into .txt 
python out_stats.py > stats.txt

cat stats.txt
rm main.profile
