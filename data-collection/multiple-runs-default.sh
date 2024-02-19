# You should take samples in multiples of eight (or less) for maximum efficiency, as there are 8 cores
#
# -N0 => means there are 0 arguments, otherwise the numbers are added as positional arguments
parallel -N0 ./main.py avg -lor 0.1 ::: {0..7} 
