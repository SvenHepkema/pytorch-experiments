if [ $# -ne 1 ]; then
	echo "Enter a filename as the first positional argument."
	exit -1
fi

cat $1 | awk -F, '{print $11}' | gnuplot -p -e "plot '-' using (0):1 with boxplot; pause -1"
