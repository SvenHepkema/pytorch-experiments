if [ $# -ne 1 ]; then
	echo "Enter a filename as the first positional argument."
	exit -1
fi

cat $1 | awk -F, '{ print $1, $6, $15 }' | gnuplot -p -e \
	"set xlabel 'epochs'; set ylabel 'middle layer size' offset 0,-2; set zlabel 'n correct' offset 4,6 ; set title 'Accuracy with multiple epoch counts and first layer sizes' offset 0,-3 ; unset key; set hidden3d; set dgrid3d 30,30 qnorm 2; splot '-' with lines ; pause -1"
