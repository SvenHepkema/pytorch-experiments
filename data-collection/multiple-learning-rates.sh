if [ $# -ne 1 ]; then
	echo "Enter a number of epochs as the first positional argument."
	exit -1
fi

parallel ./main.py avg -ep $1 -lr {} -lor 0.1 ::: 0.1 0.05 0.025 0.01 0.005 0.0025 0.0001
