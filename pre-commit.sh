# pre-commit.sh

export PATH="/home/mellor/anaconda3/bin:$PATH"

git stash -q --keep-index

echo "### Running unit tests to avoid bad commits ###"
cd ~/Dropbox/Programming/Python/eventgraphs/tests
python -m unittest test_eventgraph
RESULT=$?
cd ..

git stash pop -q
if [ $RESULT -ne 0 ]; then
	echo "### Tests failed - Aborting commit ###"
	exit 1
fi

echo "### Tests passed - commiting"
exit 0