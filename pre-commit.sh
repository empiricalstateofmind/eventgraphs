# pre-commit.sh
git stash -q --keep-index

echo "### Running unit tests to avoid bad commits ###"
cd ./tests
python -m unittest 
RESULT=$?
cd ..

git stash pop -q
if [ $RESULT -ne 0 ]; then
	echo "### Tests failed - Aborting commit ###"
	exit 1
fi
exit 0