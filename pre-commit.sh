# pre-commit.sh
git stash -q --keep-index

cd ./tests
python -m unittest 
cd ..

RESULT=$?
git stash pop -q
[ $RESULT -ne 0 ] && exit 1
exit 0