# pre-commit.sh
git stash -q --keep-index

cd ./tests
python -m unittest 
RESULT=$?
cd ..

git stash pop -q
[ $RESULT -ne 0 ] && exit 1
exit 0