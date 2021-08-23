# cd docs
if [ "$1" == "autogen" ]
then
  PYTHONPATH=../ sphinx-autogen source/*.rst
else
  make clean
  make html
  cp -r build/html/* .
fi
