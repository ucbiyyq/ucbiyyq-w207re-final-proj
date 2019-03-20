@echo off

pushd ..\..\notebooks

REM dir

jupyter nbconvert --to pdf 1.0-yyq-scratch-slides.ipynb --output ..\reports\test

popd