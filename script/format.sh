echo "===== Sorting imports ====="

isort --trailing-comma --line-width 120 --multi-line 3 environment/
isort --trailing-comma --line-width 120 --multi-line 3 model/
isort --trailing-comma --line-width 120 --multi-line 3 util/

echo ""
echo "===== Formatting via black ====="

black --line-length 120 environment/
black --line-length 120 model/
black --line-length 120 util/
