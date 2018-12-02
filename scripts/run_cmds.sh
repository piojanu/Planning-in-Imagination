#/bin/bash
export OMP_NUM_THREADS=1
CKPT_DIR=checkpoints/test
MAIN_SCRIPT_PATH=../run.py
CONFIG_FILE=test_config.json

run_test () {
  echo "Testing $1"
  python $MAIN_SCRIPT_PATH -c $CONFIG_FILE ${@:1:99} >test_${1}.log 2>error_${1}.log && echo "$1 succeeded" || { cat test_${1}.log && cat error_${1}.log && echo "$1 failed" && exit 1; }
}

run_test iter_train

run_test eval -n 1

rm -rf checkpoints
