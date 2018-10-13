#/bin/bash
CKPT_DIR=checkpoints/test/
MAIN_SCRIPT_PATH=../run.py

run_test () {
  echo "Testing $1"
  python $MAIN_SCRIPT_PATH -c test_config.json ${@:1:99} >test.log 2>error.log && echo "$1 succeeded" || { cat test.log && cat error.log && echo "$1 failed" && exit 1; }
}

