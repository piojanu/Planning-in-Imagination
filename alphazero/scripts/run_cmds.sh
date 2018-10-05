#/bin/bash
CKPT_DIR=checkpoints/test/
MAIN_SCRIPT_PATH=../src/run.py

run_test () {
  echo "Testing $1"
  python $MAIN_SCRIPT_PATH -c test_config.json ${@:1:99} >test.log 2>error.log && echo "$1 succeeded" || { cat test.log && cat error.log && echo "$1 failed" && exit 1; }
}

run_test self_play

run_test cross_play -d $CKPT_DIR

run_test clash --no-render $CKPT_DIR`ls $CKPT_DIR | cut -f 1 | head -n1` $CKPT_DIR`ls $CKPT_DIR | cut -f 1 | tail -n1`

run_test train -ckpt $CKPT_DIR`ls $CKPT_DIR | cut -f 1 | tail -n1`

#TODO: enable it
#run_test hopt -n 20

rm -rf checkpoints
