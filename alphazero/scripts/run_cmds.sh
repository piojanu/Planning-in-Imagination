#/bin/bash
CKPT_DIR=checkpoints/test/
MAIN_SCRIPT_PATH=../src/run.py

echo "Testing self_play"
echo y | python $MAIN_SCRIPT_PATH -c test_config.json self_play || { echo "self_play Failed" ; exit 1; }

echo "Testing cross_play"
python $MAIN_SCRIPT_PATH -c test_config.json cross_play -d $CKPT_DIR -sc test_config.json || { echo "cross_play Failed" ; exit 1; }

echo "Testing clash"
python $MAIN_SCRIPT_PATH -c test_config.json clash --no-render $CKPT_DIR`ls $CKPT_DIR | cut -f 1 | head -n1` $CKPT_DIR`ls $CKPT_DIR | cut -f 1 | tail -n1` || { echo "clash Failed" ; exit 1; }
