#python training.py --params utils/blackbox_words_attack.yaml > log_file_1700_5_1900 2>&1 &
python training.py --params utils/blackbox_words_attack.yaml > $1 2>&1 &

