#!/bin/bash
python runner.py --seed 46 --model wr-8-16 --optim sgdn --bn --aug --bs 200 --wd 0 --momentum 0.9
python runner.py --seed 46 --model wr-8-16 --optim adam --bn --aug --bs 200 --wd 0
python runner.py --seed 47 --model wr-8-16 --optim sgd --bn --aug --bs 200 --wd 0
python runner.py --seed 47 --model wr-8-16 --optim sgdn --bn --aug --bs 200 --wd 0 --momentum 0.9
python runner.py --seed 47 --model wr-8-16 --optim adam --bn --aug --bs 200 --wd 0
python runner.py --seed 48 --model wr-8-16 --optim sgd --bn --aug --bs 200 --wd 0
python runner.py --seed 48 --model wr-8-16 --optim sgdn --bn --aug --bs 200 --wd 0 --momentum 0.9
python runner.py --seed 48 --model wr-8-16 --optim adam --bn --aug --bs 200 --wd 0
python runner.py --seed 49 --model wr-8-16 --optim sgd --bn --aug --bs 200 --wd 0
python runner.py --seed 49 --model wr-8-16 --optim sgdn --bn --aug --bs 200 --wd 0 --momentum 0.9
python runner.py --seed 49 --model wr-8-16 --optim adam --bn --aug --bs 200 --wd 0
python runner.py --seed 50 --model wr-8-16 --optim sgd --bn --aug --bs 200 --wd 0
python runner.py --seed 50 --model wr-8-16 --optim sgdn --bn --aug --bs 200 --wd 0 --momentum 0.9
python runner.py --seed 50 --model wr-8-16 --optim adam --bn --aug --bs 200 --wd 0
python runner.py --seed 51 --model wr-8-16 --optim sgd --bn --aug --bs 200 --wd 0
python runner.py --seed 51 --model wr-8-16 --optim sgdn --bn --aug --bs 200 --wd 0 --momentum 0.9
python runner.py --seed 51 --model wr-8-16 --optim adam --bn --aug --bs 200 --wd 0
