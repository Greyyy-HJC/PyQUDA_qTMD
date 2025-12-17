cd ../../
python -m tests.pyq_vs_gpt.pyquda_main
python -m tests.pyq_vs_gpt.gpt_main

cd tests/pyq_vs_gpt/
python compare_h5.py data/c2pt/S8T8_gpt.c2pt.0.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.h5 data/c2pt/S8T8_pyq.c2pt.0.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.h5 > diff_2pt.txt